"""In-place controlled point addition in affine coordinates.

[106] Algorithm 3, which is [HJN+20] Fig. 9 with the two modifications the
paper describes, on top of [RNSL17].  This is the workhorse of Shor's ECDLP:
one of these per bit of the two scalars (or per window, once windowed).

The accumulator (x1, y1) is quantum; the addend (x2, y2) is a classical point.
When the control q is 0 the accumulator must come back untouched, and when it
is 1 it must hold the sum -- in place, with no garbage.  That "in place, no
garbage" is why affine coordinates are used despite needing an inversion:
projective addition is cheaper but cannot erase its input (see `ec_proj`).

Why the eleven steps work
-------------------------
The two divisions are the same circuit, and the second one *clears* lambda
rather than computing it.  That is not a coincidence: by step 8 the registers
hold x = x2 - x3 and y = lambda (x2 - x3), so y/x is again lambda, and the
XOR-accumulating division zeroes the register it filled at step 3.  The whole
algorithm is arranged around making that identity true.

Likewise step 4: y holds y1 - y2 and x*lambda is by definition y1 - y2, so a
bitwise XOR clears y -- no modular subtraction needed.

[106]'s second modification is step 9.  [HJN+20] did x <- x - 2*x2 followed by
a controlled x <- x + x2; this merges them into one controlled subtraction of
(2 - q) x2.  Preparing that operand is free here: 2*x2 mod p and x2 mod p are
both classical, so the register is X-gated to one of them and CNOTs from q
patch the differing bits.

Exceptional cases
-----------------
Two, one per division -- see `ec_classical.point_add_exceptional`:

  x1 == x2   step 3 divides by x1 - x2.  Doubling, adding an inverse, or the
             point at infinity.  This is the case both papers name.
  x3 == x2   step 8 divides by x2 - x3, i.e. P1 = -2 P2.  Easy to overlook,
             since it is invisible in the addition formulas and only shows up
             in what the registers hold; it is inherent to the construction and
             [1128]'s in-place-multiplier variant hits the same wall.

Both are O(1/p) for a random accumulator, which is exactly the failure rate
Shor's algorithm is built to tolerate.  The circuit does not detect them -- a
check would cost more than it saves -- so on an excluded input it returns
garbage and leaves ancillas dirty.  The tests exclude them explicitly and
report how many, rather than quietly passing.
"""

import ec_adders as A
import ec_kaliski as K
import ec_modarith as MA
import ec_mult as MU


def _csub_2x2_or_x2(m, q, x, x2, p):
    """[106] Alg. 3 step 9: x <- x - ((2-q) * x2 mod p), one subtraction.

    Both operands are classical, so building (2-q)x2 costs only CNOTs: encode
    2*x2 mod p with X gates, then flip the bits where it differs from x2 mod p,
    controlled by q.
    """
    ctx, n = m.ctx, len(x)
    c0, c1 = (2 * x2) % p, x2 % p
    diff = c0 ^ c1
    creg = m.anc(n, "c9")
    A.encode_const(ctx, creg, c0)
    for i in range(n):
        if (diff >> i) & 1:
            ctx.cx(q, creg[i])
    MA.modsub(m, creg, x, p)
    for i in range(n):
        if (diff >> i) & 1:
            ctx.cx(q, creg[i])
    A.encode_const(ctx, creg, c0)
    m.free(creg)


def point_add_ctrl(m, q, x1, y1, x2, y2, p):
    """(x1, y1) <- (x1, y1) + (x2, y2) when q; unchanged when q = 0.

    In place, no garbage.  x2, y2 are classical.
    """
    n = len(x1)
    lam = m.anc(n, "lam")

    MA.modsub_const(m, x1, x2, p)                 # 1  x <- x1 - x2
    MA.cmodsub_const(m, q, y1, y2, p)             # 2  y <- y1 - q y2
    K.mod_div(m, q, x1, y1, lam, p)               # 3  lambda <- q y/x
    MU.modmul_xor(m, x1, lam, y1, p)              # 4  y <- y XOR x lambda  (= 0)
    MA.modadd_const(m, x1, 3 * x2 % p, p)         # 5  x <- x + 3 x2
    MU.modsqr_sub(m, lam, x1, p)                  # 6  x <- x - lambda^2
    MU.modmul_add(m, x1, lam, y1, p)              # 7  y <- y + x lambda
    K.mod_div(m, q, x1, y1, lam, p)               # 8  lambda <- 0
    _csub_2x2_or_x2(m, q, x1, x2, p)              # 9  x <- -x3
    MA.cmodsub_const(m, q, y1, y2, p)             # 10 y <- y3
    MA.cmodneg(m, q, x1, p)                       # 11 x <- x3

    m.free(lam)


def point_add_ctrl_inv(m, q, x1, y1, x2, y2, p):
    """The subtraction of the classical point: `point_add_ctrl` run backwards."""
    m.emit_inverse(point_add_ctrl, m, q, x1, y1, x2, y2, p)


# =============================================================================
# The same addition on [106]'s optimised parts, in Montgomery form
# =============================================================================
def _mont_w(n, w=None):
    """A word size dividing n (the carry-save Montgomery multiplier needs one)."""
    if w:
        assert n % w == 0
        return w
    return next(k for k in (4, 3, 2, 1) if n % k == 0)


def _mont_div(m, ctrl, x, y, out, p, w, fanout):
    """out ^= ctrl * (y / x), everything in Montgomery form: the unconditional
    Kaliski inversion of [106] Sec 3.3 (which lands directly on the Montgomery
    form of x^-1) and the carry-save Montgomery multiplier of Sec 3.2."""
    import ec_kaliski_opt as KO
    import ec_montgomery as MG
    n = len(x)
    inv, lam = m.anc(n, "inv"), m.anc(n, "lam")
    start = len(m.qc.data)
    KO.mod_inv_mont_clean(m, x, inv, p, fanout, ctrl)   # inv = ctrl X^-1 2^n
    MG.mont_mul_clean(m, y, inv, lam, p, w)             # lam = ctrl (Y/X) 2^n
    body = list(m.qc.data[start:])
    for i in range(n):
        m.ctx.cx(lam[i], out[i])
    for ci in reversed(body):
        m.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)
    m.free(inv, lam)


def _mont_acc(m, a, b, acc, p, w, op):
    """acc <- op(acc, a b 2^-n): compute the Montgomery product, use, undo."""
    import ec_montgomery as MG
    n = len(a)
    t = m.anc(n, "mp")
    MG.mont_mul_clean(m, a, b, t, p, w)
    op(t, acc)
    m.emit_inverse(MG.mont_mul_clean, m, a, b, t, p, w)
    m.free(t)


def point_add_ctrl_mont(m, q, x1, y1, x2, y2, p, w=None, fanout=4):
    """`point_add_ctrl` with the accumulator in Montgomery form (X 2^n mod p).

    Every step of [106] Alg. 3 other than the division and the products is
    linear, so it is unchanged with the classical constants converted once;
    the division uses the unconditional, postponed-reduction Kaliski
    inversion (`ec_kaliski_opt`), which returns the Montgomery form of the
    inverse directly, and the products use the carry-save Montgomery
    multiplier (`ec_montgomery`).  Those were built and measured on their own;
    this is where they meet.
    """
    import ec_mult as MU
    from ec_classical import to_mont
    n = len(x1)
    w = _mont_w(n, w)
    X2, Y2 = to_mont(x2, p, n), to_mont(y2, p, n)
    lam = m.anc(n, "lam")

    MA.modsub_const(m, x1, X2, p)                                     # 1
    MA.cmodsub_const(m, q, y1, Y2, p)                                 # 2
    _mont_div(m, q, x1, y1, lam, p, w, fanout)                        # 3
    _mont_acc(m, x1, lam, y1, p, w, lambda t, a: MU.copy_reg(m.ctx, t, a))  # 4
    MA.modadd_const(m, x1, 3 * X2 % p, p)                             # 5
    lc = m.anc(n, "lc")
    MU.copy_reg(m.ctx, lam, lc)
    _mont_acc(m, lc, lam, x1, p, w, lambda t, a: MA.modsub(m, t, a, p))     # 6
    MU.copy_reg(m.ctx, lam, lc)
    m.free(lc)
    _mont_acc(m, x1, lam, y1, p, w, lambda t, a: MA.modadd(m, t, a, p))     # 7
    _mont_div(m, q, x1, y1, lam, p, w, fanout)                        # 8
    _csub_2x2_or_x2(m, q, x1, X2, p)                                  # 9
    MA.cmodsub_const(m, q, y1, Y2, p)                                 # 10
    MA.cmodneg(m, q, x1, p)                                           # 11
    m.free(lam)
