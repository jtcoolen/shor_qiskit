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

Backend selection
-----------------
`multiplier` may be `reference`, `montgomery-qcsa`, or `montgomery-lookup`;
`inversion` may be `kaliski`, `kaliski-opt`, `divsteps`,
`divsteps-decoded`, or `divsteps-arithmetic`. Montgomery multiplication
converts the ordinary field residues at its boundary, so the point-adder
formula remains unchanged. The optimized inverse likewise converts its
Montgomery-scaled result back to an ordinary residue.
The toy-width `divsteps` inverter stores all branch decisions;
`divsteps-decoded` reuses its decision qubits via an exact toy-state lookup;
`divsteps-arithmetic` uses the signed coefficients and arithmetic bit-window
decoder of Gu Appendix A. None is Gu's packed-Euclidean circuit.
"""

import ec_adders as A
import ec_divsteps as D
import ec_kaliski as K
import ec_kaliski_opt as KO
import ec_modarith as MA
import ec_montgomery as MG
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


def _mont_word_size(n, requested):
    if requested is not None:
        if requested < 1 or n % requested:
            raise ValueError(f"Montgomery word size {requested} must divide {n}")
        return requested
    return next(w for w in range(min(16, n), 0, -1) if n % w == 0)


def _modmul(m, x, y, out, p, multiplier, word_size):
    if multiplier == "reference":
        MU.modmul(m, x, y, out, p)
        return
    if multiplier not in ("montgomery-qcsa", "montgomery-lookup"):
        raise ValueError(f"unknown multiplier backend {multiplier!r}")

    n = len(x)
    w = _mont_word_size(n, word_size)
    R = pow(2, n, p)
    x_mont, y_mont = m.anc(n, "xm"), m.anc(n, "ym")
    product_mont = m.anc(n, "pm")
    kind = "qcsa" if multiplier == "montgomery-qcsa" else "lookup"
    MU.modmul_const(m, x, R, x_mont, p)
    MU.modmul_const(m, y, R, y_mont, p)
    MG.mont_mul_clean(m, x_mont, y_mont, product_mont, p, w, kind=kind)
    one = m.anc(n, "one")
    m.ctx.x(one[0])
    MG.mont_mul_clean(m, product_mont, one, out, p, w, kind=kind)
    m.ctx.x(one[0])
    m.emit_inverse(
        MG.mont_mul_clean, m, x_mont, y_mont, product_mont, p, w,
        kind)
    m.emit_inverse(MU.modmul_const, m, y, R, y_mont, p)
    m.emit_inverse(MU.modmul_const, m, x, R, x_mont, p)
    m.free(x_mont, y_mont, product_mont, one)


def _modmul_xor(m, x, y, acc, p, multiplier, word_size):
    product = m.anc(len(x), "prod")
    _modmul(m, x, y, product, p, multiplier, word_size)
    for a, b in zip(product, acc):
        m.ctx.cx(a, b)
    m.emit_inverse(_modmul, m, x, y, product, p, multiplier, word_size)
    m.free(product)


def _modsqr_sub(m, x, acc, p, multiplier, word_size):
    n = len(x)
    copy = m.anc(n, "sq")
    product = m.anc(n, "sqprod")
    for a, b in zip(x, copy):
        m.ctx.cx(a, b)
    _modmul(m, copy, x, product, p, multiplier, word_size)
    MA.modsub(m, product, acc, p)
    m.emit_inverse(_modmul, m, copy, x, product, p, multiplier, word_size)
    for a, b in zip(x, copy):
        m.ctx.cx(a, b)
    m.free(copy, product)


def _modmul_add(m, x, y, acc, p, multiplier, word_size):
    product = m.anc(len(x), "prod")
    _modmul(m, x, y, product, p, multiplier, word_size)
    MA.modadd(m, product, acc, p)
    m.emit_inverse(_modmul, m, x, y, product, p, multiplier, word_size)
    m.free(product)


def _moddiv(m, q, x, y, out, p, multiplier, word_size, inversion):
    n = len(x)
    if inversion not in ("kaliski", "kaliski-opt", "divsteps",
                         "divsteps-decoded", "divsteps-arithmetic"):
        raise ValueError(f"unknown inversion backend {inversion!r}")

    def build_inverse(machine, src, dst, modulus, ctrl):
        if inversion in ("divsteps", "divsteps-decoded", "divsteps-arithmetic"):
            D.mod_inv(machine, src, dst, modulus, ctrl,
                      history={"divsteps": "full",
                               "divsteps-decoded": "decoded",
                               "divsteps-arithmetic": "arithmetic"}[inversion])
            return []
        if inversion == "kaliski":
            recs, counter = K.mod_inv(machine, src, dst, modulus, ctrl=ctrl)
            garbage = [bit for record in recs for bit in record]
            if counter is not None:
                garbage.append(counter)
            return garbage

        raw = machine.anc(n, "inv_mont")
        KO.mod_inv_mont_clean(machine, src, raw, modulus, ctrl=ctrl)
        correction = pow(pow(2, 2 * n, modulus), -1, modulus)
        if correction == 1:
            for a, b in zip(raw, dst):
                machine.ctx.cx(a, b)
        else:
            MU.modmul_const(machine, raw, correction, dst, modulus)
        return [raw]

    def multiply(machine, a, b, dst, modulus):
        _modmul(machine, a, b, dst, modulus, multiplier, word_size)

    K.mod_div(m, q, x, y, out, p, build_inverse, multiply)


def point_add_ctrl(m, q, x1, y1, x2, y2, p, multiplier="reference",
                   inversion="kaliski", word_size=None):
    """(x1, y1) <- (x1, y1) + (x2, y2) when q; unchanged when q = 0.

    In place, no garbage.  x2, y2 are classical.
    """
    n = len(x1)
    lam = m.anc(n, "lam")

    MA.modsub_const(m, x1, x2, p)                 # 1  x <- x1 - x2
    MA.cmodsub_const(m, q, y1, y2, p)             # 2  y <- y1 - q y2
    _moddiv(m, q, x1, y1, lam, p, multiplier, word_size, inversion)  # 3
    _modmul_xor(m, x1, lam, y1, p, multiplier, word_size)             # 4
    MA.modadd_const(m, x1, 3 * x2 % p, p)         # 5  x <- x + 3 x2
    _modsqr_sub(m, lam, x1, p, multiplier, word_size)                  # 6
    _modmul_add(m, x1, lam, y1, p, multiplier, word_size)              # 7
    _moddiv(m, q, x1, y1, lam, p, multiplier, word_size, inversion)   # 8
    _csub_2x2_or_x2(m, q, x1, x2, p)              # 9  x <- -x3
    MA.cmodsub_const(m, q, y1, y2, p)             # 10 y <- y3
    MA.cmodneg(m, q, x1, p)                       # 11 x <- x3

    m.free(lam)


def point_add_ctrl_inv(m, q, x1, y1, x2, y2, p, multiplier="reference",
                       inversion="kaliski", word_size=None):
    """The subtraction of the classical point: `point_add_ctrl` run backwards."""
    m.emit_inverse(
        point_add_ctrl, m, q, x1, y1, x2, y2, p, multiplier, inversion,
        word_size)
