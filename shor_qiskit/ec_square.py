"""Squaring for step 10 of the point addition: x2 <- x2 - lambda^2 mod p.

`ec_window._csub_square` computes lambda^2 mod p with the *general* schoolbook
multiplier (n modular doublings and n controlled modular additions), subtracts
it, and runs the multiplier again to uncompute: about 12 n^2 Toffolis, 17% of a
point addition at n = 256.  Two things make it much cheaper.

A dedicated integer squarer (IonQ Sec VII.A).  With h_i = x >> (i+1),

    x^2 = -x + sum_i (2 x_i - 1) 2^(2i+1) (h_i + x_i),

and each term is an *unconditional* signed subtraction: complement h_i on x_i
(CNOTs), read the result -- padded with two copies of x_i as its sign -- as a
signed (n - i + 1)-bit number, and subtract it at offset 2i + 1.  The running
sum after step i fits in n + i + 2 signed bits (measured exhaustively in the
tests), so each subtraction only touches a window of n + 1 - i bits, entered by
one sign-extension CNOT.  Total n(n + 3)/2 ANDs, against ~n^2 for summing the
partial products and ~6 n^2 for the modular multiplier.

A pseudo-Mersenne fold.  For p = 2^n - f, 2^n == f, so with z = x^2 split as
z_hi 2^n + z_lo,   x^2 == z_lo + f z_hi   (mod p).  Subtract z_lo, then
f z_hi by walking f's bits: a copy of z_hi is doubled bit by bit (cheap, Alg 7)
and subtracted where f has a 1.  For secp256k1, f = 2^32 + 977: 33 doublings
and 7 subtractions.  The fold's inputs z_lo, z_hi lie in [0, 2^n), not [0, p);
they reach p with probability f/2^n, which is the approximation (the same one
every pseudo-Mersenne circuit here makes).
"""

import ec_adders as A
import ec_approx as AX
import ec_modarith as MA
from ec_sim import Reg


def sqr_int(m, x, z, space=False):
    """z (2n + 1 clean qubits) <- x^2.  x preserved.  n(n + 3)/2 ANDs.

    z's top qubit is the running sign and comes back to |0>.  space=True
    gives every subtraction one ancilla (CDKM, ~2x the Toffolis) instead of
    up to n (Gidney); an integer is an ancilla budget: Gidney for the
    subtractions whose carries fit in it, CDKM for the others.
    """
    ctx, n = m.ctx, len(x)
    assert len(z) == 2 * n + 1
    pad = m.anc(1, "pad")

    def width(need):
        if space is False:
            return need
        budget = 1 if space is True else space
        return need if need <= budget else 1

    # z <- -x as an (n+1)-bit two's-complement number
    zero = m.anc(1, "z0")
    anc = m.anc(width(n), "sq")
    A.sub(ctx, Reg(list(x) + list(zero)), Reg(z[:n + 1]), anc)
    m.free(anc, zero)

    for i in range(n):
        ctx.cx(z[n + i], z[n + i + 1])              # sign-extend by one bit
        h = list(x[i + 1:])
        for q in h + [pad[0]]:
            ctx.cx(x[i], q)                         # x_i ? NOT(0||h) : (0||h)
        operand = Reg(h + [pad[0], x[i]], "op")     # signed, sign = x_i
        window = Reg(z[2 * i + 1:n + i + 2], "win")
        assert len(operand) == len(window) == n + 1 - i
        anc = m.anc(width(max(len(window) - 1, 1)), "sq")
        A.sub(ctx, operand, window, anc)            # z -= signed operand
        m.free(anc)
        for q in h + [pad[0]]:
            ctx.cx(x[i], q)
    m.free(pad)


def sqr_int_cost(n):
    return n * (n + 3) // 2


# --- the pseudo-Mersenne fold ------------------------------------------------
def _csub_times_const(m, ctrl, v, c, acc, q, dbl, half, sub):
    """acc <- acc - c v mod q (when ctrl), for a small classical c.

    v is doubled in place bit by bit and subtracted where c has a 1, then
    halved back: bitlen(c) - 1 doublings and halvings, popcount(c)
    subtractions."""
    bits = [(c >> b) & 1 for b in range(c.bit_length())]
    for b, bit in enumerate(bits):
        if bit:
            sub(m, ctrl, v, acc)
        if b < len(bits) - 1:
            dbl(m, v)
    for b in range(len(bits) - 1):
        half(m, v)


def csub_square_pm(m, ctrl, src, acc, q, lsbs=None, msbs=None):
    """acc <- acc - src^2 mod q (when ctrl), q = 2^n - f pseudo-Mersenne.

    Signature of `ec_window._csub_square`, so it drops into PointAddCfg.square.
    """
    pm = AX.pseudo_mersenne(q)
    assert pm, f"{q} is not pseudo-Mersenne"
    u, f = pm
    n = len(src)
    assert u == n
    z = m.anc(2 * n + 1, "z")
    sqr_int(m, src, z)
    lo, hi = Reg(z[:n], "zlo"), Reg(z[n:2 * n], "zhi")

    def sub(mm, c, v, a):
        if c is None:
            MA.modsub(mm, v, a, q)
        else:
            MA.cmodsub(mm, c, v, a, q)

    def dbl(mm, v):
        AX.moddbl_pm(mm, v, q, lsbs)

    def half(mm, v):
        AX.modhalf_pm(mm, v, q, lsbs)

    sub(m, ctrl, lo, acc)                           # acc -= z_lo
    _csub_times_const(m, ctrl, hi, f, acc, q, dbl, half, sub)   # acc -= f z_hi
    m.emit_inverse(sqr_int, m, src, z)
    m.free(z)


def csub_square_generic(m, ctrl, src, acc, p):
    """acc <- acc - src^2 mod p (when ctrl), any odd p: the dedicated squarer
    and a fold by c = 2^n mod p.  Exact: z_lo and z_hi are first reduced into
    [0, p) (one conditional subtraction each; they are below 2^n < 2p), and the
    reduction flags are uncomputed with the square.  For generic p, c has ~n
    bits, so the fold is O(n^2) and the gain over the general multiplier is
    smaller than for pseudo-Mersenne p."""
    n = len(src)
    c = (1 << n) % p
    z = m.anc(2 * n + 1, "z")
    fl = m.anc(2, "zfl")

    def compute():
        sqr_int(m, src, z)
        for j, part in enumerate((Reg(z[:n]), Reg(z[n:2 * n]))):
            cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
            A.geq_const(m.ctx, part, p, fl[j], cp, sc)     # fl = [part >= p]
            A.csub_const(m.ctx, fl[j], part, p, cp, sc)    # part -= p
            m.free(cp, sc)

    compute()
    lo, hi = Reg(z[:n], "zlo"), Reg(z[n:2 * n], "zhi")

    def sub(mm, cc, v, a):
        if cc is None:
            MA.modsub(mm, v, a, p)
        else:
            MA.cmodsub(mm, cc, v, a, p)
    sub(m, ctrl, lo, acc)
    _csub_times_const(m, ctrl, hi, c, acc, p,
                      lambda mm, v: MA.moddbl(mm, v, p),
                      lambda mm, v: MA.modhalf(mm, v, p), sub)
    m.emit_inverse(compute)
    m.free(fl, z)


# --- ECDSA.Fail Sec 5.3.4: one Karatsuba split, applied square by square ------
def _apply_shifted(m, ctrl, v, s, acc, q, sign, dbl, half):
    """acc <- acc + sign * v * 2^s (mod q), q = 2^n - f, v a quantum integer.

    v 2^s splits at bit n into lo (v's low n - s bits above s zeros) and hi
    (the rest, worth hi * 2^n == hi * f): one modular add of lo, and f * hi by
    the doubling chain.  No gates for the shift -- it is a relabelling."""
    u, f = AX.pseudo_mersenne(q)
    n = len(acc)
    add = (lambda mm, c, x, a: (MA.modadd(mm, x, a, q) if c is None else MA.cmodadd(mm, c, x, a, q))) \
        if sign > 0 else \
        (lambda mm, c, x, a: (MA.modsub(mm, x, a, q) if c is None else MA.cmodsub(mm, c, x, a, q)))
    lo_bits = list(v[:max(0, n - s)])
    zeros = m.anc(s + (n - s - len(lo_bits)), "sh") if s or len(lo_bits) < n - s else []
    lo = Reg(list(zeros[:s]) + lo_bits + list(zeros[s:]), "lo")
    add(m, ctrl, lo, acc)
    hi_bits = list(v[n - s:]) if len(v) + s > n else []
    if hi_bits:
        _add_times_small(m, ctrl, hi_bits, f, acc, q, add, dbl, half)
    if len(zeros):
        m.free(zeros)


def _add_times_small(m, ctrl, bits, c, acc, q, add, dbl, half):
    """acc <- acc (+/-) c * v for a short quantum integer v (its `bits`).

    When v 2^b fits in n bits for every set bit b of c, c v is a sum of
    *shifted copies* -- relabellings -- and costs popcount(c) modular
    additions; otherwise fall back to the doubling chain."""
    n = len(acc)
    if len(bits) + c.bit_length() - 1 <= n:
        for b in range(c.bit_length()):
            if (c >> b) & 1:
                z = m.anc(n - len(bits), "sh")
                reg = Reg(list(z[:b]) + list(bits) + list(z[b:]), "shift")
                add(m, ctrl, reg, acc)
                m.free(z)
        return
    padq = m.anc(n - len(bits), "hp")
    _csub_times_const(m, ctrl, Reg(list(bits) + list(padq), "hi"), c, acc, q, dbl, half, add)
    m.free(padq)


def csub_square_karatsuba_pm(m, ctrl, src, acc, q, lsbs=None, msbs=None):
    """acc <- acc - src^2 mod q, q = 2^n - f, by one Karatsuba split.

    src = L + 2^h H.  With A = L^2, B = H^2, C = (L + H)^2 and c2 = 2^(2h) mod q,
        -src^2 == (1 - 2^h) A  - 2^h C  + (2^h - c2) B      (mod q)
    and each square is built, applied straight into acc, and unbuilt in turn,
    so no 2n-bit product ever exists (ECDSA.Fail's order: C, then A, then B).
    Three squares of ~n/2 bits against one of n: 3/4 of the squaring ANDs."""
    pm = AX.pseudo_mersenne(q)
    assert pm, f"{q} is not pseudo-Mersenne"
    n = len(src)
    h = (n + 1) // 2
    L, H = Reg(list(src[:h]), "L"), Reg(list(src[h:]), "H")
    c2 = pow(2, 2 * h, q)
    dbl = lambda mm, v: AX.moddbl_pm(mm, v, q, lsbs)
    half = lambda mm, v: AX.modhalf_pm(mm, v, q, lsbs)

    def with_square(x, body):
        z = m.anc(2 * len(x) + 1, "kz")
        sqr_int(m, x, z)
        body(Reg(list(z[:2 * len(x)]), "sq"))
        m.emit_inverse(sqr_int, m, x, z)
        m.free(z)

    # C = (L + H)^2:  acc -= 2^h C
    sreg = m.anc(h + 1, "LH")
    for a, b in zip(L, sreg):
        m.ctx.cx(a, b)
    zpad = m.anc(h + 1 - len(H), "hz")
    sc = m.anc(h, "ad")
    A.add(m.ctx, Reg(list(H) + list(zpad)), sreg, sc)
    m.free(sc)
    with_square(sreg, lambda C: _apply_shifted(m, ctrl, C, h, acc, q, -1, dbl, half))
    sc = m.anc(h, "ad")
    A.sub(m.ctx, Reg(list(H) + list(zpad)), sreg, sc)
    m.free(sc, zpad)
    for a, b in zip(L, sreg):
        m.ctx.cx(a, b)
    m.free(sreg)

    # A = L^2:  acc += 2^h A - A
    def appA(Asq):
        _apply_shifted(m, ctrl, Asq, h, acc, q, +1, dbl, half)
        _apply_shifted(m, ctrl, Asq, 0, acc, q, -1, dbl, half)
    with_square(L, appA)

    # B = H^2:  acc += 2^h B - c2 B
    def appB(Bsq):
        _apply_shifted(m, ctrl, Bsq, h, acc, q, +1, dbl, half)
        sub = lambda mm, c, x, a: (MA.modsub(mm, x, a, q) if c is None else MA.cmodsub(mm, c, x, a, q))
        _add_times_small(m, ctrl, list(Bsq), c2, acc, q, sub, dbl, half)
    with_square(H, appB)
