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


def sqr_int(m, x, z):
    """z (2n + 1 clean qubits) <- x^2.  x preserved.  n(n + 3)/2 ANDs.

    z's top qubit is the running sign and comes back to |0>.
    """
    ctx, n = m.ctx, len(x)
    assert len(z) == 2 * n + 1
    pad = m.anc(1, "pad")

    # z <- -x as an (n+1)-bit two's-complement number
    zero = m.anc(1, "z0")
    anc = m.anc(n, "sq")
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
        anc = m.anc(max(len(window) - 1, 1), "sq")
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
