"""Depth-optimised arithmetic for the point addition: [106]'s choice, on `ec_cla`.

Every other cell in this package ripples: an n-bit addition has Toffoli depth
~n, so a point addition at n = 256 runs at a depth within ~10% of its Toffoli
count (`hier.block_depth`).  [106] trades the other way: the Draper-Kutin-
Rains-Svore carry-lookahead adder (`ec_cla`) has Toffoli depth ~4 log2 n for
~7n Toffolis and ~2n ancillas.  This module puts it where the depth is:

  CLAArith              exact modular doubling, halving and controlled
                        addition for the Bezout replay: log-depth each
  Dialog(walk_cla=True) the GCD walk's comparison and controlled subtraction
                        (`ec_gcd.Dialog._round`)
  sqr_int_cla           the dedicated squarer with every window subtraction
                        carry-lookahead

The GCD rounds stay sequential -- round i + 1 reads round i's parity and
comparison -- so the depth becomes ~(rounds) x O(log n) instead of
(rounds) x O(n).  The table lookups do not change: a unary-iteration lookup
of 2^w entries is ~2^w deep, the floor of this configuration.
"""

import ec_cla as CL
from ec_gcd import Exact
from ec_sim import Reg


def fanout(m, c, k):
    """k handles on the value of qubit c: c itself and k - 1 copies made by
    CNOTs.  Toffolis sharing a control otherwise run one after another (a
    qubit takes part in one gate at a time); on the copies they run in one
    layer.  Cliffords only: no Toffoli depth, k - 1 ancillas while held."""
    cp = m.anc(max(k - 1, 0), "fan") if k > 1 else []
    for q in cp:
        m.ctx.cx(c, q)
    return [c] + list(cp), cp


def unfan(m, c, cp):
    for q in cp:
        m.ctx.cx(c, q)
    if cp:
        m.free(cp)


def fan_cswap(m, c, a, b):
    """cswap(c, a_i, b_i) for every i, in one Toffoli layer."""
    cs, cp = fanout(m, c, len(a))
    for ci, x, y in zip(cs, a, b):
        m.ctx.cswap(ci, x, y)
    unfan(m, c, cp)


def fan_and(m, c, a, t, undo=False):
    """t_i ^= c AND a_i for every i (onto clean t), in one layer."""
    cs, cp = fanout(m, c, len(a))
    for ci, x, y in zip(cs, a, t):
        (m.ctx.and_dg if undo else m.ctx.and_)(ci, x, y)
    unfan(m, c, cp)


def moddbl_cla(m, x, p):
    """x <- 2x mod p, exactly, at logarithmic depth: the shift is a relabelling,
    [2x >= p] a constant comparator, the subtraction of p (copied under the
    flag) a carry-lookahead subtraction, and the flag is the result's parity
    (2x is even and p odd)."""
    ctx, n = m.ctx, len(x)
    z = m.anc(1, "z")
    xe = Reg([z[0]] + list(x), "2x")                  # n + 1 bits, value 2x
    f = m.anc(1, "f")
    CL.cla_geq_const(m, xe, p, f[0])
    ones = [i for i in range(n + 1) if p >> i & 1]
    cp = m.anc(len(ones), "pc")
    xs = [0] * (n + 1)
    for j, i in enumerate(ones):
        ctx.cx(f[0], cp[j])
        xs[i] = cp[j]
    CL.cla_sub(m, xs, xe)
    for j in range(len(ones)):
        ctx.cx(f[0], cp[j])
    m.free(cp)
    ctx.cx(xe[0], f[0])                               # odd <=> p was subtracted
    m.free(f)
    for i in range(n - 1, -1, -1):                    # result into x, top -> z
        ctx.swap(xe[i], xe[i + 1])
    m.free(z)


def modhalf_cla(m, x, p):
    m.emit_inverse(moddbl_cla, m, x, p)


class CLAArith(Exact):
    """Exact replay arithmetic on carry-lookahead adders (see module doc)."""
    name = "cla"

    def dbl(self, m, reg):
        moddbl_cla(m, reg, self.q)

    def half(self, m, reg):
        modhalf_cla(m, reg, self.q)

    def cadd(self, m, c, a, b):
        if c is None:
            CL.cla_modadd(m, a, b, self.q)
            return
        t = m.anc(len(a), "ca")
        fan_and(m, c, a, t)                           # copy under control, depth 1
        CL.cla_modadd(m, t, b, self.q)
        fan_and(m, c, a, t, undo=True)
        m.free(t)

    def cswap(self, m, c, a, b):
        fan_cswap(m, c, a, b)

    def signadd(self, m, e, a, b):
        raise NotImplementedError("CLAArith serves the dialog replay (cadd, dbl)")


def cla_csub(m, ctrl, x, y):
    """y <- y - ctrl * x (mod 2^n) at logarithmic depth: copy under a fanned-out
    control (one layer of ANDs), carry-lookahead subtraction, free uncopy."""
    t = m.anc(len(x), "cs")
    fan_and(m, ctrl, x, t)
    CL.cla_sub(m, list(t), y)
    fan_and(m, ctrl, x, t, undo=True)
    m.free(t)


def cla_gt_ctrl(m, ctrl, x, y, out, k=None):
    """out ^= ctrl AND [top_k(x) > top_k(y)] at logarithmic depth."""
    ctx = m.ctx
    k = min(k or len(x), len(x))
    tx, ty = Reg(x[len(x) - k:]), Reg(y[len(y) - k:])
    t = m.anc(1, "gt")
    CL.cla_lt(m, ty, tx, t[0])                        # [y < x] = [x > y]
    ctx.ccx(ctrl, t[0], out)
    CL.cla_lt(m, ty, tx, t[0])
    m.free(t)


def sqr_int_cla(m, x, z):
    """`ec_square.sqr_int` with every window subtraction carry-lookahead:
    x^2 into the 2n + 1 clean qubits z, at depth ~n log n instead of ~n^2/2."""
    ctx, n = m.ctx, len(x)
    assert len(z) == 2 * n + 1
    pad = m.anc(1, "pad")
    zero = m.anc(1, "z0")
    CL.cla_sub(m, list(x) + [zero[0]], Reg(z[:n + 1]))
    m.free(zero)
    for i in range(n):
        ctx.cx(z[n + i], z[n + i + 1])
        h = list(x[i + 1:])
        for q in h + [pad[0]]:
            ctx.cx(x[i], q)
        operand = h + [pad[0], x[i]]
        window = Reg(z[2 * i + 1:n + i + 2], "win")
        CL.cla_sub(m, operand, window)
        for q in h + [pad[0]]:
            ctx.cx(x[i], q)
    m.free(pad)


def csub_square_cla(m, ctrl, src, acc, p):
    """acc <- acc - src^2 mod p (ctrl None only), pseudo-Mersenne fold on the
    carry-lookahead cells: the squarer above, then acc -= z_lo and acc -= f z_hi
    by f's bits, doubling a copy of z_hi (`ec_square._csub_times_const`)."""
    import ec_approx as AX
    import ec_square as SQ
    u, f = AX.pseudo_mersenne(p)
    n = len(src)
    assert u == n and ctrl is None
    z = m.anc(2 * n + 1, "z")
    sqr_int_cla(m, src, z)
    lo, hi = Reg(z[:n], "zlo"), Reg(z[n:2 * n], "zhi")

    def sub(mm, c, v, a):
        CL.cla_modsub(mm, v, a, p)

    sub(m, None, lo, acc)
    SQ._csub_times_const(m, None, hi, f, acc, p,
                         lambda mm, v: moddbl_cla(mm, v, p),
                         lambda mm, v: modhalf_cla(mm, v, p), sub)
    m.emit_inverse(sqr_int_cla, m, src, z)
    m.free(z)
