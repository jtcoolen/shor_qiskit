"""[Luo26] Sec 5: a controlled affine point addition on three field registers.

[Luo26] H. Luo et al., arXiv:2607.13816v3, Sec 5 (Figs 14 and 15).
[Gid25b] C. Gidney, "A classical-quantum adder with constant workspace and
         linear gates", arXiv:2507.23079.

`ec_luo.Luo.div` is exact and unitary, and it peaks at four field registers:
x (inside the EEA's bank W2), y, the second bank W1, and the quotient z.
[Luo26] gets to three by *venting* y: once z = y/x sits in the second bank's
place, y is a function of the other registers, so it is measured in the X
basis and the register is reused, clean, as the bank of the backward EEA.  The
measurement leaves a phase (-1)^(b.y) for outcome b; it is cancelled after
the backward EEA by recomputing y into the same register, applying Z^b and
uncomputing:

    forward EEA (A = W1 clean)       X = x^-1, S = Gamma(x), A = 0
    A <- y x^-1                      a modular multiplication
    vent Y                           b, phase (-1)^(b.y), Y = 0
    backward EEA (Y as W1)           X = x, S = 0
    Y <- x A;  Z^b;  Y <- 0          the recomputation cancels the phase
    swap Y, A                        Y = y/x, A = 0

`Luo3.mul` is the mirror image: A <- x y, vent Y, forward EEA with Y as the
bank, recompute y = A x^-1 for Z^b and uncompute it, backward EEA, swap.
Two EEAs and three multiplications each.

The multiplications take p as a classical constant, because the bank that
held it in `Luo.div` now holds z.  [Luo26] App B uses [Gid25b]'s
constant-workspace classical-quantum adder for their modular reductions, for
any p; that adder is not built here.  `arith` chooses instead: `ec_gcd.Exact`
(any odd p; its scratch is ~2n, so it shows the construction, not the width)
or `ec_gcd.PMSpace(p, lean=True)` for a pseudo-Mersenne p such as secp256k1's,
whose cells need ~38 clean qubits -- fewer than the EEA leaves idle while the
multiplications run, so the width stays at the EEA's.

`point_add_ctrl_luo` is [Luo26] Fig. 14 on those two blocks: a classical point
added under a control, with the third register doing the constant loads and
the square.  The measurement-based steps are `ec_mbu.vent` / `ec_mbu.zfix`:
`ec_sim` runs them as the XOR they stand for, `ec_mbu.run_live` performs the
measurement and the Z^b, and `live_coherent` checks the phase cancels.
"""

import itertools

import ec_adders as A
import ec_gcd as G
import ec_luo as L
import ec_mbu as MB
from ec_sim import Reg

_keys = itertools.count()


def _fwd(arith):
    """True for an arithmetic whose inverses are built forwards (`csub`,
    `half`): measurement-based cells (`ec_cqadd.GidneyArith`), which cannot
    be run backwards gate by gate."""
    return getattr(arith, "forward_inverse", False)


def _dirty(arith, *regs, skip=()):
    """The operand registers, lent to `dbl` as dirty qubits when it takes them."""
    if not _fwd(arith):
        return {}
    seen, out = set(skip), []
    for r in regs:
        for q in r:
            if q not in seen:
                seen.add(q)
                out.append(q)
    return {"dirty": out}


def mul_acc(m, arith, a, b, acc):
    """acc <- a b mod p from acc = 0: Horner over a, most significant bit first.
    Each bit of a is copied into a fresh qubit to control its addition, so a
    may be b (a square)."""
    n = len(a)
    kw = _dirty(arith, a, b, skip=acc)
    for i in reversed(range(n)):
        if i != n - 1:
            arith.dbl(m, acc, **kw)
        c = m.anc(1, "mc")
        m.ctx.cx(a[i], c[0])
        arith.cadd(m, c[0], b, acc)                  # acc += c b
        m.ctx.cx(a[i], c[0])
        m.free(c)


def mul_unacc(m, arith, a, b, acc):
    """acc <- 0 from acc = a b: `mul_acc` backwards -- gate by gate for a
    unitary arithmetic, step by step through `csub` and `half` for one whose
    inverses are built forwards."""
    if not _fwd(arith):
        m.emit_inverse(mul_acc, m, arith, a, b, acc)
        return
    n = len(a)
    kw = _dirty(arith, a, b, skip=acc)
    for i in range(n):
        c = m.anc(1, "mc")
        m.ctx.cx(a[i], c[0])
        arith.csub(m, c[0], b, acc)                  # acc -= c b
        m.ctx.cx(a[i], c[0])
        m.free(c)
        if i != n - 1:
            arith.half(m, acc, **kw)


def _quotient(p):
    def f(den, num):
        d = den % p
        return num * pow(d, -1, p) % p if d else 0
    return f


def div3(m, x, y, p, arith, steps=None, lean=True):
    """|x>|y> -> |x>|y / x mod p> for x != 0, on three field registers
    ([Luo26] Fig. 15): the forward EEA with W1 as the clean bank, A = y/x in
    W1's lanes, y vented, the backward EEA with Y as the bank, and y
    recomputed around the Z^b that cancels the vent's phase."""
    ctx, n = m.ctx, len(x)
    bk = L.Banks(m, n, x=x, lean=lean)
    L.eea_forward(m, bk, p, steps=steps)                     # W1 = 0, W2 = x^-1
    xinv, a = L._le(bk, bk.W2, 1, n), L._le(bk, bk.W1, 1, n)
    idle = bk.lt + bk.lq + bk.lr + [bk.P1, bk.P2, bk.Sign] + bk.W2[n:] + bk.W1[n:]
    with L._lent(m, idle):
        mul_acc(m, arith, xinv, y, a)                        # A = y / x
    key = ("luo3", next(_keys))
    MB.vent(m, [xinv, a], y, _quotient(p), key)              # y = A / x^-1
    bk2 = L.Banks([list(y) + bk.W1[n:]] + list(bk[1:]))       # Y is the bank now
    m.emit_inverse(L.eea_forward, m, bk2, p, None, steps, True, True)
    with L._lent(m, bk.meta() + bk.consumed + bk.W1[n:]):
        mul_acc(m, arith, x, a, y)                           # y again ...
        MB.zfix(m, y, key)                                   # ... cancels the phase
        mul_unacc(m, arith, x, a, y)
    for q1, q2 in zip(y, a):
        ctx.swap(q1, q2)
    bk.free(m)


def mul3(m, x, y, p, arith, steps=None, lean=True):
    """|x>|y> -> |x>|x y mod p> for x != 0: `div3`'s mechanism, mirrored --
    A = x y, y vented, the forward EEA with Y as the bank, y = A x^-1
    recomputed around Z^b, the backward EEA."""
    ctx, n = m.ctx, len(x)
    bk = L.Banks(m, n, x=x, lean=lean)
    a = L._le(bk, bk.W1, 1, n)
    with L._lent(m, bk.meta() + bk.consumed + bk.W1[n:]):
        mul_acc(m, arith, x, y, a)                           # A = x y
    key = ("luo3", next(_keys))
    MB.vent(m, [x, a], y, _quotient(p), key)                 # y = A / x
    bk2 = L.Banks([list(y) + bk.W1[n:]] + list(bk[1:]))
    L.eea_forward(m, bk2, p, steps=steps)                    # W2 = x^-1, Y = 0
    xinv = L._le(bk2, bk2.W2, 1, n)
    idle = bk.lt + bk.lq + bk.lr + [bk.P1, bk.P2, bk.Sign] + bk.W2[n:] + bk.W1[n:]
    with L._lent(m, idle):
        mul_acc(m, arith, xinv, a, y)                        # y again ...
        MB.zfix(m, y, key)                                   # ... cancels the phase
        mul_unacc(m, arith, xinv, a, y)
    m.emit_inverse(L.eea_forward, m, bk2, p, None, steps, True, True)
    for q1, q2 in zip(y, a):
        ctx.swap(q1, q2)
    bk.free(m)


class Luo3:
    """In-place division / multiplication on three field registers (the
    `ec_gcd` backend interface: .div(m, x, y, p), .mul(m, x, y, p))."""

    def __init__(self, arith=None, steps=None, lean=True):
        self.arith, self.steps, self.lean = arith, steps, lean

    def _ar(self, p):
        return self.arith or G.Exact(p)

    def div(self, m, x, y, p):
        """|x>|y> -> |x>|y / x mod p> for x != 0 (`div3`)."""
        div3(m, x, y, p, self._ar(p), self.steps, self.lean)

    def mul(self, m, x, y, p):
        """|x>|y> -> |x>|x y mod p> for x != 0 (`mul3`)."""
        mul3(m, x, y, p, self._ar(p), self.steps, self.lean)


# =============================================================================
# [Luo26] Fig. 14: the controlled point addition
# =============================================================================
def _const_op(m, ctrl, reg, k, p, op):
    """reg <- reg +- ctrl * k mod p, for classical k: k is loaded into a
    scratch register by CNOTs from ctrl (the third field register is clean
    at every point this is called), added or subtracted, and unloaded."""
    n = len(reg)
    kr = m.anc(n, "K")
    for i in range(n):
        if (k >> i) & 1:
            m.ctx.cx(ctrl, kr[i])
    op(m, kr, reg, p)
    for i in range(n):
        if (k >> i) & 1:
            m.ctx.cx(ctrl, kr[i])
    m.free(kr)


def _cneg(m, ctrl, x, p):
    """x <- p - x when ctrl, for x in [1, p): complement, then add p + 1 mod
    2^n (NOT x + p + 1 = 2^n + p - x).  One n-qubit constant and one carry,
    so it fits where the third field register is clean.  Wrong only on
    x = 0, which the division before it excludes."""
    n = len(x)
    for q in x:
        m.ctx.cx(ctrl, q)
    kr, cy = m.anc(n, "K"), m.anc(1, "cy")
    k = (p + 1) % (1 << n)
    for i in range(n):
        if (k >> i) & 1:
            m.ctx.cx(ctrl, kr[i])
    A.cdkm_add(m.ctx, kr, x, cy[0])
    for i in range(n):
        if (k >> i) & 1:
            m.ctx.cx(ctrl, kr[i])
    m.free(kr, cy)


def point_add_ctrl_luo(m, ctrl, x, y, x2, y2, p, backend=None):
    """(x, y) <- (x, y) + (x2, y2) when ctrl; unchanged when ctrl = 0.

    [Luo26] Fig. 14 with the in-place division and multiplication of
    `Luo3` (`backend`, default `Luo3()`), whose arithmetic also serves the
    constant additions and the square.  (x2, y2) is classical; the
    exceptional cases of `ec_pointadd` apply, and x must be nonzero when
    ctrl = 0 (the division runs uncontrolled)."""
    be = backend or Luo3()
    ar = be._ar(p)
    add = lambda mm, k, r, pp: ar.add(mm, k, r, pp)          # r += k
    sub = lambda mm, k, r, pp: ar.sub(mm, k, r, pp)          # r -= k
    _const_op(m, ctrl, x, x2 % p, p, sub)                   # x <- x1 - c x2
    _const_op(m, ctrl, y, y2 % p, p, sub)                   # y <- y1 - c y2
    be.div(m, x, y, p)                                       # y <- lambda
    _const_op(m, ctrl, x, 3 * x2 % p, p, add)               # x <- x + c 3 x2
    s = m.anc(len(x), "sq")                                  # the third register:
    mul_acc(m, ar, y, y, s)                                  #   s = lambda^2
    if _fwd(ar):
        ar.csub(m, ctrl, s, x)                               # x <- x - c lambda^2
    else:
        m.emit_inverse(ar.cadd, m, ctrl, s, x)
    mul_unacc(m, ar, y, y, s)                                #   s = 0
    m.free(s)
    be.mul(m, x, y, p)                                       # y <- lambda x
    _cneg(m, ctrl, x, p)                                     # x <- -x when c
    _const_op(m, ctrl, x, x2 % p, p, add)                   # x <- x3
    _const_op(m, ctrl, y, y2 % p, p, sub)                   # y <- y3


def square_sub(arith):
    """`PointAddCfg.square` on one field register: s = src^2 mod p by the
    Horner multiplier, acc -= (ctrl) s, s uncomputed -- n qubits where the
    integer square of `ec_square` holds 2n + 1."""
    def sq(m, ctrl, src, acc, p):
        s = m.anc(len(src), "sq")
        mul_acc(m, arith, src, src, s)
        if ctrl is None:
            arith.sub(m, s, acc, p)
        elif _fwd(arith):
            arith.csub(m, ctrl, s, acc)
        else:
            m.emit_inverse(arith.cadd, m, ctrl, s, acc)
        mul_unacc(m, arith, src, src, s)
        m.free(s)
    return sq


def neg_lean(m, ctrl, x, p):
    """x <- p - x (when ctrl), x in [1, p): `_cneg` with ctrl None allowed.
    One n-qubit constant and one carry; for `PointAddCfg.neg` and the sign
    negations of `ec_signedwin.signed_windows`."""
    if ctrl is not None:
        _cneg(m, ctrl, x, p)
        return
    n = len(x)
    for q in x:
        m.ctx.x(q)
    kr, cy = m.anc(n, "K"), m.anc(1, "cy")
    A.encode_const(m.ctx, kr, (p + 1) % (1 << n))
    A.cdkm_add(m.ctx, kr, x, cy[0])
    A.encode_const(m.ctx, kr, (p + 1) % (1 << n))
    m.free(kr, cy)


def windowed_cfg(p, arith=None, steps=None):
    """`ec_window.PointAddCfg` for [Luo26] Sec 6.4's windowed addition on three
    field registers: masked tables, the looked-up point loaded one coordinate
    at a time into the one free register (their five lookups: `serial_load`),
    `Luo3` for the division and multiplication, and `arith` (default
    `ec_gcd.Exact(p)`) for everything else.  Use it with signed windows and
    `neg_lean` for the sign."""
    import ec_window as W
    ar = arith or G.Exact(p)
    return W.PointAddCfg(lookup="mbu", serial_load=True, offsets=True, add=ar,
                         mul=Luo3(arith=ar, steps=steps), square=square_sub(ar),
                         neg=neg_lean)


def point_add_qubits(n):
    """[Luo26] Sec 6.1: 3n + 6 floor(log2 n) + 19 (their count, no control)."""
    return 3 * n + 6 * (n.bit_length() - 1) + 19


_ = Reg
