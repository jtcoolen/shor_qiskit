"""Windowed point addition -- [1128] Algorithm 1.

    |i> |R>  ->  |i> |R + P_i>

for a window of 2^w precomputed points.  Windowing is what makes the point
*multiplication* cheap: instead of 2n+2 additions, one per bit of the two
scalars, there are 2*ceil(n/w) of them, one per window.  The lookups that
replace the savings cost 2^w Toffoli each and are an order of magnitude below
the arithmetic, so w is chosen large -- [1128] and [2] both use w = 16.

Two things make this different from `ec_pointadd`.

The multiplications are in-place.  [106] Alg. 3 computes lambda with a division,
uses it, and clears it with a *second* division.  Here `ec_eea`'s in-place
multiplier does the job: step 6 runs it backwards to divide, step 11 runs it
forwards to multiply, and lambda is consumed rather than cleared.  One circuit,
used twice, and one fewer inversion.

i = 0 is handled without a special case.  The window's zeroth entry is the point
at infinity, which the lookup returns as (0, 0).  Steps 10 and 12 are the only
ones controlled by c = [i != 0]; every other step is unconditional, and on the
i = 0 branch the division at step 6 and the multiplication at step 11 are exact
inverses of each other, so the accumulator comes back untouched.  That is a much
cheaper way to add zero than gating the whole circuit.

The same exceptional cases apply as in `ec_pointadd`: x_R = x_Pi at step 6, and
x_Pi = x_3 at step 11.  Both are O(1/p).

The i = 0 branch carries a third, which follows from the (0, 0) encoding of
infinity rather than from the group law: step 6 divides by x_R - x_Pi, which for
P_i = O is x_R itself, so adding zero fails when the accumulator sits on x = 0.
That is one or two points on the curve -- O(1/p) again, and of a piece with the
others -- but it is worth naming, because it is invisible in the algebra: adding
the identity looks like it could never fail.
"""

import ec_adders as A
import ec_eea as E
import ec_modarith as MA
import ec_mult as MU
from qrom import lookup_ui


def _lookup(m, one, addr, out, table):
    """XOR table[addr] into `out`.  Self-inverse, so the same call unloads it."""
    anc = m.anc(max(1, len(addr)), "lk")
    lookup_ui(m.ctx, one, addr, out, table, anc)
    m.free(anc)


def windowed_point_add(m, addr, x2, y2, points, p, iters=None):
    """[1128] Algorithm 1.  `points[i]` is the window entry for address i.

    points[0] must be the point at infinity; it is encoded as (0, 0), which is
    what makes the i = 0 branch fall out for free.
    """
    ctx, n = m.ctx, len(x2)
    w = len(addr)
    assert len(points) == 1 << w, "the window must be full"
    assert points[0].inf, "entry 0 must be the point at infinity"

    xs = [0 if P.inf else P.x for P in points]
    ys = [0 if P.inf else P.y for P in points]
    x3s = [0 if P.inf else (3 * P.x) % p for P in points]

    one = m.anc(1, "one")
    ctx.x(one[0])
    c = m.anc(1, "c")
    MA.is_nonzero(m, addr, c[0])                       # 1

    x1, y1 = m.anc(n, "x1"), m.anc(n, "y1")
    _lookup(m, one[0], addr, x1, xs)                   # 2
    _lookup(m, one[0], addr, y1, ys)
    MA.modsub(m, x1, x2, p)                            # 3
    MA.modsub(m, y1, y2, p)                            # 4
    _lookup(m, one[0], addr, x1, xs)                   # 5
    _lookup(m, one[0], addr, y1, ys)

    E.inplace_div(m, x2, y2, p, iters)                 # 6  y2 <- y2 / x2

    xx = m.anc(n, "3x")
    _lookup(m, one[0], addr, xx, x3s)                  # 7
    MA.modadd(m, xx, x2, p)                            # 8
    _lookup(m, one[0], addr, xx, x3s)                  # 9
    m.free(xx)

    _csub_square(m, c[0], y2, x2, p)                   # 10 if c: x2 -= y2^2
    E.inplace_mul(m, x2, y2, p, iters)                 # 11 y2 <- y2 * x2
    MA.cmodneg(m, c[0], x2, p)                         # 12 if c: x2 <- -x2

    _lookup(m, one[0], addr, x1, xs)                   # 13
    _lookup(m, one[0], addr, y1, ys)
    MA.modsub(m, y1, y2, p)                            # 14
    MA.modadd(m, x1, x2, p)                            # 15
    _lookup(m, one[0], addr, x1, xs)                   # 16
    _lookup(m, one[0], addr, y1, ys)
    m.free(x1, y1)

    MA.is_nonzero(m, addr, c[0])                       # 17
    m.free(c)
    ctx.x(one[0])
    m.free(one)


def _csub_square(m, ctrl, src, acc, p):
    """acc -= src^2 mod p, when ctrl.  Ancillas are plentiful at this point."""
    n = len(src)
    t = m.anc(n, "sq")
    MU.modsqr(m, src, t, p)
    MA.cmodsub(m, ctrl, t, acc, p)
    m.emit_inverse(MU.modsqr, m, src, t, p)
    m.free(t)


def window_points(curve, base, w):
    """The window {[i] base} with entry 0 the point at infinity.

    [1128] Sec 2 adds a fixed offset instead, to dodge infinity entirely; this
    keeps the plain multiples because Algorithm 1 already handles i = 0.
    """
    from ec_classical import O, Point
    pts, acc = [O], O
    for _ in range((1 << w) - 1):
        acc = curve.add(acc, base)
        pts.append(Point(acc.x, acc.y, acc.inf))
    return pts


def n_windowed_additions(n, w):
    """Windowed point additions needed to cover both scalars.

    Delegates to `ec_classical.n_point_additions` so there is one definition
    rather than two: this module and `ec_classical` previously disagreed by one
    window (2*ceil(n/w) against 2*ceil((n+1)/w)), which is exactly the sort of
    off-by-one that survives because both look right in isolation.

    The (n+1) is deliberate: the scalars run over [0, r) with r just above 2^n
    for the standard curves, so the counting register is n+1 bits, not n.
    """
    from ec_classical import n_point_additions
    return n_point_additions(n, w)
