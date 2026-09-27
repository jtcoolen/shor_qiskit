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


# =============================================================================
# The same addition with the 2026 refinements, each switchable
# =============================================================================
from dataclasses import dataclass


@dataclass(frozen=True)
class PointAddCfg:
    """Which refinements `windowed_point_add_cfg` applies.

    The default reproduces `windowed_point_add` op for op; every field moves
    one step towards the IonQ / ECDSA.Fail constructions, and each is ablated
    separately in bench/.

    lookup      "recompute": unary iteration with Toffolis on an always-true
                control, unloaded by running it again -- ten lookups of
                2(L-1) Toffolis.  "mbu": temporary-AND loads (L-2 ANDs,
                uncontrolled) and measurement-based unloads, all repaired by
                ONE merged phase lookup per addition (IonQ Alg 6).
    merge_xy    load (x, y) as one 2n-bit word: one unary iteration instead
                of two ([1128] Sec 2 -- the data load is only CNOTs).
    offsets     the table never contains O (IonQ Alg 4 masks), so the
                c = [i != 0] flag, and the controls it puts on the square and
                the negation, disappear -- and so does the x_R = 0 failure.
    free_xy1    return x1/y1 to the pool between steps 5 and 13, so the
                division and multiplication can use them (-2n qubits, free).
    serial_load one n-qubit register, loaded with x, then y, around each use
                (ECDSA.Fail Era III): -n more qubits, one more load per use.
    mul         None: ec_eea's exact dialog.  Otherwise an object with
                .mul(m, x, y, p) and .div(m, x, y, p)  (ec_gcd backends).
    square      "general": modsqr, controlled subtract, modsqr undone.  Or a
                callable square_sub(m, ctrl, src, acc, p).
    neg         "exact": ec_modarith.cmodneg.  Or a callable neg(m, ctrl, x, p).
    add         None: exact modadd/modsub.  Or an object with
                .add(m, x, y, p) and .sub(m, x, y, p)  (y <- y +- x).
    """
    lookup: str = "recompute"
    merge_xy: bool = False
    offsets: bool = False
    free_xy1: bool = False
    serial_load: bool = False
    mul: object = None
    square: object = "general"
    neg: object = "exact"
    add: object = None
    iters: object = None


IONQ_LOOKUPS = PointAddCfg(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True)


def windowed_point_add_cfg(m, addr, x2, y2, points, p, cfg=PointAddCfg()):
    """[1128] Algorithm 1, with the refinements `cfg` selects.

    `points[i]` is the window entry for address i.  Without `cfg.offsets`,
    points[0] must be O (encoded (0, 0)); with it, no entry may be O.
    """
    import ec_mbu as MB
    from ec_sim import Reg

    ctx, n = m.ctx, len(x2)
    w = len(addr)
    assert len(points) == 1 << w, "the window must be full"
    if cfg.offsets:
        assert not any(P.inf for P in points), "masked tables never contain O"
    else:
        assert points[0].inf, "entry 0 must be the point at infinity"
    assert not (cfg.merge_xy and cfg.serial_load), "merge_xy and serial_load conflict"

    xs = [0 if P.inf else P.x for P in points]
    ys = [0 if P.inf else P.y for P in points]
    x3s = [0 if P.inf else (3 * P.x) % p for P in points]
    xys = [x | (y << n) for x, y in zip(xs, ys)]
    mbu = cfg.lookup == "mbu"
    assert cfg.lookup in ("mbu", "recompute"), cfg.lookup
    m._wpa_n = getattr(m, "_wpa_n", 0) + 1
    grp = f"wpa{m._wpa_n}"

    sub = (lambda a, b: MA.modsub(m, a, b, p)) if cfg.add is None else \
        (lambda a, b: cfg.add.sub(m, a, b, p))
    add = (lambda a, b: MA.modadd(m, a, b, p)) if cfg.add is None else \
        (lambda a, b: cfg.add.add(m, a, b, p))
    div = (lambda: E.inplace_div(m, x2, y2, p, cfg.iters)) if cfg.mul is None else \
        (lambda: cfg.mul.div(m, x2, y2, p))
    mul = (lambda: E.inplace_mul(m, x2, y2, p, cfg.iters)) if cfg.mul is None else \
        (lambda: cfg.mul.mul(m, x2, y2, p))
    square = _csub_square if cfg.square == "general" else cfg.square
    neg = MA.cmodneg if cfg.neg == "exact" else cfg.neg

    one = None
    if not mbu:
        one = m.anc(1, "one")
        ctx.x(one[0])

    def load(reg, table):
        if mbu:
            MB.lookup(m, addr, reg, table)
        else:
            _lookup(m, one[0], addr, reg, table)

    def unload(reg, table):
        if mbu:
            MB.unlookup(m, addr, reg, table, group=grp)
        else:
            _lookup(m, one[0], addr, reg, table)

    c = None
    if not cfg.offsets:
        c = m.anc(1, "c")
        MA.is_nonzero(m, addr, c[0])                   # 1
    cq = c[0] if c is not None else None

    # --- 2-5: (x2, y2) <- (x2 - x1, y2 - y1) ---------------------------------
    if cfg.serial_load:
        t = m.anc(n, "t1")
        load(t, xs); sub(t, x2); unload(t, xs)
        load(t, ys); sub(t, y2); unload(t, ys)
        m.free(t)
    else:
        x1, y1 = m.anc(n, "x1"), m.anc(n, "y1")
        if cfg.merge_xy:
            xy = Reg(list(x1) + list(y1), "xy1")
            load(xy, xys)
            sub(x1, x2)
            sub(y1, y2)
            unload(xy, xys)
        else:
            load(x1, xs)                               # 2
            load(y1, ys)
            sub(x1, x2)                                # 3
            sub(y1, y2)                                # 4
            unload(x1, xs)                             # 5
            unload(y1, ys)
        if cfg.free_xy1:
            m.free(x1, y1)

    div()                                              # 6  y2 <- y2 / x2

    xx = m.anc(n, "3x")
    load(xx, x3s)                                      # 7
    add(xx, x2)                                        # 8
    unload(xx, x3s)                                    # 9
    m.free(xx)

    square(m, cq, y2, x2, p)                           # 10 (if c) x2 -= y2^2
    mul()                                              # 11 y2 <- y2 * x2
    neg(m, cq, x2, p)                                  # 12 (if c) x2 <- -x2

    # --- 13-16: (x2, y2) <- (x2 + x1, y2 - y1) -------------------------------
    if cfg.serial_load:
        t = m.anc(n, "t1")
        load(t, ys); sub(t, y2); unload(t, ys)
        load(t, xs); add(t, x2); unload(t, xs)
        m.free(t)
    else:
        if cfg.free_xy1:
            x1, y1 = m.anc(n, "x1"), m.anc(n, "y1")
        if cfg.merge_xy:
            xy = Reg(list(x1) + list(y1), "xy1")
            load(xy, xys)
            sub(y1, y2)
            add(x1, x2)
            unload(xy, xys)
        else:
            load(x1, xs)                               # 13
            load(y1, ys)
            sub(y1, y2)                                # 14
            add(x1, x2)                                # 15
            unload(x1, xs)                             # 16
            unload(y1, ys)
        m.free(x1, y1)

    if mbu:
        MB.phase_fix(m, grp)                           # one repair for all six
    if c is not None:
        MA.is_nonzero(m, addr, c[0])                   # 17
        m.free(c)
    if one is not None:
        ctx.x(one[0])
        m.free(one)


def masked_window_points(curve, base, w, rng, avoid=()):
    """{mu + [i] base : i < 2^w} for a random mu chosen so no entry is O.

    IonQ Sec IV / Alg 4 (after Proos-Zalka): a per-table offset removes the
    point at infinity from every window, which removes the i != 0 flag and
    its controls from the addition.  The offsets are classical and sum into a
    known constant that post-processing subtracts.  Returns (points, mu).
    `avoid` lists points the entries must also stay clear of.
    """
    from ec_classical import Point
    pts = [P for P in curve.points() if not P.inf]
    while True:
        mu = rng.choice(pts)
        T = [curve.add(mu, curve.mul(i, base)) for i in range(1 << w)]
        if not any(P.inf for P in T) and not any(P in avoid for P in T):
            return [Point(P.x, P.y, P.inf) for P in T], mu
