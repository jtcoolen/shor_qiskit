"""Property-based tests over randomly generated toy curves and points.

The hand-written tests pin down two curves in detail.  These pin down the
*properties* across many curves: a random prime, random non-singular (a, b),
random points, random scalars.  That is where curve-shape assumptions hide --
a = 0, b = 0, curves with a point of order 2, primes where 2 is or is not a
quadratic residue, primes just above a power of two.

Every property compares a circuit against `ec_classical`, and every circuit run
goes through `ec_sim.run`, which also asserts that each ancilla is |0> at the
moment it is freed.  So "the answer was right" and "the circuit was reversible"
are both being checked on every case.
"""
from _ec_util import primes, scope, section
from _pbt import SKIP, check

import ec_classical as C
import ec_eea as E
import ec_kaliski as K
import ec_modarith as MA
import ec_mult as MU
import ec_pointadd as PA
import ec_proj as PJ
from ec_sim import Machine, run

SMALL_P = [p for p in primes(64) if p >= 5]
TINY_P = [p for p in primes(32) if p >= 5]
N = scope(30, 120)


# --- generators -------------------------------------------------------------
def gen_prime_pair(rnd):
    p = rnd.choice(SMALL_P)
    return {"p": p, "a": rnd.randrange(p), "b": rnd.randrange(p)}


def gen_tiny_prime_pair(rnd):
    p = rnd.choice(TINY_P)
    return {"p": p, "a": rnd.randrange(p), "b": rnd.randrange(p)}


def gen_curve_points(rnd, ps=None):
    ps = ps or TINY_P
    p = rnd.choice(ps)
    ca, cb = rnd.randrange(p), rnd.randrange(p)
    return {"p": p, "ca": ca, "cb": cb,
            "i": rnd.randrange(64), "j": rnd.randrange(64), "z": rnd.randrange(1, 64)}


def _curve(p, ca, cb):
    if (4 * ca**3 + 27 * cb**2) % p == 0:
        return None
    return C.Curve(p, ca, cb)


def _pick(curve, k):
    pts = [P for P in curve.points() if not P.inf]
    return pts[k % len(pts)] if pts else None


# --- properties -------------------------------------------------------------
def prop_modarith(p, a, b):
    # These circuits are only defined for reduced inputs.  Saying so explicitly
    # matters for shrinking: without it the shrinker lowers p, leaves a and b
    # above the new modulus, and reports that out-of-domain case as "minimal".
    if a >= p or b >= p:
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    MA.modadd(m, x, y, p)
    rd = run(m, {x: a, y: b})
    assert rd(y) == (a + b) % p, (p, a, b, rd(y))
    assert rd(x) == a

    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    MA.modsub(m, x, y, p)
    assert run(m, {x: a, y: b})(y) == (b - a) % p, ("modsub", p, a, b)

    m = Machine("and")
    x = m.alloc(n, "x")
    MA.moddbl(m, x, p)
    assert run(m, {x: a})(x) == 2 * a % p, ("moddbl", p, a)

    m = Machine("and")
    x = m.alloc(n, "x")
    MA.modneg(m, x, p)
    assert run(m, {x: a})(x) == (-a) % p, ("modneg", p, a)


def prop_mul(p, a, b):
    if a >= p or b >= p:
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    x, y, z = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "z")
    MU.modmul(m, x, y, z, p)
    rd = run(m, {x: a, y: b})
    assert rd(z) == a * b % p, (p, a, b, rd(z))
    assert rd(x) == a and rd(y) == b


def prop_inverse(p, a, b):
    if a >= p or a % p == 0:
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    x, o = m.alloc(n, "x"), m.alloc(n, "o")
    K.mod_inv(m, x, o, p)
    rd = run(m, {x: a})
    assert rd(o) == pow(a, -1, p), (p, a, rd(o))
    assert rd(x) == a


def prop_division(p, a, b):
    if a >= p or b >= p or a % p == 0:
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    c = m.alloc(1, "c")
    x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "o")
    K.mod_div(m, c[0], x, y, o, p)
    for ct in (0, 1):
        rd = run(m, {c: ct, x: a, y: b})
        assert rd(o) == ((b * pow(a, -1, p)) % p if ct else 0)


def prop_inplace_mul(p, a, b):
    if a >= p or b >= p or a % p == 0:
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    E.inplace_mul(m, x, y, p)
    rd = run(m, {x: a, y: b})
    assert rd(y) == a * b % p, (p, a, b, rd(y))
    assert rd(x) == a


def prop_affine_add(p, ca, cb, i, j, z):
    curve = _curve(p, ca, cb)
    if curve is None:
        return SKIP
    P1, P2 = _pick(curve, i), _pick(curve, j)
    if P1 is None or P2 is None or C.point_add_exceptional(curve, P1, P2):
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    q = m.alloc(1, "q")
    X, Y = m.alloc(n, "x"), m.alloc(n, "y")
    PA.point_add_ctrl(m, q[0], X, Y, P2.x, P2.y, p)
    for ct in (0, 1):
        rd = run(m, {q: ct, X: P1.x, Y: P1.y})
        want = curve.add(P1, P2) if ct else P1
        assert (rd(X), rd(Y)) == (want.x, want.y), (curve.name, P1, P2, ct)


def prop_affine_roundtrip(p, ca, cb, i, j, z):
    curve = _curve(p, ca, cb)
    if curve is None:
        return SKIP
    P1, P2 = _pick(curve, i), _pick(curve, j)
    if P1 is None or P2 is None or C.point_add_exceptional(curve, P1, P2):
        return SKIP
    n = p.bit_length()
    m = Machine("and")
    q = m.alloc(1, "q")
    X, Y = m.alloc(n, "x"), m.alloc(n, "y")
    PA.point_add_ctrl(m, q[0], X, Y, P2.x, P2.y, p)
    PA.point_add_ctrl_inv(m, q[0], X, Y, P2.x, P2.y, p)
    for ct in (0, 1):
        rd = run(m, {q: ct, X: P1.x, Y: P1.y})
        assert (rd(X), rd(Y)) == (P1.x, P1.y), "add then un-add is not the identity"


def prop_projective_add(p, ca, cb, i, j, z):
    curve = _curve(p, ca, cb)
    if curve is None:
        return SKIP
    P1, P2 = _pick(curve, i), _pick(curve, j)
    if P1 is None or P2 is None or P1.x == P2.x:
        return SKIP
    S = curve.add(P1, P2)
    if S.inf:
        return SKIP
    Z = 1 + (z % (p - 1))
    n = p.bit_length()
    m = Machine("and")
    X1, Y1, Z1 = m.alloc(n, "X1"), m.alloc(n, "Y1"), m.alloc(n, "Z1")
    X3, Y3, Z3 = m.alloc(n, "X3"), m.alloc(n, "Y3"), m.alloc(n, "Z3")
    PJ.jacobian_add(m, X1, Y1, Z1, P2.x, P2.y, X3, Y3, Z3, p)
    x1, y1 = P1.x * Z * Z % p, P1.y * pow(Z, 3, p) % p
    rd = run(m, {X1: x1, Y1: y1, Z1: Z})
    got = C.jacobian_to_affine(curve, rd(X3), rd(Y3), rd(Z3))
    assert got == S, (curve.name, P1, P2, Z, got, S)
    assert (rd(X1), rd(Y1), rd(Z1)) == (x1, y1, Z), "input point must survive"


def prop_projective_matches_affine(p, ca, cb, i, j, z):
    """The two point-addition circuits must agree with each other."""
    curve = _curve(p, ca, cb)
    if curve is None:
        return SKIP
    P1, P2 = _pick(curve, i), _pick(curve, j)
    if P1 is None or P2 is None or C.point_add_exceptional(curve, P1, P2):
        return SKIP
    n = p.bit_length()
    ma = Machine("and")
    q = ma.alloc(1, "q")
    X, Y = ma.alloc(n, "x"), ma.alloc(n, "y")
    PA.point_add_ctrl(ma, q[0], X, Y, P2.x, P2.y, p)
    rda = run(ma, {q: 1, X: P1.x, Y: P1.y})

    mp = Machine("and")
    X1, Y1, Z1 = mp.alloc(n, "X1"), mp.alloc(n, "Y1"), mp.alloc(n, "Z1")
    X3, Y3, Z3 = mp.alloc(n, "X3"), mp.alloc(n, "Y3"), mp.alloc(n, "Z3")
    PJ.jacobian_add(mp, X1, Y1, Z1, P2.x, P2.y, X3, Y3, Z3, p)
    rdp = run(mp, {X1: P1.x, Y1: P1.y, Z1: 1})
    aff = C.jacobian_to_affine(curve, rdp(X3), rdp(Y3), rdp(Z3))
    assert (rda(X), rda(Y)) == (aff.x, aff.y), (curve.name, P1, P2)


def prop_ecdlp(p, ca, cb, i, j, z):
    """End to end on a random curve, via the table oracle."""
    import ec_shor as S
    curve = _curve(p, ca, cb)
    if curve is None:
        return SKIP
    G = _pick(curve, i)
    if G is None:
        return SKIP
    order = curve.point_order(G)
    if not (3 <= order <= 12):
        return SKIP
    if any(P.x == 0 and P.y == 0 for P in curve.points() if not P.inf):
        return SKIP                       # (0,0) is on the curve: no infinity code
    k = 1 + (j % (order - 1))
    Q = curve.mul(k, G)
    top, counts, info = S.solve(curve, G, Q, order, shots=2048)
    assert top is not None, (curve.name, G, k)
    assert curve.mul(top, G) == Q, (curve.name, G, k, top)


def main():
    section("modular arithmetic over random primes")
    check("modadd / modsub / moddbl / modneg", gen_prime_pair, prop_modarith,
          n=N, shrink_keys=[("p", SMALL_P), ("a", 0), ("b", 0)])
    check("modmul", gen_tiny_prime_pair, prop_mul, n=N,
          shrink_keys=[("p", TINY_P), ("a", 0), ("b", 0)])

    section("Kaliski inversion and division over random primes")
    check("mod_inv == pow(x,-1,p)", gen_tiny_prime_pair, prop_inverse,
          n=scope(20, 80), shrink_keys=[("p", TINY_P), ("a", 1)])
    check("mod_div == y/x, both control branches", gen_tiny_prime_pair,
          prop_division, n=scope(12, 50), shrink_keys=[("p", TINY_P), ("a", 1), ("b", 0)])

    section("EEA in-place multiplication over random primes")
    check("inplace_mul == x*y mod q", gen_tiny_prime_pair, prop_inplace_mul,
          n=scope(15, 60), shrink_keys=[("p", TINY_P), ("a", 1), ("b", 0)])

    section("point addition over random curves")
    keys = [("p", TINY_P), ("ca", 0), ("cb", 0), ("i", 0), ("j", 0), ("z", 1)]
    check("affine add == curve.add, both control branches", gen_curve_points,
          prop_affine_add, n=scope(25, 100), shrink_keys=keys)
    check("affine add then un-add is the identity", gen_curve_points,
          prop_affine_roundtrip, n=scope(15, 60), shrink_keys=keys)
    check("projective add == curve.add, every representative", gen_curve_points,
          prop_projective_add, n=scope(25, 100), shrink_keys=keys)
    check("the two point-addition circuits agree with each other",
          gen_curve_points, prop_projective_matches_affine,
          n=scope(15, 60), shrink_keys=keys)

    section("end-to-end ECDLP over random curves")
    check("Shor recovers the discrete logarithm", gen_curve_points, prop_ecdlp,
          n=scope(12, 40), shrink_keys=keys)


if __name__ == "__main__":
    main()
    print("\ntest_ec_pbt: all passed")
