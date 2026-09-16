"""In-place controlled point addition in affine coordinates (2026/106 Alg. 3)."""
from _ec_util import ok, section

import ec_classical as C
import ec_pointadd as PA
from ec_sim import Machine, run


def check_curve(curve, label):
    p, n = curve.p, curve.p.bit_length()
    pts = [P for P in curve.points() if not P.inf]
    ok_n = skip = 0
    for P2 in pts:
        m = Machine("and")
        q = m.alloc(1, "q")
        X, Y = m.alloc(n, "x1"), m.alloc(n, "y1")
        PA.point_add_ctrl(m, q[0], X, Y, P2.x, P2.y, p)
        for P1 in pts:
            if C.point_add_exceptional(curve, P1, P2):
                skip += 1
                continue
            for ct in (0, 1):
                rd = run(m, {q: ct, X: P1.x, Y: P1.y})
                want = curve.add(P1, P2) if ct else P1
                assert (rd(X), rd(Y)) == (want.x, want.y), (label, P1, P2, ct)
                assert rd(q) == ct
                ok_n += 1
    ok(f"{label}: {ok_n} controlled additions exact, {skip} excluded as "
       f"exceptional, all ancillas |0>")
    return ok_n, skip


def main():
    section("every point pair on two toy curves")
    check_curve(C.CLASSIQ, "y^2=x^3+5x+4 mod 7 (Classiq tutorial)")
    check_curve(C.TOY11, "y^2=x^3+x+6 mod 11")

    section("the two exceptional cases are real, and are the only ones")
    curve = C.CLASSIQ
    p, n = curve.p, 3
    pts = [P for P in curve.points() if not P.inf]
    named = 0
    for P1 in pts:
        for P2 in pts:
            if P1.x == P2.x:
                continue
            P3 = curve.add(P1, P2)
            if not P3.inf and P3.x == P2.x:
                named += 1
    assert named > 0
    ok(f"{named} pairs on this curve satisfy P1 = -2 P2 -- the x3 == x2 case, "
       f"which the addition formulas never hint at")

    section("running the addition backwards subtracts")
    for P2 in pts:
        m = Machine("and")
        q = m.alloc(1, "q")
        X, Y = m.alloc(n, "x1"), m.alloc(n, "y1")
        PA.point_add_ctrl_inv(m, q[0], X, Y, P2.x, P2.y, p)
        for P1 in pts:
            if C.point_add_exceptional(curve, P1, P2):
                continue
            S = curve.add(P1, P2)
            rd = run(m, {q: 1, X: S.x, Y: S.y})
            assert (rd(X), rd(Y)) == (P1.x, P1.y)
    ok("point_add_ctrl_inv undoes point_add_ctrl on every non-exceptional pair")


if __name__ == "__main__":
    main()
    print("\ntest_ec_pointadd: all passed")
