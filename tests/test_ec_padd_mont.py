"""[106]'s optimised inversion and Montgomery multiplier, wired into the
eleven-step addition (point_add_ctrl_mont): exact on every non-exceptional
input, both control values; cost reported against point_add_ctrl."""
from _ec_util import ok, section

import ec_classical as C
import ec_cost as CO
import ec_pointadd as PA
from ec_classical import to_mont
from ec_sim import Machine, run


def main():
    section("point_add_ctrl_mont: Montgomery form, [106] Sec 3.2-3.3 parts")
    for curve in (C.CLASSIQ, C.TOY11):
        p = curve.p
        n = p.bit_length()
        pts = [P for P in curve.points() if not P.inf]
        for P2 in pts[:2]:
            m = Machine("and")
            q = m.alloc(1, "q")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            PA.point_add_ctrl_mont(m, q[0], x, y, P2.x, P2.y, p)
            good = 0
            for P1 in pts:
                for qv in (0, 1):
                    if qv and C.point_add_exceptional(curve, P1, P2):
                        continue
                    S = curve.add(P1, P2) if qv else P1
                    if S.inf:
                        continue
                    rd = run(m, {q: qv, x: to_mont(P1.x, p, n), y: to_mont(P1.y, p, n)})
                    assert (rd(x), rd(y)) == (to_mont(S.x, p, n), to_mont(S.y, p, n))
                    good += 1
        ref = Machine("and")
        q2 = ref.alloc(1, "q")
        x2, y2 = ref.alloc(n, "x"), ref.alloc(n, "y")
        PA.point_add_ctrl(ref, q2[0], x2, y2, P2.x, P2.y, p)
        a, b = CO.count(m), CO.count(ref)
        print(f"      {curve.name}: Montgomery {a['toffoli_paper']} Toffoli-eq / "
              f"{a['qubits']} qubits vs reference {b['toffoli_paper']} / {b['qubits']} "
              f"-- at n = {n} the carry-save multiplier's n + 2w + 2 accumulator "
              f"and the compute/uncompute wrappers dominate")
    ok("exact on every non-exceptional (P1, q) for two addends on each curve")


if __name__ == "__main__":
    main()
    print("\ntest_ec_padd_mont: all passed")
