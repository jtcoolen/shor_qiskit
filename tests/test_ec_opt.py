"""ec_opt: exact constant propagation and cancellation, re-verified."""
import random

from _ec_util import ok, random_curve, random_generator, section

import ec_classical as C
import ec_cost as CO
import ec_gcd as G
import ec_opt as O
import ec_window as W
from ec_sim import Machine, run


def same(m, m2, regs, cases):
    for c in cases:
        a = run(m, dict(zip(regs, c)))
        b = run(m2, dict(zip(regs, c)))           # also re-checks every ancilla
        assert [a(r) for r in regs] == [b(r) for r in regs], c


def main():
    section("peephole passes: same outputs, every ancilla check kept")
    rnd = random.Random(1)
    q, n = 61, 6
    for name, be in (("Dialog", G.Dialog()), ("CondInv", G.CondInv(replay="standard")),
                     ("PingPong", G.PingPong(rounds=17)), ("Jump2", G.Jump2(steps=7))):
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        be.mul(m, x, y, q)
        m2 = O.peephole(m, [x, y])
        same(m, m2, [x, y], [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(60)])
        a, b = CO.count(m)["toffoli_paper"], CO.count(m2)["toffoli_paper"]
        assert b <= a
        print(f"      {name:<9} {a:>5} -> {b:>5} Toffoli-eq ({100*(a-b)/a:.1f}%)")

    rnd = random.Random(61)
    curve, pts = random_curve(rnd, pmax=61, pmin=61)
    G0 = random_generator(rnd, curve, pts, 8)
    tab = W.masked_window_points(curve, G0, 2, random.Random(2))[0]
    m = Machine("and")
    a_, x_, y_ = m.alloc(2, "a"), m.alloc(6, "x"), m.alloc(6, "y")
    W.windowed_point_add_cfg(m, a_, x_, y_, tab, 61, W.IONQ_LOOKUPS)
    m2 = O.peephole(m, [a_, x_, y_])
    cases = [(i, R.x, R.y) for R in pts[:20] for i in range(4)
             if not C.point_add_exceptional(curve, R, tab[i]) and not curve.add(R, tab[i]).inf]
    same(m, m2, [a_, x_, y_], cases)
    a, b = CO.count(m)["toffoli_paper"], CO.count(m2)["toffoli_paper"]
    print(f"      windowed addition (IonQ cfg), p=61: {a} -> {b} Toffoli-eq ({100*(a-b)/a:.1f}%)")
    assert b < a
    ok("constant propagation + cancellation: exact, a few percent, no sampling")


if __name__ == "__main__":
    main()
    print("\ntest_ec_opt: all passed")
