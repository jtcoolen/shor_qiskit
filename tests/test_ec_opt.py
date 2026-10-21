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


def fail_rate(m, regs, want, cases):
    from ec_sim import SimError
    bad = 0
    for c in cases:
        try:
            rd = run(m, dict(zip(regs, c)))
            bad += [rd(r) for r in regs] != want(c)
        except SimError:
            bad += 1
    return bad / len(cases)


def test_census():
    section("ECDSA.Fail's fire census: measured, not exact")
    rnd = random.Random(7)
    q, n = 61, 6
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    G.Dialog(fused_cmp=True).mul(m, x, y, q)
    want = lambda c: [c[0], c[0] * c[1] % q]
    everything = [(a, b) for a in range(1, q) for b in range(q)]
    full = O.strip_unfired(m, O.census(m, [{x: a, y: b} for a, b in everything]))
    assert fail_rate(full, [x, y], want, everything) == 0.0
    t0, tf = CO.count(m)["toffoli_paper"], CO.count(full)["toffoli_paper"]
    print(f"      census over every input (exact dead code): {t0} -> {tf} Toffoli-eq")
    fresh = rnd.sample(everything, 400)
    prev = 1.0
    for k in (3, 10, 30, 100, 300):
        sample = rnd.sample(everything, k)
        mk = O.strip_unfired(m, O.census(m, [{x: a, y: b} for a, b in sample]))
        r = fail_rate(mk, [x, y], want, fresh)
        print(f"      sample of {k:>3}: {CO.count(mk)['toffoli_paper']:>5} Toffoli-eq, "
              f"fresh inputs wrong {100 * r:5.1f}%")
        assert r <= prev + 0.05
        prev = r
    ok("exact when the census covers every input; on a sample, cheaper and wrong "
       "on a measured fraction of fresh inputs")


if __name__ == "__main__":
    main()
    test_census()
    print("\ntest_ec_opt: all passed")
