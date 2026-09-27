"""ec_square: the dedicated squarer and the square-subtract of step 10."""
import random

from _ec_util import FULL, ok, section

import ec_approx as AX
import ec_cost as CO
import ec_square as SQ
import ec_window as W
from ec_sim import Machine, SimError, run


def ands(m):
    return sum(1 for i in m.qc.data if i.operation.name == "ecand")


def test_sqr_int():
    section("sqr_int: x^2 exactly, n(n+3)/2 ANDs")
    for n in range(1, 8):
        m = Machine("and")
        x, z = m.alloc(n, "x"), m.alloc(2 * n + 1, "z")
        SQ.sqr_int(m, x, z)
        for xv in range(1 << n):
            rd = run(m, {x: xv})
            assert rd(z) == xv * xv and rd(x) == xv, (n, xv)
        assert ands(m) == SQ.sqr_int_cost(n), (n, ands(m))
    ok("n = 1..7, every x; the sign qubit returns to |0>")


def build_sq(fn, q, **kw):
    n = q.bit_length()
    m = Machine("and")
    c, s, a = m.alloc(1, "c"), m.alloc(n, "s"), m.alloc(n, "a")
    fn(m, c[0], s, a, q, **kw)
    return m, c, s, a


def rate(m, c, s, a, q, cases):
    bad = 0
    for cv, sv, av in cases:
        want = (av - cv * sv * sv) % q
        try:
            rd = run(m, {c: cv, s: sv, a: av})
            bad += (rd(a), rd(s)) != (want, sv)
        except SimError:
            bad += 1
    return bad / len(cases)


def test_square_sub():
    section("square-subtract: acc -= src^2 mod p")
    rnd = random.Random(3)
    for p in (11, 13, 61, 127):
        n = p.bit_length()
        cases = [(cv, sv, av) for cv in (0, 1) for sv in range(p) for av in range(0, p, 1 if p < 64 else 7)]
        m, c, s, a = build_sq(SQ.csub_square_generic, p)
        assert rate(m, c, s, a, p, cases) == 0.0, p
        ref = build_sq(W._csub_square, p)[0]
        print(f"      p={p:>3} generic fold: {CO.count(m)['toffoli_paper']:>5} Toffoli-eq "
              f"vs general multiplier {CO.count(ref)['toffoli_paper']:>5}")
    ok("generic (exact) square-subtract: every input, p = 11, 13, 61, 127")

    for q in (61, 127, 251):
        u, f = AX.pseudo_mersenne(q)
        n = q.bit_length()
        cases = [(rnd.randrange(2), rnd.randrange(q), rnd.randrange(q)) for _ in range(400)]
        m, c, s, a = build_sq(SQ.csub_square_pm, q, msbs=n)
        r = rate(m, c, s, a, q, cases)
        ref = build_sq(W._csub_square, q)[0]
        print(f"      q=2^{u}-{f}: PM fold {CO.count(m)['toffoli_paper']:>5} Toffoli-eq vs "
              f"{CO.count(ref)['toffoli_paper']:>5} general; failure {100*r:.1f}% "
              f"(inputs reaching q: ~{100*2*f/q:.1f}%)")
        assert r <= 3 * f / q + 0.02, (q, r)
    ok("pseudo-Mersenne fold: failures only at the ~f/q non-canonical inputs")


def test_projection():
    section("cost at n = 64 and n = 256 (built, not simulated)")
    for n, q in ((64, (1 << 64) - 59), (256, 2**256 - 2**32 - 977)):
        m = Machine("and")
        s, a = m.alloc(n, "s"), m.alloc(n, "a")
        SQ.csub_square_pm(m, None, s, a, q, lsbs=None, msbs=48 if n == 256 else 32)
        t_new = CO.count(m)["toffoli_paper"]
        line = f"      n={n}: PM square-subtract {t_new} Toffoli-eq ({t_new / n**2:.2f} n^2)"
        if n == 64:
            ref = Machine("and")
            s2, a2 = ref.alloc(n, "s"), ref.alloc(n, "a")
            W._csub_square(ref, None, s2, a2, q)
            t_old = CO.count(ref)["toffoli_paper"]
            line += f" vs {t_old} ({t_old / n**2:.2f} n^2) general"
            assert t_new < t_old / 3
        print(line)
    ok("the dedicated squarer + fold is several times cheaper than the general "
       "multiplier route")


def test_karatsuba():
    section("ECDSA.Fail Sec 5.3.4: one Karatsuba split, square by square")
    rnd = random.Random(4)
    for q in (61, 127, 251):
        n = q.bit_length()
        cases = [(cv, sv, av) for cv in (0, 1) for sv in range(0, q, 1 if q < 100 else 3)
                 for av in range(0, q, 7)]
        m, c, s, a = build_sq(SQ.csub_square_karatsuba_pm, q, msbs=n)
        assert rate(m, c, s, a, q, cases) == 0.0, q
    ok("exact at q = 61, 127 (odd n) and 251")
    for n, q in ((64, (1 << 64) - 59), (256, 2**256 - 2**32 - 977)):
        out = {}
        for lab, fn in (("fold", SQ.csub_square_pm), ("karatsuba", SQ.csub_square_karatsuba_pm)):
            m = Machine("and")
            s_, a_ = m.alloc(n, "s"), m.alloc(n, "a")
            fn(m, None, s_, a_, q, msbs=48)
            out[lab] = CO.count(m)["toffoli_paper"]
        h = (n + 1) // 2
        sq_plain = 2 * SQ.sqr_int_cost(n)
        sq_kara = 2 * (2 * SQ.sqr_int_cost(h) + SQ.sqr_int_cost(h + 1))
        print(f"      n={n}: squaring ANDs {sq_plain} -> {sq_kara} "
              f"({100*(1-sq_kara/sq_plain):.0f}% fewer), but square-subtract "
              f"{out['fold']} -> {out['karatsuba']}: the three shifted terms each "
              f"need their overflow folded back through 2^n = f")
    ok("Karatsuba cuts the squaring ANDs by ~1/4 as ECDSA.Fail reports; composed "
       "from this package's modular additions its extra folds cost more than that")


def main():
    test_sqr_int()
    test_square_sub()
    test_karatsuba()
    if FULL or True:
        test_projection()


if __name__ == "__main__":
    main()
    print("\ntest_ec_square: all passed")
