"""Signed and fused primitives: ci_add, cgt_fused, gt_top, csignadd,
csignadd_pm, cmodneg_approx, modhalf_pm."""
from _ec_util import FULL, ok, section

import ec_adders as A
import ec_approx as AX
import ec_modarith as MA
from ec_sim import Machine, SimError, run


def ands(m):
    return sum(1 for i in m.qc.data if i.operation.name == "ecand")


def toff(m):
    from ec_cost import count
    return count(m)["toffoli_paper"]


def test_integer():
    section("ci_add, cgt_fused, gt_top: exhaustive, and their AND counts")
    for n in range(1, 6):
        m = Machine("and")
        e, x, y = m.alloc(1, "e"), m.alloc(n, "x"), m.alloc(n, "y")
        A.ci_add(m.ctx, e[0], x, y, m.anc(max(n - 1, 1), "a"))
        for ev in (0, 1):
            for xv in range(1 << n):
                for yv in range(1 << n):
                    rd = run(m, {e: ev, x: xv, y: yv})
                    assert rd(y) == (yv + (-1) ** ev * xv) % (1 << n) and rd(x) == xv
        assert ands(m) == max(n - 1, 0) or n == 1, (n, ands(m))

        m = Machine("and")
        c, x, y, o = m.alloc(1, "c"), m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
        A.cgt_fused(m.ctx, c[0], x, y, o[0], m.anc(n, "a"))
        for cv in (0, 1):
            for xv in range(1 << n):
                for yv in range(1 << n):
                    rd = run(m, {c: cv, x: xv, y: yv})
                    assert rd(o) == (cv and xv > yv) and rd(x) == xv and rd(y) == yv
        assert toff(m) == n + 1, (n, toff(m))
    ok("n = 1..5: y +- x in n-1 ANDs; ctrl AND [x > y] in n+1 (was 2n+1)")

    n, k = 6, 3
    m = Machine("and")
    x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
    A.gt_top(m.ctx, x, y, o[0], m.anc(n, "a"), k)
    for xv in range(64):
        for yv in range(64):
            assert run(m, {x: xv, y: yv})(o) == ((xv >> 3) > (yv >> 3))
    ok("gt_top: the comparison of the top k bits, exactly")


def test_csignadd():
    section("csignadd: exact signed modular addition, any odd p")
    for p in (29, 61, 127):
        n = p.bit_length()
        m = Machine("and")
        e, x, y = m.alloc(1, "e"), m.alloc(n, "x"), m.alloc(n, "y")
        MA.csignadd(m, e[0], x, y, p)
        step = 1 if p < 100 else 3
        for ev in (0, 1):
            for xv in range(0, p, step):
                for yv in range(0, p, step):
                    rd = run(m, {e: ev, x: xv, y: yv})
                    assert rd(y) == (yv + (-1) ** ev * xv) % p, (p, ev, xv, yv)
    ok("p = 29, 61, 127, both signs: exact")


def rate(build, q, inputs, ref):
    n = q.bit_length()
    m, regs = build(n)
    bad = 0
    for inp in inputs:
        try:
            rd = run(m, {regs[k]: v for k, v in inp.items()})
            if rd(regs["out"]) != ref(inp):
                bad += 1
        except SimError:
            bad += 1
    return bad / len(inputs), m


def test_pm():
    section("pseudo-Mersenne signed add and negation (IonQ): measured")
    for q in ((61, 127, 251, 509) if FULL else (61, 127, 251)):
        u, f = AX.pseudo_mersenne(q)
        step = 1 if q < 128 else 5
        ins = [{"e": ev, "x": xv, "y": yv} for ev in (0, 1)
               for xv in range(0, q, step) for yv in range(0, q, step)]

        def build(n, msbs):
            m = Machine("and")
            e, x, y = m.alloc(1, "e"), m.alloc(n, "x"), m.alloc(n, "y")
            AX.csignadd_pm(m, e[0], x, y, q, msbs=msbs)
            return m, {"e": e, "x": x, "y": y, "out": y}
        rates = []
        for msbs in (2, u):
            r, m = rate(lambda n: build(n, msbs), q, ins,
                        lambda i: (i["y"] + (-1) ** i["e"] * i["x"]) % q)
            rates.append(r)
        # at full width the only failures are sums in [q, 2^u) the PM adder
        # cannot reduce: bounded by the fraction of such sums, ~2f/q
        assert rates[1] <= 2.5 * f / q + 1e-12, (q, rates)
        assert rates[1] <= rates[0]
        print(f"      q = 2^{u}-{f}: csignadd_pm failure {100*rates[0]:5.2f}% "
              f"at msbs=2, {100*rates[1]:5.2f}% at msbs={u} (bound 2.5f/q = "
              f"{100*2.5*f/q:.2f}%)")

        ins = [{"c": cv, "x": xv} for cv in (0, 1) for xv in range(q)]

        def buildn(n, kappa, tau):
            m = Machine("and")
            c, x = m.alloc(1, "c"), m.alloc(n, "x")
            AX.cmodneg_approx(m, c[0], x, q, kappa, tau)
            return m, {"c": c, "x": x, "out": x}
        r_exact, m = rate(lambda n: buildn(n, u, u), q, ins,
                          lambda i: (-i["x"]) % q if i["c"] else i["x"])
        assert r_exact == 0.0, (q, r_exact)
        r_small, _ = rate(lambda n: buildn(n, 2, 2), q, ins,
                          lambda i: (-i["x"]) % q if i["c"] else i["x"])
        mx = Machine("and")
        c, x = mx.alloc(1, "c"), mx.alloc(u, "x")
        MA.cmodneg(mx, c[0], x, q)
        print(f"      q = 2^{u}-{f}: cmodneg_approx exact at kappa=tau={u} "
              f"({toff(m)} Toffoli-eq vs {toff(mx)} exact cmodneg); "
              f"{100*r_small:.2f}% wrong at kappa=tau=2")
    # the saving is at IonQ's parameters, not at full width: build (no
    # simulation) at secp256k1 size, kappa = 65, tau = 32
    P256K1 = 2**256 - 2**32 - 977
    ma, mx = Machine("and"), Machine("and")
    xa, xx = ma.alloc(256, "x"), mx.alloc(256, "x")
    ca, cx = ma.alloc(1, "c"), mx.alloc(1, "c")
    AX.cmodneg_approx(ma, ca[0], xa, P256K1, kappa=65, tau=32)
    MA.cmodneg(mx, cx[0], xx, P256K1)
    ta, tx = toff(ma), toff(mx)
    assert ta < tx / 4, (ta, tx)
    print(f"      secp256k1, kappa=65 tau=32: cmodneg_approx {ta} Toffoli-eq "
          f"vs exact cmodneg {tx}")
    ok("csignadd_pm within its bound and monotone; cmodneg_approx exact at "
       "kappa = tau = u (and no cheaper there), 4x+ cheaper at IonQ's parameters")

    section("modhalf_pm: Algorithm 7 backwards")
    q = 127
    n = 7
    m = Machine("and")
    x = m.alloc(n, "x")
    AX.modhalf_pm(m, x, q)
    for xv in range(q):
        assert (2 * run(m, {x: xv})(x)) % q == xv
    ok("q = 127: x -> x/2 mod q on every x")


def main():
    test_integer()
    test_csignadd()
    test_pm()


if __name__ == "__main__":
    main()
    print("\ntest_ec_signed: all passed")
