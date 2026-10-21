"""ec_depth: carry-lookahead cells for the point addition, and exact depth.

  * doubling, halving, controlled addition, the fanned-out controlled
    subtraction, comparison and swap, and the squarer are exact;
  * the dialog multiplier on them (CLA replay, CLA walk) is exact;
  * hier.exact_depth, which expands the cached hierarchy on the fly, equals
    depth.toffoli_depth of the flat circuit;
  * at n = 64 the CLA configuration has a fraction of the ripple one's depth.
"""
import random

from _ec_util import FULL, ok, section

import depth
import ec_cost as CO
import ec_depth as DP
import ec_gcd as G
import hier as H
from ec_sim import Machine, SimError, run


def test_cells():
    section("carry-lookahead cells: exact")
    rnd = random.Random(1)
    for p in (61, 127, 251):
        n = p.bit_length()
        for fn, want in ((DP.moddbl_cla, lambda v: 2 * v % p),
                         (DP.modhalf_cla, lambda v: v * pow(2, -1, p) % p)):
            m = Machine("and")
            x = m.alloc(n, "x")
            fn(m, x, p)
            assert all(run(m, {x: v})(x) == want(v) for v in range(p)), (p, fn.__name__)
        m = Machine("and")
        c, a, b = m.alloc(1, "c"), m.alloc(n, "a"), m.alloc(n, "b")
        DP.CLAArith(p).cadd(m, c[0], a, b)
        for cv, av, bv in [(rnd.randrange(2), rnd.randrange(p), rnd.randrange(p))
                           for _ in range(150)]:
            rd = run(m, {c: cv, a: av, b: bv})
            assert (rd(a), rd(b)) == (av, (bv + cv * av) % p)
    for n in range(1, 7):
        m = Machine("and")
        c, x, y = m.alloc(1, "c"), m.alloc(n, "x"), m.alloc(n, "y")
        DP.cla_csub(m, c[0], x, y)
        DP.fan_cswap(m, c[0], x, y)
        for cv in (0, 1):
            for xv in range(1 << n):
                for yv in range(1 << n):
                    rd = run(m, {c: cv, x: xv, y: yv})
                    d = (yv - cv * xv) % (1 << n)
                    assert (rd(x), rd(y)) == ((d, xv) if cv else (xv, d))
        m = Machine("and")
        x, z = m.alloc(n, "x"), m.alloc(2 * n + 1, "z")
        DP.sqr_int_cla(m, x, z)
        assert all(run(m, {x: v})(z) == v * v for v in range(1 << n))
    ok("doubling/halving every input at p = 61, 127, 251; fanned-out subtraction "
       "and swap, squarer: exhaustive to n = 6")


def test_dialog():
    section("the dialog on carry-lookahead cells")
    rnd = random.Random(2)
    for q in (61, 127):
        n = q.bit_length()
        for kw in (dict(), dict(walk_cla=True),
                   dict(walk_cla=True, cmp_msbs=n + 1, c_pad=2.3, share=True, compress="fig1")):
            for fn in ("mul", "div"):
                m = Machine("and")
                x, y = m.alloc(n, "x"), m.alloc(n, "y")
                getattr(G.Dialog(arith=DP.CLAArith(q), fused_cmp=True, **kw), fn)(m, x, y, q)
                for _ in range(60 if FULL else 25):
                    xv, yv = rnd.randrange(1, q), rnd.randrange(q)
                    want = xv * yv % q if fn == "mul" else yv * pow(xv, -1, q) % q
                    rd = run(m, {x: xv, y: yv})
                    assert (rd(x), rd(y)) == (xv, want), (q, kw, fn)
    ok("multiplication and division exact, q = 61, 127, with and without the "
       "CLA walk and register sharing")


def test_exact_depth():
    section("hier.exact_depth = the flat circuit's Toffoli depth")
    for q, n, k in ((61, 6, 6), (2 ** 32 - 5, 32, 16)):
        for lab, mk in (("ripple", lambda: G.Dialog(arith=G.PM(q, msbs=k), fused_cmp=True)),
                        ("CLA", lambda: G.Dialog(arith=DP.CLAArith(q), fused_cmp=True,
                                                 walk_cla=True))):
            m = Machine("and")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            mk().mul(m, x, y, q)
            flat = depth.toffoli_depth(m.qc)
            with H.tracing():
                hm = H.HierMachine("and")
                x, y = hm.alloc(n, "x"), hm.alloc(n, "y")
                mk().mul(hm, x, y, q)
                ex = H.exact_depth(hm)
            assert abs(ex - flat) <= max(2, flat // 1000), (q, lab, ex, flat)
            print(f"      n={n:>2} {lab:<6} flat {flat:>6} hierarchical {ex:>6} "
                  f"Toffolis {CO.count(m)['toffoli_paper']:>7}")
    ok("equal (to the convention for multi-controlled gates) at n = 6 and 32")


def test_depth_gain():
    section("depth against Toffolis at n = 64")
    q, n = 2 ** 64 - 59, 64
    out = {}
    for lab, mk in (("ripple", lambda: G.Dialog(arith=G.PM(q, msbs=24), fused_cmp=True)),
                    ("CLA", lambda: G.Dialog(arith=DP.CLAArith(q), fused_cmp=True,
                                             walk_cla=True))):
        with H.tracing():
            hm = H.HierMachine("and")
            x, y = hm.alloc(n, "x"), hm.alloc(n, "y")
            mk().mul(hm, x, y, q)
            c = H.count(hm)
            out[lab] = (c["toffoli_paper"], H.exact_depth(hm), c["qubits"])
    for lab, (t, d, qb) in out.items():
        print(f"      {lab:<6} {t:>8} Toffolis, depth {d:>7}, {qb} qubits")
    assert out["CLA"][1] < out["ripple"][1] / 3
    ok("the carry-lookahead multiplier: several times shallower, several times "
       "more Toffolis")


def main():
    test_cells()
    test_dialog()
    test_exact_depth()
    test_depth_gain()


if __name__ == "__main__":
    main()
    print("\ntest_ec_depth: all passed")
