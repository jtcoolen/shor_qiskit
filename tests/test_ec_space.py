"""ec_space: the qubit-lean arithmetic ([1128] Sec 3.2's space-optimised variant).

  * the building blocks (controlled CDKM at 3n, CDKM comparator, chunked
    all-ones, increment on borrowed qubits, constant adder) are exact;
  * the CDKM cells, and the lean ones, give the *same answer as ec_approx's
    cells on every input* -- including where both are wrong (the ~f/q
    pseudo-Mersenne approximation) -- with less scratch;
  * the point addition's own add/sub, the square-subtract, the squarer's and
    the dialog walk's ancilla budgets, and the GCD backends compose;
  * a whole windowed point addition with every space option on is correct on
    a Mersenne-prime curve, and uses fewer qubits than the IonQ configuration.
"""
import random

from _ec_util import FULL, ok, random_curve, random_generator, section

import ec_adders as A
import ec_approx as AX
import ec_cost as CO
import ec_gcd as G
import ec_space as SP
import ec_square as SQ
import ec_window as W
from ec_sim import Machine, SimError, run


def toff(m):
    return CO.count(m)["toffoli_paper"]


def outcome(m, init, regs):
    try:
        rd = run(m, init)
        return tuple(rd(r) for r in regs)
    except SimError:
        return "fail"


def test_blocks():
    section("building blocks: exact, at the stated cost")
    for n in range(1, 6):
        m = Machine("and")
        k, x, y, c = m.alloc(1, "k"), m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "c")
        SP.cdkm_cadd(m.ctx, k[0], x, y, c[0])
        for kv in (0, 1):
            for xv in range(1 << n):
                for yv in range(1 << n):
                    rd = run(m, {k: kv, x: xv, y: yv})
                    assert (rd(y), rd(x)) == ((yv + kv * xv) % (1 << n), xv)
        assert toff(m) == 3 * n
        m2 = Machine("and")
        x2, y2, c2, k2 = m2.alloc(n, "x"), m2.alloc(n, "y"), m2.alloc(1, "c"), m2.alloc(1, "k")
        A.cdkm_add(m2.ctx, x2, y2, c2[0], (k2[0],))
        assert toff(m2) == 4 * n                       # the control on MAJ and UMA

        m = Machine("and")
        x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
        SP.lt_cdkm(m, x, y, o[0])
        for xv in range(1 << n):
            for yv in range(1 << n):
                rd = run(m, {x: xv, y: yv})
                assert (rd(o), rd(x), rd(y)) == (int(xv < yv), xv, yv)
        assert toff(m) == 2 * n and m.qc.num_qubits == 2 * n + 2
    ok("controlled CDKM 3n (ec_adders.cdkm_add with a control: 4n); "
       "CDKM comparator 2n with one ancilla")

    for k in range(1, 14):
        m = Machine("and")
        qs, o = m.alloc(k, "q"), m.alloc(2, "o")
        SP.all_ones_lean(m, qs, [o[0], o[1]])
        for v in range(1 << k):
            rd = run(m, {qs: v})
            assert rd(o) == (3 if v == (1 << k) - 1 else 0) and rd(qs) == v
        assert m.qc.num_qubits - k - 2 <= max(k - 1, 0)
    big = Machine("and")
    qs, o = big.alloc(49, "q"), big.alloc(1, "o")
    SP.all_ones_lean(big, qs, [o[0]])
    anc49, ands49 = big.qc.num_qubits - 50, toff(big)
    ok(f"chunked all-ones exact for k <= 13; k = 49: {anc49} ancillas "
       f"(plain chain: 48), {ands49} ANDs (plain: 48)")

    for L in range(1, 7):
        m = Machine("and")
        w, g = m.alloc(L, "w"), m.alloc(L + 1, "g")
        SP.inc_borrowed(m, w, g)
        for wv in range(1 << L):
            for gv in range(0, 1 << (L + 1), 3):
                rd = run(m, {w: wv, g: gv})
                assert (rd(w), rd(g)) == ((wv + 1) % (1 << L), gv)
    rnd = random.Random(1)
    for L in range(2, 9):
        for kk in range(1, 1 << min(L, 5)):
            for nb in (0, 8):
                m = Machine("and")
                c, y = m.alloc(1, "c"), m.alloc(L, "y")
                bo = m.alloc(nb, "b") if nb else None
                SP._cadd_const_lean(m, c[0], y, kk, borrow=list(bo) if nb else [])
                for cv in (0, 1):
                    for yv in range(1 << L):
                        init = {c: cv, y: yv}
                        bv = rnd.randrange(1 << nb) if nb else 0
                        if nb:
                            init[bo] = bv
                        rd = run(m, init)
                        assert rd(y) == (yv + cv * kk) % (1 << L), (L, kk, cv, yv)
                        assert not nb or rd(bo) == bv
    ok("increment on borrowed qubits (any state, restored) and the lean "
       "constant adder: exact")


def test_cells():
    section("CDKM and lean cells = ec_approx's cells, input for input")
    rnd = random.Random(2)
    for q, lsbs in ((61, None), (127, None), (251, None), (2**31 - 1, 12)):
        n = q.bit_length()
        msbs = max(2, n // 2)
        small = q < 300
        cases = ([(cv, xv, yv) for cv in (0, 1) for xv in range(q)
                  for yv in range(0, q, 1 if FULL else 5)] if small else
                 [(rnd.randrange(2), rnd.randrange(q), rnd.randrange(q)) for _ in range(1500)]
                 + [(1, rnd.randrange(q), q - 1 - rnd.randrange(64)) for _ in range(300)])
        built = {}
        for lab, fn in (("approx", lambda m, c, x, y: AX.cmodadd_pm_q(m, c, x, y, q, lsbs, msbs)),
                        ("space", lambda m, c, x, y: SP.cmodadd_pm_q_space(m, c, x, y, q, lsbs, msbs)),
                        ("lean", lambda m, c, x, y: SP.cmodadd_pm_q_space(
                            m, c, x, y, q, lsbs, msbs, lean=True))):
            m = Machine("and")
            c, x, y = m.alloc(1, "c"), m.alloc(n, "x"), m.alloc(n, "y")
            fn(m, c[0], x, y)
            built[lab] = (m, c, x, y)
        ref = [outcome(built["approx"][0], {built["approx"][1]: cv, built["approx"][2]: xv,
                                            built["approx"][3]: yv}, built["approx"][2:])
               for cv, xv, yv in cases]
        for lab in ("space", "lean"):
            m, c, x, y = built[lab]
            got = [outcome(m, {c: cv, x: xv, y: yv}, (x, y)) for cv, xv, yv in cases]
            assert got == ref, (q, lab)
        qb = {k: v[0].qc.num_qubits for k, v in built.items()}
        tf = {k: toff(v[0]) for k, v in built.items()}
        wrong = sum(r == "fail" or r[1] != (yv + cv * xv) % q
                    for r, (cv, xv, yv) in zip(ref, cases))

        dbl = {}
        for lab, fn in (("approx", lambda m, x: AX.moddbl_pm(m, x, q, lsbs)),
                        ("space", lambda m, x: SP.moddbl_pm_space(m, x, q, lsbs)),
                        ("lean", lambda m, x: SP.moddbl_pm_space(m, x, q, lsbs, lean=True)),
                        ("half", lambda m, x: AX.modhalf_pm(m, x, q, lsbs)),
                        ("half-lean", lambda m, x: SP.modhalf_pm_space(m, x, q, lsbs, lean=True))):
            m = Machine("and")
            x = m.alloc(n, "x")
            fn(m, x)
            dbl[lab] = (m, x)
        xs = range(q) if small else [rnd.randrange(q) for _ in range(1500)]
        for a, b in (("approx", "space"), ("approx", "lean"), ("half", "half-lean")):
            assert [outcome(dbl[a][0], {dbl[a][1]: v}, [dbl[a][1]]) for v in xs] == \
                   [outcome(dbl[b][0], {dbl[b][1]: v}, [dbl[b][1]]) for v in xs], (q, a, b)

        sg = {}
        for lab, fn in (("approx", lambda m, e, x, y: AX.csignadd_pm(m, e, x, y, q, lsbs, msbs)),
                        ("lean", lambda m, e, x, y: SP.csignadd_pm_space(
                            m, e, x, y, q, lsbs, msbs, lean=True))):
            m = Machine("and")
            e, x, y = m.alloc(1, "e"), m.alloc(n, "x"), m.alloc(n, "y")
            fn(m, e[0], x, y)
            sg[lab] = (m, e, x, y)
        assert [outcome(sg["approx"][0], dict(zip(sg["approx"][1:], cs)), sg["approx"][2:])
                for cs in cases] == \
               [outcome(sg["lean"][0], dict(zip(sg["lean"][1:], cs)), sg["lean"][2:])
                for cs in cases], q
        print(f"      q={q:<10} lsbs={lsbs!s:<4} {len(cases)} inputs ({wrong} wrong in "
              f"all three alike): qubits {qb['approx']} -> {qb['space']} -> {qb['lean']}, "
              f"Toffolis {tf['approx']} -> {tf['space']} -> {tf['lean']}")
        if not small:
            assert qb["approx"] > qb["space"] > qb["lean"]
    ok("controlled addition, doubling, halving, signed addition: identical "
       "outputs (and identical failures)")


def test_addsub():
    section("the point addition's own y <- y +- x (PointAddCfg.add)")
    for q in (61, 127):
        n = q.bit_length()
        for arith in (G.Exact(q), G.PM(q, msbs=n), G.PMSpace(q, msbs=n, lean=True)):
            for op in ("add", "sub"):
                m = Machine("and")
                x, y = m.alloc(n, "x"), m.alloc(n, "y")
                getattr(arith, op)(m, x, y, q)
                f = (1 << n) - q
                bad = 0
                for xv in range(q):
                    for yv in range(q):
                        want = (yv + xv) % q if op == "add" else (yv - xv) % q
                        if outcome(m, {x: xv, y: yv}, (x, y)) != (xv, want):
                            bad += 1
                            assert arith.name != "exact"
                            # the characterised failures: an unreduced sum in
                            # [q, 2^n), or (run backwards) an input below f
                            assert (q <= xv + yv < 1 << n) if op == "add" else yv < f
                if q == 127:
                    assert bad == 0, (arith.name, op, bad)
                print(f"      q={q} {arith.name:<9} {op}: {bad}/{q * q} wrong")
    ok("exact for Exact and at the Mersenne prime 127; otherwise wrong only "
       "where x + y lands in [q, 2^n) (add) or y < f (sub)")


def test_square():
    section("square-subtract on the CDKM cells; squarer ancilla budgets")
    for n in range(1, 8):
        for space in (True, 1, 3):
            m = Machine("and")
            x, z = m.alloc(n, "x"), m.alloc(2 * n + 1, "z")
            SQ.sqr_int(m, x, z, space=space)
            for xv in range(1 << n):
                rd = run(m, {x: xv})
                assert rd(z) == xv * xv and rd(x) == xv
    ok("sqr_int exact with every subtraction CDKM, or Gidney within a budget")
    rnd = random.Random(3)
    for q in (127, 251):
        n = q.bit_length()
        _, f = AX.pseudo_mersenne(q)
        cases = [(rnd.randrange(2), rnd.randrange(q), rnd.randrange(q)) for _ in range(400)]
        for kw in (dict(), dict(lean=True, sqr_space=4)):
            m = Machine("and")
            c, s, a = m.alloc(1, "c"), m.alloc(n, "s"), m.alloc(n, "a")
            SP.csub_square_pm_space(m, c[0], s, a, q, msbs=n, **kw)
            bad = sum(outcome(m, {c: cv, s: sv, a: av}, (a, s)) != ((av - cv * sv * sv) % q, sv)
                      for cv, sv, av in cases)
            if q == 127:
                assert bad == 0, (q, kw, bad)
            else:
                assert bad / len(cases) <= 3 * f / q + 0.02, (q, kw, bad)
            print(f"      q={q} {str(kw):<32} {m.qc.num_qubits} qubits "
                  f"{toff(m):>5} Toffoli-eq, {bad}/{len(cases)} wrong")
    ok("exact at q = 127; at q = 251 wrong only on the ~f/q inputs (acc < f "
       "for the reversed PM adder, z_lo or z_hi >= q for the fold)")


def test_gcd():
    section("GCD backends on the lean cells")
    rnd = random.Random(4)
    q = 61
    pairs = [(xv, yv) for xv in range(1, q) for yv in range(q)]
    if not FULL:
        pairs = pairs[::11]
    for kw in (dict(walk_space=True), dict(walk_space=8),
               dict(walk_space=True, c_pad=2.3, share=True, compress="fig1", fused_cmp=True)):
        for fn in ("mul", "div"):
            m = Machine("and")
            x, y = m.alloc(6, "x"), m.alloc(6, "y")
            getattr(G.Dialog(**kw), fn)(m, x, y, q)
            for xv, yv in pairs:
                want = xv * yv % q if fn == "mul" else yv * pow(xv, -1, q) % q
                assert outcome(m, {x: xv, y: yv}, (x, y)) == (xv, want), (kw, fn)
    ok(f"Dialog walk on the controlled CDKM (and with a budget): exact, q = {q}")

    for q in (61, 127):
        n = q.bit_length()
        ps = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(60)]
        # cmp_msbs = n + 1: the walk's registers are n + 1 wide, so this is
        # the exact comparison and only the arithmetic differs
        for mk in (lambda a: G.Dialog(arith=a, fused_cmp=True, cmp_msbs=n + 1, c_pad=2.3,
                                      share=True, compress="fig1", walk_space=True),
                   lambda a: G.CondInv(arith=a, cmp_msbs=n + 1, c_pad=2.3, replay="standard",
                                       compress="fig1", reuse_x=True, walk_space=True)):
            res = []
            for arith in (G.PM(q, msbs=n), G.PMSpace(q, msbs=n),
                          G.PMSpace(q, msbs=n, lean=True)):
                m = Machine("and")
                x, y = m.alloc(n, "x"), m.alloc(n, "y")
                mk(arith).mul(m, x, y, q)
                res.append([outcome(m, {x: xv, y: yv}, (x, y)) for xv, yv in ps])
            assert res[0] == res[1] == res[2], q
            if q == 127:
                assert res[0] == [(xv, xv * yv % q) for xv, yv in ps]
    ok("PM, PMSpace and lean PMSpace replays agree input for input (exact at 127)")


def test_point_add():
    section("a whole windowed point addition, every space option on")
    rnd = random.Random(127)
    curve, pts = random_curve(rnd, pmax=127, pmin=127)
    G0 = random_generator(rnd, curve, pts, 8)
    q, n = 127, 7
    tab = W.masked_window_points(curve, G0, 2, random.Random(5))[0]
    pml = G.PMSpace(q, msbs=n, lean=True)
    lean = W.PointAddCfg(
        lookup="mbu", merge_xy=True, offsets=True, free_xy1=True, add=pml,
        square=lambda m, c, s, a, p: SP.csub_square_pm_space(m, c, s, a, p, msbs=n,
                                                             sqr_space=6, lean=True),
        mul=G.Dialog(arith=pml, fused_cmp=True, cmp_msbs=n, c_pad=2.3, share=True,
                     compress="fig1", walk_space=10))
    ionq = W.PointAddCfg(
        lookup="mbu", merge_xy=True, offsets=True, free_xy1=True,
        square=lambda m, c, s, a, p: SQ.csub_square_pm(m, c, s, a, p, msbs=n),
        mul=G.CondInv(arith=G.PM(q, msbs=n), cmp_msbs=n, c_pad=2.3, replay="ci"))
    counts = {}
    for lab, cfg in (("IonQ-style", ionq), ("space, lean", lean)):
        m = Machine("and")
        a, x, y = m.alloc(2, "a"), m.alloc(n, "x"), m.alloc(n, "y")
        W.windowed_point_add_cfg(m, a, x, y, tab, q, cfg)
        counts[lab] = CO.count(m)
        if lab != "space, lean":
            continue
        good = 0
        import ec_classical as C
        Rs = [R for R in curve.points() if not R.inf]
        for R in (Rs if FULL else rnd.sample(Rs, min(24, len(Rs)))):
            for i, T in enumerate(tab):
                if C.point_add_exceptional(curve, R, T):
                    continue
                S = curve.add(R, T)
                if S.inf:
                    continue
                assert outcome(m, {a: i, x: R.x, y: R.y}, (x, y, a)) == (S.x, S.y, i), (R, i)
                good += 1
    for lab, c in counts.items():
        print(f"      {lab:<12} {c['qubits']:>4} qubits {c['toffoli_paper']:>7} Toffoli-eq")
    assert counts["space, lean"]["qubits"] < counts["IonQ-style"]["qubits"]
    ok(f"{good} (R, i) pairs on {curve.name} (p = 127) exact; fewer qubits than "
       f"the IonQ configuration at toy size too (n = 256: bench/ec_space_256.py)")


def main():
    test_blocks()
    test_cells()
    test_addsub()
    test_square()
    test_gcd()
    test_point_add()


if __name__ == "__main__":
    main()
    print("\ntest_ec_space: all passed")
