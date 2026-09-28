"""ec_proj_q: [106] Sec 4.2 with a quantum addend -- the Jacobian addition
(mixed and full, plain and controlled), its zig-zag chain, and the projective
ECDLP circuit that converts to affine once at the end.

Every exceptional input is run too: the circuit computes the *formula* exactly
and cleanly on all of them, and the test states what that formula gives when it
is not the group law, with the count, rather than skipping them.
"""
import random

from _ec_util import FULL, ok, scope, section

import ec_classical as C
import ec_cost as CO
import ec_modarith as MA
import ec_mult as MU
import ec_proj_q as PQ
import ec_shor as SH
from ec_classical import zigzag_registers, zigzag_schedule
from ec_sim import Machine, run

CURVE13 = C.Curve(13, 1, 1, "p13")               # 18 points, cyclic


def build_add(curve, full, ctrl):
    """One addition on fresh registers: (m, P1, P2, P3, q)."""
    n = curve.p.bit_length()
    m = Machine("and")
    P1 = tuple(m.alloc(n, c + "1") for c in "XYZ")
    P2 = tuple(m.alloc(n, c + "2") for c in ("XYZ" if full else "XY"))
    P3 = tuple(m.alloc(n, c + "3") for c in "XYZ")
    q = m.alloc(1, "q")
    if ctrl:
        PQ.jacobian_add_q_ctrl(m, q[0], P1, P2, P3, curve.p)
    else:
        PQ.jacobian_add_q(m, P1, P2, P3, curve.p)
    return m, P1, P2, P3, q


def check_add(curve, built, A, a, B, b, qv, kinds):
    """Run one input; assert the answer and the input registers; tally kinds.
    Ancilla cleanliness is enforced by `run` at every `free`."""
    m, P1, P2, P3, q = built
    init = {**dict(zip(P1, a)), **dict(zip(P2, b)), q: qv}
    rd = run(m, init)
    got = tuple(rd(R) for R in P3)
    assert tuple(rd(R) for R in P1) == tuple(a)
    assert tuple(rd(R) for R in P2) == tuple(b)
    if not qv:
        assert got == tuple(a), (A, B, a, b, got)       # a bit copy of P1
        return
    assert got == PQ.jacobian_add_q_ref(curve, a, b), (A, B, a, b, got)
    why = PQ.jacobian_exceptional(curve, A, B)
    kinds[why] = kinds.get(why, 0) + 1
    if why is None:
        assert PQ.from_jacobian(curve, *got) == curve.add(A, B), (A, B, got)
    elif why == "P1 = -P2":
        assert got[2] == 0 and PQ.from_jacobian(curve, *got) == C.O, got
    else:                                               # P1 = P2, or an O
        assert got == (0, 0, 0), (why, got)


def fmt(kinds):
    good = kinds.get(None, 0)
    bad = ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items(), key=str) if k)
    return f"{good} generic exact; exceptional {bad}"


def test_formula():
    section("reference: the 16-multiplication law, and Alg. 4 as its Z2 = 1 case")
    r = random.Random(1)
    for curve in (C.CLASSIQ, C.TOY11, CURVE13):
        pts, p = curve.points(), curve.p
        for A in pts:
            for B in pts:
                a = PQ.jacobian_rep(curve, A, r.randrange(1, p))
                b = PQ.jacobian_rep(curve, B, r.randrange(1, p))
                got = PQ.jacobian_add_q_ref(curve, a, b)
                if PQ.jacobian_exceptional(curve, A, B) is None:
                    assert PQ.from_jacobian(curve, *got) == curve.add(A, B)
                if not B.inf:
                    assert PQ.jacobian_add_q_ref(curve, a, (B.x, B.y)) == \
                        C.jacobian_add_ref(curve, *a, B.x, B.y)
                    assert PQ.jacobian_add_q_ref(curve, a, (B.x, B.y, 1)) == \
                        PQ.jacobian_add_q_ref(curve, a, (B.x, B.y))
    ok("group law on every generic pair at p = 7, 11, 13; the mixed form equals "
       "ec_classical.jacobian_add_ref and the full form at Z2 = 1")

    calls = [0]
    orig = MU.modmul

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    MU.modmul = counting
    try:
        for full, want in ((False, 11), (True, 16)):
            for ctrl in (False, True):
                calls[0] = 0
                build_add(C.CLASSIQ, full, ctrl)
                assert calls[0] == want, (full, ctrl, calls[0])
    finally:
        MU.modmul = orig
    ok("multiplications: 11 with an affine addend ([106] Alg. 4), 16 with a "
       "Jacobian one; the control adds none")


def test_add_mixed():
    section("jacobian_add_q, quantum affine addend: [106] Alg. 4 as a lookup feeds it")
    curve = C.CLASSIQ
    p, pts = curve.p, curve.points()
    built, kinds = build_add(curve, False, False), {}
    for A in pts:                                        # every representative
        for lam in range(1, p):
            a = PQ.jacobian_rep(curve, A, lam)
            for B in pts[1:]:
                check_add(curve, built, A, a, B, (B.x, B.y), 1, kinds)
    ok(f"p = 7, every (X1:Y1:Z1) x every affine P2: {fmt(kinds)}")

    r, curve = random.Random(2), CURVE13
    p, pts = curve.p, curve.points()
    built, kinds = build_add(curve, False, False), {}
    for _ in range(scope(150, 1500)):
        A, B = r.choice(pts), r.choice(pts[1:])
        a = PQ.jacobian_rep(curve, A, r.randrange(1, p))
        check_add(curve, built, A, a, B, (B.x, B.y), 1, kinds)
    ok(f"p = 13, random pairs: {fmt(kinds)}")


def test_add_jacobian():
    section("jacobian_add_q, quantum Jacobian addend: both Z's live")
    curve = C.CLASSIQ
    p, pts = curve.p, curve.points()
    r = random.Random(3)
    built, kinds = build_add(curve, True, False), {}
    lams = range(1, p) if FULL else None
    for A in pts:                                        # every pair of points
        for B in pts:
            if lams:
                reps = [(l1, l2) for l1 in lams for l2 in lams]
            else:
                reps = [(r.randrange(1, p), r.randrange(1, p)) for _ in range(2)]
            for l1, l2 in reps:
                check_add(curve, built, A, PQ.jacobian_rep(curve, A, l1),
                          B, PQ.jacobian_rep(curve, B, l2), 1, kinds)
    ok(f"p = 7, every pair of points incl. O, "
       f"{'every' if FULL else '2 random'} scalings: {fmt(kinds)}")

    r, curve = random.Random(4), C.TOY11
    p, pts = curve.p, curve.points()
    built, kinds = build_add(curve, True, False), {}
    for _ in range(scope(100, 1000)):
        A, B = r.choice(pts), r.choice(pts)
        check_add(curve, built, A, PQ.jacobian_rep(curve, A, r.randrange(1, p)),
                  B, PQ.jacobian_rep(curve, B, r.randrange(1, p)), 1, kinds)
    ok(f"p = 11, random pairs: {fmt(kinds)}")


def test_add_ctrl():
    section("jacobian_add_q_ctrl: out = P1 + P2 if q else a bit copy of P1")
    r = random.Random(5)
    for curve in (C.CLASSIQ, CURVE13):
        p, pts = curve.p, curve.points()
        for full in (False, True):
            built, kinds = build_add(curve, full, True), {}
            for _ in range(scope(60, 400)):
                A = r.choice(pts)
                B = r.choice(pts if full else pts[1:])
                a = PQ.jacobian_rep(curve, A, r.randrange(1, p))
                b = PQ.jacobian_rep(curve, B, r.randrange(1, p)) if full else (B.x, B.y)
                for qv in (0, 1):
                    check_add(curve, built, A, a, B, b, qv, kinds)
            ok(f"p = {p}, {'Jacobian' if full else 'affine'} addend, q = 0 and 1 "
               f"on each input: {fmt(kinds)}; q = 0 exact on all, exceptional or not")

    for p in (7, 13, 251):
        n = p.bit_length()
        m = Machine("and")
        MA.modsub(m, m.alloc(n, "a"), m.alloc(n, "b"), p)
        sub = CO.count(m)["toffoli_paper"]
        rows = {row[0]: row[3] for row in PQ.cost_rows(p)}
        for form in ("affine", "Jacobian"):
            base = rows[f"Jacobian + quantum {form}"]
            assert rows[f"Jacobian + quantum {form}, ctrl"] - base == 3 * n + 2 * sub
    ok("the control costs exactly 3n ANDs + 2 modular subtractions (p = 7, 13, 251)")


def test_zigzag():
    section("zigzag_chain_q: 2-4 additions with quantum addends on m register triples")
    r = random.Random(6)
    for curve in (C.CLASSIQ, CURVE13):
        p, n = curve.p, curve.p.bit_length()
        pts = [P for P in curve.points() if not P.inf]
        for N in (2, 3, 4):
            for kind in ("jacobian", "affine", "ctrl"):
                m = Machine("and")
                start = tuple(m.alloc(n, c + "0") for c in "XYZ")
                width = "XY" if kind == "affine" else "XYZ"
                adds = [tuple(m.alloc(n, f"{c}a{j}") for c in width) for j in range(N)]
                qs = m.alloc(N, "q") if kind == "ctrl" else None
                final, resid, trips = PQ.zigzag_chain_q(
                    m, start, adds, p, ctrls=list(qs) if qs else None)
                assert len(trips) == zigzag_registers(N)
                keep = sorted(zigzag_schedule(N)[2])
                assert len(resid) == len(keep)
                done, tries = 0, 0
                while done < scope(5, 25):
                    tries += 1
                    assert tries < 200, (N, kind)
                    S0, Bs = r.choice(pts), [r.choice(pts) for _ in range(N)]
                    bits = [r.randrange(2) for _ in range(N)] if qs else [1] * N
                    acc, path, bad = S0, [S0], False
                    for B, bit in zip(Bs, bits):
                        if bit:
                            bad = bad or PQ.jacobian_exceptional(curve, acc, B)
                            acc = curve.add(acc, B)
                        path.append(acc)
                    if bad:
                        continue
                    ins = dict(zip(start, PQ.jacobian_rep(curve, S0, r.randrange(1, p))))
                    for regs, B in zip(adds, Bs):
                        rep = PQ.jacobian_rep(curve, B, r.randrange(1, p))
                        ins.update(zip(regs, (B.x, B.y) if kind == "affine" else rep))
                    if qs:
                        ins[qs] = sum(b << i for i, b in enumerate(bits))
                    rd = run(m, ins)
                    got = PQ.from_jacobian(curve, *(rd(R) for R in final))
                    assert got == acc, (N, kind, got, acc)
                    for j, t in zip(keep, resid):          # the schedule's residue
                        assert PQ.from_jacobian(curve, *(rd(R) for R in t)) == path[j]
                    assert all(rd(R) == v for R, v in ins.items())   # inputs kept
                    done += 1
        ok(f"p = {p}: N = 2, 3, 4 additions with Jacobian / affine / controlled "
           f"addends on {', '.join(str(zigzag_registers(N)) for N in (2, 3, 4))} "
           f"triples; the final point and every residual point as the tape predicts")


def check_ecdlp(curve, P, Q, plan, pairs, label, **kw):
    order = plan["order"]
    m, info = PQ.ecdlp_projective(curve, P, Q, order, plan=plan, **kw)
    good, bad = PQ.check_ecdlp_projective(curve, P, Q, m, info, pairs)
    assert good, label
    c = CO.count(m)
    exc = (f"exceptional {bad}, each run and seen to fail loudly" if bad
           else "none exceptional")
    ok(f"{label}: {good}/{len(pairs)} (k, l) give target + [k]P + [l]Q after the "
       f"one inversion ({info['additions']} additions on {info['m_regs']} triples, "
       f"{info['pa_calls']} PA/PA-dg, {c['qubits']} qubits, "
       f"{c['toffoli_paper']} Toffolis); {exc}")
    return m, info


def test_ecdlp():
    section("ecdlp_projective: projective throughout, one Z-inversion, affine out")
    r = random.Random(7)
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    plan = PQ.pick_plan(curve, P, Q, order)
    grid = [(k, l) for k in range(1 << plan["bits_k"]) for l in range(1 << plan["bits_l"])]
    assert PQ.exceptional_count(curve, plan)[0] == 0      # S outside <P>
    check_ecdlp(curve, P, Q, plan, grid if FULL else r.sample(grid, 24),
                "p = 7, w = 1 lookups")

    plan = PQ.pick_plan(curve, P, Q, order, form="ctrl")
    assert PQ.exceptional_count(curve, plan)[0] == 0
    check_ecdlp(curve, P, Q, plan, r.sample(grid, scope(8, 64)),
                "p = 7, controlled additions")

    plan = PQ.pick_plan(curve, P, Q, order, w=2, seeds=range(8))
    exc, why = PQ.exceptional_count(curve, plan)
    grid2 = [(k, l) for k in range(1 << plan["bits_k"]) for l in range(1 << plan["bits_l"])]
    check_ecdlp(curve, P, Q, plan, r.sample(grid2, scope(8, 40)), "p = 7, w = 2 lookups")
    ok(f"   (w = 2 grid: {exc}/{len(grid2)} exceptional: {why})")

    curve, P = C.TOY11, C.TOY11_G
    Q = curve.mul(5, P)
    order = curve.point_order(P)
    plan = PQ.pick_plan(curve, P, Q, order, seeds=range(16))
    exc, why = PQ.exceptional_count(curve, plan)
    grid = [(k, l) for k in range(1 << plan["bits_k"]) for l in range(1 << plan["bits_l"])]
    check_ecdlp(curve, P, Q, plan, r.sample(grid, scope(8, 60)), "p = 11, w = 1 lookups")
    ok(f"   (p = 11 has prime order 13, so no coset to hide in: {exc}/{len(grid)} "
       f"of the grid exceptional: {why})")

    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    plan = PQ.pick_plan(curve, P, Q, curve.point_order(P))
    grid = [(k, l) for k in range(1 << plan["bits_k"]) for l in range(1 << plan["bits_l"])]
    m, info = PQ.ecdlp_projective(curve, P, Q, plan["order"], plan=plan, uncompute=False)
    good, _ = PQ.check_ecdlp_projective(curve, P, Q, m, info, r.sample(grid, 4))
    assert good == 4
    ok(f"uncompute=False ([106]'s forward half): same answers, "
       f"{CO.count(m)['toffoli_paper']} Toffolis, {info['m_regs']} triples + the start "
       f"left as path-dependent garbage")


def test_costs():
    section("cost: one addition, projective against affine in place")
    ps = (7, 13, 65521) if FULL else (7, 13)
    for p in ps:
        rows = PQ.cost_rows(p)
        print(CO.table(rows, ["circuit", "n", "qubits", "Toffoli"]))
        t = {row[0]: row[3] for row in rows}
        assert t["Jacobian + quantum affine"] < t["affine in place, ctrl (ec_pointadd)"]
        assert t["Jacobian + classical affine (ec_proj)"] < t["Jacobian + quantum affine"]
    ok("per addition, Alg. 4 with a quantum addend is cheaper than the affine "
       "in-place addition and dearer than with a constant addend")

    import hier as H
    targets = H.DEFAULT_TARGETS + [
        ("ec_proj_q", ["jacobian_add_q", "jacobian_add_q_ctrl", "to_affine"])]
    flat = PQ.cost_rows(13)
    with H.tracing(targets):
        tree = PQ.cost_rows(13, count=H.count)
    assert tree == flat, (tree, flat)
    ok("hier.tracing with ec_proj_q's builders as boundaries counts exactly the "
       "flat qubits and Toffolis (so n = 256 is a 25 s build)")

    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    ma, _ = SH.ecdlp_circuit(curve, P, Q, order, oracle="arith")
    plan = PQ.pick_plan(curve, P, Q, order)
    rows = [["affine (ec_shor, oracle=arith)", *(CO.count(ma)[k] for k in ("qubits", "toffoli_paper"))]]
    for un in (False, True):
        mp, info = PQ.ecdlp_projective(curve, P, Q, order, plan=plan, uncompute=un)
        rows.append([f"projective, {'with' if un else 'no'} unwinding ({info['pa_calls']} PA)",
                     *(CO.count(mp)[k] for k in ("qubits", "toffoli_paper"))])
    print(CO.table(rows, ["p = 7 ECDLP oracle", "qubits", "Toffoli"]))
    ok("whole oracle: the zig-zag re-runs and the Bennett unwinding cost more "
       "additions than the per-addition saving buys back at this arithmetic")


def main():
    test_formula()
    test_add_mixed()
    test_add_jacobian()
    test_add_ctrl()
    test_zigzag()
    test_ecdlp()
    test_costs()


if __name__ == "__main__":
    main()
    print("\ntest_ec_proj_q: all passed")
