"""ec_mbu: measurement-based uncomputation as logical gates, and proof that the
literal measure-and-repair realisation is phase-correct.

The logical gates are checked the way the rest of the EC stack is: exhaustively
on basis states with ec_sim, ancillas asserted clean where freed.  Then
`run_live` executes the real X-basis measurements on the sparse simulator and
applies the repairs built from the actual outcomes; the uncompute is correct iff
superpositions come back intact and all basis inputs come back with one phase.
"""
import math
import random

from _ec_util import ok, section

import depth as D
import ec_adders as A
import ec_cost as CO
import ec_mbu as MB
from ec_gates import Ctx
from ec_sim import Machine, run
from qiskit.circuit import QuantumCircuit, QuantumRegister
from sparse_sim import SparseSimError, simulate_sparse, uniform_state


def ands_in(op):
    return sum(1 for ci in op.definition.data if ci.operation.name == "ecand")


def test_lookup_gate():
    section("LookupGate: cost = the ANDs of its own definition; function exact")
    rnd = random.Random(1)
    for w in range(1, 5):
        for ctl in (True, False):
            nout = 3
            table = [rnd.randrange(8) for _ in range(1 << w)]
            g = MB.LookupGate(w, nout, table, ctl)
            assert ands_in(g) == g.ec_cost["toffoli"] == MB.walk_ands(w, ctl), (w, ctl)
            m = Machine("and")
            c = m.alloc(1, "c") if ctl else None
            a, o = m.alloc(w, "a"), m.alloc(nout, "o")
            MB.lookup(m, a, o, table, ctrl=c[0] if ctl else None)
            for av in range(1 << w):
                for cv in ((0, 1) if ctl else (1,)):
                    for ov in (0, 5):
                        ins = {a: av, o: ov}
                        if ctl:
                            ins[c] = cv
                        rd = run(m, ins)
                        assert rd(o) == ov ^ (table[av] if cv else 0) and rd(a) == av
            assert CO.count(m)["toffoli_paper"] == MB.walk_ands(w, ctl)
    ok("w = 1..4, controlled (L-1 ANDs) and uncontrolled (L-2): exhaustive")


def test_phase_lookup():
    section("phase_lookup: diag((-1)^F) at every split, cost as declared")
    rnd = random.Random(2)
    for w in range(1, 6):
        for ctl in (True, False):
            for l in range(0, w + 1):
                F = [rnd.getrandbits(1) for _ in range(1 << w)]
                k = MB.phase_ancillas(w, l, ctl)
                qc = QuantumCircuit(QuantumRegister(1, "c"), QuantumRegister(w, "a"),
                                    QuantumRegister(max(k, 1), "k"))
                c, a, anc = qc.qregs[0][0], list(qc.qregs[1]), list(qc.qregs[2])
                MB.phase_lookup(Ctx(qc, "and"), c if ctl else None, a, F, anc[:k], l)
                assert D.profile(qc)["toffoli"] == MB.phase_cost(w, l, ctl), (w, l, ctl)
                st = uniform_state(qc, None, c=([c], (0, 1)), a=(a, range(1 << w)))
                br = simulate_sparse(qc, state=st).only()
                br.assert_clean(anc)
                amps = br.amplitudes([c], a)
                amp0 = 1 / math.sqrt(2 << w)
                for (cv, av), amp in amps.items():
                    sign = (-1) ** (F[av] * (cv if ctl else 1))
                    assert abs(amp - sign * amp0) < 1e-12, (w, l, ctl, cv, av)
    ok("w = 1..5, every l, controlled and not: phases exact, ancillas clean")
    for w in (8, 12, 16):
        l, cost = MB.best_split(w, 10**9)
        print(f"      w={w:>2}: best split l={l}, {cost} ANDs "
              f"({MB.phase_ancillas(w, l)} ancillas) vs {MB.walk_ands(w)} unsplit")
    assert MB.best_split(16, 10**9)[1] <= 3 * 256


def build_pair(w, nout, table, grouped, fix=True, ctl=False, two=False):
    m = Machine("and")
    c = m.alloc(1, "c") if ctl else None
    a, o = m.alloc(w, "a"), m.alloc(nout, "o")
    cq = c[0] if ctl else None
    MB.lookup(m, a, o, table, cq)
    o2 = None
    if two:
        o2 = m.alloc(nout, "o2")
        MB.lookup(m, a, o2, [t ^ 5 for t in table], cq)
    MB.unlookup(m, a, o, table, cq, group="g" if grouped else None)
    if two:
        MB.unlookup(m, a, o2, [t ^ 5 for t in table], cq, group="g" if grouped else None)
    if grouped and fix:
        MB.phase_fix(m, "g")
    return m, c, a, o, o2


def test_unlookup_live():
    section("unlookup: exact logically, phase-correct when run literally")
    rnd = random.Random(3)
    outcome_sets = [MB.all_ones_outcome(), lambda i, c: 0] + \
        [MB.random_outcomes(s) for s in range(6)]
    for w in (1, 2, 3, 4):
        for ctl in (False, True):
            for grouped, two in ((False, False), (True, False), (True, True)):
                table = [rnd.randrange(1, 16) for _ in range(1 << w)]
                m, c, a, o, o2 = build_pair(w, 4, table, grouped, ctl=ctl, two=two)
                # logical: exhaustive on basis states, output cleared where freed
                for av in range(1 << w):
                    ins = {a: av}
                    if ctl:
                        ins[c] = 1
                    rd = run(m, ins)
                    assert rd(o) == 0 and rd(a) == av
                # live: address (and control) in superposition, comes back intact
                regs = {"a": (a, range(1 << w))}
                if ctl:
                    regs["c"] = (c, (0, 1))
                st = uniform_state(m.qc, None, **regs)
                for oc in outcome_sets[:4]:
                    S = MB.run_live(m.qc, state=st, outcomes=oc, checks=m.checks)
                    amps = S.branch.amplitudes(a, *( [c] if ctl else [] ))
                    vals = list(amps.values())
                    assert len(vals) == len(st), (w, ctl, grouped)
                    assert max(abs(v - vals[0]) for v in vals) < 1e-12
                    S.branch.assert_clean(o)
                # and every basis input shares one phase, under 8 outcome patterns
                inputs = [{a: av, **({c: cv} if ctl else {})}
                          for av in range(1 << w) for cv in ((0, 1) if ctl else (0,))]
                MB.live_coherent(m.qc, inputs, outcome_sets, checks=m.checks)
    ok("w = 1..4, controlled or not, ungrouped / grouped / two grouped loads: "
       "superpositions restored exactly, one phase across all inputs")

    table = [3, 8, 12, 7]                  # parities 0,1,0,1 under m = 1111
    m, c, a, o, _ = build_pair(2, 4, table, grouped=True, fix=False)
    try:
        MB.live_coherent(m.qc, [{a: v} for v in range(4)], [MB.all_ones_outcome()])
        raise AssertionError("a missing phase fix went unnoticed")
    except SparseSimError:
        pass
    ok("a group whose phase fix is left out is caught")


def test_cost_and_inverse():
    section("costs, and running a builder backwards")
    w, nout = 4, 5
    table = list(range(3, 3 + 16))
    m, *_ = build_pair(w, nout, table, grouped=False)
    cst = CO.count(m)
    l, fix = MB.best_split(w, MB.phase_ancillas(w, w // 2, False), False)
    assert cst["toffoli_paper"] == MB.walk_ands(w, False) + fix
    assert cst["measure"] == nout
    ok(f"lookup + unlookup, w={w}: {cst['toffoli_paper']} Toffoli-eq "
       f"= {MB.walk_ands(w, False)} load + {fix} repair; {nout} measurements")

    def body(mm, a, o, x):
        MB.lookup(mm, a, o, table)
        A.add(mm.ctx, o, x, mm.anc(nout - 1, "ad"))      # use the loaded value
        MB.unlookup(mm, a, o, table)

    m = Machine("and")
    a, o, x = m.alloc(w, "a"), m.alloc(nout, "o"), m.alloc(nout, "x")
    m.emit_inverse(body, m, a, o, x)
    for av in range(16):
        rd = run(m, {a: av, x: 7})
        assert rd(x) == (7 - table[av]) % 32 and rd(o) == 0
    names = [ci.operation.name for ci in m.qc.data]
    assert names[0] == "ec_lookup" and "ec_unlookup" in names[-3:]
    st = uniform_state(m.qc, {x: 7}, a=(a, range(16)))
    S = MB.run_live(m.qc, state=st, outcomes=MB.random_outcomes(9))
    amps = list(S.branch.amplitudes(a).values())
    assert len(amps) == 16 and max(abs(v - amps[0]) for v in amps) < 1e-12
    ok("emit_inverse turns unlookup into lookup and back: the reversed "
       "builder subtracts, and is phase-correct run literally")


def test_mbu_flag():
    section("MbuFlagGate: exact and approximate phase repairs")
    n = 5

    def recompute(qc, qs):
        x, y, f, anc = qs[:n], qs[n:2 * n], qs[2 * n], qs[2 * n + 1:]
        A.gt_uint(Ctx(qc, "and"), x, y, f, anc)

    def phase_exact(qc, qs):
        x, y, anc = qs[:n], qs[n:2 * n], qs[2 * n + 1:]
        t, sc = anc[0], anc[1:]
        A.gt_uint(Ctx(qc, "and"), x, y, t, sc)
        qc.z(t)
        A.gt_uint(Ctx(qc, "and"), x, y, t, sc)

    def phase_top(k):
        def ph(qc, qs):
            x, y, anc = qs[:n], qs[n:2 * n], qs[2 * n + 1:]
            t, sc = anc[0], anc[1:]
            A.gt_uint(Ctx(qc, "and"), x[n - k:], y[n - k:], t, sc)
            qc.z(t)
            A.gt_uint(Ctx(qc, "and"), x[n - k:], y[n - k:], t, sc)
        return ph

    def build(phase):
        m = Machine("and")
        x, y, f = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "f")
        sc = m.anc(n, "sc")
        A.gt_uint(m.ctx, x, y, f[0], sc)
        m.free(sc)
        MB.mbu_flag(m, f[0], list(x) + list(y), recompute, phase, nanc=n + 1)
        return m, x, y, f

    m, x, y, f = build(phase_exact)
    inputs = [{x: a, y: b} for a in range(1 << n) for b in range(1 << n)]
    for i in inputs[::37]:
        assert run(m, i)(f) == 0
    MB.live_coherent(m.qc, inputs, [MB.all_ones_outcome()], checks=m.checks)
    assert CO.count(m)["measure"] == 1
    ok(f"flag = [x > y] cleared by measurement, exact comparator repair: "
       f"all {len(inputs)} inputs coherent")

    for k in (1, 2, 3, 5):
        m, x, y, f = build(phase_top(k))
        frac = MB.live_faults(m.qc, inputs, MB.all_ones_outcome())
        wrong = sum(1 for a in range(1 << n) for b in range(1 << n)
                    if (a > b) != ((a >> (n - k)) > (b >> (n - k))))
        want = min(wrong, len(inputs) - wrong) / len(inputs)
        assert abs(frac - want) < 1e-12, (k, frac, want)
        print(f"      repair on the top {k} bits: {100 * frac:5.2f}% of inputs "
              f"left with the wrong phase (= inputs where the truncated "
              f"comparison disagrees)")
    ok("an approximate repair's phase-fault rate is exactly the disagreement "
       "rate of its truncated comparator, and 0 at full width")


def test_anf_phaseup():
    section("[G25] ANF lookup and power-product phaseup")
    rnd = random.Random(8)
    for w in range(1, 6):
        table = [rnd.randrange(16) for _ in range(1 << w)]
        g = MB.LookupGate(w, 4, table, controlled=False, anf=True)
        assert ands_in(g) == g.ec_cost["toffoli"] == MB.anf_ands(w)
        m = Machine("and")
        a, o = m.alloc(w, "a"), m.alloc(4, "o")
        MB.lookup(m, a, o, table, anf=True)
        for av in range(1 << w):
            assert run(m, {a: av, o: 3})(o) == 3 ^ table[av]
    ok("ANF lookup: 2^w - w - 1 ANDs (declared = built), exact, w = 1..5")

    for w in range(1, 7):
        for l in range(0, w + 1):
            F = [rnd.getrandbits(1) for _ in range(1 << w)]
            k = MB.phaseup_ancillas(w, l)
            qc = QuantumCircuit(QuantumRegister(w, "a"), QuantumRegister(max(k, 1), "k"))
            a, anc = list(qc.qregs[0]), list(qc.qregs[1])
            MB.phaseup(Ctx(qc, "and"), a, F, anc[:k], l)
            assert D.profile(qc)["toffoli"] == MB.phaseup_ands(w, l)
            br = simulate_sparse(qc, state=uniform_state(qc, None, a=(a, range(1 << w)))).only()
            br.assert_clean(anc)
            amps = br.amplitudes(a)
            ref = amps[(0,)] * (-1) ** F[0]
            for (av,), amp in amps.items():
                assert abs(amp - ref * (-1) ** F[av]) < 1e-12, (w, l, av)
    ok("phaseup = diag((-1)^F) up to global phase, every split, w = 1..6")
    for w in (8, 12, 16):
        print(f"      w={w:>2}: phaseup {MB.phaseup_ands(w):>4} ANDs "
              f"({MB.phaseup_ancillas(w)} ancillas) vs one-hot split "
              f"{MB.best_split(w, 10**9)[1]:>4}")
    assert MB.phaseup_ands(16) < MB.best_split(16, 10**9)[1]

    outcome_sets = [MB.all_ones_outcome()] + [MB.random_outcomes(s_) for s_ in range(4)]
    for w in (2, 3):
        for ctl in (False, True):
            for grouped in (False, True):
                table = [rnd.randrange(1, 16) for _ in range(1 << w)]
                m = Machine("and")
                c = m.alloc(1, "c") if ctl else None
                a, o = m.alloc(w, "a"), m.alloc(4, "o")
                cq = c[0] if ctl else None
                MB.lookup(m, a, o, table, cq)
                MB.unlookup(m, a, o, table, cq, group="g" if grouped else None,
                            repair="phaseup")
                if grouped:
                    MB.phase_fix(m, "g", repair="phaseup")
                inputs = [{a: av, **({c: cv} if ctl else {})}
                          for av in range(1 << w) for cv in ((0, 1) if ctl else (0,))]
                MB.live_coherent(m.qc, inputs, outcome_sets, checks=m.checks)
    ok("unlookup with the phaseup repair: phase-correct run literally, "
       "grouped or not, controlled or not")


def main():
    test_lookup_gate()
    test_anf_phaseup()
    test_phase_lookup()
    test_unlookup_live()
    test_cost_and_inverse()
    test_mbu_flag()


if __name__ == "__main__":
    main()
    print("\ntest_ec_mbu: all passed")
