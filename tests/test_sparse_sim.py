"""sparse_sim: agrees with every other simulator where they overlap.

  * the closed-form order-finding distribution (full counting register);
  * _dynsim's exact branch enumeration (one-control semiclassical circuits);
  * qiskit's dense Statevector, amplitude for amplitude including phase, on
    random circuits -- and on the same circuits embedded in 70 qubits, where
    the keys stop fitting in uint64;
  * ec_sim on the reversible EC builders (same output bits, one basis state,
    every ancilla check passes).

Then the property it exists for: assert_coherent accepts a measurement-based
AND uncompute with its CZ repair and rejects the same circuit without it.
"""
import math
import random

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

import sparse_sim as SS
from _dynsim import outcome_distribution
from ec_sim import Machine, simulate
from onectrl import order_circuit_1c
from shor_essentials import order_circuit
from shor_stats import multiplicative_order, order_finding_probs


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def close_dist(a, b, tol=1e-10):
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) < tol


def test_order_finding():
    for A in (7, 2):
        qc = order_circuit(A, 15)
        d = SS.simulate_sparse(qc).distribution()
        ref = order_finding_probs(A, 15, 8)
        assert close_dist(d, {y: float(p) for y, p in enumerate(ref) if p > 1e-15}), A
    ok("full order-finding circuit, N=15: distribution = closed form (1e-10)")
    for A in (2, 4, 7, 8, 11, 13, 14):
        qc = order_circuit_1c(A, 15)
        d = SS.simulate_sparse(qc).distribution()
        assert close_dist(d, outcome_distribution(qc)), A
    ok("one-control circuit, N=15, every base: distribution = _dynsim (1e-10)")


def random_circuit(nq, depth, rnd, ncl=0):
    qc = QuantumCircuit(QuantumRegister(nq, "q"), ClassicalRegister(max(ncl, 1), "c"))
    one = ["h", "x", "z", "s", "sdg", "t", "tdg", "sx", "y"]
    for _ in range(depth):
        g = rnd.random()
        qs = rnd.sample(range(nq), 3)
        if g < 0.35:
            getattr(qc, rnd.choice(one))(qs[0])
        elif g < 0.5:
            qc.cx(qs[0], qs[1])
        elif g < 0.6:
            qc.ccx(*qs)
        elif g < 0.7:
            qc.cp(rnd.uniform(-3, 3), qs[0], qs[1])
        elif g < 0.78:
            qc.cswap(*qs)
        elif g < 0.86:
            qc.rz(rnd.uniform(-3, 3), qs[0])
        elif g < 0.93:
            qc.ry(rnd.uniform(-3, 3), qs[0])
        else:
            qc.mcx(qs[:2], qs[2], ctrl_state=rnd.randrange(4))
    return qc


def test_statevector():
    rnd = random.Random(5)
    worst = 0.0
    for trial in range(40):
        nq = rnd.randint(3, 7)
        qc = random_circuit(nq, rnd.randint(5, 40), rnd)
        init = rnd.randrange(1 << nq)
        want = Statevector.from_int(init, 1 << nq).evolve(qc).data
        br = SS.simulate_sparse(qc, init={i: (init >> i) & 1 for i in range(nq)}).only()
        got = np.zeros(1 << nq, dtype=complex)
        for k, a in zip(br.keys, br.amps):
            got[int(k)] = a
        worst = max(worst, float(np.max(np.abs(got - want))))
        # the same circuit embedded at the top of a 70-qubit register
        wide = QuantumCircuit(QuantumRegister(70, "w"))
        wide.compose(qc.remove_final_measurements(inplace=False),
                     qubits=list(range(70 - nq, 70)), inplace=True)
        brw = SS.simulate_sparse(
            wide, init={70 - nq + i: (init >> i) & 1 for i in range(nq)}).only()
        gotw = np.zeros(1 << nq, dtype=complex)
        for k, a in zip(brw.keys, brw.amps):
            assert int(k) & ((1 << (70 - nq)) - 1) == 0
            gotw[int(k) >> (70 - nq)] = a
        worst = max(worst, float(np.max(np.abs(gotw - want))))
    assert worst < 1e-10, worst
    ok(f"40 random circuits (H, T, CCX, CSWAP, CP, RY, open-control MCX): "
       f"amplitudes = Statevector, max |d| = {worst:.1e}; same at 70 qubits")


def test_ec_builders():
    import ec_eea as E
    import ec_window as W
    import ec_classical as C
    rnd = random.Random(3)
    q = 13
    n = q.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    E.inplace_mul(m, x, y, q)
    idx = {qb: i for i, qb in enumerate(m.qc.qubits)}
    for _ in range(20):
        xv, yv = rnd.randrange(1, q), rnd.randrange(q)
        init = {}
        for i, qb in enumerate(x):
            init[qb] = (xv >> i) & 1
        for i, qb in enumerate(y):
            init[qb] = (yv >> i) & 1
        bits = simulate(m.qc, init, m.checks)
        br = SS.simulate_sparse(m.qc, init=init, checks=m.checks).only()
        br.assert_basis()
        assert br.value(y) == xv * yv % q == sum(bits[idx[b]] << i for i, b in enumerate(y))
        assert br.value(x) == xv
    ok(f"in-place multiplication mod {q} ({m.qc.num_qubits} qubits): "
       f"same bits as ec_sim, one basis state, ancilla checks pass")

    curve, G, p = C.CLASSIQ, C.CLASSIQ_G, 7
    pts = W.window_points(curve, G, 2)
    m = Machine("and")
    addr, x2, y2 = m.alloc(2, "a"), m.alloc(3, "x"), m.alloc(3, "y")
    W.windowed_point_add(m, addr, x2, y2, pts, p)
    for R in [P for P in curve.points() if not P.inf][:4]:
        for i in (1, 2):
            if C.point_add_exceptional(curve, R, pts[i]):
                continue
            S = curve.add(R, pts[i])
            if S.inf:
                continue
            br = SS.simulate_sparse(m.qc, init={addr: i, x2: R.x, y2: R.y},
                                    checks=m.checks).only()
            assert (br.value(x2), br.value(y2)) == (S.x, S.y)
    ok(f"windowed point addition p=7 ({m.qc.num_qubits} qubits): correct, one basis state")


def and_mbu(repair=True):
    """AND into t, then uncompute it by X-measurement (+ CZ repair)."""
    a, b, t = QuantumRegister(1, "a"), QuantumRegister(1, "b"), QuantumRegister(1, "t")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(a, b, t, c)
    qc.h(a)                           # put a in superposition so a phase shows
    qc.ccx(a[0], b[0], t[0])
    qc.h(t)
    qc.measure(t, c)
    with qc.if_test((c[0], 1)):
        if repair:
            qc.cz(a[0], b[0])
        qc.x(t[0])
    qc.h(a)
    return qc, a, b, t


def test_coherence():
    for outcome in (0, 1):
        for bv in (0, 1):
            qc, a, b, t = and_mbu(True)
            br = SS.simulate_sparse(qc, init={b: bv}, measure=[outcome]).only()
            br.assert_basis()               # H . (phase-free) . H = identity on a
            assert br.value(a) == 0 and br.value(t) == 0
    ok("X-measure uncompute with CZ repair: a returns to |0> exactly, both outcomes")

    qc, a, b, t = and_mbu(False)
    br = SS.simulate_sparse(qc, init={b: 1}, measure=[1]).only()
    assert br.value(a) == 1                 # the unrepaired phase kicks a to |1>
    ok("without the repair the kicked-back phase is visible (a is left in |1>)")

    # assert_coherent on basis inputs: the gadget without the pre/post H
    def gadget(repair):
        a, b, t = QuantumRegister(1, "a"), QuantumRegister(1, "b"), QuantumRegister(1, "t")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(a, b, t, c)
        qc.ccx(a[0], b[0], t[0])
        qc.h(t)
        qc.measure(t, c)
        with qc.if_test((c[0], 1)):
            if repair:
                qc.cz(a[0], b[0])
            qc.x(t[0])
        return qc, a, b
    qc, a, b = gadget(True)
    inputs = [{a: av, b: bv} for av in (0, 1) for bv in (0, 1)]
    SS.assert_coherent(qc, inputs, [[0], [1]])
    qc, a, b = gadget(False)
    try:
        SS.assert_coherent(qc, inputs, [[0], [1]])
        raise AssertionError("missing repair not detected")
    except SS.SparseSimError:
        pass
    ok("assert_coherent: accepts the repaired gadget, rejects the unrepaired one")


def test_session():
    # lookup two bits into a register, X-measure it, repair the phase by hand
    a, o = QuantumRegister(1, "a"), QuantumRegister(2, "o")
    T = [0b01, 0b11]
    tmpl = QuantumCircuit(a, o)
    seg = QuantumCircuit(a, o)
    seg.cx(a[0], o[1])                # T[a] = 1 + 2a: o0 always 1, o1 = a
    seg.x(o[0])
    seen = set()
    for seed in range(32):
        s = SS.Session(tmpl, state={0: 1 / math.sqrt(2), 1: 1 / math.sqrt(2)}, seed=seed)
        s.run(seg)
        mm = s.mx(o)
        seen.add(mm)
        F = [bin(mm & T[v]).count("1") & 1 for v in (0, 1)]
        fix = QuantumCircuit(a, o)
        if F[0] != F[1]:
            fix.z(a[0])
        s.run(fix)
        amps = s.branch.amplitudes(a)
        v = np.array([amps.get((0,), 0), amps.get((1,), 0)])
        assert abs(abs(np.vdot(v, [1 / math.sqrt(2)] * 2)) - 1) < 1e-12
        s.branch.assert_clean(o)
    assert len(seen) == 4
    ok("Session: lookup, X-measure (all 4 outcomes seen), phase repair -> "
       "address superposition restored exactly, output register clean")


def main():
    test_order_finding()
    test_statevector()
    test_ec_builders()
    test_coherence()
    test_session()


if __name__ == "__main__":
    main()
    print("\ntest_sparse_sim: all passed")
