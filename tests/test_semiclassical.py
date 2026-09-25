"""The semiclassical inverse QFT, as a unit and then on the real oracles.

  1. Fourier states, exhaustively, on Aer: every y is read back exactly
  2. exact equivalence to qft(t).inverse() on entangled inputs, arbitrary rungs
  3. the checks have teeth: three plausible bugs are each caught
  4. Aer's sampling of the dynamic circuit matches the exact distribution
  5. the real oracles, exactly: factoring N=15 and ECDLP p=7, the one-control
     circuit's distribution equals the closed form, as does the full circuit's
  6. the shape: one counting qubit, t measurements, t(t-1)/2 conditioned phases

"Exact" means computed by `_dynsim`, which enumerates measurement branches
instead of sampling them -- the literal semantics of measurement and
feed-forward, independent of Aer.
"""
import math

import numpy as np
from _ec_util import ok, section

from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, random_statevector, random_unitary
from qiskit_aer import AerSimulator

import ec_classical as C
import ec_shor as S
import shor_stats as ST
from _dynsim import outcome_distribution
from onectrl import order_circuit_1c
from semiclassical import semiclassical_iqft
from shor_essentials import order_circuit, qft

SIM = AerSimulator(method="statevector", seed_simulator=11)
EXACT = 1e-10


def as_array(dist, size):
    a = np.zeros(size)
    for v, p in dist.items():
        a[v] += p
    return a


# =============================================================================
# Reference and candidate for arbitrary rungs on a small system register
# =============================================================================
def full_reference(psi, unitaries, order):
    """t counting qubits, rungs applied in `order`, qft(t).inverse(): exact P(y)."""
    t, s = len(unitaries), int(math.log2(len(psi)))
    qc = QuantumCircuit(t + s)
    qc.h(range(t))
    for k in order:
        qc.append(UnitaryGate(unitaries[k]).control(1), [k, *range(t, t + s)])
    qc.append(qft(t).inverse(), range(t))
    init = Statevector(np.kron(psi, np.eye(2**t)[0]))
    return init.evolve(qc).probabilities(range(t))


def one_control(psi, unitaries, driver=semiclassical_iqft):
    """The same thing on one recycled control qubit.  Returns (circuit, init)."""
    t, s = len(unitaries), int(math.log2(len(psi)))
    ctr, sysr = QuantumRegister(1, "c"), QuantumRegister(s, "sys")
    out = ClassicalRegister(t, "out")
    qc = QuantumCircuit(ctr, sysr, out)
    rungs = [lambda qc, c, U=U: qc.append(UnitaryGate(U).control(1), [c, *sysr])
             for U in unitaries]
    driver(qc, ctr[0], out, rungs)
    return qc, Statevector(np.kron(psi, [1, 0]))


def one_control_exact(psi, unitaries, driver=semiclassical_iqft):
    qc, init = one_control(psi, unitaries, driver)
    return as_array(outcome_distribution(qc, init), 2 ** len(unitaries))


def random_case(t, s, seed):
    psi = random_statevector(2**s, seed=seed).data
    return psi, [random_unitary(2**s, seed=seed * 100 + k).data for k in range(t)]


def commuting_case(t, s, seed):
    """Shor's case: rung k is V^(2^k) for one V."""
    psi = random_statevector(2**s, seed=seed).data
    V = random_unitary(2**s, seed=seed).data
    return psi, [np.linalg.matrix_power(V, 2**k) for k in range(t)]


# =============================================================================
def test_fourier_states_exhaustive():
    section("1. Fourier states: |y~> reads back as exactly y, every y, t = 1..6")
    circs, want = [], []
    for t in range(1, 7):
        for y in range(2**t):
            ctr, out = QuantumRegister(1, "c"), ClassicalRegister(t, "out")
            qc = QuantumCircuit(ctr, out)
            rungs = [lambda qc, c, k=k, y=y, t=t: qc.p(2 * math.pi * y * 2**k / 2**t, c)
                     for k in range(t)]
            semiclassical_iqft(qc, ctr[0], out, rungs)
            circs.append(qc)
            want.append(y)
    res = SIM.run(transpile(circs, SIM, optimization_level=0), shots=32).result()
    for i, y in enumerate(want):
        counts = res.get_counts(i)
        assert list(counts) == [format(y, f"0{circs[i].num_clbits}b")], (y, counts)
    ok(f"all {len(circs)} Fourier states decoded deterministically on Aer "
       f"(the counting register is one qubit throughout)")


def test_exact_equivalence():
    section("2. exact equivalence to qft(t).inverse() on entangled inputs")
    worst, n = 0.0, 0
    for t in range(1, 6):
        for seed in range(1, 4):
            psi, Us = random_case(t, 2, seed)
            ref = full_reference(psi, Us, order=reversed(range(t)))
            worst = max(worst, np.abs(one_control_exact(psi, Us) - ref).max())
            n += 1
    assert worst < EXACT, worst
    ok(f"arbitrary non-commuting rungs, t = 1..5, {n} cases: max |dP| = {worst:.1e} "
       f"against the full circuit run heaviest-rung-first")

    worst = 0.0
    for t in range(1, 6):
        for seed in range(1, 4):
            psi, Us = commuting_case(t, 2, seed)
            ref = full_reference(psi, Us, order=range(t))   # as order_circuit does
            worst = max(worst, np.abs(one_control_exact(psi, Us) - ref).max())
    assert worst < EXACT, worst
    ok(f"Shor's case, rungs V^(2^k) in the full circuit's natural order: "
       f"max |dP| = {worst:.1e}")

    psi, Us = random_case(3, 2, 7)
    d = ST.tvd(full_reference(psi, Us, order=range(3)), one_control_exact(psi, Us))
    assert d > 1e-3, d
    ok(f"and the rung ORDER is what the equivalence rests on: non-commuting rungs "
       f"lightest-first give a different distribution (TVD {d:.3f}); Shor's commute")


# --- deliberately broken drivers, for section 3 ------------------------------
def _mutant(corrections=True, heaviest_first=True, reset=True):
    def driver(qc, ctrl, out, rungs):
        t = len(rungs)
        for i in range(t):
            qc.h(ctrl)
            rungs[t - 1 - i if heaviest_first else i](qc, ctrl)
            for j in range(i if corrections else 0):
                with qc.if_test((out[j], 1)):
                    qc.p(-math.pi / 2 ** (i - j), ctrl)
            qc.h(ctrl)
            qc.measure(ctrl, out[i])
            if reset:
                with qc.if_test((out[i], 1)):
                    qc.x(ctrl)
    return driver


def test_checks_have_teeth():
    section("3. each plausible bug is caught by the exact check")
    psi, Us = commuting_case(4, 2, 5)
    ref = full_reference(psi, Us, order=range(4))
    assert np.abs(one_control_exact(psi, Us, _mutant()) - ref).max() < EXACT
    for label, kw in (("phase corrections dropped", {"corrections": False}),
                      ("rungs lightest-first (bit order reversed)", {"heaviest_first": False}),
                      ("control qubit not reset", {"reset": False})):
        d = ST.tvd(ref, one_control_exact(psi, Us, _mutant(**kw)))
        assert d > 0.05, (label, d)
        ok(f"{label:42s} TVD from the right answer {d:.3f}")


def test_aer_matches_exact():
    section("4. Aer's shot-by-shot sampling matches the exact distribution")
    shots = 8192
    for t, seed in ((3, 1), (4, 2), (5, 3)):
        psi, Us = random_case(t, 2, seed)
        qc, _ = one_control(psi, Us)
        prep = QuantumCircuit(qc.qubits, qc.clbits)
        prep.initialize(psi, qc.qubits[1:])
        run = prep.compose(qc)
        counts = SIM.run(transpile(run, SIM, optimization_level=0),
                         shots=shots).result().get_counts()
        emp = ST.empirical({int(b, 2): c for b, c in counts.items()}, 2**t)
        exact = full_reference(psi, Us, order=reversed(range(t)))
        d, bound = ST.tvd(emp, exact), ST.tvd_null(exact, shots)
        assert d <= bound, (t, d, bound)
        ok(f"t={t}: TVD {d:.4f} <= {bound:.4f}, the shot-noise bound at {shots} shots")


def test_real_oracles_exact():
    section("5. the real oracles: one-control == full circuit == closed form, exactly")
    # The reference itself: Aer's exact statevector of the full-register circuit.
    qc = order_circuit(7, 15).remove_final_measurements(inplace=False)
    qc.save_probabilities(list(range(8)))
    full = np.asarray(SIM.run(transpile(qc, SIM, optimization_level=0))
                      .result().data(0)["probabilities"])
    d = np.abs(full - ST.order_finding_probs(7, 15, 8)).max()
    assert d < EXACT, d
    ok(f"order_circuit(7, 15), 18 qubits: Aer exact probabilities == closed form "
       f"(max |dP| {d:.1e})")

    worst = 0.0
    coprime = [A for A in range(2, 15) if math.gcd(A, 15) == 1]
    for A in coprime:
        got = as_array(outcome_distribution(order_circuit_1c(A, 15)), 2**8)
        worst = max(worst, np.abs(got - ST.order_finding_probs(A, 15, 8)).max())
    assert worst < EXACT, worst
    ok(f"order_circuit_1c(A, 15) for every A in {coprime}, 11 qubits: "
       f"== closed form (max |dP| {worst:.1e})")

    cu, G = C.CLASSIQ, C.CLASSIQ_G
    r = cu.point_order(G)
    worst_full = worst_1c = 0.0
    for k in range(1, r):
        Q = cu.mul(k, G)
        exact = ST.ecdlp_probs(r, k, 3)
        qc, info = S.ecdlp_circuit(cu, G, Q, r, oracle="table")
        mb, q = info["m_bits"], 1 << info["m_bits"]
        qc = qc.remove_final_measurements(inplace=False)
        qc.save_probabilities(list(range(2 * mb)))
        full = np.asarray(SIM.run(transpile(qc, SIM, optimization_level=0))
                          .result().data(0)["probabilities"]).reshape(q, q).T
        worst_full = max(worst_full, np.abs(full - exact).max())
        qc1, _ = S.ecdlp_circuit_1c(cu, G, Q, r)
        one = as_array(outcome_distribution(qc1), q * q).reshape(q, q).T
        worst_1c = max(worst_1c, np.abs(one - exact).max())
    assert worst_full < EXACT and worst_1c < EXACT, (worst_full, worst_1c)
    ok(f"ECDLP on y^2=x^3+5x+4 mod 7, every k: full circuit (12 qubits) and "
       f"one-control (7 qubits) == closed form (max |dP| {max(worst_full, worst_1c):.1e})")


def _count_ops(qc):
    tally = {}
    for inst in qc.data:
        op = inst.operation
        tally[op.name] = tally.get(op.name, 0) + 1
        for b in getattr(op, "blocks", ()):
            for k, v in _count_ops(b).items():
                tally["if/" + k] = tally.get("if/" + k, 0) + v
    return tally


def test_shape():
    section("6. shape: what the transform costs and what it saves")
    for A, N in ((7, 15), (2, 21), (5, 33)):
        n = math.ceil(math.log2(N))
        t = 2 * n
        qc, full = order_circuit_1c(A, N), order_circuit(A, N)
        ops = _count_ops(qc)
        assert qc.num_qubits == 2 * n + 3 and full.num_qubits == 4 * n + 2
        assert ops["measure"] == t and ops["if/p"] == t * (t - 1) // 2
        assert ops["if/x"] == t                        # the resets
        ok(f"N={N}: {qc.num_qubits} qubits instead of {full.num_qubits}; "
           f"{t} measurements, {t*(t-1)//2} conditioned phases, {t} conditioned resets")
    for cu, G in ((C.CLASSIQ, C.CLASSIQ_G), (C.TOY11, C.TOY11_G)):
        r = cu.point_order(G)
        qc, info = S.ecdlp_circuit_1c(cu, G, G, r)
        full = S.ecdlp_circuit(cu, G, G, r, oracle="table")[0]
        mb, n = info["m_bits"], info["n"]
        ops = _count_ops(qc)
        assert qc.num_qubits == 1 + 2 * n and full.num_qubits == 2 * mb + 2 * n
        assert ops["measure"] == 2 * mb and ops["if/p"] == 2 * (mb * (mb - 1) // 2)
        ok(f"{cu.name}: {qc.num_qubits} qubits instead of {full.num_qubits} "
           f"(two {mb}-bit registers -> one qubit)")


if __name__ == "__main__":
    test_fourier_states_exhaustive()
    test_exact_equivalence()
    test_checks_have_teeth()
    test_aer_matches_exact()
    test_real_oracles_exact()
    test_shape()
    print("\ntest_semiclassical: all passed")
