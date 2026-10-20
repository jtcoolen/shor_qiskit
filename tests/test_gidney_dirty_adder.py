"""Gidney 2507.23079 dirty-ancilla constant adder on Qiskit Aer dynamic circuits."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import gidney_dirty_adder as G

SIM = AerSimulator(method="statevector")


def _run(n, d, psi, shots=24, seed=1, carry_in=0):
    """psi: amplitudes over (target, dirty) indexed x | h << n.  Returns per-shot states."""
    adder = G.dirty_const_adder_circuit(n, d, carry_in)
    k = n + max(n - 2, 0)
    qc = QuantumCircuit(*adder.qregs, *adder.cregs)
    qc.prepare_state(psi, list(range(k)))
    qc.compose(adder, inplace=True)
    qc.save_statevector(pershot=True)
    res = SIM.run(transpile(qc, SIM), shots=shots, seed_simulator=seed).result()
    return [np.asarray(s) for s in res.data()["statevector"]], qc.num_qubits - k


def _expected(n, d, psi, nclean):
    out = np.zeros(len(psi) * 2 ** nclean, dtype=complex)
    mask = (1 << n) - 1
    for i, a in enumerate(psi):
        x, h = i & mask, i >> n
        out[((x + d) & mask) | (h << n)] = a
    return out


def _same_up_to_phase(a, b):
    return abs(abs(np.vdot(a, b)) - 1) < 1e-9 and np.allclose(a * np.vdot(b, a) / abs(np.vdot(b, a)), b, atol=1e-9)


def _all_basis(n):
    for d in range(1 << n):
        for i in range(1 << (n + max(n - 2, 0))):
            psi = np.zeros(1 << (n + max(n - 2, 0)), dtype=complex)
            psi[i] = 1
            sts, nc = _run(n, d, psi, shots=6, seed=i + d)
            exp = _expected(n, d, psi, nc)
            for s in sts:
                assert _same_up_to_phase(s, exp), (n, d, i)


def _random_sup(n):
    rng = np.random.default_rng(n)
    k = n + max(n - 2, 0)
    for d in [0, 1, (1 << n) - 1, 5 % (1 << n), (1 << n) // 2 + 1] + list(rng.integers(0, 1 << n, 3)):
        psi = rng.normal(size=1 << k) + 1j * rng.normal(size=1 << k)
        psi /= np.linalg.norm(psi)
        sts, nc = _run(n, int(d), psi, shots=16, seed=int(d) + 7)
        exp = _expected(n, int(d), psi, nc)
        for s in sts:
            assert _same_up_to_phase(s, exp), (n, d)


def test_vent_outcomes_are_random_so_corrections_are_exercised():
    n = 4
    psi = np.ones(1 << (2 * n - 2), dtype=complex)
    psi /= np.linalg.norm(psi)
    adder = G.dirty_const_adder_circuit(n, 11)
    qc = QuantumCircuit(*adder.qregs, *adder.cregs)
    qc.prepare_state(psi, list(range(2 * n - 2)))
    qc.compose(adder, inplace=True)
    res = SIM.run(transpile(qc, SIM), shots=200, seed_simulator=3).result()
    assert len(res.get_counts()) == 8


def test_all_basis_cases():
    for n in (1, 2, 3, 4):
        _all_basis(n)


def test_random_superpositions_exact_phase():
    for n in (2, 3, 4, 5):
        _random_sup(n)


def test_toffoli_count_and_ancillas():
    for n in range(3, 12):
        for d in (0, 1, (1 << n) - 1, 0b10110101101 % (1 << n)):
            qc = G.dirty_const_adder_circuit(n, d)
            assert qc.num_qubits == n + (n - 2) + 2
            assert G.toffoli_count(qc) <= 3 * n - 7


def test_carry_in_and_validation():
    n = 3
    psi = np.zeros(1 << 4, dtype=complex)
    psi[0b0_101] = 1
    adder = G.dirty_const_adder_circuit(n, 2, carry_in=1)
    assert adder.num_qubits == 3 + 1 + 2
    states, nclean = _run(n, 2, psi, carry_in=1)
    expected = _expected(n, 3, psi, nclean)
    for state in states:
        assert _same_up_to_phase(state, expected)
    for dirty, clean in (([], [4, 5]), ([3], [4]), ([2], [4, 5])):
        try:
            G.append_dirty_const_adder(QuantumCircuit(8), [0, 1, 2], dirty, clean, 1)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    for args in (([], [], [], 1), ([0], [], [], 1, 2)):
        try:
            G.append_dirty_const_adder(QuantumCircuit(4), *args)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    try:
        G.dirty_const_adder_circuit(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
