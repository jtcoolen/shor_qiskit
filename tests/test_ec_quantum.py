"""Cross-check the fast basis-state simulator against a real quantum simulator.

Everything else in this suite uses `ec_sim.simulate`, which pushes one basis
state through a circuit.  That is sound *if* the circuits really are basis-state
permutations -- but "if" is doing work there, and a bug in the simulator would
hide itself from every other test in the package.

So this file evolves the same circuits on genuine superpositions with Qiskit's
Statevector and checks two things the fast path cannot see:

  * the output state is exactly sum_x |x> |f(x)> with the right amplitudes, so
    no relative phase has crept in (a diagonal gate the fast simulator skips
    would show up here as a phase, and would ruin Shor's interference);
  * every ancilla is *unentangled* and back in |0>, not merely zero on each
    basis state separately.

It is deliberately small -- statevector cost is exponential -- but it is the
only thing anchoring the rest.
"""
import numpy as np
from qiskit.quantum_info import Statevector

from _ec_util import ok, section

import ec_adders as A
import ec_modarith as MA
from ec_gates import AndDgGate, AndGate
from ec_sim import Machine
from qiskit import QuantumCircuit


def _check_superposition(m, inreg, outreg, values, f, label):
    """Put `inreg` in a uniform superposition over `values` and check the result."""
    qc = QuantumCircuit(*m.qc.qregs)
    idx = {q: i for i, q in enumerate(m.qc.qubits)}
    amp = np.zeros(2 ** m.qc.num_qubits, dtype=complex)
    for v in values:
        pos = sum(((v >> i) & 1) << idx[q] for i, q in enumerate(inreg))
        amp[pos] = 1 / np.sqrt(len(values))
    sv = Statevector(amp).evolve(m.qc)

    want = np.zeros_like(amp)
    inplace = list(inreg) == list(outreg)
    for v in values:
        pos = 0
        if not inplace:                    # the input register survives
            pos |= sum(((v >> i) & 1) << idx[q] for i, q in enumerate(inreg))
        pos |= sum(((f(v) >> i) & 1) << idx[q] for i, q in enumerate(outreg))
        want[pos] = 1 / np.sqrt(len(values))
    got = np.asarray(sv)
    fid = abs(np.vdot(want, got))
    assert fid > 1 - 1e-9, f"{label}: fidelity {fid} -- phases or values wrong"
    # every amplitude outside the intended support must vanish, which is the
    # statement that no ancilla is left entangled
    mask = want == 0
    assert np.max(np.abs(got[mask])) < 1e-9, f"{label}: leaked amplitude (dirty ancilla)"
    ok(f"{label}: exact on superposition, fidelity {fid:.12f}, no ancilla entanglement")


def main():
    section("the AND gadget really is a permutation with no phase")
    qc = QuantumCircuit(3)
    qc.append(AndGate(), [0, 1, 2])
    from qiskit.quantum_info import Operator
    U = Operator(qc).data
    for a in range(2):
        for b in range(2):
            i = a + 2 * b
            j = i + 4 * (a & b)
            assert abs(U[j, i] - 1) < 1e-9, f"AND put a phase on |{a},{b}>"
    qc2 = QuantumCircuit(3)
    qc2.append(AndGate(), [0, 1, 2])
    qc2.append(AndDgGate(), [0, 1, 2])
    assert np.allclose(Operator(qc2).data, np.eye(8))
    ok("AND is the clean permutation |a,b,0> -> |a,b,a AND b>, phase-free; AND-dagger inverts it")

    section("gidney_add on a superposition")
    n = 3
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    a = m.anc(n - 1, "a")
    A.gidney_add(m.ctx, x, y, a)
    m.free(a)
    _check_superposition(m, x, y, range(2**n), lambda v: v, "gidney_add (y=0, so y<-x)")

    section("modular addition on a superposition")
    for p in (3, 5, 7):
        n = p.bit_length()
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        MA.modadd(m, x, y, p)
        if m.qc.num_qubits > 20:
            continue
        _check_superposition(m, x, y, range(p), lambda v: v % p,
                             f"modadd mod {p} ({m.qc.num_qubits} qubits)")

    section("modular doubling on a superposition")
    for p in (3, 5, 7):
        n = p.bit_length()
        m = Machine("and")
        x = m.alloc(n, "x")
        MA.moddbl(m, x, p)
        if m.qc.num_qubits > 20:
            continue
        _check_superposition(m, x, x, range(p), lambda v: 2 * v % p,
                             f"moddbl mod {p} ({m.qc.num_qubits} qubits)")


if __name__ == "__main__":
    main()
    print("\ntest_ec_quantum: all passed")
