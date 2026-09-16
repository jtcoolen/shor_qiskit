"""Measurement-based uncomputation, end to end, numerically.

Claim: on  sum_a |a>|T[a]>, applying H to every output qubit and observing m
leaves  sum_a (-1)^{parity(m & T[a])} |a>  (x) |m>.  So the output register is
disentangled by construction, and the only damage is a phase that
phase_fixup(F[a] = parity(m & T[a])) removes exactly.
"""
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qrom import lookup_ui, phase_fixup

def parity(v):
    return bin(v).count("1") & 1

for m_bits, W in ((2, 3), (3, 4)):
    L = 1 << m_bits
    table = [(5 * j + 2) % (1 << W) for j in range(L)]
    addr = QuantumRegister(m_bits, "a"); one = QuantumRegister(1, "one")
    out = QuantumRegister(W, "o"); anc = QuantumRegister(m_bits, "anc")
    qc = QuantumCircuit(addr, one, out, anc)
    qc.h(addr)                                   # uniform superposition
    qc.x(one)                                    # lookup control = 1
    lookup_ui(qc, one[0], list(addr), list(out), table, list(anc))
    qc.h(out)                                    # X-basis measurement, part 1
    sv = Statevector.from_instruction(qc).data

    nq = qc.num_qubits
    # qubit order: addr(0..m-1), one(m), out(m+1..m+W), anc(rest)
    ok = 0
    for meas in range(1 << W):
        amps = np.zeros(L, dtype=complex)
        for a in range(L):
            idx = a | (1 << m_bits) | (meas << (m_bits + 1))   # anc = 0, one = 1
            amps[a] = sv[idx]
        want = np.array([(-1.0)**parity(meas & table[a]) for a in range(L)])
        want = want / np.sqrt(L) / np.sqrt(1 << W)
        assert np.allclose(amps, want), (m_bits, meas, amps, want)
        ok += 1
    print(f"  m={m_bits}, W={W}: all {ok} measurement outcomes leave exactly "
          f"phase (-1)^parity(m & T[a]); output disentangled")

    # and the fixup removes precisely that phase
    meas = 5
    F = [parity(meas & table[a]) for a in range(L)]
    l = m_bits // 2
    a2 = QuantumRegister(m_bits, "a"); o2 = QuantumRegister(1, "one")
    ul = QuantumRegister(1 << l, "ul"); ah = QuantumRegister(max(m_bits - l, 1), "ah")
    fx2 = QuantumCircuit(a2, o2, ul, ah)
    fx2.h(a2)
    phase_fixup(fx2, list(a2), F, o2[0], list(ul), list(ah), l)
    got = Statevector.from_instruction(fx2).data
    damaged = np.array([(-1.0)**F[a] for a in range(L)]) / np.sqrt(L)
    block = np.array([got[a] for a in range(L)])
    assert np.allclose(block, damaged), "fixup did not apply the intended phase"
    print(f"  m={m_bits}: fixup applies exactly (-1)^F[a] -- so composing it with "
          f"the measurement restores the uniform state")
print("measurement-based uncomputation: ALL PASSED")
