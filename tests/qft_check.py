"""Hand-rolled QFT, checked against Qiskit's QFTGate as a unitary."""
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator


def qft(n, inverse=False, approx=0):
    """QFT on n qubits, Qiskit convention (qubit 0 = LSB).
    approx: drop controlled rotations with angle below pi/2^approx."""
    qc = QuantumCircuit(n, name="qft" + ("_dg" if inverse else ""))
    for j in reversed(range(n)):
        qc.h(j)
        for k in range(j):
            if approx and (j - k) > approx:
                continue
            qc.cp(np.pi / 2 ** (j - k), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc.inverse() if inverse else qc


for n in range(1, 7):
    mine = Operator(qft(n)).data
    theirs = Operator(QFTGate(n)).data
    assert np.allclose(mine, theirs), f"n={n} mismatch"
    # also check the direct definition |j> -> sum_k e^{2 pi i j k / 2^n} |k>
    ref = np.array([[np.exp(2j * np.pi * j * k / 2**n) for j in range(2**n)]
                    for k in range(2**n)]) / np.sqrt(2**n)
    assert np.allclose(mine, ref), f"n={n} differs from the definition"
    inv = Operator(qft(n, inverse=True)).data
    assert np.allclose(inv @ mine, np.eye(2**n)), f"n={n} inverse wrong"
print("qft matches QFTGate AND the textbook definition, n = 1..6; inverse OK")

# gate counts, exact vs approximate
for n in (8, 16):
    d = max(1, int(np.ceil(np.log2(n))) + 2)
    ex, ap = qft(n), qft(n, approx=d)
    print(f"n={n:3d}: exact {ex.size():4d} gates, "
          f"approx(d={d}) {ap.size():4d} gates")
