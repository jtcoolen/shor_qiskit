"""Separate the two control savings: (a) uncontrolled QFT layers (conjugation),
(b) uncontrolled blocks (compensating residue)."""
import math
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from shor_essentials import add_const, c_add_const, c_add_mod

def add_mod_uncontrolled(qc, X, y, sign, flag, N):
    """No outer control anywhere -- to be wrapped in .control(k)."""
    yw = list(y) + [sign]
    add_const(qc, X, yw)
    add_const(qc, -N, yw)
    qc.cx(sign, flag)
    c_add_const(qc, [flag], N, yw)
    add_const(qc, -X, yw)
    qc.cx(sign, flag)
    qc.x(flag)
    add_const(qc, X, yw)

def tp(qc):
    return transpile(qc, basis_gates=["cx", "u"], optimization_level=1)

for N in (15, 33):
    n = math.ceil(math.log2(N))
    for k in (1, 2):
        # (A) fully naive: build uncontrolled, then control EVERY gate
        y = QuantumRegister(n); sf = QuantumRegister(2)
        inner = QuantumCircuit(y, sf)
        add_mod_uncontrolled(inner, 7 % N, list(y), sf[0], sf[1], N)
        full = QuantumCircuit(QuantumRegister(k), y, sf)
        full.append(inner.to_gate().control(k), full.qubits)
        A = tp(full)
        # (B) this note's version
        c = QuantumRegister(k); y2 = QuantumRegister(n); sf2 = QuantumRegister(2)
        smart = QuantumCircuit(c, y2, sf2)
        c_add_mod(smart, list(c), 7 % N, list(y2), sf2[0], sf2[1], N)
        B = tp(smart)
        ca = A.count_ops().get("cx", 0); cb = B.count_ops().get("cx", 0)
        print(f"N={N} n={n} k={k}: every-gate-controlled {ca:6d} CNOT/depth {A.depth():6d}"
              f"   vs this note {cb:5d}/{B.depth():5d}   => {ca/cb:5.1f}x")
