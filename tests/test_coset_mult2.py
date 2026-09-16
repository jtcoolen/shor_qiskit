"""Coset multiplier trend, on a small modulus so the registers stay simulable."""
import math
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
import coset

SIM = AerSimulator(method="statevector")

def mult_err(N, A, x, cpad, shots=2000):
    m = coset.coset_width(N, cpad)
    xr = QuantumRegister(m); yr = QuantumRegister(m); c = QuantumRegister(1)
    qc = QuantumCircuit(c, xr, yr)
    qc.x(c[0])
    coset.encode(qc, xr, x, N, cpad)
    coset.encode(qc, yr, 0, N, cpad)
    coset.c_mult_acc(qc, c[0], A, list(xr), list(yr), N)
    qc.measure_all()
    res = SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots).result()
    want = (A * x) % N
    bad = sum(cnt for b, cnt in res.get_counts().items()
              if coset.decode((int(b.replace(" ", ""), 2) >> (1 + m)) & ((1 << m) - 1), N) != want)
    return bad / shots

N = 7
print(f"N={N} (n=3):  y += A*x mod N, both registers coset-encoded")
print(f"{'cpad':>5} {'m':>3} {'qubits':>7} {'adds A':>7} {'worst err':>11} {'bound A/2^cpad':>15}")
for cpad in (3, 4, 5, 6, 7):
    m = coset.coset_width(N, cpad)
    worst = max(mult_err(N, A, x, cpad) for A in (3, 5) for x in (2, 6))
    print(f"{cpad:>5} {m:>3} {2*m+1:>7} {m:>7} {worst:>11.4f} {m/(1<<cpad):>15.4f}")
