"""When does the coset approximation actually bite, and does the error follow
GE's subadditive  A * 2^-cpad  law?"""
import math
import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
import coset

SIM = AerSimulator(method="statevector")

def err(N, k, adds, cpad, shots=8000):
    m = coset.coset_width(N, cpad)
    reg = QuantumRegister(m)
    qc = QuantumCircuit(reg)
    coset.encode(qc, reg, k, N, cpad)
    for a in adds:
        coset.add(qc, a, reg, N)
    qc.measure_all()
    res = SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots).result()
    want = (k + sum(adds)) % N
    return sum(c for b, c in res.get_counts().items()
               if coset.decode(int(b, 2), N) != want) / shots

print("Headroom H = 2^cpad*(2^n - N) + N - 1 - k.  The approximation only bites")
print("once the accumulated offset exceeds it, so N near 2^n is the hard case.\n")
print(f"{'N':>4} {'2^n-N':>6} {'A':>3} {'cpad':>5} {'measured':>10} {'A/2^cpad':>10}")
rng = np.random.default_rng(3)
for N, n in ((15, 4), (31, 5)):
    for A in (8,):
        adds = [int(x) for x in rng.integers(1, N, size=A)]
        for cpad in (2, 3, 4, 5, 6):
            e = err(N, 1, adds, cpad)
            print(f"{N:>4} {2**n - N:>6} {A:>3} {cpad:>5} {e:>10.4f} "
                  f"{A / (1 << cpad):>10.4f}")
    print()
