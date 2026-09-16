"""Coset representation: does a plain adder really do modular arithmetic,
and does the error follow the 2^-cpad law?"""
import math, time
import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
import coset
from shor_essentials import c_add_mod

SIM = AerSimulator(method="statevector")

def run_adds(N, k, adds, cpad, shots=4000):
    """Encode k, apply a sequence of plain additions, measure; return the
    fraction of shots whose residue is wrong."""
    m = coset.coset_width(N, cpad)
    reg = QuantumRegister(m, "r")
    qc = QuantumCircuit(reg)
    coset.encode(qc, reg, k, N, cpad)
    for a in adds:
        coset.add(qc, a, reg, N)
    qc.measure_all()
    res = SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots).result()
    want = (k + sum(adds)) % N
    bad = sum(c for b, c in res.get_counts().items()
              if coset.decode(int(b, 2), N) != want)
    return bad / shots

# --- 1. one addition, exhaustive over k and a ------------------------------
t0 = time.time()
for N in (15, 21):
    cpad = 5
    worst = 0.0
    for k in range(N):
        for a in range(N):
            worst = max(worst, run_adds(N, k, [a], cpad, shots=400))
    print(f"N={N} cpad={cpad}: worst error over all (k,a) = {worst:.4f} "
          f"(bound {coset.deviation_bound(1, cpad):.4f})", flush=True)

# --- 2. the 2^-cpad law ----------------------------------------------------
print(f"\n{'cpad':>5} {'m':>3} {'measured err':>13} {'bound 2^-cpad':>14}")
N, k = 21, 5
adds = [13, 8, 17, 4]                      # four additions, each < N
for cpad in (1, 2, 3, 4, 5, 6, 7):
    e = run_adds(N, k, adds, cpad, shots=8000)
    print(f"{cpad:>5} {coset.coset_width(N,cpad):>3} {e:>13.5f} "
          f"{coset.deviation_bound(len(adds), cpad):>14.5f}")

# --- 3. what it saves ------------------------------------------------------
def cost(qc):
    d = transpile(qc, basis_gates=["cx","u"], optimization_level=1)
    return d.count_ops().get("cx",0), d.depth()

print(f"\n{'N':>5} {'coset add (plain)':>19} {'7-block modular add':>21} {'saving':>8}")
for N in (15, 63, 255, 1023):
    n = math.ceil(math.log2(N)); cpad = 5
    r = QuantumRegister(n + cpad); a = QuantumCircuit(r)
    coset.add(a, 7 % N, r, N)
    ca, da = cost(a)
    c = QuantumRegister(1); y = QuantumRegister(n); sf = QuantumRegister(2)
    b = QuantumCircuit(c, y, sf)
    c_add_mod(b, [c[0]], 7 % N, list(y), sf[0], sf[1], N)
    cb, db = cost(b)
    print(f"{N:>5} {ca:>10} CNOT {da:>4}d {cb:>12} CNOT {db:>5}d {cb/ca:>7.1f}x")
print(f"\n[{time.time()-t0:.0f}s]  COSET: done")
