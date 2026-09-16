"""Test the essentials implementation on N=33 (n=6, 26 qubits, orders 2/5/10)."""
import math
import sys
import time
from fractions import Fraction

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from shor_essentials import c_add_mod, c_ua, order_circuit, order_from_counts

N = 33
n = math.ceil(math.log2(N))          # 6


def set_value(qc, val, qubits):
    for i, q in enumerate(qubits):
        if (val >> i) & 1:
            qc.x(q)


def exact_outcome(qc):
    sv = Statevector.from_instruction(qc).data
    idx = int(np.argmax(np.abs(sv)))
    assert abs(abs(sv[idx]) - 1.0) < 1e-8, f"not a basis state, |amp|={abs(sv[idx])}"
    return idx


# ---------------- Level 2: c_add_mod, exhaustive ---------------------------
t0 = time.time()
for ctrl_on in (True, False):
    for y0 in range(N):
        for X in range(N):
            c = QuantumRegister(1)
            y = QuantumRegister(n)
            sf = QuantumRegister(2)
            qc = QuantumCircuit(c, y, sf)
            if ctrl_on:
                qc.x(c[0])
            set_value(qc, y0, y)
            c_add_mod(qc, [c[0]], X, list(y), sf[0], sf[1], N)
            v = exact_outcome(qc)
            got_y = (v >> 1) & (2**n - 1)
            got_sf = v >> (1 + n)
            want = (y0 + X) % N if ctrl_on else y0
            assert got_y == want and got_sf == 0, \
                (ctrl_on, y0, X, got_y, got_sf, want)
print(f"Level 2  c_add_mod N=33: OK, all {N*N} (y,X) pairs x control on/off "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------------- Level 4: c_ua, all coprime A, all x ----------------------
t0 = time.time()
coprime = [a for a in range(2, N) if math.gcd(a, N) == 1]
for A in coprime:
    for ctrl_on in (True, False):
        for x0 in range(N):
            c = QuantumRegister(1)
            x = QuantumRegister(n)
            yq = QuantumRegister(n)
            sf = QuantumRegister(2)
            qc = QuantumCircuit(c, x, yq, sf)
            if ctrl_on:
                qc.x(c[0])
            set_value(qc, x0, x)
            c_ua(qc, c[0], A, list(x), list(yq), sf[0], sf[1], N)
            v = exact_outcome(qc)
            got_x = (v >> 1) & (2**n - 1)
            rest = v >> (1 + n)
            want = (A * x0) % N if ctrl_on else x0
            assert got_x == want and rest == 0, \
                (A, ctrl_on, x0, got_x, rest, want)
    print(f"  Level 4 c_ua N=33: A={A:2d} OK ({time.time()-t0:.0f}s)", flush=True)
print(f"Level 4  c_ua N=33: OK, all {len(coprime)} coprime bases, all x<33, "
      f"control on/off, scratch clean ({time.time()-t0:.0f}s)", flush=True)

# ---------------- Level 5: the full 26-qubit circuit -----------------------
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
SAMPLER = Sampler(seed=13)
t = 2 * n

for A in (5, 10):                    # r=10 (factors to 11); r=2 (cheap check)
    t0 = time.time()
    qc = order_circuit(A, N)
    print(f"\nA={A}: {qc.num_qubits} qubits, transpiling ...", flush=True)
    isa = PM.run(qc)
    print(f"  depth {isa.depth()}  ({time.time()-t0:.0f}s); simulating ...",
          flush=True)
    t1 = time.time()
    counts = SAMPLER.run([isa], shots=2048).result()[0].data.out.get_counts()
    print(f"  simulated in {time.time()-t1:.0f}s", flush=True)
    r_true = 1
    while pow(A, r_true, N) != 1:
        r_true += 1
    print(f"  true r={r_true}; ideal peaks near "
          f"{[round(2**t*s/r_true) for s in range(r_true)]}")
    for b, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        y = int(b, 2)
        print(f"    y={y:5d}  {cnt:5d} shots   {y/2**t:.4f}  ~ "
              f"{Fraction(y, 2**t).limit_denominator(N-1)}")
    r = order_from_counts(counts, A, N, t)
    print(f"  order_from_counts -> r = {r} (true {r_true}) "
          f"{'OK' if r == r_true else 'MISMATCH'}", flush=True)
    assert r == r_true, (A, r, r_true)
    if r % 2 == 0:
        h = pow(A, r // 2, N)
        d = math.gcd(h - 1, N)
        print(f"  gcd(A^(r/2)-1, N) = {d}"
              + (f"  -> 33 = {d} x {N//d}" if 1 < d < N else "  (trivial)"))

print("\nN=33 ALL TESTS PASSED")
