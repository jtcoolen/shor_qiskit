"""Test the essentials implementation on N=21 (n=5, 22 qubits, orders 2/3/6)."""
import math
import sys
import time

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from shor_essentials import c_add_mod, c_ua, order_circuit, order_from_counts

N = 21
n = math.ceil(math.log2(N))          # 5


def set_value(qc, val, qubits):
    for i, q in enumerate(qubits):
        if (val >> i) & 1:
            qc.x(q)


def exact_outcome(qc):
    """Circuit is a permutation on a basis state: return the unique output index."""
    sv = Statevector.from_instruction(qc).data
    idx = int(np.argmax(np.abs(sv)))
    assert abs(abs(sv[idx]) - 1.0) < 1e-8, f"not a basis state, |amp|={abs(sv[idx])}"
    return idx


# ---------------- Level 2: c_add_mod, exhaustive over y and X ---------------
t0 = time.time()
bad = 0
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
            got_y = (v >> 1) & (2**n - 1)       # qubit 0 is the control
            got_sf = v >> (1 + n)
            want = (y0 + X) % N if ctrl_on else y0
            if got_y != want or got_sf != 0:
                bad += 1
                print(f"  FAIL ctrl={ctrl_on} y={y0} X={X}: "
                      f"got {got_y} sf={got_sf}, want {want}")
assert bad == 0
print(f"Level 2  c_add_mod N=21: OK, all {N*N} (y,X) pairs x control on/off "
      f"({time.time()-t0:.0f}s)")

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
            rest = v >> (1 + n)                 # y register + sign + flag
            want = (A * x0) % N if ctrl_on else x0
            assert got_x == want and rest == 0, \
                (A, ctrl_on, x0, got_x, rest, want)
print(f"Level 4  c_ua N=21: OK, A in {coprime}, all x<21, control on/off, "
      f"scratch clean ({time.time()-t0:.0f}s)")
sys.stdout.flush()

# ---------------- Level 5: the full 22-qubit order-finding circuit ---------
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

SIM = AerSimulator()
PM = generate_preset_pass_manager(backend=SIM, optimization_level=1)
SAMPLER = Sampler(seed=11)

for A in (2, 8):                                # r=6 (hard) and r=2 (exact)
    t0 = time.time()
    qc = order_circuit(A, N)
    print(f"\nA={A}: building/transpiling {qc.num_qubits} qubits ...")
    sys.stdout.flush()
    isa = PM.run(qc)
    print(f"  transpiled: {isa.depth()} depth, "
          f"{isa.count_ops().get('cx', 0)} CNOT ({time.time()-t0:.0f}s)")
    sys.stdout.flush()
    t1 = time.time()
    counts = SAMPLER.run([isa], shots=2048).result()[0].data.out.get_counts()
    print(f"  simulated in {time.time()-t1:.0f}s")
    t = 2 * n
    r_true = 1
    while pow(A, r_true, N) != 1:
        r_true += 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    print(f"  true r={r_true}; ideal peaks near "
          f"{[round(2**t*s/r_true) for s in range(r_true)]}")
    print("  top outcomes (y, shots, y/2^t, nearest s/r):")
    for b, cnt in top:
        y = int(b, 2)
        from fractions import Fraction
        fr = Fraction(y, 2**t).limit_denominator(N - 1)
        print(f"    y={y:4d}  {cnt:5d} shots   {y/2**t:.4f}   ~ {fr}")
    r = order_from_counts(counts, A, N, t)
    print(f"  order_from_counts -> r = {r}  "
          f"{'OK' if r == r_true else 'MISMATCH (expected ' + str(r_true) + ')'}")
    assert r == r_true, (A, r, r_true)
    # factor
    if r % 2 == 0:
        d = math.gcd(pow(A, r // 2, N) - 1, N)
        print(f"  gcd(A^(r/2)-1, N) = {d}  -> 21 = {d} x {N//d}")

print("\nN=21 ALL TESTS PASSED")
