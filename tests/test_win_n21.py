"""Windowed Shor end-to-end on N=21 (order 6 -- continued fractions matter)."""
import math, time
from fractions import Fraction
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_from_counts
from windowed import windowed_order_circuit

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=29); N = 21
# t=7 first (26 qubits, quick); then the full t=2n=10 (29 qubits, ~8.6 GB).
for A, w, t in ((2, 2, 7), (2, 2, 10)):
    t0 = time.time()
    qc = windowed_order_circuit(A, N, w=w, t=t)
    isa = PM.run(qc)
    print(f"\nA={A} w={w} t={t}: {qc.num_qubits} qubits, depth {isa.depth()}", flush=True)
    counts = S.run([isa], shots=2048).result()[0].data.out.get_counts()
    r_true = 1
    while pow(A, r_true, N) != 1:
        r_true += 1
    print(f"  ideal peaks ~ {[round(2**t*s/r_true) for s in range(r_true)]}")
    for b, c in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        y = int(b, 2)
        print(f"    y={y:5d} {c:5d} shots  {y}/{2**t} -> {Fraction(y,2**t).limit_denominator(N-1)}")
    r = order_from_counts(counts, A, N, t)
    print(f"  r = {r} (true {r_true}) {'OK' if r==r_true else 'MISMATCH'} [{time.time()-t0:.0f}s]", flush=True)
    assert r == r_true, (A, w, t, r, r_true)
    d = math.gcd(pow(A, r//2, N) - 1, N)
    print(f"  gcd(A^(r/2)-1,21) = {d} -> 21 = {d} x {N//d}", flush=True)
print("\nWINDOWED N=21: PASSED")
