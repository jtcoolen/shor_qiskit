"""Windowed Shor at the proper t=2n, now that the QROM is space-efficient."""
import math, time
from fractions import Fraction
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_from_counts
from windowed import windowed_order_circuit

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=23); N = 15
for A, w in ((7, 2), (2, 2)):
    t0 = time.time(); t = 2 * 4
    qc = windowed_order_circuit(A, N, w=w, t=t)
    isa = PM.run(qc)
    print(f"\nA={A} w={w} t={t}: {qc.num_qubits} qubits, depth {isa.depth()}", flush=True)
    counts = S.run([isa], shots=2048).result()[0].data.out.get_counts()
    for b, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        y = int(b, 2)
        print(f"    y={y:4d} {c:5d} shots  {y}/{2**t} -> {Fraction(y,2**t).limit_denominator(N-1)}")
    r = order_from_counts(counts, A, N, t)
    print(f"  r = {r} [{time.time()-t0:.0f}s]", flush=True)
    assert r == 4, r
    d = math.gcd(pow(A, r//2, N) - 1, N)
    print(f"  gcd = {d} -> 15 = {d} x {N//d}")
print("\nFULL-PRECISION WINDOWED: PASSED")
