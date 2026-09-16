"""Nested-windowed Shor, end to end on N=15."""
import math, time
from fractions import Fraction
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_from_counts
from nested import nested_order_circuit

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=31); N = 15
for A, we, wm, t in ((7, 2, 2, 4), (2, 2, 2, 4)):
    t0 = time.time()
    qc = nested_order_circuit(A, N, we=we, wm=wm, t=t)
    isa = PM.run(qc)
    print(f"\nA={A} we={we} wm={wm} t={t}: {qc.num_qubits} qubits, depth {isa.depth()}"
          f"  ({math.ceil(t/we)} multiplications, not {t})", flush=True)
    counts = S.run([isa], shots=2048).result()[0].data.out.get_counts()
    for b, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        y = int(b, 2)
        print(f"    y={y:3d} {c:5d} shots  {y}/{2**t} -> {Fraction(y,2**t).limit_denominator(N-1)}")
    r = order_from_counts(counts, A, N, t)
    print(f"  r = {r} [{time.time()-t0:.0f}s]", flush=True)
    assert r == 4, r
    d = math.gcd(pow(A, r//2, N) - 1, N)
    print(f"  gcd = {d} -> 15 = {d} x {N//d}", flush=True)
print("\nNESTED WINDOWED END-TO-END: PASSED")
