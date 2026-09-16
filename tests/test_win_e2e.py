"""End-to-end Shor with windowed arithmetic, N=15."""
import math, time
from fractions import Fraction
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_from_counts
from windowed import windowed_order_circuit

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=17)
N = 15

# (w=2, t=4): real windowing.  t=4 is legitimate here because r=4 divides 2^4.
# (w=1, t=8): full-precision counting register, degenerate window -- cross-check.
for A, w, t in ((7, 2, 4), (2, 2, 4), (7, 1, 8)):
    t0 = time.time()
    qc = windowed_order_circuit(A, N, w=w, t=t)
    isa = PM.run(qc)
    print(f"\nA={A} w={w} t={t}: {qc.num_qubits} qubits, depth {isa.depth()}",
          flush=True)
    counts = S.run([isa], shots=2048).result()[0].data.out.get_counts()
    el = time.time() - t0
    r_true = 1
    while pow(A, r_true, N) != 1:
        r_true += 1
    print(f"  ideal peaks: {[round(2**t*s/r_true) for s in range(r_true)]}")
    for b, c in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        y = int(b, 2)
        print(f"    y={y:4d}  {c:5d} shots  {y}/{2**t} -> "
              f"{Fraction(y, 2**t).limit_denominator(N-1)}")
    r = order_from_counts(counts, A, N, t)
    print(f"  r = {r} (true {r_true}) {'OK' if r == r_true else 'MISMATCH'} "
          f"[{el:.0f}s]", flush=True)
    assert r == r_true, (A, w, t, r, r_true)
    if r % 2 == 0:
        d = math.gcd(pow(A, r // 2, N) - 1, N)
        print(f"  gcd(A^(r/2)-1,15) = {d} -> 15 = {d} x {N//d}"
              if 1 < d < N else f"  gcd trivial ({d}) -- retry base")
print("\nWINDOWED END-TO-END: ALL PASSED")
