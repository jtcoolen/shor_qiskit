"""Does t=n really fail where t=2n works?  N=21, A=2, true r=6."""
import math
from fractions import Fraction
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_circuit, order_from_counts

N, A, r_true = 21, 2, 6
PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=11)

for t in (5, 7):                      # n and n+2, both short of 2n=10
    qc = order_circuit(A, N, t=t)
    counts = S.run([PM.run(qc)], shots=2048).result()[0].data.out.get_counts()
    r = order_from_counts(counts, A, N, t)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"t={t} ({qc.num_qubits} qubits): recovered r={r} (true {r_true}) "
          f"{'OK' if r==r_true else '<-- FAILS'}")
    for b, c in top:
        y = int(b, 2)
        print(f"    y={y:3d}/{2**t}  {c:5d} shots  -> {Fraction(y,2**t).limit_denominator(N-1)}")
