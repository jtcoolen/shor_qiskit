"""One counting qubit: 2n+3 qubits.  Reaches moduli the 4n+2 circuit cannot."""
import math, time
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_from_counts
from onectrl import order_circuit_1c

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=41)
# Default to the two quick moduli.  Mid-circuit measurement forces Aer to
# simulate shot by shot, so cost grows fast: N=33 takes ~17 min, N=143 hours.
# Set SHOR_1C_BIG=1 to include the larger ones.
import os
CASES = [(15, 7), (15, 2), (21, 2)]
if os.environ.get("SHOR_1C_BIG"):
    CASES += [(33, 5), (35, 2), (51, 5), (143, 2)]
for N, A in CASES:
    n = math.ceil(math.log2(N)); t = 2*n
    if math.gcd(A,N) != 1:
        print(f"N={N} A={A}: gcd>1, skip"); continue
    r_true = 1
    while pow(A, r_true, N) != 1: r_true += 1
    t0 = time.time()
    qc = order_circuit_1c(A, N)
    counts = S.run([PM.run(qc)], shots=2048).result()[0].data.out.get_counts()
    r = order_from_counts(counts, A, N, t)
    el = time.time()-t0
    ok = "OK" if r == r_true else f"got {r}"
    msg = ""
    if r and r % 2 == 0:
        d = math.gcd(pow(A, r//2, N) - 1, N)
        msg = f" -> {N} = {d} x {N//d}" if 1 < d < N else " -> gcd trivial (retry base)"
    print(f"N={N:4d} A={A:2d} n={n} : {qc.num_qubits:2d} qubits (plain {4*n+2}), "
          f"r={r_true} {ok}{msg}  [{el:.0f}s]", flush=True)
print("ONE-CONTROL: done")
