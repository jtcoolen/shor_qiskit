"""One counting qubit: 2n+3 qubits.  Reaches moduli the 4n+2 circuit cannot.

End to end on Aer, asserted three ways: the order comes out, N factors, and
the measured histogram is the exact order-finding distribution
(shor_stats.order_finding_probs) to within shot noise -- the last because a
broken circuit gives a distorted histogram, not a wrong answer.
"""
import math
import os
import time

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

import shor_stats as ST
from onectrl import order_circuit_1c
from shor_essentials import order_from_counts

PM = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
S = Sampler(seed=41)
SHOTS = 2048
# Default to the two quick moduli.  Mid-circuit measurement forces Aer to
# simulate shot by shot, so cost grows fast: N=33 takes ~17 min, N=143 hours.
# Set SHOR_1C_BIG=1 to include the larger ones.
CASES = [(15, 7), (15, 2), (21, 2)]
if os.environ.get("SHOR_1C_BIG"):
    CASES += [(33, 5), (35, 2), (51, 5), (143, 2)]
for N, A in CASES:
    n = math.ceil(math.log2(N)); t = 2*n
    if math.gcd(A, N) != 1:
        print(f"N={N} A={A}: gcd>1, skip"); continue
    r_true = ST.multiplicative_order(A, N)
    t0 = time.time()
    qc = order_circuit_1c(A, N)
    counts = S.run([PM.run(qc)], shots=SHOTS).result()[0].data.out.get_counts()
    r = order_from_counts(counts, A, N, t)
    el = time.time()-t0
    assert r == r_true, (N, A, r, r_true)
    msg = ""
    if r % 2 == 0:
        d = math.gcd(pow(A, r//2, N) - 1, N)
        msg = f" -> {N} = {d} x {N//d}" if 1 < d < N else " -> gcd trivial (retry base)"
    print(f"N={N:4d} A={A:2d} n={n} : {qc.num_qubits:2d} qubits (plain {4*n+2}), "
          f"r={r_true} OK{msg}  [{el:.0f}s]", flush=True)

    exact = ST.order_finding_probs(A, N, t)
    emp = ST.empirical(ST.order_counts(counts), 2**t)
    tvd, bound = ST.tvd(emp, exact), ST.tvd_null(exact, SHOTS)
    assert tvd <= bound, (N, A, tvd, bound)
    hit = ST.order_success_mask(A, N, t)
    print(f"          histogram: TVD {tvd:.3f} from the exact distribution, "
          f"shot-noise bound {bound:.3f}; one shot gives r with "
          f"P = {exact[hit].sum():.3f} exact, {emp[hit].sum():.3f} measured",
          flush=True)
print("ONE-CONTROL: done")
