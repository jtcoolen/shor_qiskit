"""The windowed accumulate with EVERY lookup uncomputed by measurement."""
import time
from qiskit_aer import AerSimulator
from unlookup import run_windowed_acc_mbu

SIM = AerSimulator(method="statevector")
N, n, w = 15, 4, 2
t0 = time.time(); nm = 0
for k in (1, 7, 13):
    for c_val in (1, 0):
        for x_val in range(N):
            xo, yo, scratch, meas = run_windowed_acc_mbu(
                SIM, N, k, x_val, c_val, w, n)
            want = (k * x_val) % N if c_val else 0
            assert (xo, yo, scratch) == (x_val, want, 0), \
                (k, c_val, x_val, xo, yo, scratch, want)
            nm += meas
    print(f"  k={k:2d}: OK all x, control on/off; x preserved, y=k*x mod 15, "
          f"scratch clean", flush=True)
print(f"windowed accumulate with measurement-based uncomputation: ALL PASSED "
      f"({nm} mid-circuit measurements, {time.time()-t0:.0f}s)")
