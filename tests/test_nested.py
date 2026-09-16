"""Nested windowing: |e>|1>|0> -> |e>|k^e mod N>|0> for every exponent."""
import math, time
from qiskit.circuit import QuantumCircuit, QuantumRegister
from _fastsim import outcome, setv
from nested import windowed_exponentiate

N, n, k = 15, 4, 7
for we, wm in ((2, 2), (1, 2), (2, 1)):
    t0 = time.time(); te = 4
    for ev in range(2**te):
        e = QuantumRegister(te); a = QuantumRegister(n); b = QuantumRegister(n)
        sf = QuantumRegister(2); tmp = QuantumRegister(n)
        anc = QuantumRegister(we + wm); one = QuantumRegister(1)
        qc = QuantumCircuit(e, a, b, sf, tmp, anc, one)
        setv(qc, ev, e)
        qc.x(a[0])                                    # a = 1
        windowed_exponentiate(qc, k, N, list(e), list(a), list(b),
                              sf[0], sf[1], list(tmp), list(anc), one[0], we, wm)
        v = outcome(qc)
        got_e = v & (2**te - 1)
        got_a = (v >> te) & (2**n - 1)
        rest = v >> (te + n)
        want = pow(k, ev, N)
        assert (got_e, got_a, rest) == (ev, want, 0), \
            (we, wm, ev, got_e, got_a, rest, want)
    mults = math.ceil(te / we)
    print(f"  we={we} wm={wm}: OK all {2**te} exponents, k^e mod 15 correct, "
          f"scratch clean; {mults} multiplications vs {te} ({qc.num_qubits} qubits, "
          f"{time.time()-t0:.0f}s)")
print("NESTED WINDOWING: ALL PASSED")
