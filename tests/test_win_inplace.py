"""Windowed in-place multiplication x *= k (mod 2^L): exact, scratch clean."""
from qiskit.circuit import QuantumCircuit, QuantumRegister
from _fastsim import outcome, setv
from windowed_inplace import times_equal

for L, w, ks in [(4, 2, (3, 5, 7, 11, 13, 15)), (5, 2, (3, 9, 21, 31)), (5, 3, (5, 27)),
                 (4, 4, (7,)), (4, 1, (7, 13))]:
    for k in ks:
        for ctrl_on in (True, False):
            for x0 in range(1 << L):
                c = QuantumRegister(1); x = QuantumRegister(L); t = QuantumRegister(L)
                a = QuantumRegister(w); cy = QuantumRegister(1)
                qc = QuantumCircuit(c, x, t, a, cy)
                if ctrl_on:
                    qc.x(c[0])
                setv(qc, x0, x)
                times_equal(qc, c[0], list(x), k, w, list(t), list(a), cy[0])
                v = outcome(qc)
                want = (k * x0) % (1 << L) if ctrl_on else x0
                got = (v >> 1) & ((1 << L) - 1)
                assert (got, v >> (1 + L)) == (want, 0), (L, w, k, ctrl_on, x0, got, want)
    print(f"  L={L} w={w}: OK for k in {ks}, all x, control on/off", flush=True)
print("windowed in-place multiplication: OK")
