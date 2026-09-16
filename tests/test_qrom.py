"""Unary-iteration QROM: every address, both control values, ancillas clean."""
import time
from qiskit.circuit import QuantumCircuit, QuantumRegister
from _fastsim import outcome, setv
from qrom import lookup_ui

t0 = time.time()
for m in (1, 2, 3, 4):
    W = 4
    table = [(11 * j + 5) % (2**W) for j in range(2**m)]
    for ctrl_on in (True, False):
        for a in range(2**m):
            c = QuantumRegister(1); ad = QuantumRegister(m)
            out = QuantumRegister(W); anc = QuantumRegister(m)
            qc = QuantumCircuit(c, ad, out, anc)
            if ctrl_on: qc.x(c[0])
            setv(qc, a, ad)
            lookup_ui(qc, c[0], list(ad), list(out), table, list(anc))
            v = outcome(qc)
            got = ((v >> 1) & (2**m - 1), (v >> (1+m)) & (2**W - 1), v >> (1+m+W))
            want = (a, table[a] if ctrl_on else 0, 0)
            assert got == want, (m, ctrl_on, a, got, want)
    # cost + space check
    c = QuantumRegister(1); ad = QuantumRegister(m); out = QuantumRegister(W); anc = QuantumRegister(m)
    qc = QuantumCircuit(c, ad, out, anc)
    lookup_ui(qc, c[0], list(ad), list(out), table, list(anc))
    tof = qc.count_ops().get("ccx", 0)
    print(f"  m={m}: OK both controls; {tof:3d} Toffoli (2(L-1)={2*(2**m-1):3d}), "
          f"{m} ancillas vs {1<<m:2d} for one-hot")
print(f"unary-iteration QROM: ALL PASSED ({time.time()-t0:.0f}s)")
