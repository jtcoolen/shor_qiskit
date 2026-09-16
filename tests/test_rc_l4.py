"""Levels 3-4 over the ripple-carry adder: exhaustive at N=15."""
import math, time
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from rc_adder import rc_c_ua

N = 15; n = math.ceil(math.log2(N))
def setv(qc, v, qs):
    for i, q in enumerate(qs):
        if (v >> i) & 1: qc.x(q)

t0 = time.time()
for A in (2, 4, 7, 8, 11, 13):
    for ctrl_on in (True, False):
        for x0 in range(N):
            c=QuantumRegister(1); x=QuantumRegister(n); y=QuantumRegister(n)
            sf=QuantumRegister(2); anc=QuantumRegister(n+2)
            qc=QuantumCircuit(c,x,y,sf,anc)
            if ctrl_on: qc.x(c[0])
            setv(qc, x0, x)
            rc_c_ua(qc, c[0], A, list(x), list(y), sf[0], sf[1], N, list(anc))
            sv = Statevector.from_instruction(qc).data
            v = int(np.argmax(np.abs(sv)))
            assert abs(abs(sv[v])-1) < 1e-9, "not a basis state"
            got_x = (v >> 1) & (2**n - 1); rest = v >> (1+n)
            want = (A*x0) % N if ctrl_on else x0
            assert (got_x, rest) == (want, 0), (A, ctrl_on, x0, got_x, rest, want)
    print(f"  rc_c_ua N=15 A={A:2d}: OK ({qc.num_qubits} qubits)", flush=True)
print(f"Levels 3-4 over ripple-carry: OK, all A, all x, control on/off, "
      f"scratch clean ({time.time()-t0:.0f}s)")
