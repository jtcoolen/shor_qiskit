"""Level 4 over the windowed accumulate, N=15."""
import math, time
from qiskit.circuit import QuantumCircuit, QuantumRegister
from _fastsim import outcome, setv
from windowed import windowed_c_ua

N, n, w = 15, 4, 2
t0 = time.time()
for A in (2, 7, 13):
    for ctrl_on in (True, False):
        for x0 in range(N):
            c=QuantumRegister(1); x=QuantumRegister(n); y=QuantumRegister(n)
            sf=QuantumRegister(2); tmp=QuantumRegister(n); un=QuantumRegister(w)
            qc=QuantumCircuit(c,x,y,sf,tmp,un)
            if ctrl_on: qc.x(c[0])
            setv(qc, x0, x)
            windowed_c_ua(qc, c[0], A, list(x), list(y), sf[0], sf[1], N,
                          list(tmp), list(un), w)
            v = outcome(qc)
            got_x = (v>>1)&(2**n-1); rest = v>>(1+n)
            want = (A*x0)%N if ctrl_on else x0
            assert (got_x, rest)==(want,0), (A,ctrl_on,x0,got_x,rest,want)
    print(f"  windowed_c_ua N=15 w=2 A={A:2d}: OK ({qc.num_qubits} qubits)", flush=True)
print(f"Level 4 windowed: OK all A, all x, control on/off, scratch clean ({time.time()-t0:.0f}s)")
