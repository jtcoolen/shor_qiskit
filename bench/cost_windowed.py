"""Does windowing actually pay at these sizes?  Count, honestly."""
import math
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from shor_essentials import c_mult_acc
from windowed import windowed_c_mult_acc

def counts(qc):
    d = transpile(qc, basis_gates=["cx","u"], optimization_level=1)
    return d.count_ops().get("cx",0), d.depth()

print(f"{'N':>4} {'n':>2} {'method':>12} {'mod adds':>9} {'qubits':>7} {'CNOT':>7} {'depth':>7}")
for N in (15, 63, 255):
    n = math.ceil(math.log2(N)); k = 7 % N
    # reference: n modular additions
    c=QuantumRegister(1);x=QuantumRegister(n);y=QuantumRegister(n);sf=QuantumRegister(2)
    ref=QuantumCircuit(c,x,y,sf)
    c_mult_acc(ref,c[0],k,list(x),list(y),sf[0],sf[1],N)
    cx,dep = counts(ref)
    print(f"{N:>4} {n:>2} {'shift-add':>12} {n:>9} {ref.num_qubits:>7} {cx:>7} {dep:>7}")
    for w in (2, 3, 4):
        if w > n: continue
        c2=QuantumRegister(1);x2=QuantumRegister(n);y2=QuantumRegister(n)
        sf2=QuantumRegister(2);t2=QuantumRegister(n);un=QuantumRegister(w)
        win=QuantumCircuit(c2,x2,y2,sf2,t2,un)
        windowed_c_mult_acc(win,c2[0],k,list(x2),list(y2),sf2[0],sf2[1],N,
                            list(t2),list(un),w)
        cx,dep = counts(win)
        print(f"{'':>4} {'':>2} {'window w='+str(w):>12} {math.ceil(n/w):>9} "
              f"{win.num_qubits:>7} {cx:>7} {dep:>7}")
