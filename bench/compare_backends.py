"""Fourier vs ripple-carry, same modular adder, same modulus."""
import math
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from shor_essentials import c_add_mod
from rc_adder import rc_c_add_mod

for N in (15, 63):
    n = math.ceil(math.log2(N)); X = 7 % N
    # Fourier
    c=QuantumRegister(1); y=QuantumRegister(n); sf=QuantumRegister(2)
    f=QuantumCircuit(c,y,sf); c_add_mod(f,[c[0]],X,list(y),sf[0],sf[1],N)
    # Ripple-carry
    c2=QuantumRegister(1); y2=QuantumRegister(n); sf2=QuantumRegister(2); a2=QuantumRegister(n+2)
    r=QuantumCircuit(c2,y2,sf2,a2); rc_c_add_mod(r,[c2[0]],X,list(y2),sf2[0],sf2[1],N,list(a2))

    print(f"\n=== N={N} (n={n}), one doubly-usable modular adder ===")
    for name, qc in (("Fourier", f), ("ripple-carry", r)):
        # decompose to primitive gates, count rotations vs Toffoli
        d = transpile(qc, basis_gates=["x","cx","ccx","h","p","cp","swap"],
                      optimization_level=0)
        ops = d.count_ops()
        rot = sum(v for k,v in ops.items() if k in ("p","cp","rz","u"))
        tof = ops.get("ccx",0)
        cx  = transpile(qc, basis_gates=["cx","u"], optimization_level=1)
        print(f"  {name:13s} qubits {qc.num_qubits:3d}   rotations {rot:5d}   "
              f"Toffoli-ish {tof:4d}   -> CNOT {cx.count_ops().get('cx',0):5d} "
              f"depth {cx.depth():5d}")
    # smallest rotation angle in the Fourier version
    print(f"  smallest Fourier angle: 2*pi/2^{n+1} = {2*math.pi/2**(n+1):.5f} rad")
