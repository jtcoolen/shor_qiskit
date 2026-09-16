"""Measure the saving from NOT controlling every block of the modular adder.
Uses the real call pattern: c_add_mod is invoked with TWO controls (outer
control c and multiplicand bit x_i) from inside c_mult_acc."""
import math

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister

from shor_essentials import c_add_const, c_add_mod


def c_add_mod_all_controlled(qc, ctrls, X, y, sign, flag, N):
    """Naive version: every block carries the outer control."""
    yw = list(y) + [sign]
    c_add_const(qc, ctrls, X, yw)
    c_add_const(qc, ctrls, -N, yw)
    qc.mcx(list(ctrls) + [sign], flag)
    c_add_const(qc, list(ctrls) + [flag], N, yw)
    c_add_const(qc, ctrls, -X, yw)
    qc.mcx(list(ctrls) + [sign], flag)
    qc.mcx(ctrls, flag)
    c_add_const(qc, ctrls, X, yw)


def build(fn, N, n, n_ctrl):
    c = QuantumRegister(n_ctrl)
    y = QuantumRegister(n)
    sf = QuantumRegister(2)
    qc = QuantumCircuit(c, y, sf)
    fn(qc, list(c), 7 % N, list(y), sf[0], sf[1], N)
    return transpile(qc, basis_gates=["cx", "u"], optimization_level=1)


for N in (15, 33, 63):
    n = math.ceil(math.log2(N))
    print(f"\nN={N} (n={n})")
    for n_ctrl in (1, 2):
        a = build(c_add_mod, N, n, n_ctrl)
        b = build(c_add_mod_all_controlled, N, n, n_ctrl)
        ca, cb = a.count_ops().get("cx", 0), b.count_ops().get("cx", 0)
        print(f"  {n_ctrl} control(s): "
              f"2-of-7 {ca:5d} CNOT / depth {a.depth():5d}   |   "
              f"all-controlled {cb:5d} / {b.depth():5d}   "
              f"=> {cb / ca:.2f}x CNOT")
