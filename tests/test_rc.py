"""Verify the ripple-carry arithmetic, exhaustively, by exact statevector."""
import math
import time

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector

from rc_adder import _maj, _uma, rc_add, rc_add_const, rc_c_add_mod


def set_value(qc, val, qubits):
    for i, q in enumerate(qubits):
        if (val >> i) & 1:
            qc.x(q)


def outcome(qc):
    sv = Statevector.from_instruction(qc).data
    i = int(np.argmax(np.abs(sv)))
    assert abs(abs(sv[i]) - 1.0) < 1e-9, f"not a basis state ({abs(sv[i])})"
    return i


# --- MAJ / UMA truth tables and the UMA.MAJ identity -----------------------
for c in (0, 1):
    for b in (0, 1):
        for a in (0, 1):
            q = QuantumRegister(3)
            qc = QuantumCircuit(q)
            set_value(qc, c + 2 * b + 4 * a, q)   # q0=c, q1=b, q2=a
            _maj(qc, q[0], q[1], q[2])
            v = outcome(qc)
            gc, gb, ga = v & 1, (v >> 1) & 1, (v >> 2) & 1
            assert (gc, gb, ga) == (c ^ a, b ^ a, (a & b) ^ (a & c) ^ (b & c)), \
                (c, b, a, gc, gb, ga)
            # UMA . MAJ = |c, a^b^c, a>
            _uma(qc, q[0], q[1], q[2])
            v = outcome(qc)
            gc, gb, ga = v & 1, (v >> 1) & 1, (v >> 2) & 1
            assert (gc, gb, ga) == (c, a ^ b ^ c, a), (c, b, a, gc, gb, ga)
print("MAJ truth table = (c^a, b^a, MAJ) and UMA.MAJ = (c, a^b^c, a): OK, all 8")

# --- rc_add: exhaustive, n = 2..5, uncontrolled and controlled -------------
for n in range(2, 6):
    for ctrl in (None, 0, 1):
        for xv in range(2**n):
            for yv in range(2**n):
                c = QuantumRegister(1)
                x = QuantumRegister(n)
                y = QuantumRegister(n)
                a = QuantumRegister(1)
                qc = QuantumCircuit(c, x, y, a)
                if ctrl == 1:
                    qc.x(c[0])
                set_value(qc, xv, x)
                set_value(qc, yv, y)
                ctrls = () if ctrl is None else (c[0],)
                rc_add(qc, list(x), list(y), a[0], ctrls)
                v = outcome(qc)
                gx = (v >> 1) & (2**n - 1)
                gy = (v >> (1 + n)) & (2**n - 1)
                ga = v >> (1 + 2 * n)
                want = (yv + xv) % 2**n if ctrl != 0 else yv
                assert (gx, gy, ga) == (xv, want, 0), \
                    (n, ctrl, xv, yv, gx, gy, ga, want)
print("rc_add: OK exhaustive n=2..5, x preserved, carry clean, control on/off")

# --- gate counts match the CDKM claim (2n Toffoli, 4n+1 CNOT with overflow) -
for n in (4, 8):
    x = QuantumRegister(n); y = QuantumRegister(n); a = QuantumRegister(1)
    qc = QuantumCircuit(x, y, a)
    rc_add(qc, list(x), list(y), a[0])
    ops = qc.count_ops()
    print(f"  n={n}: {ops.get('ccx',0)} Toffoli, {ops.get('cx',0)} CNOT, "
          f"{2*n+1} qubits  (CDKM: 2n={2*n} Toffoli)")
    assert ops.get("ccx", 0) == 2 * n

# --- rc_add_const: exhaustive incl. negative constants ---------------------
for n in (3, 4):
    for yv in range(2**n):
        for X in (0, 1, 5, -3, 2**n - 1, 7 - 2**n):
            y = QuantumRegister(n); anc = QuantumRegister(n + 1)
            qc = QuantumCircuit(y, anc)
            set_value(qc, yv, y)
            rc_add_const(qc, X, list(y), list(anc))
            v = outcome(qc)
            gy, ganc = v & (2**n - 1), v >> n
            assert (gy, ganc) == ((yv + X) % 2**n, 0), (n, yv, X, gy, ganc)
print("rc_add_const: OK exhaustive n=3,4, negative constants, ancillas clean")

# --- rc_c_add_mod: exhaustive for N=9 and N=15, control on and off ---------
for N in (9, 15):
    n = math.ceil(math.log2(N))
    t0 = time.time()
    for ctrl_on in (True, False):
        for yv in range(N):
            for X in range(N):
                c = QuantumRegister(1)
                y = QuantumRegister(n)
                sf = QuantumRegister(2)
                anc = QuantumRegister(n + 2)      # value reg is n+1 wide
                qc = QuantumCircuit(c, y, sf, anc)
                if ctrl_on:
                    qc.x(c[0])
                set_value(qc, yv, y)
                rc_c_add_mod(qc, [c[0]], X, list(y), sf[0], sf[1], N, list(anc))
                v = outcome(qc)
                gy = (v >> 1) & (2**n - 1)
                rest = v >> (1 + n)               # sign, flag, all ancillas
                want = (yv + X) % N if ctrl_on else yv
                assert (gy, rest) == (want, 0), \
                    (N, ctrl_on, yv, X, gy, rest, want)
    print(f"rc_c_add_mod N={N}: OK all {N*N} (y,X) pairs x control on/off, "
          f"everything clean ({time.time()-t0:.0f}s)")

print("\nRIPPLE-CARRY: ALL TESTS PASSED")
