"""Gidney windowing (arXiv:1905.07682) on top of shor_essentials.py.

Windowed modular product addition: instead of n doubly-controlled modular
additions (one per bit of x), iterate x in windows of w bits, look the whole
window's contribution up from a precomputed table, and do one modular addition
per window -- ceil(n/w) additions instead of n.  The control bit is folded into
the lookup address (Gidney Fig. 5), so the addition itself is uncontrolled.
"""

import math

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister

from qrom import lookup_ui
from shor_essentials import add_const, c_add_const, qft


# --- QROM: table lookup via unary iteration (Gidney Fig. 2/4) --------------
def _unary_gate(m):
    """binary(m qubits) -> one-hot(2^m qubits), using 2^m - 1 Fredkins."""
    addr, unary = QuantumRegister(m, "a"), QuantumRegister(1 << m, "u")
    qc = QuantumCircuit(addr, unary)
    qc.x(unary[0])
    for i in reversed(range(m)):                  # MSB down to LSB
        step = 1 << i
        for j in range(0, 1 << m, 2 * step):
            qc.cswap(addr[i], unary[j], unary[j + step])
    return qc.to_gate(label="unary")


def lookup(qc, addr, out, table, unary):
    """|addr>|out> -> |addr>|out ^ table[addr]>.
    unary: 2^len(addr) clean ancillas, returned clean.
    Once the address is one-hot, loading the data is pure CNOTs."""
    g = _unary_gate(len(addr))
    qc.append(g, list(addr) + list(unary))
    for j in range(len(unary)):
        Tj = table[j] if j < len(table) else 0
        for b in range(len(out)):
            if (Tj >> b) & 1:
                qc.cx(unary[j], out[b])
    qc.append(g.inverse(), list(addr) + list(unary))


# --- adding a QUANTUM register in Fourier space ----------------------------
def add_quantum(qc, a, y, sign=+1):
    """y += sign*a  (mod 2^len(y)); a and y both quantum.  The quantum twin of
    add_const: same phase ramp, but each rotation is controlled by a bit of a."""
    n = len(y)
    qc.append(qft(n), y)
    for i in range(len(a)):
        for j in range(n):
            if i + j < n:                          # higher terms are trivial
                qc.cp(sign * 2 * np.pi * 2.0 ** (i + j - n), a[i], y[j])
    qc.append(qft(n).inverse(), y)


def add_quantum_mod(qc, t, y, sign, flag, N):
    """|y>|t> -> |(y+t) mod N>|t>, for 0 <= y,t < N.
    Beauregard's seven blocks again -- only blocks (1),(5),(7) change, from
    'add a classical constant' to 'add a quantum register'."""
    yw = list(y) + [sign]
    add_quantum(qc, t, yw, +1)            # (1) + t
    add_const(qc, -N, yw)                 # (2) - N
    qc.cx(sign, flag)                     # (3) sign -> flag
    c_add_const(qc, [flag], N, yw)        # (4) + N  if flag
    add_quantum(qc, t, yw, -1)            # (5) - t
    qc.cx(sign, flag)                     # (6) [R < t] == not flag, so this
    qc.x(flag)                            #     pair returns flag to |0>
    add_quantum(qc, t, yw, +1)            # (7) + t


# --- windowed accumulate: y += c * (k*x) mod N -----------------------------
def windowed_c_mult_acc(qc, c, k, x, y, sign, flag, N, t, anc, w):
    """Drop-in for c_mult_acc, but ceil(len(x)/w) modular additions instead of
    len(x).  c is the top-level control of each lookup, so when c=0 no data
    loads at all and the modular addition -- which is uncontrolled -- adds 0.

    t:   len(y) clean qubits holding one looked-up value (returned clean)
    anc: w clean qubits for the unary-iteration QROM     (returned clean)
    w:   window size -- bits of x consumed per lookup
    """
    n = len(y)
    for i in range(0, len(x), w):
        win = list(x[i:i + w])
        # window value j contributes (k * j * 2^i) mod N
        table = [((k * j) << i) % N for j in range(2**len(win))]
        lookup_ui(qc, c, win, t[:n], table, anc[:len(win)])
        add_quantum_mod(qc, t[:n], y, sign, flag, N)
        lookup_ui(qc, c, win, t[:n], table, anc[:len(win)])   # uncompute


# --- Levels 4-5 over the windowed accumulate -------------------------------
def windowed_c_ua(qc, c, A, x, y, sign, flag, N, t, anc, w):
    """|c>|x>|0> -> |c>|A^c x mod N>|0>.  Same multiply/swap/uncompute as
    c_ua; only the accumulate is windowed."""
    windowed_c_mult_acc(qc, c, A % N, x, y, sign, flag, N, t, anc, w)
    for i in range(len(x)):
        qc.cswap(c, x[i], y[i])
    B = pow(A, -1, N)                        # classical inverse, unchanged
    windowed_c_mult_acc(qc, c, (-B) % N, x, y, sign, flag, N, t, anc, w)


def windowed_order_circuit(A, N, w, t=None):
    """Order-finding circuit with windowed modular arithmetic."""
    from qiskit.circuit import ClassicalRegister
    n = math.ceil(math.log2(N))
    t = t if t is not None else 2 * n
    ctr = QuantumRegister(t, "ctr"); tgt = QuantumRegister(n, "tgt")
    anc = QuantumRegister(n, "anc"); sf = QuantumRegister(2, "sf")
    tmp = QuantumRegister(n, "tmp"); un = QuantumRegister(w, "qanc")
    out = ClassicalRegister(t, "out")
    qc = QuantumCircuit(ctr, tgt, anc, sf, tmp, un, out)
    qc.h(ctr)
    qc.x(tgt[0])
    for i in range(t):
        windowed_c_ua(qc, ctr[i], pow(A, 2**i, N), list(tgt), list(anc),
                      sf[0], sf[1], N, list(tmp), list(un), w)
    qc.append(qft(t).inverse(), ctr)
    qc.measure(ctr, out)
    return qc
