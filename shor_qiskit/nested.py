"""Nested windowed modular exponentiation -- Gidney arXiv:1905.07682 Sec 3.5.

Windowing the multiplication (windowed.py) saves one log factor.  Windowing the
*exponent* as well saves a second: instead of one controlled multiplication per
exponent qubit, take w_e exponent bits at a time and let the lookup select the
factor k^(v * 2^i).  Two consequences:

  * ceil(t / w_e) multiplications instead of t;
  * the multiplications need no control at all -- when the exponent window is 0
    the factor is k^0 = 1, and multiply-by-1 is the identity;
  * the swap between accumulator registers becomes a *relabelling*, i.e. free,
    where the controlled version needed n Fredkin gates.

The lookup address is the exponent window bits followed by the multiplicand
window bits, so one table serves both loops.
"""

import math

from qiskit.circuit import (ClassicalRegister, QuantumCircuit, QuantumRegister)

from qrom import lookup_ui
from shor_essentials import qft
from windowed import add_quantum_mod


def _madd_gate(n_t, n_y, N, inverse=False):
    """Gate for |t>|y>|sign,flag> -> |t>|(y+t) mod N>|sign,flag> (or its inverse)."""
    t = QuantumRegister(n_t, "t"); y = QuantumRegister(n_y, "y")
    sf = QuantumRegister(2, "sf")
    qc = QuantumCircuit(t, y, sf)
    add_quantum_mod(qc, list(t), list(y), sf[0], sf[1], N)
    g = qc.to_gate(label="y+=t mod N")
    return g.inverse() if inverse else g


def windowed_exponentiate(qc, k, N, e, a, b, sign, flag, tmp, anc, one, we, wm):
    """|e>|a=x>|b=0> -> |e>|k^e * x mod N>|b=0>, nested windowing.

    e:   exponent (counting) qubits          a, b: n-qubit registers, b starts |0>
    tmp: n clean qubits for lookup output    anc:  we+wm clean qubits for the QROM
    one: one clean qubit used as always-true lookup control (returned clean)
    """
    n = len(a)
    A, B = list(a), list(b)
    madd = _madd_gate(n, n, N)
    msub = _madd_gate(n, n, N, inverse=True)
    relabels = 0
    qc.x(one)                                     # always-true control
    for i in range(0, len(e), we):
        ei = list(e[i:i + we])
        ne = len(ei)
        kes = [pow(k, (2**i) * v, N) for v in range(2**ne)]
        kes_inv = [pow(v, -1, N) for v in kes]

        # B += A * k_e   (mod N):  maps (x, 0) -> (x, x*k_e)
        for j in range(0, n, wm):
            mi = A[j:j + wm]
            addr = ei + list(mi)
            table = [((kes[ad & ((1 << ne) - 1)] * (ad >> ne)) << j) % N
                     for ad in range(1 << len(addr))]
            lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])
            qc.append(madd, tmp[:n] + B + [sign, flag])
            lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])

        # A -= B * k_e^{-1} (mod N): maps (x, x*k_e) -> (0, x*k_e)
        for j in range(0, n, wm):
            mi = B[j:j + wm]
            addr = ei + list(mi)
            table = [((kes_inv[ad & ((1 << ne) - 1)] * (ad >> ne)) << j) % N
                     for ad in range(1 << len(addr))]
            lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])
            qc.append(msub, tmp[:n] + A + [sign, flag])
            lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])

        A, B = B, A                               # relabelling swap -- free
        relabels += 1
    qc.x(one)
    if relabels % 2:                              # odd: result sits in b, move it
        for p, q in zip(list(a), list(b)):
            qc.swap(p, q)


def nested_order_circuit(A, N, we, wm, t=None):
    """Order-finding circuit with windowing over both exponent and multiplicand."""
    n = math.ceil(math.log2(N))
    t = t if t is not None else 2 * n
    ctr = QuantumRegister(t, "ctr"); areg = QuantumRegister(n, "a")
    breg = QuantumRegister(n, "b"); sf = QuantumRegister(2, "sf")
    tmp = QuantumRegister(n, "tmp"); anc = QuantumRegister(we + wm, "anc")
    one = QuantumRegister(1, "one"); out = ClassicalRegister(t, "out")
    qc = QuantumCircuit(ctr, areg, breg, sf, tmp, anc, one, out)
    qc.h(ctr)
    qc.x(areg[0])                                  # a = |1>
    windowed_exponentiate(qc, A, N, list(ctr), list(areg), list(breg),
                          sf[0], sf[1], list(tmp), list(anc), one[0], we, wm)
    qc.append(qft(t).inverse(), ctr)
    qc.measure(ctr, out)
    return qc
