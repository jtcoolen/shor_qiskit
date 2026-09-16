"""Order finding with ONE counting qubit: 4n+2 -> 2n+3 qubits.

In the inverse QFT every controlled rotation is controlled by a qubit that is
about to be measured anyway.  So measure it first and make the control
classical (Griffiths-Niu semiclassical QFT; Mosca-Ekert, Parker-Plenio,
Beauregard).  The counting register never exists all at once: one qubit is
Hadamarded, drives its rung, receives phase corrections conditioned on every
previously measured bit, is measured, and is reset.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from shor_essentials import c_ua


def order_circuit_1c(A, N, t=None):
    """Same distribution as order_circuit, on 2n+3 qubits instead of 4n+2.
    Needs mid-circuit measurement + feed-forward (dynamic circuits)."""
    n = math.ceil(math.log2(N))
    t = t if t is not None else 2 * n
    ctr = QuantumRegister(1, "ctr")                  # the whole counting register
    tgt = QuantumRegister(n, "tgt")
    anc = QuantumRegister(n, "anc")
    sf = QuantumRegister(2, "sf")                    # sign, flag
    out = ClassicalRegister(t, "out")
    qc = QuantumCircuit(ctr, tgt, anc, sf, out)
    qc.x(tgt[0])                                     # target = |1>
    for i in range(t):
        qc.h(ctr[0])
        # rungs run most-significant exponent first
        c_ua(qc, ctr[0], pow(A, 2 ** (t - 1 - i), N),
             list(tgt), list(anc), sf[0], sf[1], N)
        for j in range(i):                           # semiclassical inverse QFT
            with qc.if_test((out[j], 1)):
                qc.p(-math.pi / 2 ** (i - j), ctr[0])
        qc.h(ctr[0])
        qc.measure(ctr[0], out[i])
        with qc.if_test((out[i], 1)):                # reset for the next rung
            qc.x(ctr[0])
    return qc


# Readout note: with the rungs run most-significant-exponent first, the FIRST
# measured bit is the LEAST significant bit of the phase -- which is exactly
# Qiskit's own bit order.  So the plain `order_from_counts` reads this circuit
# unchanged; no reversal is needed.  (Checked against known r at N=15,21,33.)
