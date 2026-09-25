"""Order finding with ONE counting qubit: 4n+2 -> 2n+3 qubits.

In the inverse QFT every controlled rotation is controlled by a qubit that is
about to be measured anyway.  So measure it first and make the control
classical (Griffiths-Niu semiclassical QFT; Mosca-Ekert, Parker-Plenio,
Beauregard).  The counting register never exists all at once: one qubit is
Hadamarded, drives its rung, receives phase corrections conditioned on every
previously measured bit, is measured, and is reset.  `semiclassical.py` holds
that loop; it is shared with the ECDLP circuit.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from semiclassical import semiclassical_iqft
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

    def rung(k):                                     # controlled U^(2^k)
        return lambda qc, c: c_ua(qc, c, pow(A, 2 ** k, N),
                                  list(tgt), list(anc), sf[0], sf[1], N)
    semiclassical_iqft(qc, ctr[0], out, [rung(k) for k in range(t)])
    return qc


# Readout note: the semiclassical transform measures the LEAST significant bit
# of the phase first and stores bit i in out[i] -- Qiskit's own bit order, and
# the same as order_circuit's.  So the plain `order_from_counts` reads this
# circuit unchanged; no reversal is needed.  (Checked exactly against the
# closed-form distribution in tests/test_semiclassical.py.)
