"""The semiclassical inverse QFT (Griffiths-Niu): a counting register of ONE qubit.

Shor's circuits -- order finding and the elliptic-curve discrete log alike --
end the same way: t counting qubits in |+>, qubit k controls a rung of weight
2^k, then qft(t).inverse() and measure.  Every controlled rotation in that
inverse QFT is controlled by a qubit that is about to be measured anyway.  So
measure it first and make the control classical.  The counting register then
never exists all at once: one qubit is Hadamarded, drives its rung, receives
phase corrections conditioned on every previously measured bit, is measured,
and is reset.  t qubits become 1, for t mid-circuit measurements and
t(t-1)/2 classically conditioned phases (Mosca-Ekert, Parker-Plenio,
Beauregard).

Why the bit order comes out right.  Write the measured integer y as
0.b_1 b_2 ... b_t in units of 2^t, so b_t is its least significant bit.  The
qubit that drives the rung of weight 2^(t-1) carries phase 0.b_t, which one
Hadamard reads off exactly; so rungs run heaviest first and the FIRST
measurement is the LEAST significant bit.  Step i then carries
0.b_(t-i) b_(t-i+1) ... b_t; subtracting the already-measured tail
b_(t-j) / 2^(i-j+1) for each j < i leaves 0.b_(t-i), one Hadamard reads it,
and it lands in out[i] -- bit i of y, exactly where the full circuit's
qft(t).inverse() + measure puts it.  So the readout is unchanged: the same
post-processing reads both circuits.

The equivalence holds for arbitrary rungs, not only commuting powers of one
unitary: it is the deferred-measurement principle, applied after the heaviest
rung is moved first.  In Shor the rungs commute, so the order is free.
`tests/test_semiclassical.py` checks all of this exactly.
"""

import math


def semiclassical_iqft(qc, ctrl, out, rungs):
    """Append rungs + inverse QFT + measurement, on one recycled control qubit.

    `rungs[k](qc, ctrl)` appends the operation that counting bit k (weight
    2^k) would control.  `ctrl` must be |0> on entry and is |0> again on exit.
    `out[i]` receives bit i of the measured integer, as the full circuit's
    qft(len(rungs)).inverse() followed by measuring qubit i into out[i].

    Needs mid-circuit measurement and feed-forward (dynamic circuits).
    """
    t = len(rungs)
    assert len(out) >= t, "one classical bit per rung"
    for i in range(t):
        qc.h(ctrl)
        rungs[t - 1 - i](qc, ctrl)                   # heaviest rung first
        for j in range(i):                           # the inverse QFT, classically
            with qc.if_test((out[j], 1)):
                qc.p(-math.pi / 2 ** (i - j), ctrl)
        qc.h(ctrl)
        qc.measure(ctrl, out[i])
        with qc.if_test((out[i], 1)):                # reset for the next rung
            qc.x(ctrl)
