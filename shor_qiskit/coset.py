"""Zalka's coset representation of modular integers (Gidney-Ekera 2021, Sec 2.4).

The seven-block modular adder exists only to turn addition mod 2^m into
addition mod N.  The coset representation removes the need: store k not as
|k> but as the periodic superposition

    |k>_coset  =  2^(-cpad/2) * sum_{j=0}^{2^cpad - 1} |jN + k>

in a register of m = n + cpad qubits.  Every branch is congruent to k mod N,
so a PLAIN, NON-MODULAR addition of a already gives a state whose every branch
is congruent to k + a.  When k + a >= N the sum simply re-indexes,

    sum_j |jN + (k+a)>  =  sum_{j'=1}^{2^cpad} |j'N + (k+a-N)>,

which differs from the ideal coset state of (k+a) mod N in only the two edge
terms out of 2^cpad.  That is the whole approximation: the deviation per
addition is ~2^-cpad, and it is subadditive over a sequence of additions.

So Level 2 disappears into Level 1, and the sign and flag qubits with it.
"""

import math

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister

from shor_essentials import add_const, c_add_const


def coset_width(N, cpad):
    """Register width m holding the coset representation."""
    return math.ceil(math.log2(N)) + cpad


def coset_state(k, N, cpad):
    """The ideal coset state as an amplitude vector on m = n + cpad qubits."""
    m = coset_width(N, cpad)
    v = np.zeros(1 << m, dtype=complex)
    J = 1 << cpad
    for j in range(J):
        val = j * N + (k % N)
        if val < (1 << m):
            v[val] = 1.0
    nrm = np.linalg.norm(v)
    assert nrm > 0, "empty coset state"
    return v / nrm


def encode(qc, reg, k, N, cpad):
    """Load |k>_coset into reg.  Done here by state preparation, which is exact
    and keeps the demonstration honest; a real implementation builds it with
    cpad controlled additions of N (a one-off cost, paid once per run)."""
    qc.initialize(coset_state(k, N, cpad), list(reg))


def add(qc, X, reg, N=None):
    """Modular addition -- as a PLAIN addition.  This is the entire point:
    no compare, no subtract, no restore, no flag qubit, no sign qubit.
    X should be canonicalised into [0, N) to keep the deviation small."""
    if N is not None:
        X %= N
    add_const(qc, X, list(reg))


def c_add(qc, ctrls, X, reg, N=None):
    """Controlled version -- again just the plain controlled adder."""
    if N is not None:
        X %= N
    c_add_const(qc, list(ctrls), X, list(reg))


def decode(value, N):
    """Read a measured register value back as a residue."""
    return value % N


def deviation_bound(n_additions, cpad, n=None, csep=None):
    """Gidney-Ekera Sec 2.9: deviation per addition <= n/(csep * 2^cpad),
    subadditive, so total <= additions * that.  Without carry runways there is
    one piece, csep = n, and the per-addition bound is just 2^-cpad."""
    per = 1.0 / (1 << cpad) if (n is None or csep is None) else n / (csep * (1 << cpad))
    return n_additions * per


def padding_for(n, ne, delta_off=10):
    """cpad = 2 lg n + lg ne + delta_off  (Gidney-Ekera Sec 3.1)."""
    return math.ceil(2 * math.log2(n) + math.log2(ne) + delta_off)


# --- multiplication on the coset representation ----------------------------
def c_mult_acc(qc, ctrl, A, x_reg, y_reg, N):
    """|x>_coset |y>_coset  ->  |x>_coset |y + A*x mod N>_coset.

    Subtle point: in branch j the x register holds v = jN + x, so its BITS
    differ between branches.  Shift-and-add is still correct, because

        sum_i v_i * (A*2^i mod N)  ==  A*v  ==  A*(jN + x)  ==  A*x   (mod N),

    i.e. every branch accumulates something congruent to A*x.  Each added
    offset is canonicalised into [0, N), which is what keeps the deviation at
    2^-cpad per addition (Gidney-Ekera Sec 2.7).
    """
    for i in range(len(x_reg)):
        c_add(qc, [ctrl, x_reg[i]], (A << i) % N, y_reg, N)
