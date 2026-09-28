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


# =============================================================================
# The full coset order-finding circuit, on reversible (Gidney) arithmetic
# =============================================================================
# Everything above uses the Fourier adder and `initialize`.  Below, the same
# representation is built from `ec_adders` on a `Machine`: a real encoding
# circuit, windowed coset multiplication with measurement-uncomputed lookups,
# and order finding -- Gidney-Ekera 2021 Sec 2.4-2.7 end to end.

def _restoring_quotient(m, reg, N, cpad, qbits):
    """qbits ^= floor(reg / N) for reg < N 2^cpad, reg restored.

    Restoring division, top quotient bit first: subtract N 2^i if it fits,
    record the flag, and at the end add everything back."""
    import ec_adders as A
    from ec_sim import Reg
    W = len(reg)
    flags = m.anc(cpad, "qf")
    steps = []
    for i in reversed(range(cpad)):
        cp, sc = m.anc(W, "cp"), m.anc(W, "sc")
        A.geq_const(m.ctx, reg, N << i, flags[i], cp, sc)      # flag = [reg >= N 2^i]
        A.csub_const(m.ctx, flags[i], reg, N << i, cp, sc)
        m.free(cp, sc)
        steps.append(i)
    for i in range(cpad):
        m.ctx.cx(flags[i], qbits[i])
    for i in reversed(steps):                                  # undo, bottom first
        cp, sc = m.anc(W, "cp"), m.anc(W, "sc")
        A.cadd_const(m.ctx, flags[i], reg, N << i, cp, sc)
        A.geq_const(m.ctx, reg, N << i, flags[i], cp, sc)
        m.free(cp, sc)
    m.free(flags)


def encode_circuit(m, reg, k, N, cpad):
    """|0> -> the coset state of k (reg: coset_width(N, cpad) qubits).

    Load k; put cpad control qubits in uniform superposition; add j N under
    them (plain additions); recover j = floor(reg / N) by restoring division
    and XOR it into the controls, which clears them.  A one-off cost of
    O(cpad (n + cpad)) Toffolis per register."""
    import ec_adders as A
    for i in range(len(reg)):
        if (k >> i) & 1:
            m.ctx.x(reg[i])
    js = m.anc(cpad, "j")
    for q in js:
        m.qc.h(q)
    W = len(reg)
    for i in range(cpad):
        cp, sc = m.anc(W, "cp"), m.anc(W, "sc")
        A.cadd_const(m.ctx, js[i], reg, N << i, cp, sc)
        m.free(cp, sc)
    _restoring_quotient(m, reg, N, cpad, js)
    m.free(js)


def coset_mult_add(m, ctrl, x, y, k, N, w):
    """y += ctrl * k * x, all in the coset representation: plain additions.

    x holds j N + x in every branch, and k (j N + x) == k x (mod N), so a
    window of x's *bits* can be looked up as sum_b (k 2^b mod N) directly --
    every entry canonical, which is what keeps the deviation at 2^-cpad per
    addition (Gidney-Ekera Sec 2.7)."""
    import ec_adders as A
    import ec_mbu as MB
    from ec_sim import Reg
    W = len(y)
    t = m.anc(W, "T")
    for j in range(0, len(x), w):
        win = list(x[j:j + w])
        addr = win + ([ctrl] if ctrl is not None else [])
        table = []
        for a in range(1 << len(addr)):
            c = 1 if ctrl is None else (a >> len(win)) & 1
            v = sum(((k << (j + b)) % N) for b in range(len(win)) if (a >> b) & 1) % N
            table.append(v if c else 0)
        MB.lookup(m, Reg(addr), t, table)
        sc = m.anc(W - 1, "ad")
        A.add(m.ctx, t, y, sc)                           # a PLAIN addition
        m.free(sc)
        MB.unlookup(m, Reg(addr), t, table)
    m.free(t)


def coset_c_mult_inplace(m, ctrl, x, zero, k, N, w):
    """x <- k^ctrl x on coset registers; `zero` holds the coset state of 0 and
    is returned holding it (approximately), so it serves every multiplication."""
    coset_mult_add(m, ctrl, x, zero, k, N, w)            # zero += k x
    for a, b in zip(x, zero):
        m.ctx.cswap(ctrl, a, b)
    kinv = pow(k, -1, N)
    coset_mult_add(m, ctrl, x, zero, (-kinv) % N, N, w)  # back to ~coset(0)


def order_circuit_coset(A_, N, cpad, t=None, w=2):
    """Order finding in the coset representation.  Returns (Machine, info)."""
    from qiskit.circuit import ClassicalRegister
    from ec_sim import Machine
    n = math.ceil(math.log2(N))
    t = t or 2 * n
    W = coset_width(N, cpad)
    m = Machine("and", "order-coset")
    ctr = m.alloc(t, "ctr")
    x, z = m.alloc(W, "x"), m.alloc(W, "z")
    out = ClassicalRegister(t, "out")
    m.qc.add_register(out)
    encode_circuit(m, x, 1, N, cpad)
    encode_circuit(m, z, 0, N, cpad)
    for q in ctr:
        m.qc.h(q)
    for i in range(t):
        coset_c_mult_inplace(m, ctr[i], x, z, pow(A_, 1 << i, N), N, w)
    from shor_essentials import qft
    m.qc.append(qft(t).inverse(), list(ctr))
    m.qc.measure(list(ctr), out)
    return m, {"t": t, "W": W, "qubits": m.qc.num_qubits, "additions": 2 * t * math.ceil(W / w)}
