"""The full coset order-finding circuit (Zalka coset representation + oblivious
carry runways + windowed lookup arithmetic), after Gidney-Ekera 2021 Sec. 2.
Nothing here is modular: every addition is a plain (piecewise) addition.

Layout.  Both registers X (multiplicand) and Y (accumulator) hold coset
states on m = ceil(lg N) + cpad qubits.  One controlled multiplication
(`c_ua_coset`) is the same multiply / swap / un-multiply as the exact
circuit, with the coset twist that decides what has to be true of Y:

    Y = |0>_coset -> Y += A x -> swap -> Y += -A^-1 x  ->  Y = |0>_coset

The final Y is the coset state of 0 again (not merely "some multiple of N"):
the number of N's in each branch depends on the x-branch, but Y's coset index
ranges over the full 2^cpad values, so shifting it by that dependence is a
bijection and Y factors out of X (up to the edge terms the deviation bound
counts).  The runways live only on the accumulator and only during an
accumulation: encode (|+>, subtract), add piecewise, remove (add back, H).
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

import runway as rw
from coset import coset_width
from qrom import lookup_ui
from semiclassical import semiclassical_iqft
from shor_essentials import add_const, c_add_const


# --- encoding |k>_coset with a circuit (Gidney 1905.08488 Fig. 1) ----------
def encode_coset(qc, reg, k, N, cpad, pad, flag):
    """reg (m qubits, |0>) -> |k>_coset = 2^(-cpad/2) sum_j |jN + k>, 0 <= k < N.

    Step h puts `pad` in |+> and adds 2^h N when it is set; at that point every
    branch is below 2^(h+1) N, and pad == [reg >= 2^h N], so a comparison
    returns it to |0>.  pad, flag: clean qubits, returned clean."""
    reg = list(reg)
    for i in range(len(reg)):
        if (k >> i) & 1:
            qc.x(reg[i])
    for h in range(cpad):
        c = N << h
        qc.h(pad)
        c_add_const(qc, [pad], c, reg)
        add_const(qc, -c, reg + [flag])        # flag = [reg < c]
        qc.cx(flag, pad)
        qc.x(pad)                              # pad ^= [reg >= c]  ->  |0>
        add_const(qc, c, reg + [flag])


def decode_coset(value, N):
    return value % N


# --- accumulate: Y += A * X, X and Y coset-encoded ---------------------------
def coset_mult_acc(qc, ctrl, A, X, Y, N, runways=(), window=None,
                   tmp=None, anc=None):
    """|x>_coset |y>_coset -> |x>_coset |y + A x>_coset when ctrl is 1.

    runways: oblivious carry runways on Y, as (position, qubits) pairs.
    window:  None for one controlled constant per bit of x; w > 1 for one
             table lookup + one quantum addition per w bits (needs tmp, a
             len(ceil lg N)-qubit clean register, and anc, w clean qubits).
    """
    runways = list(runways)
    rw.encode_runways(qc, Y, runways)
    if window is None:
        for i in range(len(X)):
            rw.piecewise_add_const(qc, (A << i) % N, Y, runways, ctrls=[ctrl, X[i]])
    else:
        t = list(tmp)
        for i in range(0, len(X), window):
            win = list(X[i:i + window])
            table = [(A * j << i) % N for j in range(1 << len(win))]
            lookup_ui(qc, ctrl, win, t, table, list(anc)[:len(win)])
            rw.piecewise_add_quantum(qc, t, Y, runways)
            lookup_ui(qc, ctrl, win, t, table, list(anc)[:len(win)])
    rw.remove_runways(qc, Y, runways)


def c_ua_coset(qc, ctrl, A, X, Y, N, runways=(), window=None, tmp=None, anc=None):
    """|c>|A^? x>_coset|0>_coset -> |c>|A^c x>_coset|0>_coset (Level 4, coset form)."""
    coset_mult_acc(qc, ctrl, A % N, X, Y, N, runways, window, tmp, anc)
    for i in range(len(X)):
        qc.cswap(ctrl, X[i], Y[i])
    coset_mult_acc(qc, ctrl, (-pow(A, -1, N)) % N, X, Y, N, runways, window, tmp, anc)


# --- the order-finding circuit ----------------------------------------------
def coset_order_circuit(A, N, cpad, t=None, runway_spacing=None, runway_len=None,
                        window=None):
    """Order finding with a one-qubit semiclassical counting register, both
    arithmetic registers in coset form, optional runways on the accumulator
    (every `runway_spacing` bits, `runway_len` long) and optional windowing.

    Returns (circuit, layout) with layout = dict(m=, runway_positions=)."""
    n = math.ceil(math.log2(N))
    m = coset_width(N, cpad)
    t = t if t is not None else 2 * n
    ctr = QuantumRegister(1, "ctr")
    X = QuantumRegister(m, "x")
    Y = QuantumRegister(m, "y")
    pad = QuantumRegister(2, "enc")                    # encoder scratch: pad, flag
    pos = rw.runway_positions(m, runway_spacing, runway_len) if runway_spacing else []
    rws = [(p, QuantumRegister(runway_len, f"rw{p}")) for p in pos]
    tmp = QuantumRegister(n, "tmp") if window else None
    anc = QuantumRegister(window, "qrom") if window else None
    out = ClassicalRegister(t, "out")
    regs = [ctr, X, Y, pad] + [r for _, r in rws] + [r for r in (tmp, anc) if r]
    qc = QuantumCircuit(*regs, out)
    encode_coset(qc, X, 1, N, cpad, pad[0], pad[1])
    encode_coset(qc, Y, 0, N, cpad, pad[0], pad[1])
    runways = [(p, list(r)) for p, r in rws]

    def rung(k):
        return lambda qc, c: c_ua_coset(
            qc, c, pow(A, 2 ** k, N), list(X), list(Y), N, runways, window,
            list(tmp) if tmp else None, list(anc) if anc else None)
    semiclassical_iqft(qc, ctr[0], out, [rung(k) for k in range(t)])
    return qc, dict(m=m, runway_positions=pos)
