"""Minimal, complete Shor implementation -- exactly the code shown in the
essentials document. Fourier-basis arithmetic after Draper/Beauregard."""

import math
from fractions import Fraction

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister


# --- Level 0: the quantum Fourier transform ---------------------------------
def qft(n, approx=0):
    """QFT on n qubits, Qiskit convention (qubit 0 = LSB):
        |j> -> 2^(-n/2) sum_k exp(2*pi*i*j*k/2^n) |k>.
    approx > 0 drops controlled rotations by less than pi/2^approx."""
    qc = QuantumCircuit(n, name="qft")
    for j in reversed(range(n)):
        qc.h(j)
        for k in range(j):
            if approx and (j - k) > approx:
                continue                     # angle too small to matter
            qc.cp(np.pi / 2 ** (j - k), k, j)
    for i in range(n // 2):                  # reverse the qubit order
        qc.swap(i, n - 1 - i)
    return qc.to_gate()


# --- Level 1: addition of a classical constant, in Fourier space ------------
def add_const(qc, X, y):
    """|y> -> |y + X mod 2^n> on the qubit list y (LSB first)."""
    n = len(y)
    qc.append(qft(n), y)
    for i in range(n):
        qc.p(2 * np.pi * X * 2.0 ** (i - n), y[i])
    qc.append(qft(n).inverse(), y)


def c_add_const(qc, ctrls, X, y):
    """add_const controlled on every qubit in ctrls."""
    n = len(y)
    qc.append(qft(n), y)
    for i in range(n):
        qc.mcp(2 * np.pi * X * 2.0 ** (i - n), ctrls, y[i])
    qc.append(qft(n).inverse(), y)


# --- Level 2: modular addition of a classical constant ----------------------
def c_add_mod(qc, ctrls, X, y, sign, flag, N):
    """|y> -> |(y + X) mod N> when all ctrls are 1; identity otherwise.
    Needs 0 <= X < N, 0 <= y < N, sign and flag in |0>."""
    yw = list(y) + [sign]                     # (n+1)-wide value register
    c_add_const(qc, ctrls, X, yw)             # (1) + X          (controlled)
    add_const(qc, -N, yw)                     # (2) - N
    qc.cx(sign, flag)                         # (3) sign -> flag
    c_add_const(qc, [flag], N, yw)            # (4) + N  if flag
    add_const(qc, -X, yw)                     # (5) - X
    qc.mcx(list(ctrls) + [sign], flag)        # (6) clear flag from the answer
    qc.x(flag)
    add_const(qc, X, yw)                      # (7) + X

# --- Levels 3-4: controlled in-place multiplication -------------------------
def c_mult_acc(qc, c, A, x, y, sign, flag, N):
    """|c>|x>|y> -> |c>|x>|(y + A*x) mod N> : shift-and-add accumulate."""
    for i in range(len(x)):
        c_add_mod(qc, [c, x[i]], (A << i) % N, y, sign, flag, N)


def c_ua(qc, c, A, x, y, sign, flag, N):
    """|c>|x>|0> -> |c>|A^c x mod N>|0> : multiply, swap, uncompute."""
    c_mult_acc(qc, c, A % N, x, y, sign, flag, N)
    for i in range(len(x)):
        qc.cswap(c, x[i], y[i])
    B = pow(A, -1, N)                        # classical inverse, gcd(A,N)=1
    c_mult_acc(qc, c, (-B) % N, x, y, sign, flag, N)


# --- Level 5: the order-finding circuit -------------------------------------
def order_circuit(A, N, t=None):
    """Counting register of t qubits (default 2n), target, n+2 scratch."""
    n = math.ceil(math.log2(N))
    t = t if t is not None else 2 * n
    ctr, tgt = QuantumRegister(t, "ctr"), QuantumRegister(n, "tgt")
    anc = QuantumRegister(n, "anc")
    sf = QuantumRegister(2, "sf")            # sign qubit, flag qubit
    out = ClassicalRegister(t, "out")
    qc = QuantumCircuit(ctr, tgt, anc, sf, out)
    qc.h(ctr)                                # Hadamard layer
    qc.x(tgt[0])                             # target = |1>
    for i in range(t):                       # the ladder: t rungs
        c_ua(qc, ctr[i], pow(A, 2**i, N), tgt, anc, sf[0], sf[1], N)
    qc.append(qft(t).inverse(), ctr)     # inverse QFT
    qc.measure(ctr, out)
    return qc


# --- Classical post-processing ----------------------------------------------
def order_from_counts(counts, A, N, t):
    for bits in sorted(counts, key=counts.get, reverse=True):
        y = int(bits, 2)
        if y == 0:
            continue
        r = Fraction(y, 2**t).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            return r
    return 0


def find_factor(N, run):                     # run: circuit -> counts dict
    if N % 2 == 0:
        return 2
    for k in range(2, N.bit_length() + 1):   # prime-power check
        d = round(N ** (1 / k))
        if d > 1 and d**k == N:
            return d
    import random
    while True:
        a = random.randrange(2, N)
        if (d := math.gcd(a, N)) > 1:
            return d                         # lucky gcd
        t = 2 * math.ceil(math.log2(N))
        r = order_from_counts(run(order_circuit(a, N)), a, N, t)
        if r and r % 2 == 0:
            d = math.gcd(pow(a, r // 2, N) - 1, N)
            if 1 < d < N:
                return d
