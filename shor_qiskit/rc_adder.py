"""Ripple-carry (CDKM) arithmetic: a drop-in replacement for the Fourier
Level 1 of shor_essentials.py, using only X, CNOT and Toffoli.

Cuccaro, Draper, Kutin, Moulton, quant-ph/0410184.
"""


# --- Level 0: the two three-qubit blocks ------------------------------------
def _maj(qc, c, b, a, ctrls=()):
    """MAJ: |c,b,a> -> |c^a, b^a, MAJ(a,b,c)>.  Only the b-target takes ctrls."""
    qc.mcx([*ctrls, a], b)               # plain CNOT when ctrls is empty
    qc.cx(a, c)
    qc.ccx(c, b, a)


def _uma(qc, c, b, a, ctrls=()):
    """UMA: undoes MAJ's carry and writes the sum bit into b."""
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.mcx([*ctrls, c], b)               # plain CNOT when ctrls is empty


# --- Level 1a: add one quantum register into another ------------------------
def rc_add(qc, x, y, carry, ctrls=()):
    """|x>|y> -> |x>|(y + x) mod 2^n>.  carry: one clean ancilla, returned |0>.
    With ctrls, acts only when all controls are 1 (see the telescoping note)."""
    n = len(y)
    assert len(x) == n, "x and y must be the same width"
    _maj(qc, carry, y[0], x[0], ctrls)
    for i in range(n - 1):
        _maj(qc, x[i], y[i + 1], x[i + 1], ctrls)
    for i in range(n - 1, 0, -1):
        _uma(qc, x[i - 1], y[i], x[i], ctrls)
    _uma(qc, carry, y[0], x[0], ctrls)


# --- Level 1b: add a classical constant (the interface Level 2 wants) -------
def rc_add_const(qc, X, y, anc, ctrls=()):
    """|y> -> |y + X mod 2^n>.  anc: n+1 clean ancillas, returned clean.
    Drop-in for add_const / c_add_const, plus the ancilla register."""
    n = len(y)
    carry, xreg = anc[0], anc[1:n + 1]
    Xm = X % (2**n)                      # negative X just works, as in Fourier
    for i in range(n):                   # encode the constant -- uncontrolled
        if (Xm >> i) & 1:
            qc.x(xreg[i])
    rc_add(qc, xreg, y, carry, ctrls)
    for i in range(n):                   # unencode
        if (Xm >> i) & 1:
            qc.x(xreg[i])


# --- Level 2: the same seven blocks, verbatim, over the RC adder ------------
def rc_c_add_mod(qc, ctrls, X, y, sign, flag, N, anc):
    """|y> -> |(y + X) mod N> when all ctrls are 1; identity otherwise.
    Identical in structure to c_add_mod -- only the adder changed."""
    yw = list(y) + [sign]
    rc_add_const(qc, X, yw, anc, ctrls)  # (1) + X          (controlled)
    rc_add_const(qc, -N, yw, anc)        # (2) - N
    qc.cx(sign, flag)                    # (3) sign -> flag
    rc_add_const(qc, N, yw, anc, [flag])  # (4) + N  if flag
    rc_add_const(qc, -X, yw, anc)        # (5) - X
    qc.mcx(list(ctrls) + [sign], flag)    # (6) clear flag from the answer
    qc.x(flag)
    rc_add_const(qc, X, yw, anc)         # (7) + X


# --- Levels 3-5: unchanged in structure, only the adder differs -------------
def rc_c_mult_acc(qc, c, A, x, y, sign, flag, N, anc):
    """|c>|x>|y> -> |c>|x>|(y + A*x) mod N>."""
    for i in range(len(x)):
        rc_c_add_mod(qc, [c, x[i]], (A << i) % N, y, sign, flag, N, anc)


def rc_c_ua(qc, c, A, x, y, sign, flag, N, anc):
    """|c>|x>|0> -> |c>|A^c x mod N>|0>."""
    rc_c_mult_acc(qc, c, A % N, x, y, sign, flag, N, anc)
    for i in range(len(x)):
        qc.cswap(c, x[i], y[i])
    B = pow(A, -1, N)                        # classical inverse, unchanged
    rc_c_mult_acc(qc, c, (-B) % N, x, y, sign, flag, N, anc)
