"""Quantum Carry-Save Adder: multi-operand addition at low depth.

[106] Sec 3.2 builds its Montgomery multiplication on the QCSA of [KLJ+25].
The idea is the classical one: a 3:2 compressor turns three addends into two
without propagating a single carry, so a tree of them reduces w operands to two
in O(log w) depth, and only the *last* addition has to be a real, carry-
propagating adder.  That is what buys [106] its depth numbers -- against
[HJN+20]'s w sequential controlled additions, this is one adder plus a shallow
tree.

The compressor, per bit position:

    S_i = A_i XOR B_i XOR C_i           three CNOTs, no Toffoli
    M_i = MAJ(A_i, B_i, C_i)            one AND, using
          MAJ(a,b,c) = a XOR ((a XOR b) AND (a XOR c))

and A + B + C = S + 2M.  So a compression costs n ANDs and no carry chain at
all; its depth is a constant, independent of n.

Reversibility.  (A,B,C) -> (S,M) throws away n bits, so it cannot be done in
place.  This module computes S and M into fresh registers, leaves the inputs
alone, and gives the caller `qcsa_sum`, which builds the tree, does the one
real addition, and then unwinds the tree to give every ancilla back -- the
QCSA / Add / QCSA-dagger shape of [106] Fig. 5.
"""

import ec_adders as A_


def csa_compress(m, A, B, C):
    """(A,B,C) -> (S, M) with A + B + C == S + M, inputs preserved.

    M is returned already shifted (its MAJ bits start at index 1), so it reads
    directly as 2*MAJ and the caller never has to track the offset.  The top
    bit of each input must be 0, which the padding in `qcsa_sum` guarantees;
    otherwise the shift would drop a carry.
    """
    ctx, W = m.ctx, len(A)
    assert len(B) == len(C) == W
    S, M = m.anc(W, "csaS"), m.anc(W, "csaM")
    for i in range(W):                       # S = A ^ B ^ C
        ctx.cx(A[i], S[i])
        ctx.cx(B[i], S[i])
        ctx.cx(C[i], S[i])
    for i in range(W - 1):                   # M = MAJ, written one place up
        ctx.cx(A[i], B[i])
        ctx.cx(A[i], C[i])
        ctx.and_(B[i], C[i], M[i + 1])
        ctx.cx(A[i], M[i + 1])
        ctx.cx(A[i], B[i])
        ctx.cx(A[i], C[i])
    return S, M


def csa_layers(w):
    """How many 3:2 compressions and how many tree layers for w operands."""
    n_comp, layers, cur = 0, 0, w
    while cur > 2:
        nxt = 0
        c = cur
        while c >= 3:
            c -= 3
            n_comp += 1
            nxt += 2
        nxt += c
        cur = nxt
        layers += 1
    return n_comp, layers


def qcsa_sum(m, ops, out, keep=False):
    """out (|0>) <- sum(ops).  Inputs preserved; all tree scratch returned.

    The tree runs down to two operands, one carry-propagating adder produces
    the answer, and then the tree is run backwards.  That last step is what
    makes this usable inside a reversible multiplier: with `keep=False` the
    only thing left behind is the answer.
    """
    W = len(out)
    assert all(len(o) == W for o in ops), "pad every operand to the output width"
    assert len(ops) >= 1

    start = len(m.qc.data)
    cur, made = list(ops), []
    while len(cur) > 2:
        nxt = []
        while len(cur) >= 3:
            S, M = csa_compress(m, cur.pop(), cur.pop(), cur.pop())
            made += [S, M]
            nxt += [S, M]
        nxt += cur
        cur = nxt
    tree = list(m.qc.data[start:])

    # the one real addition
    anc = m.anc(W, "qadd")
    for q_i, o_i in zip(cur[0], out):
        m.ctx.cx(q_i, o_i)
    if len(cur) == 2:
        A_.add(m.ctx, cur[1], out, anc)
    m.free(anc)

    if not keep:
        for ci in reversed(tree):                 # QCSA-dagger
            m.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)
        for r in made:
            m.free(r)
    return made


def qcsa_acc(m, ops, acc):
    """acc += sum(ops), inputs preserved, all tree scratch returned.

    The accumulating form: the tree reduces w addends to two, those two go into
    the accumulator with ordinary adders, and the tree is unwound.  Two
    carry-propagating additions in place of w -- which is the whole reason
    [106] Fig. 5 has a QCSA in it.
    """
    W = len(acc)
    assert all(len(o) == W for o in ops), "pad operands to the accumulator width"
    start = len(m.qc.data)
    cur, made = list(ops), []
    while len(cur) > 2:
        nxt = []
        while len(cur) >= 3:
            S, M = csa_compress(m, cur.pop(), cur.pop(), cur.pop())
            made += [S, M]
            nxt += [S, M]
        nxt += cur
        cur = nxt
    tree = list(m.qc.data[start:])

    anc = m.anc(W, "qadd")
    for r in cur:
        A_.add(m.ctx, r, acc, anc)
    m.free(anc)

    for ci in reversed(tree):
        m.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)
    for r in made:
        m.free(r)


def qcsa_width(n, w):
    """Output width for w operands of n bits: n + ceil(log2 w), plus one spare.

    The spare is not slack -- `csa_compress` shifts its carry word up by one and
    needs the top bit of every operand to be zero for that to be lossless.
    """
    return n + max(1, (w - 1).bit_length()) + 1
