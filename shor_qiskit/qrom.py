"""Space-efficient QROM by unary iteration (Gidney Fig. 2 / Babbush et al.).

The one-hot lookup materialises all 2^m address states at once, so it needs
2^m ancillas.  Unary iteration walks the address tree instead, holding only
the AND of the current prefix: m ancillas, same Toffoli count, and the
uncompute Toffolis are measurement-friendly ANDs.
"""


def _walk(qc, ctrl, addr, out, table, offset, anc):
    """out ^= table[offset + value(addr)], all conditioned on ctrl."""
    if not addr:
        T = table[offset] if offset < len(table) else 0
        for b in range(len(out)):
            if (T >> b) & 1:
                qc.cx(ctrl, out[b])           # leaf: data load is CNOTs
        return
    a, rest = addr[-1], addr[:-1]             # split on the most significant bit
    half, node = 1 << len(rest), anc[0]
    qc.ccx(ctrl, a, node)                     # node = ctrl AND a      (upper half)
    _walk(qc, node, rest, out, table, offset + half, anc[1:])
    qc.cx(ctrl, node)                         # node = ctrl AND NOT a  (lower half)
    _walk(qc, node, rest, out, table, offset, anc[1:])
    qc.cx(ctrl, node)                         # put node back to ctrl AND a ...
    qc.ccx(ctrl, a, node)                     # ... and clear it


def lookup_ui(qc, ctrl, addr, out, table, anc):
    """|addr>|out> -> |addr>|out ^ table[addr]> when ctrl is 1, else identity.
    anc: len(addr) clean ancillas, returned clean.  2(L-1) Toffolis."""
    assert len(anc) >= len(addr), "unary iteration needs one ancilla per level"
    _walk(qc, ctrl, list(addr), list(out), list(table), 0, list(anc))


# --- measurement-based uncomputation (Gidney Fig. 3) -----------------------
def _walk_cz(qc, ctrl, hi, unary_lo, F, hi_val, anc):
    """Unary-iterate the high address bits; at the leaf for high value v apply
    CZ(ctrl, unary_lo[j]) wherever F[j + v*2^l] is set."""
    if not hi:
        L_lo = len(unary_lo)
        for j in range(L_lo):
            if F[j + hi_val * L_lo]:
                qc.cz(ctrl, unary_lo[j])
        return
    a, rest = hi[-1], hi[:-1]
    half, node = 1 << len(rest), anc[0]
    qc.ccx(ctrl, a, node)
    _walk_cz(qc, node, rest, unary_lo, F, hi_val + half, anc[1:])
    qc.cx(ctrl, node)
    _walk_cz(qc, node, rest, unary_lo, F, hi_val, anc[1:])
    qc.cx(ctrl, node)
    qc.ccx(ctrl, a, node)


def phase_fixup(qc, addr, F, one, unary_lo, anc_hi, l):
    """Apply the phase (-1)^F[a] to every basis state |a> of addr.

    Splits addr into l low bits (unary-encoded, 2^l Fredkins) and the rest
    (unary-iterated, 2(2^h - 1) Toffolis).  With l = h = m/2 that is O(sqrt(L))
    instead of the O(L) a direct phase lookup would cost.

    one: one clean ancilla (used as an always-true control, returned clean)
    unary_lo: 2^l clean ancillas;  anc_hi: len(addr)-l clean ancillas
    """
    from windowed import _unary_gate
    lo, hi = list(addr[:l]), list(addr[l:])
    g = _unary_gate(l)
    qc.append(g, lo + list(unary_lo[:1 << l]))
    qc.x(one)                                        # always-true control
    _walk_cz(qc, one, hi, list(unary_lo[:1 << l]), F, 0, list(anc_hi))
    qc.x(one)
    qc.append(g.inverse(), lo + list(unary_lo[:1 << l]))


# --- temporary AND: 4 T to compute, 0 T to uncompute (Gidney 1709.06648) ---
def and_compute(qc, a, b, t):
    """t must start in |0>; leaves t = a AND b using 4 T gates, no phase.
    A full Toffoli costs 7 T -- this is cheaper only because t starts clean."""
    qc.h(t)
    qc.t(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.h(t)
    qc.sdg(t)


def and_uncompute(qc, a, b, t, cbit):
    """Erase t = a AND b for FREE: measure in the X basis, then repair the
    phase with a Clifford CZ.  Zero T gates.  Needs one classical bit, which
    may be reused by every later uncompute."""
    qc.h(t)
    qc.measure(t, cbit)
    with qc.if_test((cbit, 1)):
        qc.cz(a, b)
        qc.x(t)                                   # return the ancilla to |0>


def _walk_ta(qc, ctrl, addr, out, table, offset, anc, cbit):
    """_walk, with every AND made temporary."""
    if not addr:
        T = table[offset] if offset < len(table) else 0
        for b in range(len(out)):
            if (T >> b) & 1:
                qc.cx(ctrl, out[b])
        return
    a, rest = addr[-1], addr[:-1]
    half, node = 1 << len(rest), anc[0]
    and_compute(qc, ctrl, a, node)                # 4 T instead of 7
    _walk_ta(qc, node, rest, out, table, offset + half, anc[1:], cbit)
    qc.cx(ctrl, node)
    _walk_ta(qc, node, rest, out, table, offset, anc[1:], cbit)
    qc.cx(ctrl, node)
    and_uncompute(qc, ctrl, a, node, cbit)        # 0 T instead of 7


def lookup_ui_ta(qc, ctrl, addr, out, table, anc, cbit):
    """lookup_ui with temporary ANDs: same map, ~3.5x fewer T gates.
    cbit: one classical bit, reused by every uncompute."""
    _walk_ta(qc, ctrl, list(addr), list(out), list(table), 0, list(anc), cbit)
