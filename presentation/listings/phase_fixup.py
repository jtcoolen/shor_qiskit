def phase_fixup(qc, addr, F, one, unary_lo, anc_hi, l):
    from windowed import _unary_gate
    lo, hi = list(addr[:l]), list(addr[l:])
    g = _unary_gate(l)
    qc.append(g, lo + list(unary_lo[:1 << l]))
    qc.x(one)                                        # always-true control
    _walk_cz(qc, one, hi, list(unary_lo[:1 << l]), F, 0, list(anc_hi))
    qc.x(one)
    qc.append(g.inverse(), lo + list(unary_lo[:1 << l]))
