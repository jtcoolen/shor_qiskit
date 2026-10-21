def vent(m, data, reg, f, key):
    g = VentGate([len(d) for d in data], len(reg), f, key)
    m.qc.append(g, [q for d in data for q in d] + list(reg))


def zfix(m, reg, key):
    m.qc.append(ZFixGate(len(reg), key), list(reg))
