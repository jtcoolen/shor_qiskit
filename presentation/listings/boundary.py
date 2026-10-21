def wrapper(*args, **kwargs):
    ...
    (ea, ek), qubits = sh
    key = (label, ea, ek)
    node = _CACHE.get(key)
    if node is None:
        node = _build(fn, label, key, args, kwargs, qubits, m)
        ...
        _CACHE[key] = node
    anc = m.anc(node.n_anc, "h") if node.n_anc else []
    m.qc.append(NodeGate(node), list(qubits) + list(anc))
