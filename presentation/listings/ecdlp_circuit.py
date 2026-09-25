qc = QuantumCircuit(kr, lr, px, py, ck, cl)
...                                     # accumulator <- S  (X gates)
qc.h(kr)
qc.h(lr)
pack = list(px) + list(py)
Ps, Qs = _rungs(curve, P, Q, mb)
for i, R in enumerate(Ps):
    permutation(qc, pack, point_perm(curve, R, n), ctrls=[kr[i]])
for i, R in enumerate(Qs):
    permutation(qc, pack, point_perm(curve, R, n), ctrls=[lr[i]])
qc.append(qft(mb).inverse(), list(kr))
qc.append(qft(mb).inverse(), list(lr))
qc.measure(kr, ck)
qc.measure(lr, cl)
