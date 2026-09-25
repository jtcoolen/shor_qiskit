def ecdlp_circuit(curve, P, Q, m, S):
    n = curve.p.bit_length()
    u, v = QuantumRegister(m, "u"), QuantumRegister(m, "v")
    pt = QuantumRegister(2 * n, "point")               # x, then y
    qc = QuantumCircuit(u, v, pt, ClassicalRegister(m, "j1"), ClassicalRegister(m, "j2"))
    for i in range(2 * n):                             # accumulator <- S
        if ((S.x + (S.y << n)) >> i) & 1:
            qc.x(pt[i])
    qc.h(u)
    qc.h(v)
    for i in range(m):                                 # controlled +[2^i]P and +[2^i]Q
        permutation(qc, pt, point_perm(curve, curve.mul(2**i, P), n), ctrls=[u[i]])
        permutation(qc, pt, point_perm(curve, curve.mul(2**i, Q), n), ctrls=[v[i]])
    qc.append(qft(m).inverse(), u)
    qc.append(qft(m).inverse(), v)
    qc.measure(u, qc.cregs[0])
    qc.measure(v, qc.cregs[1])
    return qc
