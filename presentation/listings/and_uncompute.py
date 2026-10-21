def and_uncompute(qc, a, b, t, cbit):
    qc.h(t)
    qc.measure(t, cbit)
    with qc.if_test((cbit, 1)):
        qc.cz(a, b)
        qc.x(t)                                   # return the ancilla to |0>
