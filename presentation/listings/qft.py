def qft(n, approx=0):
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
