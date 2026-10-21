for i in range(0, len(e), we):
    ei = list(e[i:i + we])
    ne = len(ei)
    kes = [pow(k, (2**i) * v, N) for v in range(2**ne)]
    ...
    for j in range(0, n, wm):
        mi = A[j:j + wm]
        addr = ei + list(mi)
        table = [((kes[ad & ((1 << ne) - 1)] * (ad >> ne)) << j) % N
                 for ad in range(1 << len(addr))]
        lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])
        qc.append(madd, tmp[:n] + B + [sign, flag])
        lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])
    ...                                       # A -= B * k_e^-1, the same way
    A, B = B, A                               # relabelling swap -- free
