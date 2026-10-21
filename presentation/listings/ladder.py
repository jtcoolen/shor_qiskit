for i in range(t):                       # the ladder: t rungs
    c_ua(qc, ctr[i], pow(A, 2**i, N), tgt, anc, sf[0], sf[1], N)
