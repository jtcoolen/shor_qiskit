for i in range(t):
    c_ua(qc, ctr[i], pow(A, 2**i, N), list(tgt), list(anc), sf[0], sf[1], N)
if garbage == "x":
    for i in range(t):
        qc.cx(ctr[i], g[i])               # a copy of x left behind
qc.append(qft(t).inverse(), ctr)
