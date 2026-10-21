def bezout_replay(m, r, s, recs, q, dbl=None, cadd=None):
    dbl = dbl or (lambda mm, reg: MA.moddbl(mm, reg, q))
    cadd = cadd or (lambda mm, c, a, b: MA.cmodadd(mm, c, a, b, q))
    for b0, b0b1 in reversed(recs):
        dbl(m, s)                                        # s <- 2s mod q
        cadd(m, b0[0], r, s)                             # if b0: s += r mod q
        for i in range(len(r)):
            m.ctx.cswap(b0b1[0], r[i], s[i])             # if b0b1: swap r, s
