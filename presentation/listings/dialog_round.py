def dialog_round(m, u, v, rec):
    ctx, n = m.ctx, len(u)
    b0, b0b1 = rec
    ctx.cx(v[0], b0[0])                              # b0 = v mod 2

    t, sc = m.anc(1, "t"), m.anc(n, "sc")
    A.gt_uint(ctx, u, v, t[0], sc)                   # t = [u > v]
    ctx.and_(b0[0], t[0], b0b1[0])                   # b0b1 = b0 AND b1
    A.gt_uint(ctx, u, v, t[0], sc)
    m.free(t, sc)

    for i in range(n):
        ctx.cswap(b0b1[0], u[i], v[i])               # if b0b1: swap u, v

    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b0[0], u, v, cp, sc)                 # if b0: v -= u
    m.free(cp, sc)

    for i in range(n - 1):                           # v /= 2 (v is even now)
        ctx.swap(v[i], v[i + 1])
