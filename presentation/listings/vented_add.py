for k in range(n):
    ...
        nxt = m.anc(1, "cq")[0]               # c_{k+1} = maj(x_k, d_k, c_k)
        ...
            ctx.cx(xs[k], cur)                # cur = x ^ c
            _flip(ctx, xs[k], _bit(d, k), ctrl)
            ctx.and_(xs[k], cur, nxt)         # (x ^ d)(x ^ c)
            _flip(ctx, xs[k], _bit(d, k), ctrl)
            ctx.cx(xs[k], nxt)                # maj = x ^ (x ^ d)(x ^ c)
            ctx.cx(xs[k], cur)                # cur = c again
    ...
        ctx.cx(cur, xs[k])
    _flip(ctx, xs[k], _bit(d, k), ctrl)
    if cur is not None and cur is not cin:    # vent c_k (k >= 1)
        ...
        m.free(Reg([cur]))
    cur = nxt
