def inc_borrowed(m, w, g):
    ctx, k = m.ctx, len(w)
    ...
    cy = m.anc(1, "icy")
    A.sub(ctx, Reg(g), Reg(list(w)), cy)
    for q in g:
        ctx.x(q)
    A.sub(ctx, Reg(g), Reg(list(w)), cy)
    for q in g:
        ctx.x(q)
    m.free(cy)
