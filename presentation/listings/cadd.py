def cadd(ctx, ctrl, x, y, copy, anc):
    assert len(copy) == len(x)
    for i, q in enumerate(x):
        ctx.and_(ctrl, q, copy[i])
    add(ctx, copy, y, anc)
    for i, q in enumerate(x):
        ctx.and_dg(ctrl, q, copy[i])
