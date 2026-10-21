def ci_add(ctx, e, x, y, anc):
    for q in y:
        ctx.cx(e, q)
    add(ctx, x, y, anc)
    for q in y:
        ctx.cx(e, q)
