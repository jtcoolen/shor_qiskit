def _round(self, m, U, VT, rec, w):
    ctx = m.ctx
    a, mm = rec
    u, vt = Reg(U[:w + 1]), Reg(VT[:w + 1])
    ctx.cx(vt[1], a[0])
    ctx.cx(u[1], a[0])                               # a = v~[1] ^ u[1]
    ...
    A.ci_add(ctx, a[0], u, vt, anc)                  # v~ += (-1)^a u
    ...
    for i in range(len(vt) - 1):                     # v~ /= 2 (even)
        ctx.swap(vt[i], vt[i + 1])
    ...
    A.gt_top(ctx, u, vt, mm[0], sc, k, ctrl=a[0])    # m = a AND [u > v~]
    ...
    for i in range(len(u)):
        ctx.cswap(mm[0], u[i], vt[i])
