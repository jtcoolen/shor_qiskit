def moddbl_pm(m, x, q, lsbs=None):
    ...
    u, f = pm
    ctx, n = m.ctx, len(x)
    ...
    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])                      # n+1 bits; xe[n] is the overflow
    cp, sc = m.anc(lsbs, "cp"), m.anc(lsbs, "sc")
    A.cadd_const(ctx, xe[n], Reg(list(xe[:lsbs])), f, cp, sc)
    m.free(cp, sc)
    ctx.cx(xe[0], xe[n])                          # clears the bit == subtracts 2^u
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
