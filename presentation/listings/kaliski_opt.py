cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
A.csub(ctx, b[0], u, v, cp, sc)               # v -= u
m.free(cp, sc)
cp, sc = m.anc(W, "cp"), m.anc(W, "sc")
A.cadd(ctx, b[0], r, s, cp, sc)               # s += r   (plain, not mod p)
m.free(cp, sc)
_rot_down(ctx, v)                             # v /= 2
_rot_up(ctx, r)                               # r *= 2   (free: no reduction)
