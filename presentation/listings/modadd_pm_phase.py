ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
sc = m.anc(n, "sc")
A.add(ctx, x + ax, y + ay, sc)                   # ay = carry-out
m.free(sc)
cp, sc2 = m.anc(kappa, "cp"), m.anc(kappa, "sc2")
A.cadd_const(ctx, ay[0], Reg(list(y[:kappa])), f, cp, sc2)   # 2^u -> f
m.free(cp, sc2)
...
def phase(qc, qs):
    ...
    qc.h(out)                                    # |->: the CNOT kicks back
    A.lt_uint(Ctx(qc, "and"), top_bits(yy, delta), top_bits(xx, delta), out,
              anc[:delta])
    ...
MB.mbu_flag(m, ay[0], list(x) + list(y), recompute, phase, nanc=delta + 1)
