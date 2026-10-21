def div3(m, x, y, p, arith, steps=None, lean=True):
    ctx, n = m.ctx, len(x)
    bk = L.Banks(m, n, x=x, lean=lean)
    L.eea_forward(m, bk, p, steps=steps)                     # W1 = 0, W2 = x^-1
    xinv, a = L._le(bk, bk.W2, 1, n), L._le(bk, bk.W1, 1, n)
    idle = bk.lt + bk.lq + bk.lr + [bk.P1, bk.P2, bk.Sign] + bk.W2[n:] + bk.W1[n:]
    with L._lent(m, idle):
        mul_acc(m, arith, xinv, y, a)                        # A = y / x
    key = ("luo3", next(_keys))
    MB.vent(m, [xinv, a], y, _quotient(p), key)              # y = A / x^-1
    bk2 = L.Banks([list(y) + bk.W1[n:]] + list(bk[1:]))       # Y is the bank now
    m.emit_inverse(L.eea_forward, m, bk2, p, None, steps, True, True)
    with L._lent(m, bk.meta() + bk.consumed + bk.W1[n:]):
        mul_acc(m, arith, x, a, y)                           # y again ...
        MB.zfix(m, y, key)                                   # ... cancels the phase
        mul_unacc(m, arith, x, a, y)
    for q1, q2 in zip(y, a):
        ctx.swap(q1, q2)
    bk.free(m)
