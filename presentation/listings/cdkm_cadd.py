def _cuma(ctx, ctrl, c, b, a):
    ctx.ccx(c, b, a)                     # a <- x
    ctx.ccx(ctrl, c, b)                  # b ^= ctrl (c_i ^ x)
    ctx.cx(a, c)                         # c <- c_i
    ctx.cx(a, b)                         # b = y ^ ctrl (x ^ c_i)

def cdkm_cadd(ctx, ctrl, x, y, carry):
    n = len(y)
    assert len(x) == n and n >= 1
    _maj(ctx, carry, y[0], x[0])
    for i in range(n - 1):
        _maj(ctx, x[i], y[i + 1], x[i + 1])
    for i in range(n - 1, 0, -1):
        _cuma(ctx, ctrl, x[i - 1], y[i], x[i])
    _cuma(ctx, ctrl, carry, y[0], x[0])
