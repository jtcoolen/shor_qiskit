for i in range(n):
    ctx.cx(z[n + i], z[n + i + 1])              # sign-extend by one bit
    h = list(x[i + 1:])
    for q in h + [pad[0]]:
        ctx.cx(x[i], q)                         # x_i ? NOT(0||h) : (0||h)
    operand = Reg(h + [pad[0], x[i]], "op")     # signed, sign = x_i
    window = Reg(z[2 * i + 1:n + i + 2], "win")
    ...
    A.sub(ctx, operand, window, anc)            # z -= signed operand
    ...
    for q in h + [pad[0]]:
        ctx.cx(x[i], q)
