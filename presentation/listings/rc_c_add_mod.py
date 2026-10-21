def rc_add_const(qc, X, y, anc, ctrls=()):
    n = len(y)
    carry, xreg = anc[0], anc[1:n + 1]
    Xm = X % (2**n)                      # negative X just works, as in Fourier
    for i in range(n):                   # encode the constant -- uncontrolled
        if (Xm >> i) & 1:
            qc.x(xreg[i])
    rc_add(qc, xreg, y, carry, ctrls)
    for i in range(n):                   # unencode
        if (Xm >> i) & 1:
            qc.x(xreg[i])

def rc_c_add_mod(qc, ctrls, X, y, sign, flag, N, anc):
    yw = list(y) + [sign]
    rc_add_const(qc, X, yw, anc, ctrls)  # (1) + X          (controlled)
    rc_add_const(qc, -N, yw, anc)        # (2) - N
    qc.cx(sign, flag)                    # (3) sign -> flag
    rc_add_const(qc, N, yw, anc, [flag])  # (4) + N  if flag
    rc_add_const(qc, -X, yw, anc)        # (5) - X
    qc.mcx(list(ctrls) + [sign], flag)    # (6) clear flag from the answer
    qc.x(flag)
    rc_add_const(qc, X, yw, anc)         # (7) + X
