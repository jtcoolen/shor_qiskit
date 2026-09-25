def c_add_mod(qc, ctrls, X, y, sign, flag, N):
    yw = list(y) + [sign]                     # (n+1)-wide value register
    c_add_const(qc, ctrls, X, yw)             # (1) + X          (controlled)
    add_const(qc, -N, yw)                     # (2) - N
    qc.cx(sign, flag)                         # (3) sign -> flag
    c_add_const(qc, [flag], N, yw)            # (4) + N  if flag
    add_const(qc, -X, yw)                     # (5) - X
    qc.mcx(list(ctrls) + [sign], flag)        # (6) clear flag from the answer
    qc.x(flag)
    add_const(qc, X, yw)                      # (7) + X
