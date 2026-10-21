def add_quantum_mod(qc, t, y, sign, flag, N):
    yw = list(y) + [sign]
    add_quantum(qc, t, yw, +1)            # (1) + t
    add_const(qc, -N, yw)                 # (2) - N
    qc.cx(sign, flag)                     # (3) sign -> flag
    c_add_const(qc, [flag], N, yw)        # (4) + N  if flag
    add_quantum(qc, t, yw, -1)            # (5) - t
    qc.cx(sign, flag)                     # (6) [R < t] == not flag, so this
    qc.x(flag)                            #     pair returns flag to |0>
    add_quantum(qc, t, yw, +1)            # (7) + t
