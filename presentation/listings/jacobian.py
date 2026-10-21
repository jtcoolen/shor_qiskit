Z1sq = A(n, "Z1sq"); _, b1 = m.step(sqr, Z1, Z1sq)                    # 1
U2 = A(n, "U2");     _, b2 = m.step(MU.modmul_const, m, Z1sq, x2, U2, p)   # 2
Z1cu = A(n, "Z1cu"); _, b3 = m.step(mul, Z1sq, Z1, Z1cu)              # 3
H = A(n, "H");       _, b4 = m.step(sub_into, U2, X1, H)              # 4
...
T0 = A(n, "T0");     _, b12 = m.step(sub_into, Rsq, Hcu, T0)          # 12
_, b13 = m.step(sub_into, T0, twoV, X3)                        # 13  OUTPUT
T1 = A(n, "T1");     _, b14 = m.step(sub_into, V, X3, T1)             # 14
T2 = A(n, "T2");     _, b15 = m.step(mul, R, T1, T2)                  # 15
T3 = A(n, "T3");     _, b16 = m.step(mul, Y1, Hcu, T3)                # 16
_, b17 = m.step(sub_into, T2, T3, Y3)                          # 17  OUTPUT
_, b18 = m.step(mul, Z1, H, Z3)                                # 18  OUTPUT

# unwind everything but the three outputs.  Note T0 goes back via step 12,
# not step 13 -- step 13 wrote X3, which stays.
for body in (b16, b15, b14, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1):
    m.undo(body)
