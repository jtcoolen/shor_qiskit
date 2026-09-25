def c_mult_acc(qc, c, A, x, y, sign, flag, N):
    for i in range(len(x)):
        c_add_mod(qc, [c, x[i]], (A << i) % N, y, sign, flag, N)

def c_ua(qc, c, A, x, y, sign, flag, N):
    c_mult_acc(qc, c, A % N, x, y, sign, flag, N)
    for i in range(len(x)):
        qc.cswap(c, x[i], y[i])
    B = pow(A, -1, N)                        # classical inverse, gcd(A,N)=1
    c_mult_acc(qc, c, (-B) % N, x, y, sign, flag, N)
