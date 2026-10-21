def add(qc, X, reg, N=None):
    if N is not None:
        X %= N
    add_const(qc, X, list(reg))
