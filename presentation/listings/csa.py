def csa_compress(m, A, B, C):
    ctx, W = m.ctx, len(A)
    assert len(B) == len(C) == W
    S, M = m.anc(W, "csaS"), m.anc(W, "csaM")
    for i in range(W):                       # S = A ^ B ^ C
        ctx.cx(A[i], S[i])
        ctx.cx(B[i], S[i])
        ctx.cx(C[i], S[i])
    for i in range(W - 1):                   # M = MAJ, written one place up
        ctx.cx(A[i], B[i])
        ctx.cx(A[i], C[i])
        ctx.and_(B[i], C[i], M[i + 1])
        ctx.cx(A[i], M[i + 1])
        ctx.cx(A[i], B[i])
        ctx.cx(A[i], C[i])
    return S, M
