for k in range(K):                            # y_k in place, k < K
    _flip(ctx, xs[k], _bit(d, k), ctrl)
for k in range(K - 1, 0, -1):                 # Q^{-1}: g_{k+1} ^= y_k g_k
    ctx.ccx(xs[k], g[k - 1], g[k])
# P, k = 0: g_1 ^= d_0 ^ x_1 ^ y_0 (c_0 ^ d_0)
...
for k in range(1, K):                         # P: g_{k+1} ^= x_k ^ x_{k+1} ^ y_k g_k
    orig(k, g[k])
    orig(k + 1, g[k])
    ctx.ccx(xs[k], g[k - 1], g[k])
for k in range(1, K + 1):                     # w -> c: g_k ^= x_k
    orig(k, g[k - 1])
for k in range(K):
    _flip(ctx, xs[k], _bit(d, k), ctrl)
