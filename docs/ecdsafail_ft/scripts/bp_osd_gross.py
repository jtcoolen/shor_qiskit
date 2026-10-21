"""Minimal BP+OSD decoder (min-sum belief propagation, then order-0 ordered-statistics
post-processing) for the [[144,12,12]] gross code under code-capacity bit-flip noise.
Z-type checks H_Z=[B^T|A^T] detect X errors; logical failure = residual error not in
the stabiliser group (i.e. anticommutes with some logical Z).  Output: data/bb_bposd.dat
"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from bb_codes import xy, mpow, rank2

def gf2_solve_rows(H, s, order):
    """OSD-0: Gaussian elimination on columns in the given reliability order; return e with H e = s."""
    m, n = H.shape
    M = np.concatenate([H[:, order] % 2, s.reshape(-1, 1)], axis=1)
    pivcols = []
    r = 0
    for c in range(n):
        piv = [i for i in range(r, m) if M[i, c]]
        if not piv: continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        for i in range(m):
            if i != r and M[i, c]: M[i] ^= M[r]
        pivcols.append(c); r += 1
        if r == m: break
    e = np.zeros(n, dtype=int)
    for i, c in enumerate(pivcols):
        e[c] = M[i, n]
    out = np.zeros(n, dtype=int); out[order] = e
    return out

def bp_minsum(H, s, p, iters=60, alpha=0.8):
    m, n = H.shape
    llr0 = np.log((1 - p) / p)
    rows = [np.flatnonzero(H[i]) for i in range(m)]
    cols = [np.flatnonzero(H[:, j]) for j in range(n)]
    msg_v2c = {(i, j): llr0 for i in range(m) for j in rows[i]}
    msg_c2v = {(i, j): 0.0 for i in range(m) for j in rows[i]}
    for it in range(iters):
        for i in range(m):
            js = rows[i]
            vals = np.array([msg_v2c[(i, j)] for j in js])
            sg = np.prod(np.sign(vals)) * (-1 if s[i] else 1)
            ab = np.abs(vals)
            for idx, j in enumerate(js):
                others = np.delete(ab, idx)
                msg_c2v[(i, j)] = alpha * sg * np.sign(vals[idx]) * others.min()
        post = np.full(n, llr0)
        for j in range(n):
            post[j] += sum(msg_c2v[(i, j)] for i in cols[j])
        for i in range(m):
            for j in rows[i]:
                msg_v2c[(i, j)] = post[j] - msg_c2v[(i, j)]
        e = (post < 0).astype(int)
        if np.array_equal((H @ e) % 2, s):
            return e, post, True
    return e, post, False

def main(out):
    l, m = 12, 6
    x, y = xy(l, m)
    A = (mpow(x,3) + mpow(y,1) + mpow(y,2)) % 2
    B = (mpow(y,3) + mpow(x,1) + mpow(x,2)) % 2
    HZ = np.concatenate([B.T, A.T], axis=1) % 2       # detects X errors
    HX = np.concatenate([A, B], axis=1) % 2
    n = 2 * l * m
    # logical Z operators = kernel(HX) mod rowspace(HZ); test failure via commutation with a basis of ker(HX)
    # compute ker(HX) basis
    def nullspace(Mx):
        Mx = Mx.copy() % 2; rows, cols = Mx.shape; piv = []; r = 0
        for c in range(cols):
            pv = [i for i in range(r, rows) if Mx[i, c]]
            if not pv: continue
            Mx[[r, pv[0]]] = Mx[[pv[0], r]]
            for i in range(rows):
                if i != r and Mx[i, c]: Mx[i] ^= Mx[r]
            piv.append(c); r += 1
            if r == rows: break
        free = [c for c in range(cols) if c not in piv]
        basis = []
        for f in free:
            v = np.zeros(cols, dtype=int); v[f] = 1
            for i, c in enumerate(piv):
                if Mx[i, f]: v[c] = 1
            basis.append(v)
        return np.array(basis)
    K = nullspace(HX)                                  # all Z-type operators commuting with X checks (n - rank HX of them)
    rng = np.random.default_rng(7)
    rows_out = []
    for p in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        trials = 2000 if p <= 0.04 else 1000
        fails_bp = fails_osd = 0
        for t in range(trials):
            e = (rng.random(n) < p).astype(int)
            s = (HZ @ e) % 2
            ehat, post, conv = bp_minsum(HZ, s, p)
            if not conv:
                order = np.argsort(-np.abs(post) * np.where(post < 0, 1.0, 1.0))  # most reliable first
                # OSD: columns sorted by decreasing probability of being in error (post small/negative first)
                order = np.argsort(post)
                ehat = gf2_solve_rows(HZ, s, order)
            res = (e + ehat) % 2
            # the residual X error is harmless iff it lies in the X-stabiliser group = rowspace(HX);
            # (it is syndrome-free by construction) -> logical failure iff rank([HX; res]) > rank(HX)
            if rank2(np.vstack([HX, res])) > rank2(HX):
                fails_osd += 1
            if not conv:
                fails_bp += 1
        rows_out.append((p, trials, fails_bp, fails_osd))
        print(f"p={p} trials={trials} BP-nonconverged={fails_bp} logical failures (BP+OSD0)={fails_osd} -> {fails_osd/trials:.4f}", flush=True)
    with open(out, "w") as f:
        f.write("p trials bp_fail osd_fail pL\n")
        for r in rows_out: f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[3]/r[1]:.5f}\n")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bb_bposd.dat")
