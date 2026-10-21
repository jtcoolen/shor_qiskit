"""Square-subtract X <- X - lambda^2 mod p [ECDSA.Fail Sec 5.3.4].

The point addition needs |lambda>|X> -> |lambda>|X - lambda^2>.  The reference
(`ec_window._csub_square`) multiplies lambda by a copy of itself with the
generic double-and-add multiplier, subtracts, and multiplies again to clear.
ECDSA.Fail replaces this with three ideas, all implemented here:

Symmetric squaring.  z^2 = sum z_i 2^(2i) + sum_{i<j} z_i z_j 2^(i+j+1): the
diagonal is free (z_i^2 = z_i) and each off-diagonal product is formed once,
so a k-bit square has k(k-1)/2 AND gates against k^2 for a generic product.
Row i adds [z_i, 0, z_i z_(i+1), ..., z_i z_(k-1)] into W[2i:], exactly.

Karatsuba.  With lambda = L + beta H, beta = 2^k,
    lambda^2 = A + beta (C - A - B) + beta^2 B,  A = L^2, B = H^2, C = (L+H)^2
and beta^2 = F mod p is a classical constant (for secp256k1 F = 2^32 + 977;
here it is whatever 2^(2k) mod p happens to be).  The three pieces go
straight into X, never into a reduced lambda^2 register:
    X <- X - beta C;   X <- X - A + beta A;   X <- X + beta B - F B,
each as build (square into clean workspace), apply, unbuild, so A, B and C
are never live together and L + H exists only while C is.

Apply.  A wide integer W goes into X as sum_j W_j (c 2^j mod p), with c the
classical coefficient (beta, 1 - beta, F - beta).  The constants are summed
without reduction into an (n + e)-bit fold register, reduced once by a binary
ladder, subtracted from X, and unbuilt (`_apply_wide`); the per-bit modular
form (`_apply_bitwise`) is kept for comparison.  This step is where the
paper's pseudo-Mersenne shifts live at n = 256: its cost is linear in the
number of product bits, so Karatsuba's 3 x 2(n/2) = 3n bits against 2n for
one square is a real price that the saved partial products must repay.

Control.  The windowed addition only subtracts when its addend is not the
point at infinity.  ANDing the control into the copies of L and H makes every
square zero when it is off, so nothing downstream needs a control.
"""

import ec_adders as A
import ec_modarith as MA


def sym_square(m, z, W):
    """W += z^2 (integer, no reduction).  W must have >= 2 len(z) qubits."""
    ctx, k = m.ctx, len(z)
    assert len(W) >= 2 * k
    for i in range(k):
        rest = len(W) - 2 * i
        if rest <= 0:
            break
        prods = m.anc(max(0, k - i - 1), "pp")
        for j in range(i + 1, k):
            ctx.and_(z[i], z[j], prods[j - i - 1])
        pad = m.anc(rest - 1 - len(prods), "pad") if rest - 1 > len(prods) \
            else None
        addend = [z[i]]
        if rest > 1:
            addend += [pad[0]] + list(prods)
            addend += list(pad[1:])
        addend = addend[:rest]
        anc = m.anc(max(1, rest - 1), "sqc")
        A.add(ctx, addend, W[2 * i:], anc)
        m.free(anc)
        if pad is not None:
            m.free(pad)
        for j in reversed(range(i + 1, k)):
            ctx.and_dg(z[i], z[j], prods[j - i - 1])
        m.free(prods)


def _apply_bitwise(m, W, acc, c, p):
    """acc <- acc - c W mod p, one controlled modular constant subtraction
    per bit of W (about 5n Toffolis each)."""
    for j, w in enumerate(W):
        k = c * pow(2, j, p) % p
        if k:
            MA.cmodsub_const(m, w, acc, k, p)


def _fold(m, W, T, flags, c, p):
    """T (clean) <- sum_j W_j (c 2^j mod p) reduced mod p; `flags` keep the
    reduction decisions.  Non-modular constant additions, then one binary
    reduction ladder T -= p 2^i when T >= p 2^i."""
    ctx, w = m.ctx, len(T)
    for j, q in enumerate(W):
        k = c * pow(2, j, p) % p
        if k:
            cp, sc = m.anc(w, "cp"), m.anc(w, "sc")
            A.cadd_const(ctx, q, T, k, cp, sc)
            m.free(cp, sc)
    for i in reversed(range(len(flags))):
        cp, sc = m.anc(w, "cp"), m.anc(w, "sc")
        A.geq_const(ctx, T, p << i, flags[i], cp, sc)
        A.csub_const(ctx, flags[i], T, p << i, cp, sc)
        m.free(cp, sc)


def _apply_wide(m, W, acc, c, p):
    """acc <- acc - c W mod p via one wide fold: n + e bits, e = log2 |W|.

    About 2 |W| (n + e) Toffolis for the fold and its unbuild plus one modular
    subtraction, against 5n per bit for `_apply_bitwise`.
    """
    n = len(acc)
    e = max(1, len(W).bit_length())
    T, flags = m.anc(n + e, "fold"), m.anc(e, "rf")
    _fold(m, W, T, flags, c, p)
    MA.modsub(m, T[:n], acc, p)
    m.emit_inverse(_fold, m, W, T, flags, c, p)
    m.free(T, flags)


_apply = _apply_wide


def _term(m, z, acc, c, p):
    """Build z^2, apply -c z^2 to acc, unbuild."""
    W = m.anc(2 * len(z), "sq")
    sym_square(m, z, W)
    _apply(m, W, acc, c, p)
    m.emit_inverse(sym_square, m, z, W)
    m.free(W)


def _gated(m, ctrl, lam):
    g = m.anc(len(lam), "lg")
    for a, b in zip(lam, g):
        m.ctx.and_(ctrl, a, b)
    return g


def _ungate(m, ctrl, lam, g):
    for a, b in zip(lam, g):
        m.ctx.and_dg(ctrl, a, b)
    m.free(g)


def csub_square_symmetric(m, ctrl, lam, acc, p):
    """acc <- acc - lam^2 mod p if ctrl: one symmetric square, no split."""
    g = _gated(m, ctrl, lam)
    _term(m, g, acc, 1, p)
    _ungate(m, ctrl, lam, g)


def csub_square_karatsuba(m, ctrl, lam, acc, p, split=None):
    """acc <- acc - lam^2 mod p if ctrl: one-level Karatsuba, build-apply-unbuild."""
    ctx, n = m.ctx, len(lam)
    k = split or (n + 1) // 2
    assert 1 <= k < n
    beta = pow(2, k, p)
    F = beta * beta % p
    g = _gated(m, ctrl, lam)
    L, H = g[:k], g[k:]

    S = m.anc(k + 1, "LpH")                       # C = (L + H)^2: X -= beta C
    for a, b in zip(L, S):
        ctx.cx(a, b)
    hp = m.anc(k + 1 - len(H), "hp")
    Hp = list(H) + list(hp)
    anc = m.anc(k, "sc")
    A.add(ctx, Hp, S, anc)
    _term(m, S, acc, beta, p)
    A.sub(ctx, Hp, S, anc)
    m.free(anc)
    m.free(hp)
    for a, b in zip(L, S):
        ctx.cx(a, b)
    m.free(S)

    _term(m, L, acc, (1 - beta) % p, p)           # X -= A - beta A
    _term(m, H, acc, (F - beta) % p, p)           # X -= F B - beta B
    _ungate(m, ctrl, lam, g)
