"""Gidney's classical-quantum adder with constant workspace [Gid25b].

[Gid25b] C. Gidney, "A classical-quantum adder with constant workspace and
         linear gates", arXiv:2507.23079.
[HRS17]  T. Haener, M. Roetteler, K. Svore, "Factoring using 2n+2 qubits with
         Toffoli based modular multiplication", QIC 17 (2017).
[Luo26]  H. Luo et al., arXiv:2607.13816, App. B.

x += d for a classical d, with O(1) clean qubits and O(n) Toffolis.  The
paper gives the construction in prose and its circuits as figures; this
module rebuilds it from the prose, and the tests check every piece on every
input and, with the measurements performed, every phase.

Carries.  With c_0 the carry in, c_{k+1} = maj(x_k, d_k, c_k), and
carry(x, d, c_0) = x XOR d XOR (x + d + c_0).  [Gid25b] eq. (8): the carries
of adding d into the *complement of the sum* are the same carries,

    carry(~x', d, c_0) = carry(x, d, c_0),     x' = x + d + c_0,

so after the addition every carry is a function of the result: it can be
*vented* (X-measured, `ec_mbu.vent`), leaving a phase flip by that carry to
be done later.

`vented_add` is the streaming adder ([Gid25b] Fig. 2): the carry into bit k
lives in one of two clean qubits while bit k is summed, then is vented.
n - 1 ANDs, via maj(x, d, c) = x XOR (x XOR d)(x XOR c).

`carry_xor` is g <- g XOR (the carries c_1..c_K of x + d + c_0) for a
*dirty* g ([Gid25b] Fig. 3, after [HRS17]).  The construction here is our
own derivation, not a transcription.  Write w_k = c_k XOR x_k; then

    w_{k+1} = (x_k XOR x_{k+1}) XOR y_k w_k,     y_k = x_k XOR d_k,

a recurrence whose only product is y_k w_k.  Let P be the ascending pass
g_{k+1} ^= (x_k XOR x_{k+1}) XOR y_k g_k, and Q the same pass without the
constant terms.  On any g, P(g) = w XOR Q(g), and Q is linear and
invertible, so P(Q^{-1}(g)) = g XOR w: run Q descending (its inverse), then
P, then XOR x_k into g_k to turn w into c.  About 2K Toffolis, no ancilla,
and d enters only through X or CNOT gates -- so a control costs nothing
extra ([Gid25b] Sec 2.4: d_k = 1 becomes the control qubit).

The phase fix ([Gid25b] Sec 2.3): Z_q CX_{c->q} Z_q CX_{c->q} = Z_c.  With
the vented carries' outcomes m on a dirty g, Z^m(g), g ^= c, Z^m(g),
g ^= c leaves g as it was and applies (-1)^(m.c): exactly the vents' phase.

  cq_add_dirty   n - 1 dirty qubits, 2 clean, ~3n Toffolis: the streaming
                 adder also XORs its carries into g as it goes, so one
                 carry_xor finishes the fix ([Gid25b] Fig. 4).
  cq_add         3 clean, ~4n Toffolis: the two halves borrow each other as
                 the dirty register ([Gid25b] Sec 2.4, Fig. 5).

`GidneyArith` is [Luo26] App. B's modular arithmetic on these adders: the
`ec_gcd` interface (cadd, dbl, add, sub) for a classical modulus p, exact
for every odd p, on a constant number of clean qubits.
"""

import ec_adders as A
import ec_gcd as G
import ec_mbu as MB
import ec_space as SP
from ec_sim import Reg


def _bit(v, k):
    return (v >> k) & 1


def _flip(ctx, q, dk, ctrl):
    """q ^= d_k (times ctrl): X, CNOT from the control, or nothing."""
    if dk:
        if ctrl is None:
            ctx.x(q)
        else:
            ctx.cx(ctrl, q)


def carry(xv, d, cin, k):
    """The carry into bit k of x + d + cin (classical)."""
    m = (1 << k) - 1
    return ((xv & m) + (d & m) + cin) >> k & 1


# =============================================================================
# carry-xor: g <- g XOR carries, g dirty
# =============================================================================
def carry_xor(ctx, x, d, g, ctrl=None, cin=None):
    """g[k-1] ^= c_k, the carry into bit k of x + ctrl d + cin, for k = 1..len(g).

    x, ctrl, cin are restored; g may hold anything.  Bits of x at index
    >= len(x) count as 0 (so len(g) = len(x) also gives the carry out).
    `cin` is None (0) or a qubit."""
    nb, K = len(x), len(g)
    assert 1 <= K <= nb
    xs = list(x)

    def orig(k, out):                             # out ^= x_k (original)
        if k >= nb:
            return
        ctx.cx(xs[k], out)
        if k < K:                                 # x_k holds y_k = x_k ^ d_k now
            _flip(ctx, out, _bit(d, k), ctrl)

    for k in range(K):                            # y_k in place, k < K
        _flip(ctx, xs[k], _bit(d, k), ctrl)
    for k in range(K - 1, 0, -1):                 # Q^{-1}: g_{k+1} ^= y_k g_k
        ctx.ccx(xs[k], g[k - 1], g[k])
    # P, k = 0: g_1 ^= d_0 ^ x_1 ^ y_0 (c_0 ^ d_0)
    _flip(ctx, g[0], _bit(d, 0), ctrl)
    orig(1, g[0])
    if cin is None:
        if _bit(d, 0):
            if ctrl is None:
                ctx.cx(xs[0], g[0])
            else:
                ctx.ccx(ctrl, xs[0], g[0])
    else:
        _flip(ctx, cin, _bit(d, 0), ctrl)         # cin ^ d_0 in place
        ctx.ccx(cin, xs[0], g[0])
        _flip(ctx, cin, _bit(d, 0), ctrl)
    for k in range(1, K):                         # P: g_{k+1} ^= x_k ^ x_{k+1} ^ y_k g_k
        orig(k, g[k])
        orig(k + 1, g[k])
        ctx.ccx(xs[k], g[k - 1], g[k])
    for k in range(1, K + 1):                     # w -> c: g_k ^= x_k
        orig(k, g[k - 1])
    for k in range(K):
        _flip(ctx, xs[k], _bit(d, k), ctrl)


# =============================================================================
# the streaming adder that vents its carries
# =============================================================================
def vented_add(m, x, d, ctrl=None, cin=None, g=None, keep_carry=False):
    """x <- x + ctrl d + cin mod 2^n; the carries c_1..c_{n-1} are vented.

    Returns (keys, carry_out): the vent key of each c_k (k = 1..n-1), and
    with `keep_carry` the carry out c_n in a clean qubit the caller owns
    (not vented).  With `g`, each c_k is also XORed into g[k-1] as it is
    made ([Gid25b] Fig. 4's merged carry-xor).  Two clean qubits, n - 1
    ANDs (one more if the carry in or the control meets bit 0)."""
    ctx, n = m.ctx, len(x)
    xs = list(x)
    keys = []
    cur = cin                                     # the qubit holding c_k (None: 0)
    data_extra = ([ctrl] if ctrl is not None else []) + ([cin] if cin is not None else [])
    out = None
    for k in range(n):
        nxt = None
        if k < n - 1 or keep_carry:
            nxt = m.anc(1, "cq")[0]               # c_{k+1} = maj(x_k, d_k, c_k)
            if cur is None:                       # c_k = 0: c_{k+1} = x_k d_k
                if _bit(d, k):
                    if ctrl is None:
                        ctx.cx(xs[k], nxt)
                    else:
                        ctx.and_(ctrl, xs[k], nxt)
            else:
                ctx.cx(xs[k], cur)                # cur = x ^ c
                _flip(ctx, xs[k], _bit(d, k), ctrl)
                ctx.and_(xs[k], cur, nxt)         # (x ^ d)(x ^ c)
                _flip(ctx, xs[k], _bit(d, k), ctrl)
                ctx.cx(xs[k], nxt)                # maj = x ^ (x ^ d)(x ^ c)
                ctx.cx(xs[k], cur)                # cur = c again
            if g is not None and k < len(g):
                ctx.cx(nxt, g[k])
        if cur is not None:                       # x_k ^= c_k ^ d_k
            ctx.cx(cur, xs[k])
        _flip(ctx, xs[k], _bit(d, k), ctrl)
        if cur is not None and cur is not cin:    # vent c_k (k >= 1)
            key = ("cq", id(m), len(m.qc.data), k)
            MB.vent(m, [xs[:k]] + [[q] for q in data_extra], [cur],
                    _vent_fn(d, k, ctrl is not None, cin is not None), key)
            keys.append(key)
            m.free(Reg([cur]))
        if nxt is not None and k == n - 1:
            out = nxt
        cur = nxt
    return keys, out


def _vent_fn(d, k, has_ctrl, has_cin):
    """c_k as a function of the finished low bits: carry(~x', d, c_0)_k."""
    mask = (1 << k) - 1

    def f(xl, *rest):
        i = 0
        c = 1
        if has_ctrl:
            c = rest[i]
            i += 1
        c0 = rest[i] if has_cin else 0
        return carry((~xl) & mask, (d * c) & mask, c0, k)
    return f


def _fix(m, x, d, g, keys, ctrl, cin, merged):
    """Cancel the vents' phase (-1)^(m.c) on the dirty g (Z^m around
    carry-xors of ~x'), leaving g as it was.  `merged`: g already holds
    G ^ c (the streaming adder XORed its carries in), so one carry-xor is
    enough; otherwise two."""
    ctx = m.ctx
    for q in x:
        ctx.x(q)                                  # carries of ~x' + d = carries
    if merged:
        MB.zfix(m, g, keys)                       # (-1)^(m.(G ^ c))
        carry_xor(ctx, x, d, g, ctrl, cin)        # g = G
        MB.zfix(m, g, keys)                       # (-1)^(m.G)
    else:
        MB.zfix(m, g, keys)
        carry_xor(ctx, x, d, g, ctrl, cin)
        MB.zfix(m, g, keys)
        carry_xor(ctx, x, d, g, ctrl, cin)
    for q in x:
        ctx.x(q)


# =============================================================================
# the adders
# =============================================================================
def cq_add_dirty(m, x, d, dirty, ctrl=None, cin=None):
    """x <- x + ctrl d (+ cin) mod 2^n, borrowing n - 1 dirty qubits
    (returned as they were) and two clean ones.  ~3n Toffolis."""
    n = len(x)
    if n == 1:
        _flip(m.ctx, x[0], _bit(d, 0), ctrl)
        if cin is not None:
            m.ctx.cx(cin, x[0])
        return
    g = list(dirty)[:n - 1]
    assert len(g) == n - 1 and not set(g) & set(x), "need n - 1 dirty qubits beside x"
    keys, _ = vented_add(m, x, d, ctrl, cin, g=g)
    _fix(m, x, d, g, keys, ctrl, cin, merged=True)


def cq_add(m, x, d, ctrl=None, cin=None):
    """x <- x + ctrl d (+ cin) mod 2^n on three clean qubits, ~4n Toffolis:
    each half of x is the other's dirty register ([Gid25b] Sec 2.4)."""
    ctx, n = m.ctx, len(x)
    if n < 4:                                     # too short to split: borrow clean
        g = m.anc(max(n - 1, 1), "cqg")
        cq_add_dirty(m, x, d, g, ctrl, cin)
        m.free(g)
        return
    L = n // 2
    lo, hi = list(x[:L]), list(x[L:])
    H = n - L
    dlo, dhi = d & ((1 << L) - 1), d >> L
    klo, C = vented_add(m, lo, dlo, ctrl, cin, keep_carry=True)      # C = c_L, kept
    khi, _ = vented_add(m, hi, dhi, ctrl, C, g=lo[:H - 1])           # lo borrowed
    _fix(m, hi, dhi, lo[:H - 1], khi, ctrl, C, merged=True)          # top half done
    extra = ([[ctrl]] if ctrl is not None else []) + ([[cin]] if cin is not None else [])
    kc = ("cq", id(m), len(m.qc.data), L)
    MB.vent(m, [lo] + extra, [C], _vent_fn(dlo, L, ctrl is not None, cin is not None), kc)
    m.free(Reg([C]))
    _fix(m, lo, dlo, hi[:L], klo + [kc], ctrl, cin, merged=False)    # hi borrowed


# =============================================================================
# [Luo26] App. B: modular arithmetic with a classical modulus
# =============================================================================
class GidneyArith(G.Exact):
    """b += c a mod p, 2 x mod p, for a classical odd p, exact on every input,
    on a constant number of clean qubits: the quantum-quantum additions are
    CDKM (one carry), and every addition of +-p is `cq_add_dirty` borrowing
    the addend (or `cq_add` on clean qubits when there is none).  The
    structure is `ec_luo.cmodadd_reg` / `moddbl_reg` with the register that
    held p replaced by the classical constant.

    A vented adder's inverse is not its gates reversed -- the phase fix would
    come before its measurement -- so the inverses are built forwards:
    `csub` and `half` run the same steps backwards, each unitary step
    inverted and each classical-quantum addition of d replaced by one of
    2^w - d.  `forward_inverse` tells `ec_luo3` to uncompute through them."""
    name = "gidney-cq"
    forward_inverse = True

    def __init__(self, q):
        self.q = q

    # -- steps: ("u", fn, args) unitary, ("cq", reg, d, dirty, ctrl) --------------
    def _run(self, m, steps, inverse):
        for st in (reversed(steps) if inverse else steps):
            if st[0] == "u":
                if inverse:
                    m.emit_inverse(st[1], *st[2])
                else:
                    st[1](*st[2])
            else:
                _, reg, d, dirty, ctrl = st
                w = len(reg)
                d = ((1 << w) - d) % (1 << w) if inverse else d
                dirty = [q for q in (dirty or []) if q not in set(reg) and q is not ctrl]
                if len(dirty) >= w - 1:
                    cq_add_dirty(m, reg, d, dirty, ctrl)
                else:
                    cq_add(m, reg, d, ctrl)

    def _cadd(self, m, c, a, b, inverse):
        ctx, n, p = m.ctx, len(b), self.q
        h, zb, cy = (m.anc(1, nm)[0] for nm in ("h", "zb", "cy"))
        Ah = Reg(list(b) + [h])
        az = Reg(list(a) + [zb])

        def qq(cc):                                  # Ah += c a (overflow into h)
            if cc is None:
                A.cdkm_add(ctx, az, Ah, cy)
            else:
                SP.cdkm_cadd(ctx, cc, az, Ah, cy)

        def clear(cc):                               # h ^= NOT(c [b < a])
            from ec_luo import _carry_tap
            for q in b:
                ctx.x(q)
            _carry_tap(ctx, list(a), list(b), h, cy, cc)
            for q in b:
                ctx.x(q)
            ctx.x(h)
        steps = [("u", qq, (c,)),
                 ("cq", Ah, (1 << (n + 1)) - p, list(az), None),     # S - p: h = [S < p]
                 ("cq", Reg(b), p, list(a), h),                       # + p if negative
                 ("u", clear, (c,))]
        self._run(m, steps, inverse)
        m.free(Reg([h, zb, cy]))

    def cadd(self, m, c, a, b):
        """b <- b + c a mod p (c None: uncontrolled); a restored."""
        self._cadd(m, c, a, b, False)

    def csub(self, m, c, a, b):
        """b <- b - c a mod p, built forwards."""
        self._cadd(m, c, a, b, True)

    def add(self, m, a, b, p=None):
        self._cadd(m, None, a, b, False)

    def sub(self, m, a, b, p=None):
        self._cadd(m, None, a, b, True)

    def _dbl(self, m, reg, dirty, inverse):
        ctx, n, p = m.ctx, len(reg), self.q
        z0, hb = m.anc(1, "z0")[0], m.anc(1, "hb")[0]
        V = Reg([z0] + list(reg) + [hb])                             # 2 reg on n + 2 bits

        def parity():                                                # hb ^= NOT z0 ...
            ctx.x(z0)
            ctx.cx(z0, hb)
            ctx.x(z0)

        def rotate():
            for i in range(n - 1, -1, -1):
                ctx.swap(V[i], V[i + 1])
        steps = [("cq", V, (1 << (n + 2)) - p, dirty, None),           # 2 reg - p
                 ("cq", Reg(V[:n + 1]), p, dirty, hb),                 # + p if negative
                 ("u", parity, ()),                                    # no reduction <=> even
                 ("u", rotate, ())]
        self._run(m, steps, inverse)
        m.free(Reg([z0, hb]))

    def dbl(self, m, reg, dirty=None):
        """reg <- 2 reg mod p.  `dirty`: n + 1 qubits beside reg to borrow
        (the 3n adder); without them, the 4n one on clean qubits."""
        self._dbl(m, reg, dirty, False)

    def half(self, m, reg, dirty=None):
        """reg <- reg / 2 mod p: `dbl` built backwards."""
        self._dbl(m, reg, dirty, True)
