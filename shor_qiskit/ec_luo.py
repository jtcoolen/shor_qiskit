"""Luo et al.'s register-shared extended Euclidean algorithm, as a circuit.

[Luo26] H. Luo et al., "Quantum Algorithm for Elliptic Curve Discrete
        Logarithms with Space-Efficient Point Addition", arXiv:2607.13816v3.
[EF26]  ECDSA.Fail contributors, arXiv:2609.09582, Sec 5.3.5 -- the same EEA
        inside their 851-qubit point addition.
[PZ03]  Proos and Zalka, QIC 3(4) 2003 -- register sharing, four phases.
[BBG+18] Babbush et al., PRX 8, 041015 -- unary iteration.
[CDKM04] Cuccaro et al., quant-ph/0410184 -- the MAJ/UMA ripple adder.

The point of this construction is width.  An inversion by the extended
Euclidean algorithm keeps four numbers alive -- two remainders that shrink and
two cofactors that grow -- and [PZ03]'s observation is that a shrinking and a
growing number can share one register.  [Luo26] turns that into exact
bookkeeping: two (n+3)-qubit banks

    W1 = t | 0 | q | r        W2 = rot_ls(t' | r')

plus four floor(log2 n)+1-bit length registers (lt, lq, lr, ls) and four flags
(P1, P2, Sign, Iter).  `ec_luo_classical` is the bit-level specification; this
module transcribes it microstep by microstep, and the tests compare the two
after every step.  Measured (built through `hier` at n = 128, 256): the
inversion is exactly [Luo26] Sec 6.1's 2n + 6 floor(log2 n) + 19 qubits at
every n tried (3..256; 579 at n = 256), with 20.4M Toffolis at n = 256 against
[Luo26] Table 6's 15.35M -- the difference is the borrowed-qubit C3X of the
cells (`lean`); with an ancilla instead (one qubit more) n = 64 takes 1.11M
against their 1.20M.

One microstep (`eea_step`)
--------------------------
  pre      phase 1: W2 rotates left, ls += 1;  phase 2: right, ls -= 1
  r-block  Sign ^= [r < 2^ls r'] on the window [lt+lq+2, n+3-ls]  (phases 1,2)
           phase 2: Sign ^= 1 (the quotient bit); if set, r -= 2^ls r'
  q lane   phase 2: lq += 1;  phases 2,3: Sign <-> W1[lt+lq+1];  phase 3: lq -= 1
  t-block  phase 3: if Sign, t' += t on the window [1, lt+1];
           phases 3,4: Sign ^= [t' >= t] on [1, lt+1] / [1, n+3-lr-ls]
  post     phase 3: rotate left, ls += 1;  phase 4: right, ls -= 1
  update   [Luo26] Alg. 2's phase rules; at T = 0 mod 4, if lq = ls = 0:
           swap W1 <-> W2, recompute lt and lr from the banks, Iter ^= 1.

Everything whose position depends on the lengths is *location-controlled*
([Luo26] Sec 4.4): a pruned unary iteration [BBG+18] over a length register
produces, one position at a time, the bit "the window starts / ends here", and
an accumulator qubit s that is 1 exactly inside the window gates the ripple
cells.  Only step-dependent *active windows* are scanned: [Luo26] App. A.2's
bounds, which `ec_luo_classical.window_violations` checks against the exact
per-step ranges.  Sums a window needs (lt + lq, lr + ls) are formed in place
in one of the length registers, offset so they never need a carry bit; the
two t-window boundaries are merged into one register ([Luo26] Alg. 3 swaps lt
and lr for the same reason), so one unary iteration serves phases 3 and 4.

The cells.  A CDKM MAJ/UMA pair [CDKM04] in which the sum-writing CNOTs and the
carry Toffoli take s as an extra control; the one uncontrolled CNOT (carry-in
^= addend) is undone by the matching UMA, so a cell with s = 0 is the identity.
The carry-in of a cell is the addend lane one place less significant; where the
window's least significant end moves (the r window ends at n+3-ls), a
location-controlled swap parks a clean carry qubit in that lane for the
duration of the pass.  The carry-out is read from the addend lane at the
window's most significant end, at the moment the scan passes it.  The
three-control Toffoli borrows an idle length qubit (four Toffolis, `lean`,
default) or takes an ancilla (two): the borrowed version is what makes the
count land exactly on [Luo26]'s formula.

Flags.  At most one control qubit is alive during each location-controlled
block: phase controls are formed where they are used (a three-way AND keeps
only its result), the r-block's control is built in place in the "done" flag,
and the q-lane swap's control in P2.  Transients peak at 2 floor(log2 n) + 5
(two unary iterations, the accumulator, the carry, one control).

Lengths ([Luo26] Sec 4.4, Figs 12-13).  After the bank swap, lt ^= len(t_old)
^ len(t_new) and lr ^= len(r_old') ^ len(r_new'), each "highest set bit"
computed by a telescoping XOR of classical constants whose controls
z_j = [no set bit at or beyond j] are toggled into *borrowed* qubits of the
other bank by a Toffoli ladder, applied twice so the borrowed qubits come back.

Padding.  Every input runs the same Nmax = 4 ceil(c n) steps.  A finished run
(lr = 0) keeps its banks still and counts idle four-step blocks in ls (see
`ec_luo_classical`, "Padding"), so t' ends at a fixed place: W2's first n lanes.

Division and multiplication
---------------------------
`Luo.div` is |x>|y> -> |x>|y/x>, the interface of `ec_gcd.Dialog`, exact and
reversible end to end:

    x into W2, EEA forward          W2 = x^{-1} (sign fixed), W1 = p | 0 | 0 1
    z = y x^{-1}                    double-and-add mod p, with P = W1's p
    EEA backward                    W2 = x again, W1 = 0
    y -= z x  (= 0)                 the same multiplier backwards, P = W1 again
    swap y, z

and `Luo.mul` is that circuit reversed.  The EEA's terminal constant is the
modulus register the multiplier needs, and the multiplier's few scratch qubits
are lent from metadata that is |0> at that point, so the division peaks at
4n + 6 floor(log2 n) + 19: x-in-W2, W1, y, z and the EEA's own O(log n).
That is n above [Luo26]'s point addition (Fig. 15): theirs measures y in the
X basis to reuse its register as the second bank of the backward EEA, and uses
a constant adder with O(1) workspace ([Gid25b], itself measurement-vented) so
z can live in the bank.  Both steps are measurements; this package's circuits
are unitary and verified basis state by basis state, so they are not taken.
"""

import ec_adders as A
import ec_luo_classical as LC
import ec_modarith as MA
import ec_space as SP
from ec_sim import Reg
from rc_adder import _maj


# =============================================================================
# Small helpers
# =============================================================================
def _q(m, name):
    return m.anc(1, name)[0]


def _free(m, *qs):
    m.free(Reg(list(qs)))


def _and(m, a, b, na=False, nb=False, name="c"):
    """A fresh qubit holding (a ^ na) AND (b ^ nb)."""
    ctx, t = m.ctx, _q(m, name)
    if na:
        ctx.x(a)
    if nb:
        ctx.x(b)
    ctx.and_(a, b, t)
    if na:
        ctx.x(a)
    if nb:
        ctx.x(b)
    return t


def _unand(m, t, a, b, na=False, nb=False):
    ctx = m.ctx
    if na:
        ctx.x(a)
    if nb:
        ctx.x(b)
    ctx.and_dg(a, b, t)
    if na:
        ctx.x(a)
    if nb:
        ctx.x(b)
    _free(m, t)


def _cinc(m, ctrl, reg):
    """reg += ctrl (mod 2^len): len - 1 ANDs, their uncompute free.
    ctrl None: an unconditional increment (one temporary |1>)."""
    ctx, w = m.ctx, len(reg)
    if ctrl is None:
        one = _q(m, "one")
        ctx.x(one)
        _cinc(m, one, reg)
        ctx.x(one)
        _free(m, one)
        return
    if w == 1:
        ctx.cx(ctrl, reg[0])
        return
    a = m.anc(w - 1, "inc")
    ctx.and_(ctrl, reg[0], a[0])
    for i in range(1, w - 1):
        ctx.and_(a[i - 1], reg[i], a[i])
    for i in range(w - 1, 0, -1):
        ctx.cx(a[i - 1], reg[i])
        ctx.and_dg(a[i - 2] if i >= 2 else ctrl, reg[i - 1], a[i - 1])
    ctx.cx(ctrl, reg[0])
    m.free(a)


def _cdec(m, ctrl, reg):
    m.emit_inverse(_cinc, m, ctrl, reg)


def _add_const_small(m, reg, k):
    """reg += k (mod 2^len) for a classical k: one increment per set bit."""
    for i in range(len(reg)):
        if (k >> i) & 1:
            _cinc(m, None, reg[i:])


def _add_small(m, x, y):
    """y += x (mod 2^len) for the length registers: CDKM, one carry."""
    cy = _q(m, "cy")
    A.cdkm_add(m.ctx, Reg(list(x)), Reg(list(y)), cy)
    _free(m, cy)


def _sub_small(m, x, y):
    m.emit_inverse(_add_small, m, x, y)


def _and3(m, a, b, c, na=False, nb=False, nc=False, name="c"):
    """A fresh qubit holding a AND b AND c (with optional negations): the
    intermediate a AND b is uncomputed at once, so only one qubit stays."""
    t = _and(m, a, b, na, nb, "t3")
    r = _and(m, t, c, False, nc, name)
    _unand(m, t, a, b, na, nb)
    return r


def _unand3(m, r, a, b, c, na=False, nb=False, nc=False):
    t = _and(m, a, b, na, nb, "t3")
    _unand(m, r, t, c, False, nc)
    _unand(m, t, a, b, na, nb)


def _zero(m, reg, name="z"):
    """A fresh qubit holding [reg = 0]."""
    t = _q(m, name)
    MA.is_zero(m, Reg(list(reg)), t)
    return t


def _unzero(m, t, reg):
    MA.is_zero(m, Reg(list(reg)), t)
    _free(m, t)


def _xor_const(ctx, ctrl, reg, k):
    for i in range(len(reg)):
        if (k >> i) & 1:
            ctx.cx(ctrl, reg[i]) if ctrl is not None else ctx.x(reg[i])


# =============================================================================
# Unary iteration ([BBG+18], pruned to a promised interval: [Luo26] Sec 4.3)
# =============================================================================
def uiter(m, ctrl, reg, lo, hi, desc=False):
    """Yield (v, q) for v = lo..hi (or hi..lo), where q = ctrl AND [reg = v].

    Promise: reg is in [lo, hi] whenever ctrl = 1 (outside it a leaf may fire
    on a value it does not name).  hi - lo ANDs; the uncomputes are free.
    q is valid until the generator is advanced.
    """
    ctx = m.ctx
    if lo > hi:
        return
    assert hi < (1 << len(reg)), (hi, len(reg))

    def rec(g, lo, hi):
        if lo == hi:
            yield lo, g
            return
        b = (lo ^ hi).bit_length() - 1
        mid = (hi >> b) << b
        h = _q(m, "ui")
        if not desc:
            ctx.x(reg[b]); ctx.and_(g, reg[b], h); ctx.x(reg[b])
            yield from rec(h, lo, mid - 1)
            ctx.cx(g, h)
            yield from rec(h, mid, hi)
            ctx.and_dg(g, reg[b], h)
        else:
            ctx.and_(g, reg[b], h)
            yield from rec(h, mid, hi)
            ctx.cx(g, h)
            yield from rec(h, lo, mid - 1)
            ctx.x(reg[b]); ctx.and_dg(g, reg[b], h); ctx.x(reg[b])
        _free(m, h)

    yield from rec(ctrl, lo, hi)


class _Leaves:
    """A unary iteration consumed in lockstep with a scan over positions.

    `pos(v)` maps a register value to the bank position it marks.  The scan
    calls `at(j)` for every position in order; a leaf is taken when its
    position comes up.  `close` checks every leaf was taken.
    """

    def __init__(self, m, ctrl, reg, lo, hi, pos, scan):
        order = [j for j in scan]
        vals = [v for v in range(lo, hi + 1)]
        ps = [pos(v) for v in vals]
        # the leaf order must follow the scan order
        idx = {j: i for i, j in enumerate(order)}
        assert all(p in idx for p in ps), ("leaf outside the scan", ps, order)
        asc = len(ps) < 2 or idx[ps[0]] < idx[ps[-1]]
        self.gen = uiter(m, ctrl, reg, lo, hi, desc=not asc)
        self.pos = pos
        self.cur = next(self.gen, None)

    def at(self, j):
        if self.cur is not None and self.pos(self.cur[0]) == j:
            return self.cur[1]
        return None

    def step(self, j):
        if self.at(j) is not None:
            self.cur = next(self.gen, None)

    def close(self):
        assert self.cur is None, "unconsumed leaf"
        for _ in self.gen:
            raise AssertionError("unconsumed leaf")


# =============================================================================
# Location-controlled ripple cells
# =============================================================================
def _c3x(m, s, c, b, a, g=None):
    """a ^= s AND c AND b.  With a borrowed qubit g (any state, restored):
    four Toffolis and no ancilla; without: one AND into a temporary and one
    Toffoli."""
    ctx = m.ctx
    if g is not None:
        ctx.ccx(b, g, a)
        ctx.ccx(s, c, g)
        ctx.ccx(b, g, a)
        ctx.ccx(s, c, g)
        return
    t = _and(m, s, c, name="c3")
    ctx.ccx(t, b, a)
    _unand(m, t, s, c)


def _maj_s(m, s, c, b, a, g=None, gate_c=False):
    """[CDKM04] MAJ gated by s; identity (with `_uma_s`) when s = 0.
    `gate_c`: the carry-in update is gated too, so c is untouched when s = 0
    (used where c is a qubit the scan needs clean later)."""
    ctx = m.ctx
    ctx.ccx(s, a, b)
    ctx.ccx(s, a, c) if gate_c else ctx.cx(a, c)
    _c3x(m, s, c, b, a, g)


def _uma_s(m, s, c, b, a, g=None, gate_c=False):
    ctx = m.ctx
    _c3x(m, s, c, b, a, g)
    ctx.ccx(s, a, c) if gate_c else ctx.cx(a, c)
    ctx.ccx(s, c, b)


# =============================================================================
# The banks
# =============================================================================
class Banks(tuple):
    """W1, W2 (n + 3 qubits each, index j-1 = [Luo26]'s position j) and the
    metadata.  With `x`, W2's last n lanes *are* x's qubits (big-endian), as
    in [Luo26] Algorithm 1: the inversion consumes its input.

    A tuple of registers (W1, W2, lt, lq, lr, ls, flags, consumed, sizes), so
    that `hier.boundary` can rebuild it on a child machine and cache every
    builder that takes one.  `Banks(m, n, x=None, lean=True)` allocates.
    `lean`: the location-controlled cells borrow an idle qubit for their
    three-control Toffoli (four Toffolis) instead of an ancilla (two).
    """

    def __new__(cls, *args, x=None, bits=None, lean=True):
        if len(args) == 1:                              # rebuilt from parts
            return tuple.__new__(cls, args[0])
        m, n = args
        B = bits or LC.meta_bits(n)
        W1 = list(m.anc(n + 3, "W1"))
        if x is None:
            W2, consumed = list(m.anc(n + 3, "W2")), []
        else:
            assert len(x) == n
            consumed = list(m.anc(3, "W2hi"))
            W2 = consumed + [x[n - 1 - i] for i in range(n)]
        regs = [list(m.anc(B, nm)) for nm in ("lt", "lq", "lr", "ls")]
        flags = [_q(m, nm) for nm in ("P1", "P2", "Sign", "Iter")]
        return tuple.__new__(cls, [W1, W2] + regs + [flags, consumed, (n, B, int(lean))])

    W1 = property(lambda s: s[0])
    W2 = property(lambda s: s[1])
    lt = property(lambda s: s[2])
    lq = property(lambda s: s[3])
    lr = property(lambda s: s[4])
    ls = property(lambda s: s[5])
    P1 = property(lambda s: s[6][0])
    P2 = property(lambda s: s[6][1])
    Sign = property(lambda s: s[6][2])
    Iter = property(lambda s: s[6][3])
    consumed = property(lambda s: s[7])
    n = property(lambda s: s[8][0])
    B = property(lambda s: s[8][1])
    lean = property(lambda s: bool(s[8][2]))
    N = property(lambda s: s[8][0] + 3)

    def u(self, j):
        return self.W1[j - 1]

    def v(self, j):
        return self.W2[j - 1]

    def meta(self):
        return self.lt + self.lq + self.lr + self.ls + list(self[6])

    def free(self, m, keep=()):
        own = self.W2 if not self.consumed else self.consumed
        qs = [q for q in self.W1 + own + self.meta() if q not in set(keep)]
        m.free(Reg(qs))


# =============================================================================
# r-block: the big-endian window [L, R], L = lt + lq + 2, R = n + 3 - ls
# =============================================================================
def _r_leaves(m, bk, ctrl, lsum, k, scan):
    """R = n + 3 - ls and L = lt + lq + 2 = lsum + 3 (lsum = lt + lq - 1)."""
    N = bk.N
    return (_Leaves(m, ctrl, bk.ls, 0, min(N - k, bk.n), lambda v: N - v, scan),
            _Leaves(m, ctrl, lsum, k - 3, bk.n, lambda v: v + 3, scan))


def _r_maj_pass(m, bk, ctrl, lsum, k, s, c, tap):
    """LSB -> MSB: complement r in the window and MAJ it with r'.  With `tap`,
    Sign ^= the carry-out, i.e. [r < 2^ls r'] ([~r + r'] carries iff r' > r).

    The carry into position R is the clean qubit c: at R = n + 3 directly,
    otherwise swapped into lane R + 1 (which holds t'(l)) for the pass.  The
    cell at n + 3 gates its carry-in update so c stays clean until then."""
    ctx, N = m.ctx, bk.N
    g = bk.lt[0] if bk.lean else None                  # idle in the r-block
    scan = list(range(N, k - 1, -1))
    lamR, muL = _r_leaves(m, bk, ctrl, lsum, k, scan)
    for j in scan:
        lam, mu = lamR.at(j), muL.at(j)
        if lam is not None:
            ctx.cx(lam, s)
            if j < N:
                ctx.cswap(lam, c, bk.v(j + 1))
        cin = bk.v(j + 1) if j < N else c
        ctx.cx(s, bk.u(j))
        _maj_s(m, s, cin, bk.u(j), bk.v(j), g, gate_c=(j == N))
        if mu is not None:
            if tap:
                ctx.ccx(mu, bk.v(j), bk.Sign)
            ctx.cx(mu, s)
        lamR.step(j)
        muL.step(j)
    lamR.close()
    muL.close()


def _r_uma_pass(m, bk, ctrl, lsum, k, s, c):
    """MSB -> LSB: UMA, then un-complement: r <- ~(~r + r') = r - 2^ls r'."""
    ctx, N = m.ctx, bk.N
    g = bk.lt[0] if bk.lean else None
    scan = list(range(k, N + 1))
    lamR, muL = _r_leaves(m, bk, ctrl, lsum, k, scan)
    for j in scan:
        mu, lam = muL.at(j), lamR.at(j)
        if mu is not None:
            ctx.cx(mu, s)
        cin = bk.v(j + 1) if j < N else c
        _uma_s(m, s, cin, bk.u(j), bk.v(j), g, gate_c=(j == N))
        ctx.cx(s, bk.u(j))
        if lam is not None:
            if j < N:
                ctx.cswap(lam, c, bk.v(j + 1))
            ctx.cx(lam, s)
        muL.step(j)
        lamR.step(j)
    muL.close()
    lamR.close()


def r_compare(m, bk, ctrl, lsum, k):
    """Sign ^= ctrl AND [r < 2^ls r'] on the r window."""
    s, c = _q(m, "s"), _q(m, "c")
    _r_maj_pass(m, bk, ctrl, lsum, k, s, c, tap=True)
    m.emit_inverse(_r_maj_pass, m, bk, ctrl, lsum, k, s, c, False)
    _free(m, s, c)


def r_subtract(m, bk, ctrl, lsum, k):
    """r -= 2^ls r' on the r window, when ctrl."""
    s, c = _q(m, "s"), _q(m, "c")
    _r_maj_pass(m, bk, ctrl, lsum, k, s, c, tap=False)
    _r_uma_pass(m, bk, ctrl, lsum, k, s, c)
    _free(m, s, c)


# =============================================================================
# t-block: the little-endian window [1, B], t' (W2) against t (W1)
# =============================================================================
def _t_maj_pass(m, bk, start, stop, K, s, c, complement, tap):
    """1 -> K: MAJ t' with t while s = [j <= B]; `stop` = (ctrl, reg, lo, hi,
    pos) is the unary iteration marking B.  With `tap`, Sign ^= carry-out."""
    ctx = m.ctx
    gq = bk.lq[0] if bk.lean else None                 # idle in the t-block
    scan = list(range(1, K + 1))
    gen = _Leaves(m, *stop, scan)
    ctx.cx(start, s)
    for j in scan:
        cin = c if j == 1 else bk.u(j - 1)
        if complement:
            ctx.cx(s, bk.v(j))
        _maj_s(m, s, cin, bk.v(j), bk.u(j), gq)
        q = gen.at(j)
        if q is not None:
            if tap:
                ctx.ccx(q, bk.u(j), bk.Sign)
            ctx.cx(q, s)
        gen.step(j)
    gen.close()


def _t_uma_pass(m, bk, start, stop, K, s, c):
    ctx = m.ctx
    gq = bk.lq[0] if bk.lean else None
    scan = list(range(K, 0, -1))
    gen = _Leaves(m, *stop, scan)
    for j in scan:
        q = gen.at(j)
        if q is not None:
            ctx.cx(q, s)
        gen.step(j)
        cin = c if j == 1 else bk.u(j - 1)
        _uma_s(m, s, cin, bk.v(j), bk.u(j), gq)
    ctx.cx(start, s)
    gen.close()


def t_add(m, bk, ctrl, K):
    """t' += t on [1, lt + 1], when ctrl (phase 3 with the quotient bit)."""
    s, c = _q(m, "s"), _q(m, "c")
    stop = (ctrl, bk.lt, 1, min(K - 1, bk.n), lambda v: v + 1)
    _t_maj_pass(m, bk, ctrl, stop, K, s, c, False, False)
    _t_uma_pass(m, bk, ctrl, stop, K, s, c)
    _free(m, s, c)


def _merge_bounds(m, bk):
    """Put the phase-3 and phase-4 right boundaries in one register, so one
    unary iteration serves both ([Luo26] Alg. 3 swaps lt and lr likewise):
    lt <- ~lt = 2^B - (lt + 1)  and  lr <- lr + ls - (n + 3) = 2^B - B4
    (mod 2^B), then swap them when P2 = 1.  Either way lt = 2^B - B."""
    ctx, B = m.ctx, bk.B
    for q in bk.lt:
        ctx.x(q)
    _add_small(m, bk.ls, bk.lr)
    _add_const_small(m, bk.lr, (-bk.N) % (1 << B))
    for a, b in zip(bk.lt, bk.lr):
        ctx.cswap(bk.P2, a, b)


def t_compare(m, bk, K):
    """Sign ^= P1 AND [t' < t] on [1, lt + 1] (phase 3) or [1, n+3-lr-ls]
    (phase 4)."""
    B = bk.B
    s, c = _q(m, "s"), _q(m, "c")
    _merge_bounds(m, bk)
    stop = (bk.P1, bk.lt, max((1 << B) - K, 0), (1 << B) - 1,
            lambda v: (1 << B) - v)
    _t_maj_pass(m, bk, bk.P1, stop, K, s, c, True, True)
    m.emit_inverse(_t_maj_pass, m, bk, bk.P1, stop, K, s, c, True, False)
    m.emit_inverse(_merge_bounds, m, bk)
    _free(m, s, c)


# =============================================================================
# The location-controlled swap ([Luo26] Fig. 9)
# =============================================================================
def loc_swap(m, bk, ctrl, lsum, k, K):
    """Sign <-> W1[lt + lq + 1] = W1[lsum + 2], when ctrl."""
    scan = list(range(k, K + 1))
    g = _Leaves(m, ctrl, lsum, max(k - 2, 0), min(K - 2, bk.n), lambda v: v + 2, scan)
    for j in scan:
        q = g.at(j)
        if q is not None:
            m.ctx.cswap(q, bk.Sign, bk.u(j))
        g.step(j)
    g.close()


# =============================================================================
# Lengths ([Luo26] Sec 4.4, Figs 12-13)
# =============================================================================
def _ladder(m, ctrl, bits, dirty, scan, bnd):
    """dirty[s_i] ^= z_i for every i, z_i = AND_{h <= i} NOT a_h,
    a_h = inside(s_h) AND bits[s_h].  Toffoli ladder on borrowed qubits.

    `bnd` = (reg, lo, hi, pos): the region starts (in scan order) at the
    position pos(reg); or None for "all inside".  `ctrl` gates the region.
    """
    ctx, M = m.ctx, len(scan)
    b = None
    if bnd is not None:
        b = _q(m, "in")
        reg, lo, hi, pos = bnd

    def a_on(j):
        if b is None and ctrl is None:
            ctx.x(bits(j))
            return bits(j)
        t = _and(m, b if b is not None else ctrl, bits(j), name="a")
        ctx.x(t)
        return t

    def a_off(j, t):
        if b is None and ctrl is None:
            ctx.x(bits(j))
            return
        ctx.x(t)
        _unand(m, t, b if b is not None else ctrl, bits(j))

    # first half: reverse scan order, ending on the base
    if b is not None:
        ctx.cx(ctrl, b)
        rev = list(reversed(scan))
        g = _Leaves(m, ctrl, reg, lo, hi, pos, rev)
    for i in range(M - 1, -1, -1):
        j = scan[i]
        t = a_on(j)
        if i > 0:
            ctx.ccx(t, dirty(scan[i - 1]), dirty(j))
        else:
            ctx.cx(t, dirty(j))
        a_off(j, t)
        if b is not None:
            q = g.at(j)
            if q is not None:
                ctx.cx(q, b)
            g.step(j)
    if b is not None:
        g.close()
        g = _Leaves(m, ctrl, reg, lo, hi, pos, scan)
        q = g.at(scan[0])
        if q is not None:
            ctx.cx(q, b)
        g.step(scan[0])
    for i in range(1, M):
        j = scan[i]
        if b is not None:
            q = g.at(j)
            if q is not None:
                ctx.cx(q, b)
            g.step(j)
        t = a_on(j)
        ctx.ccx(t, dirty(scan[i - 1]), dirty(j))
        a_off(j, t)
    if b is not None:
        g.close()
        ctx.cx(ctrl, b)
        _free(m, b)


def xor_length(m, ctrl, bits, dirty, target, scan, val, bnd=None):
    """target ^= val(the first set position, in scan order, inside the region),
    or 0 if there is none.  [Luo26]'s telescoping XOR with borrowed controls.

    bits, dirty: position -> qubit (dirty ones are borrowed and restored).
    With ctrl = 0 every z_j is 1 and the telescope cancels: no change.
    """
    ctx, M = m.ctx, len(scan)
    consts = [val(scan[i]) ^ val(scan[i + 1]) for i in range(M - 1)] + [val(scan[-1])]
    _xor_const(ctx, None, target, val(scan[0]))

    def vx():
        for i, j in enumerate(scan):
            _xor_const(ctx, dirty(j), target, consts[i])

    vx()
    _ladder(m, ctrl, bits, dirty, scan, bnd)
    vx()
    _ladder(m, ctrl, bits, dirty, scan, bnd)


def update_lengths(m, bk, e, T):
    """After the bank swap (under e): lt <- len(t), lr <- len(r')."""
    n, N = bk.n, bk.N
    w = LC.luo_windows(n, T)
    k4, K4 = w["lt"]
    k5, K5 = w["lr"]
    scan_t = list(range(K4, k4 - 1, -1))
    bnd_t = (bk.lr, max(N - K4, 1), min(N - k4, n),
             lambda v: N - v)                              # region [1, N - lr]
    val_t = lambda j: j
    # lt ^= len(old t, now in W2), then ^= len(new t, in W1)
    xor_length(m, e, bk.v, bk.u, bk.lt, scan_t, val_t, bnd_t)
    xor_length(m, e, bk.u, bk.v, bk.lt, scan_t, val_t, bnd_t)
    scan_r = list(range(k5, K5 + 1))
    bnd_r = (bk.lt, max(k5 - 2, 1), n, lambda v: v + 2)    # region [lt + 2, N]
    val_r = lambda j: N - j + 1
    # lr ^= len(new r = old r', in W1), then ^= len(new r', in W2)
    xor_length(m, e, bk.u, bk.v, bk.lr, scan_r, val_r, bnd_r)
    xor_length(m, e, bk.v, bk.u, bk.lr, scan_r, val_r, bnd_r)


# =============================================================================
# One microstep
# =============================================================================
def _rot(ctx, ctrl, W, left):
    idx = range(len(W) - 1)
    for j in (idx if left else reversed(idx)):
        ctx.cswap(ctrl, W[j], W[j + 1])


def eea_step(m, bk, T):
    """Microstep T (1-based) of the schedule: `ec_luo_classical.step`.

    Flags are computed where they are used and uncomputed at once, so at most
    one control qubit is alive during each location-controlled block.  They
    lean on the reachable-state invariant "lr = 0 implies P1 = P2 = 0" (a run
    only finishes at the end of phase 4, which resets the phase)."""
    ctx, n, N = m.ctx, bk.n, bk.N
    w = LC.luo_windows(n, T)
    P1, P2, Sign = bk.P1, bk.P2, bk.Sign

    d = _zero(m, bk.lr, "done")                         # d = [lr = 0]
    if T % 4 == 1:
        _cinc(m, d, bk.ls)                              # count idle blocks

    # --- pre-shift: phase 1 (running) left, phase 2 right ---------------------
    c1 = _and(m, P1, P2, True, True, "c1")
    ctx.cx(d, c1)                                       # = !P1 !P2 !d
    _rot(ctx, c1, bk.W2, True)
    _cinc(m, c1, bk.ls)
    ctx.cx(d, c1)
    _unand(m, c1, P1, P2, True, True)
    c2 = _and(m, P1, P2, True, False, "c2")
    _rot(ctx, c2, bk.W2, False)
    _cdec(m, c2, bk.ls)
    _unand(m, c2, P1, P2, True, False)

    # --- r-block: compare, quotient bit, subtract --------------------------
    lsum = bk.lq                                        # holds lt + lq - 1
    _add_small(m, bk.lt, lsum)
    _cdec(m, None, lsum)
    k1 = w["r"][0]
    ctx.cx(P1, d)                                       # d <- !P1 xor d
    ctx.x(d)                                            #    = phases 1, 2
    r_compare(m, bk, d, lsum, k1)                       # Sign ^= [r < 2^ls r']
    ctx.x(d)
    ctx.cx(P1, d)
    _unzero(m, d, bk.lr)
    ctx.x(P1)
    ctx.ccx(P1, P2, Sign)                               # phase 2: quotient bit
    ctx.x(P1)
    cs = _and3(m, P1, P2, Sign, True, False, False, "cs")
    r_subtract(m, bk, cs, lsum, k1)
    _unand3(m, cs, P1, P2, Sign, True, False, False)

    # --- the quotient lane ------------------------------------------------------
    c2 = _and(m, P1, P2, True, False, "c2")
    _cinc(m, c2, lsum)                                  # phase 2: lq += 1
    _unand(m, c2, P1, P2, True, False)
    ctx.cx(P1, P2)                                      # P2 <- P1 xor P2
    loc_swap(m, bk, P2, lsum, *w["swap"])               # phases 2, 3
    ctx.cx(P1, P2)
    c3 = _and(m, P1, P2, False, True, "c3")
    _cdec(m, c3, lsum)                                  # phase 3: lq -= 1
    _unand(m, c3, P1, P2, False, True)
    _cinc(m, None, lsum)
    _sub_small(m, bk.lt, lsum)

    # --- t-block ------------------------------------------------------------------
    K3 = w["t"][1]
    ca = _and3(m, P1, P2, Sign, False, True, False, "ca")
    t_add(m, bk, ca, K3)                                # phase 3: t' += t
    _unand3(m, ca, P1, P2, Sign, False, True, False)
    ctx.cx(P1, Sign)
    t_compare(m, bk, K3)                                # Sign ^= [t' >= t]

    # --- post-shift: phase 3 left, phase 4 right --------------------------------
    c3 = _and(m, P1, P2, False, True, "c3")
    _rot(ctx, c3, bk.W2, True)
    _cinc(m, c3, bk.ls)
    _unand(m, c3, P1, P2, False, True)
    c4 = _and(m, P1, P2, False, False, "c4")
    _rot(ctx, c4, bk.W2, False)
    _cdec(m, c4, bk.ls)
    _unand(m, c4, P1, P2, False, False)

    # --- phase update ([Luo26] Alg. 2) ---------------------------------------------
    zq = _zero(m, bk.lq, "zq")
    d = _zero(m, bk.lr, "done")
    f = _and(m, zq, d, False, True, "f")                # lq = 0 and lr > 0
    ctx.ccx(f, Sign, P2)
    ctx.ccx(f, P1, P2)
    ctx.ccx(f, P2, Sign)
    _unand(m, f, zq, d, False, True)
    _unzero(m, d, bk.lr)
    _unzero(m, zq, bk.lq)
    zs = _zero(m, bk.ls, "zs")
    ctx.cx(zs, P1)
    ctx.cx(zs, P2)
    _unzero(m, zs, bk.ls)

    # --- end of an iteration: swap the banks, recompute the lengths ----------
    if T % 4 == 0:
        e = _zero(m, bk.lq + bk.ls, "e")                # lq = ls = 0
        for a, b in zip(bk.W1, bk.W2):
            ctx.cswap(e, a, b)
        update_lengths(m, bk, e, T)
        ctx.cx(e, bk.Iter)
        _unzero(m, e, bk.lq + bk.ls)


# =============================================================================
# Initialisation ([Luo26] Algorithm 1) and the whole forward pass
# =============================================================================
def _le(bk, W, a, b):
    """Positions a..b of a bank as a little-endian register (a least sig.)."""
    return [W[j - 1] for j in range(a, b + 1)]


def _be(bk, W, a, b):
    """Positions a..b as little-endian qubits of a big-endian field."""
    return [W[j - 1] for j in range(b, a - 1, -1)]


def eea_init(m, bk, p, x=None):
    """W1 = 1 | 0 | p, W2 = x (copied from `x`, or already in place);
    Iter = [x > p/2] and then x <- p - x; lt = 1, lr = len(x)."""
    ctx, n, N = m.ctx, bk.n, bk.N
    if x is not None:
        for i in range(n):
            ctx.cx(x[i], bk.v(N - i))
    ctx.x(bk.u(1))
    for i in range(n):
        if (p >> i) & 1:
            ctx.x(bk.u(N - i))
    # Iter ^= [p >> 1 < x]: W1 positions 3..N-1 hold p >> 1
    xs = _be(bk, bk.W2, 4, N)
    ph = _be(bk, bk.W1, 3, N - 1)
    SP.lt_cdkm(m, Reg(ph), Reg(xs), bk.Iter)
    # if Iter: x <- p - x = NOT(x + NOT p), on n + 1 bits (positions 3..N)
    X1, Pp = _be(bk, bk.W2, 3, N), _be(bk, bk.W1, 3, N)
    for q in Pp:
        ctx.cx(bk.Iter, q)
    cy = _q(m, "cy")
    SP.cdkm_cadd(ctx, bk.Iter, Reg(Pp), Reg(X1), cy)
    _free(m, cy)
    for q in X1:
        ctx.cx(bk.Iter, q)
    for q in Pp:
        ctx.cx(bk.Iter, q)
    ctx.x(bk.lt[0])                                    # lt = 1
    xor_length(m, None, bk.v, bk.u, bk.lr, list(range(4, N + 1)),
               lambda j: N - j + 1)                    # lr = len(x)


def eea_finish(m, bk, p, clear=True):
    """At the normalised end: t' <- p - t' unless Iter, so W2's first n lanes
    hold x^{-1} (W1 still holds p: p - t' = NOT(t' + NOT p) is one adder).
    With `clear`, then reset W1 = p | 0 | 0 1 and lt = n to zero."""
    ctx, n, N = m.ctx, bk.n, bk.N
    V, P = _le(bk, bk.W2, 1, n + 1), _le(bk, bk.W1, 1, n + 1)
    ctx.x(bk.Iter)
    for q in P:
        ctx.cx(bk.Iter, q)
    cy = _q(m, "cy")
    SP.cdkm_cadd(ctx, bk.Iter, Reg(P), Reg(V), cy)
    _free(m, cy)
    for q in V:
        ctx.cx(bk.Iter, q)
    for q in P:
        ctx.cx(bk.Iter, q)
    ctx.x(bk.Iter)
    if clear:
        clear_terminal(m, bk, p)


def clear_terminal(m, bk, p):
    """W1 = p | 0 | 0 1 and lt = n are input-independent: X them away."""
    ctx, n = m.ctx, bk.n
    for i in range(n):
        if (p >> i) & 1:
            ctx.x(bk.u(1 + i))
    ctx.x(bk.u(bk.N))
    _xor_const(ctx, None, bk.lt, n)


def eea_forward(m, bk, p, x=None, steps=None, finish=True, clear=True):
    """[Luo26] Algorithm 1: init, Nmax microsteps, sign fix, clear W1."""
    eea_init(m, bk, p, x)
    for T in range(1, (steps or LC.step_bound(bk.n)) + 1):
        eea_step(m, bk, T)
    if finish:
        eea_finish(m, bk, p, clear)


def luo_inv(m, x, p, steps=None, lean=True):
    """[Luo26] eq. (1): |x>|0> -> |x^{-1}>|Gamma(x)>, x consumed into W2.

    Returns (out, garbage): `out` is W2's first n lanes (little-endian),
    holding x^{-1}; `garbage` is ls (the idle-block count) and Iter.  Every
    other workspace qubit is freed, checked |0>.
    """
    n = len(x)
    bk = Banks(m, n, x=x, lean=lean)
    eea_forward(m, bk, p, steps=steps)
    out = Reg(_le(bk, bk.W2, 1, n), "xinv")
    garbage = Reg(bk.ls + [bk.Iter], "gamma")
    keep = set(out) | set(garbage)
    m.free(Reg([q for q in bk.W1 + bk.consumed + bk.meta() if q not in keep]))
    idle = [q for q in x if q not in keep]            # the caller's qubits:
    m.checks.append((len(m.qc.data), idle, 0))        # checked, not pooled
    return out, garbage


# =============================================================================
# Modular arithmetic with p held in a register (for the division)
# =============================================================================
def _carry_tap(ctx, a, b, out, c, ctrl=None):
    """out ^= ctrl AND carry-out(a + b); a, b restored.  CDKM MAJ chain."""
    w = len(a)
    _maj(ctx, c, b[0], a[0])
    for i in range(w - 1):
        _maj(ctx, a[i], b[i + 1], a[i + 1])
    if ctrl is None:
        ctx.cx(a[w - 1], out)
    else:
        ctx.ccx(ctrl, a[w - 1], out)
    for i in range(w - 2, -1, -1):
        SP._maj_dg(ctx, a[i], b[i + 1], a[i + 1])
    SP._maj_dg(ctx, c, b[0], a[0])


def moddbl_reg(m, acc, P):
    """acc <- 2 acc mod p, p in the register P (n qubits)."""
    ctx, n = m.ctx, len(acc)
    z0, hb, e0, e1, cy = (_q(m, nm) for nm in ("z0", "hb", "e0", "e1", "cy"))
    V = [z0] + list(acc) + [hb]                        # 2 acc on n + 2 bits
    for q in V:
        ctx.x(q)
    A.cdkm_add(ctx, Reg(list(P) + [e0, e1]), Reg(V), cy)
    for q in V:
        ctx.x(q)                                       # V = 2acc - p
    SP.cdkm_cadd(ctx, hb, Reg(list(P) + [e0]), Reg(V[:n + 1]), cy)
    ctx.x(z0)
    ctx.cx(z0, hb)                                     # no reduction <=> even
    ctx.x(z0)
    for i in range(n - 1, -1, -1):
        ctx.swap(V[i], V[i + 1])
    _free(m, z0, hb, e0, e1, cy)


def cmodadd_reg(m, ctrl, b, acc, P):
    """acc <- acc + ctrl * b mod p, p in the register P.  ~10n Toffolis."""
    ctx, n = m.ctx, len(acc)
    h, zb, zp, cy = (_q(m, nm) for nm in ("h", "zb", "zp", "cy"))
    Ah = Reg(list(acc) + [h])
    if ctrl is None:
        A.cdkm_add(ctx, Reg(list(b) + [zb]), Ah, cy)
    else:
        SP.cdkm_cadd(ctx, ctrl, Reg(list(b) + [zb]), Ah, cy)
    for q in Ah:
        ctx.x(q)
    A.cdkm_add(ctx, Reg(list(P) + [zp]), Ah, cy)
    for q in Ah:
        ctx.x(q)                                       # h = [acc + b < p]
    SP.cdkm_cadd(ctx, h, Reg(P), Reg(acc), cy)
    for q in acc:                                      # h ^= NOT(ctrl [acc < b])
        ctx.x(q)
    _carry_tap(ctx, list(b), list(acc), h, cy, ctrl)
    for q in acc:
        ctx.x(q)
    ctx.x(h)
    _free(m, h, zb, zp, cy)


def mul_acc_reg(m, a, b, acc, P):
    """acc <- 2^(n-1) acc + a b mod p (double-and-add over a, MSB first).

    From acc = 0 this is the product; run backwards on acc = a b it clears
    acc -- which is how the division erases y."""
    n = len(a)
    for i in reversed(range(n)):
        if i != n - 1:
            moddbl_reg(m, acc, P)
        cmodadd_reg(m, a[i], b, acc, P)


# =============================================================================
# The backend
# =============================================================================
class Luo:
    """In-place division / multiplication through [Luo26]'s EEA, with the
    `ec_gcd` backend interface: .mul(m, x, y, p), .div(m, x, y, p).

    `lean` (default) borrows an idle qubit for the location-controlled cells'
    three-control Toffoli (four Toffolis) instead of an ancilla (two)."""

    def __init__(self, steps=None, lean=True):
        self.steps, self.lean = steps, lean

    def div(self, m, x, y, p):
        """|x>|y> -> |x>|y / x mod p> for x != 0.

        x is consumed into W2; the forward EEA leaves x^{-1} there and p in W1,
        which is exactly the modulus register the multiplication needs:

            EEA forward                 W2 = x^{-1}, W1 = p | 0 | 0 1
            z = y x^{-1}                (P = W1)
            EEA backward                W2 = x, W1 = 0
            y -= z x  (= 0)             (P = W1 again, written with X gates)
            swap y, z

        Four n-bit registers at the peak (x-in-W2, W1, y, z), all exact."""
        ctx, n = m.ctx, len(x)
        assert len(y) == n and p.bit_length() == n
        bk = Banks(m, n, x=x, lean=self.lean)
        eea_forward(m, bk, p, steps=self.steps, clear=False)
        P = Reg(_le(bk, bk.W1, 1, n), "P")
        inv = Reg(_le(bk, bk.W2, 1, n), "xinv")
        z = m.anc(n, "z")
        # lend the metadata that is |0> here to the multiplier's scratch
        spare = (bk.lq + bk.lr + [bk.P1, bk.P2, bk.Sign] + bk.W2[n:]
                 + bk.W1[n:n + 2])
        with _lent(m, spare):
            mul_acc_reg(m, inv, y, z, P)               # z = y / x
        m.emit_inverse(eea_forward, m, bk, p, None, self.steps, True, False)
        A.encode_const(ctx, P, p)
        spare = bk.meta() + bk.consumed + bk.W1[n:]
        with _lent(m, spare):
            m.emit_inverse(mul_acc_reg, m, z, x, y, P)  # y -= z x  (= 0)
        A.encode_const(ctx, P, p)
        for a, b in zip(y, z):
            ctx.swap(a, b)
        m.free(z)
        bk.free(m)

    def mul(self, m, x, y, p):
        """|x>|y> -> |x>|x y mod p>: the division run backwards."""
        m.emit_inverse(self.div, m, x, y, p)


def luo_div(m, x, y, p, steps=None, lean=True):
    """|x>|y>|0> -> |x>|y / x mod p>|0>  (x != 0): `Luo.div`."""
    Luo(steps, lean).div(m, x, y, p)


def luo_mul(m, x, y, p, steps=None, lean=True):
    """|x>|y>|0> -> |x>|x y mod p>|0>  (x != 0): `luo_div` run backwards."""
    Luo(steps, lean).mul(m, x, y, p)


class _lent:
    """Return known-|0> qubits to the pool for a while (checked), then claim
    the same qubits back: the multiplier's scratch costs no new qubits."""

    def __init__(self, m, qubits):
        self.m, self.qs = m, list(qubits)

    def __enter__(self):
        self.m.free(Reg(self.qs))

    def __exit__(self, *exc):
        if exc[0] is None:
            self.m.claim(self.qs)
        return False


def div_qubits(n):
    """Width of `Luo.div` by construction: x (in W2), y, z, W1, W2's three
    extra lanes, 4 length registers, 4 flags, and the step's transients."""
    return 4 * n + 6 * LC.meta_bits(n) + 13
