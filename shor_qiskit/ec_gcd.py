"""In-place modular multiplication by a recorded GCD -- pluggable variants.

`ec_eea` builds [1128]'s construction exactly as the paper states it.  This
module makes each of its refinements -- and the rival GCDs of IonQ and
ECDSA.Fail -- a switch, so they can be ablated against each other on one
interface:

    backend.mul(m, x, y, q)    |x, y>  ->  |x, y * x mod q>
    backend.div(m, x, y, q)    |x, y>  ->  |x, y / x mod q>   (mul run backwards)

A backend pairs a *walk* (the GCD that consumes x and leaves a record) with an
*arithmetic* for the replay (exact, approximate, or pseudo-Mersenne).

`Dialog` is [1128] Algorithm 2-4.  With every option at its default it emits
exactly what `ec_eea.inplace_mul` emits; the options are

  fused_cmp   b0b1 = b0 AND [u > v] from one fused comparator (n + 1 Toffoli)
              instead of compare / AND / compare-again (2n + 1).
  cmp_msbs    decide [u > v] on the top k bits of the *current* width only
              ([1128] Sec 4) -- k + 1 Toffoli; approximate.
  c_pad       run each round on the scheduled width n - 0.7075 i + c_pad sqrt(n)
              of [1128] Sec 3.1 rather than on all n + 1 bits: u and v shrink as
              the GCD proceeds, so the swaps, subtractions and comparisons get
              cheaper.  Probabilistic: a value that outgrows its schedule gives
              a wrong answer, which the tests count.
  share       additionally *free* the high qubits the schedule no longer uses,
              so the record grows into them -- the space half of Sec 3.1.
              The ancilla checks enforce that they really are |0>.
  compress    "fig1": pack each three records (b0, b0 b1) into five qubits with
              [1128] Fig. 1's five-Toffoli circuit, freeing one qubit per three
              rounds (2.83n -> 2.355n record qubits).
  reuse_x     the replay's second register is x's own qubits, which the walk
              has emptied: -n qubits, exact.  (Not with `share`, which hands
              those qubits to the record instead.)
"""

import math

import ec_adders as A
import ec_approx as AX
import ec_modarith as MA
from ec_classical import eea_iterations
from ec_sim import Reg


# =============================================================================
# Replay arithmetic
# =============================================================================
class Exact:
    """Exact modular arithmetic, any odd q ([RNSL17] / [1128] Alg 5, 8)."""
    name = "exact"

    def __init__(self, q):
        self.q = q

    def dbl(self, m, reg):
        MA.moddbl(m, reg, self.q)

    def cadd(self, m, c, a, b):
        MA.cmodadd(m, c, a, b, self.q)

    def signadd(self, m, e, a, b):
        MA.csignadd(m, e, a, b, self.q)

    def half(self, m, reg):
        MA.modhalf(m, reg, self.q)


class Approx(Exact):
    """[1128] Alg 6 and 9: comparisons on the top msbs bits."""
    name = "approx"

    def __init__(self, q, msbs=None):
        super().__init__(q)
        self.msbs = msbs

    def dbl(self, m, reg):
        AX.moddbl_approx(m, reg, self.q, self.msbs)

    def cadd(self, m, c, a, b):
        AX.cmodadd_approx(m, c, a, b, self.q, self.msbs)


class PM(Exact):
    """[1128] Alg 7 and 11 for q = 2^u - f, plus IonQ's signed add."""
    name = "pm"

    def __init__(self, q, lsbs=None, msbs=None):
        super().__init__(q)
        assert AX.pseudo_mersenne(q), f"{q} is not pseudo-Mersenne"
        self.lsbs, self.msbs = lsbs, msbs

    def dbl(self, m, reg):
        AX.moddbl_pm(m, reg, self.q, self.lsbs)

    def cadd(self, m, c, a, b):
        AX.cmodadd_pm_q(m, c, a, b, self.q, self.lsbs, self.msbs)

    def signadd(self, m, e, a, b):
        AX.csignadd_pm(m, e, a, b, self.q, self.lsbs, self.msbs)

    def half(self, m, reg):
        AX.modhalf_pm(m, reg, self.q, self.lsbs)


def arith_for(q, kind="exact", **kw):
    return {"exact": Exact, "approx": Approx, "pm": PM}[kind](q, **kw)


# =============================================================================
# [1128] Fig. 1: three (b0, b0 b1) pairs in five qubits, five Toffolis
# =============================================================================
def fig1_compress(ctx, w):
    """w = six wires, pairs (w0,w1), (w2,w3), (w4,w5), each (b0, b0 AND b1).
    Afterwards w5 is |0> on every one of the 27 reachable inputs."""
    ctx.cx(w[1], w[0]); ctx.cx(w[3], w[2]); ctx.cx(w[5], w[4])
    ctx.cx(w[0], w[2]); ctx.cx(w[5], w[3])
    ctx.x(w[4])
    ctx.ccx(w[1], w[3], w[5])
    ctx.cx(w[1], w[4])
    ctx.x(w[2])
    ctx.ccx(w[3], w[4], w[5])
    ctx.ccx(w[4], w[5], w[1])
    ctx.ccx(w[2], w[5], w[0])
    ctx.ccx(w[0], w[1], w[5])


def fig1_decompress(m, w):
    m.emit_inverse(fig1_compress, m.ctx, w)


# =============================================================================
# The Dialog backend
# =============================================================================
class _Records:
    """Round records, with Fig. 1 groups that may be packed.

    pairs[i] is the (b0, b0b1) wire pair of round i while its group is
    unpacked.  A packed group keeps wires 0..4 and has returned wire 5 (the
    `spare`) to the pool; unpacking claims the same spare back.
    """

    def __init__(self):
        self.pairs = []
        self.groups = []          # (start, spare qubit or None while unpacked)

    def group_of(self, i):
        for g, (s, _) in enumerate(self.groups):
            if s <= i < s + 3:
                return g
        return None

    def wires(self, g):
        s, _ = self.groups[g]
        out = []
        for j in range(s, s + 3):
            out += [self.pairs[j][0][0], self.pairs[j][1][0]]
        return out

    def pack(self, m, g):
        w = self.wires(g)
        fig1_compress(m.ctx, w)
        spare = w[5]
        m.free(Reg([spare]))
        self.groups[g] = (self.groups[g][0], spare)

    def unpack(self, m, g, claim=True, reserved=()):
        """Restore the three pairs.  claim=True takes the original spare back
        (the reverse walk, where it is free again); claim=False takes a fresh
        qubit (the replay, where later records may occupy the spare) -- one
        that the reverse walk will not need to claim first."""
        s, spare = self.groups[g]
        t = (m.claim([spare])[0] if claim
             else _anc_excluding(m, 1, set(reserved), "fig1")[0])
        # rewire the third pair's b0b1 onto t
        b0, _ = self.pairs[s + 2]
        self.pairs[s + 2] = (b0, Reg([t]))
        fig1_decompress(m, self.wires(g))
        self.groups[g] = (s, None)
        return t

    def repack(self, m, g, spare_was_claimed):
        s, _ = self.groups[g]
        w = self.wires(g)
        fig1_compress(m.ctx, w)
        m.free(Reg([w[5]]))
        self.groups[g] = (s, w[5])


class Dialog:
    """[1128] in-place multiplication, with the refinements as options."""

    def __init__(self, arith=None, fused_cmp=False, cmp_msbs=None, c_pad=None,
                 share=False, compress=None, reuse_x=False, iters=None,
                 c_iter=2.4):
        assert not (share and reuse_x), "share hands x's qubits to the record"
        assert not share or c_pad is not None, "share needs the width schedule"
        assert compress in (None, "fig1"), compress
        self.arith, self.fused_cmp, self.cmp_msbs = arith, fused_cmp, cmp_msbs
        self.c_pad, self.share, self.compress = c_pad, share, compress
        self.reuse_x, self._iters, self.c_iter = reuse_x, iters, c_iter

    # -- schedule ------------------------------------------------------------
    def iters(self, n):
        return self._iters or eea_iterations(n, self.c_iter)

    def widths(self, n):
        """Width of u and v during each round (n + 1 without a schedule)."""
        it = self.iters(n)
        if self.c_pad is None:
            return [n + 1] * it
        step = 0.5 * math.log2(8 / 3)
        pad = self.c_pad * math.sqrt(n)
        return [max(2, min(n + 1, math.ceil(n - step * i + pad))) for i in range(it)]

    # -- one round of Algorithm 2 on the current widths ---------------------------
    def _round(self, m, u, v, rec):
        ctx, w = m.ctx, len(u)
        b0, b0b1 = rec
        ctx.cx(v[0], b0[0])                              # b0 = v mod 2
        if self.fused_cmp or self.cmp_msbs:
            k = min(self.cmp_msbs or w, w)
            sc = m.anc(k, "sc")
            A.gt_top(ctx, u, v, b0b1[0], sc, k, ctrl=b0[0])
            m.free(sc)
        else:
            t, sc = m.anc(1, "t"), m.anc(w, "sc")
            A.gt_uint(ctx, u, v, t[0], sc)               # t = [u > v]
            ctx.and_(b0[0], t[0], b0b1[0])
            A.gt_uint(ctx, u, v, t[0], sc)
            m.free(t, sc)
        for i in range(w):
            ctx.cswap(b0b1[0], u[i], v[i])
        cp, sc = m.anc(w, "cp"), m.anc(w, "sc")
        A.csub(ctx, b0[0], u, v, cp, sc)                 # if b0: v -= u
        m.free(cp, sc)
        for i in range(w - 1):                           # v /= 2
            ctx.swap(v[i], v[i + 1])

    # -- the walk: x -> record ------------------------------------------------
    def record(self, m, x, q):
        n = len(x)
        widths = self.widths(n)
        hi = m.anc(1, "vhi")
        u = m.anc(n + 1, "u")
        v = Reg(list(x) + list(hi), "v")
        A.encode_const(m.ctx, u, q)                      # u <- q
        U, V = list(u), list(v)
        recs = _Records()
        shrunk = {}                                      # round -> (u hi, v hi)
        for i, w in enumerate(widths):
            if self.share and len(U) > w:
                uh, vh = U[w:], V[w:]
                U, V = U[:w], V[:w]
                m.free(Reg(uh), Reg(vh))                 # must be |0>: checked
                shrunk[i] = (uh, vh)
            rec = (m.anc(1, "b0"), m.anc(1, "b1"))
            recs.pairs.append(rec)
            self._round(m, Reg(U[:w]), Reg(V[:w]), rec)
            if self.compress and i % 3 == 2:
                recs.groups.append((i - 2, None))
                recs.pack(m, len(recs.groups) - 1)
        return {"recs": recs, "U": U, "V": V, "hi": hi, "shrunk": shrunk,
                "widths": widths, "n": n}

    def clear_end(self, m, st):
        """u = 1 and v = 0 on termination."""
        m.ctx.x(st["U"][0])
        m.free(Reg(st["U"]))
        st["U"] = []
        if not self.share:
            m.free(Reg([st["V"][-1]]))                   # v's borrowed high bit

    def unrecord(self, m, x, q, st):
        """The walk backwards: rebuild x in its own qubits, consume the record.

        Without sharing this mirrors `ec_eea.inplace_mul` step 3 exactly (same
        allocations in the same order, records freed at the end).  With
        sharing every qubit the forward walk freed is *claimed* back at the
        mirror-image point, so v regrows into x's own qubits; the fresh u is
        drawn from the pool with those reserved qubits held aside.
        """
        n, widths, recs = st["n"], st["widths"], st["recs"]
        if self.share or self.compress:
            reserved = {q_ for uh, vh in st["shrunk"].values() for q_ in uh + vh}
            reserved |= {sp for _, sp in recs.groups if sp is not None}
        if self.share:
            V = list(st["V"])
            U = list(_anc_excluding(m, len(V), reserved, "u"))
        else:
            hi = (m.anc(1, "vhi") if not self.compress
                  else _anc_excluding(m, 1, reserved, "vhi"))
            V = list(x) + list(hi)
            U = list(m.anc(n + 1, "u") if not self.compress
                     else _anc_excluding(m, n + 1, reserved, "u"))
        m.ctx.x(U[0])                                    # the walk ends at u = 1
        for i in reversed(range(len(widths))):
            w = widths[i]
            g = recs.group_of(i) if self.compress else None
            if g is not None and recs.groups[g][1] is not None:
                recs.unpack(m, g, claim=True)
            rec = recs.pairs[i]
            m.emit_inverse(self._round, m, Reg(U[:w]), Reg(V[:w]), rec)
            if self.share or self.compress:
                m.free(*rec)
            if i in st["shrunk"]:
                uh, vh = st["shrunk"][i]
                m.claim(list(uh) + list(vh))
                U, V = U + list(uh), V + list(vh)
        A.encode_const(m.ctx, Reg(U), q)                 # the walk started at u = q
        m.free(Reg(U), Reg([V[-1]]))
        if not (self.share or self.compress):
            for rec in recs.pairs:                       # as ec_eea: at the end
                m.free(*rec)
        assert V[:n] == list(x)

    # -- Algorithm 3 on the record ---------------------------------------------
    def replay(self, m, r, s, st, q):
        arith = self.arith or Exact(q)
        recs = st["recs"]
        claimed = {}
        shrunk = {q_ for uh, vh in st["shrunk"].values() for q_ in uh + vh}
        for i in reversed(range(len(st["widths"]))):
            g = recs.group_of(i) if self.compress else None
            if g is not None and recs.groups[g][1] is not None:
                claimed[g] = recs.unpack(m, g, claim=False, reserved=shrunk)
            b0, b0b1 = recs.pairs[i]
            arith.dbl(m, s)                              # s <- 2s
            arith.cadd(m, b0[0], r, s)                   # if b0: s += r
            for a, b in zip(r, s):
                m.ctx.cswap(b0b1[0], a, b)               # if b0b1: swap
            if g is not None and i == recs.groups[g][0]:
                recs.repack(m, g, False)

    # -- the whole multiplication ----------------------------------------------
    def mul(self, m, x, y, q):
        n = len(x)
        st = self.record(m, x, q)
        self.clear_end(m, st)
        if self.reuse_x:
            s = Reg(list(x), "bz")                       # emptied by the walk
        elif self.share or self.compress:
            reserved = {q_ for uh, vh in st["shrunk"].values() for q_ in uh + vh}
            reserved |= {sp for _, sp in st["recs"].groups if sp is not None}
            s = _anc_excluding(m, n, reserved, "bz")     # spares must stay free
        else:
            s = m.anc(n, "bz")
        self.replay(m, y, s, st, q)
        for a, b in zip(y, s):
            m.ctx.swap(a, b)
        if not self.reuse_x:
            m.free(s)
        self.unrecord(m, x, q, st)

    def div(self, m, x, y, q):
        m.emit_inverse(self.mul, m, x, y, q)


def _anc_excluding(m, k, reserved, name):
    """m.anc(k), never handing out a qubit in `reserved`."""
    held = [q for q in m._pool if q in reserved]
    m._pool = [q for q in m._pool if q not in reserved]
    try:
        return m.anc(k, name)
    finally:
        m._pool = held + m._pool


def record_qubits(backend, n):
    """Qubits the record occupies at the end of the walk, for the tables."""
    it = backend.iters(n)
    raw = 2 * it
    return raw - (it // 3 if backend.compress == "fig1" else 0)


# =============================================================================
# IonQ Sec VI: the binary GCD with conditionally-inverted additions
# =============================================================================
class CondInv:
    """The [1128] walk rewritten so no step is a *controlled* addition.

    Store (u, v~) with v~ = v + (1 - a) u, a = v mod 2.  Both are odd, so the
    next parity is free -- a' = v~[1] XOR u[1], two CNOTs -- and the update

        v~' = (v~ + (-1)^a' u) / 2

    is one *conditionally inverted* addition (n - 1 ANDs, `ec_adders.ci_add`)
    where [1128] needs a controlled subtraction (2n).  The swap that orients
    the pair for the next round is a' AND [u > v~'].  The recorded bits (a', m)
    are exactly [1128]'s (b0, b0 b1) -- checked on every x in the tests -- so
    the record, its replay and Fig. 1's packing all carry over unchanged.

    The replay can run [1128]'s way (`replay="standard"`: reverse order,
    s <- 2s, controlled add, swap) or IonQ's (`replay="ci"`): forwards, in the
    division direction, where it has the walk's own shape

        s~' = (s~ + (-1)^a r) / 2  (mod q),  then swap on m,

    one signed modular addition and one halving per round; it ends at
    (r, s~) = (z/x, z/x), and one subtraction clears s~.  Multiplication is
    that division run backwards.  The signed addition is cheap only for
    pseudo-Mersenne q (`ec_approx.csignadd_pm`: complements around one
    adder); with exact generic arithmetic it costs two negations, and the
    standard replay is the better choice.

    Registers: u needs n + 2 bits and v~ n + 2 (v~ < 2q, and v~ + u < 3q
    before the halving).
    """

    def __init__(self, arith=None, cmp_msbs=None, c_pad=None, replay="ci",
                 iters=None, c_iter=2.4):
        assert replay in ("ci", "standard"), replay
        self.arith, self.cmp_msbs, self.c_pad = arith, cmp_msbs, c_pad
        self.replay_kind, self._iters, self.c_iter = replay, iters, c_iter

    iters = Dialog.iters
    widths = Dialog.widths

    # -- the walk -----------------------------------------------------------------
    def _first(self, m, U, VT, rec, q):
        """a0 = x mod 2;  v~ = x + (1 - a0) q;  swap iff a0 (then x < q = u)."""
        ctx = m.ctx
        a, mm = rec
        ctx.cx(VT[0], a[0])
        ctx.x(a[0])
        cp, sc = m.anc(len(VT), "cp"), m.anc(len(VT), "sc")
        A.cadd_const(ctx, a[0], VT, q, cp, sc)          # if not a0: v~ += q
        m.free(cp, sc)
        ctx.x(a[0])
        ctx.cx(a[0], mm[0])                              # m0 = a0
        for i in range(len(U)):
            ctx.cswap(mm[0], U[i], VT[i])

    def _round(self, m, U, VT, rec, w):
        """One round on the low w + 1 bits of u and v~ (w: the schedule)."""
        ctx = m.ctx
        a, mm = rec
        u, vt = Reg(U[:w + 1]), Reg(VT[:w + 1])
        ctx.cx(vt[1], a[0])
        ctx.cx(u[1], a[0])                               # a = v~[1] ^ u[1]
        anc = m.anc(max(len(u) - 1, 1), "ci")
        A.ci_add(ctx, a[0], u, vt, anc)                  # v~ += (-1)^a u
        m.free(anc)
        for i in range(len(vt) - 1):                     # v~ /= 2 (even)
            ctx.swap(vt[i], vt[i + 1])
        k = min(self.cmp_msbs or len(u), len(u))
        sc = m.anc(k, "sc")
        A.gt_top(ctx, u, vt, mm[0], sc, k, ctrl=a[0])    # m = a AND [u > v~]
        m.free(sc)
        for i in range(len(u)):
            ctx.cswap(mm[0], u[i], vt[i])

    def _walk(self, m, U, VT, recs, q, widths):
        self._first(m, U, VT, recs[0], q)
        for rec, w in zip(recs[1:], widths[1:]):
            self._round(m, U, VT, rec, w)

    def _registers(self, m, x):
        n = len(x)
        U = m.anc(n + 2, "u")
        hi = m.anc(2, "vhi")
        return U, hi, Reg(list(x) + list(hi), "v~")

    def record(self, m, x, q):
        n = len(x)
        widths = self.widths(n)
        U, hi, VT = self._registers(m, x)
        A.encode_const(m.ctx, U, q)
        recs = [(m.anc(1, "a"), m.anc(1, "m")) for _ in widths]
        self._walk(m, U, VT, recs, q, widths)
        m.ctx.x(U[0])                                    # ends at (1, 1)
        m.ctx.x(VT[0])
        m.free(U, hi)
        return recs

    def unrecord(self, m, x, q, recs):
        widths = self.widths(len(x))
        U, hi, VT = self._registers(m, x)
        m.ctx.x(U[0])
        m.ctx.x(VT[0])
        m.emit_inverse(self._walk, m, U, VT, recs, q, widths)
        A.encode_const(m.ctx, U, q)
        m.free(U, hi)
        for rec in recs:
            m.free(*rec)

    # -- replays --------------------------------------------------------------------
    def _replay_div(self, m, r, s, recs, q):
        """Forwards: (r, s) = (0, z)  ->  (z/x, z/x)."""
        arith = self.arith or Exact(q)
        _, m0 = recs[0]
        for a_, b_ in zip(r, s):
            m.ctx.cswap(m0[0], a_, b_)                   # r = 0: only the swap
        for a, mm in recs[1:]:
            arith.signadd(m, a[0], r, s)                 # s += (-1)^a r
            arith.half(m, s)                             # s /= 2
            for a_, b_ in zip(r, s):
                m.ctx.cswap(mm[0], a_, b_)

    def div(self, m, x, z, q):
        """|x, z> -> |x, z / x mod q>."""
        if self.replay_kind == "standard":
            m.emit_inverse(self.mul, m, x, z, q)
            return
        n = len(x)
        recs = self.record(m, x, q)
        r = m.anc(n, "r")
        self._replay_div(m, r, z, recs, q)
        MA.modsub(m, r, z, q)                            # z~ = r: clear it
        for a_, b_ in zip(r, z):
            m.ctx.swap(a_, b_)
        m.free(r)
        self.unrecord(m, x, q, recs)

    def mul(self, m, x, y, q):
        """|x, y> -> |x, y x mod q>."""
        if self.replay_kind == "ci":
            m.emit_inverse(self.div, m, x, y, q)
            return
        n = len(x)
        recs = self.record(m, x, q)
        s = m.anc(n, "bz")
        st = {"recs": _Records(), "widths": self.widths(n), "shrunk": {}}
        st["recs"].pairs = list(recs)
        Dialog(arith=self.arith).replay(m, y, s, st, q)
        for a_, b_ in zip(y, s):
            m.ctx.swap(a_, b_)
        m.free(s)
        self.unrecord(m, x, q, recs)


# =============================================================================
# ECDSA.Fail Sec 5.3.3: the comparison-free "ping-pong" GCD
# =============================================================================
class PingPong:
    """ECDSA.Fail Algorithm 5: no comparison, no swap, one bit per round.

    Both operands are kept *signed and odd*: rho0 = q, rho1 = x (or x - q when
    x is even).  Round i updates the other operand each time (rho1 on even i,
    rho0 on odd i) as

        e = t[1] XOR s[1],     t <- (t + (-1)^e s) / 2,

    which is odd again because the numerator is 2 mod 4 -- so the parity bit
    that [1128] must record and the magnitude comparison it must compute are
    both gone.  The walk ends at (eta0, eta1) in {+1, -1}^2, a fixed point, so
    padding rounds are harmless.  Per round: two CNOTs, one conditionally
    inverted addition (n + 1 ANDs on the n + 2-bit two's-complement
    registers), and a shift.

    Replaying the same linear rounds modulo q on (0, y) gives
    (eta0, eta1) * y / x; correcting the two signs and clearing one copy is a
    division.  Multiplication is that run backwards.

    The price is rounds: ECDSA.Fail uses L = 2.75n (704 at n = 256) and states
    no worst-case bound.  Measured here exhaustively, the worst case is 2.0n at
    n = 4, 3.0n at n = 8 and 3.9n at n = 16, so `rounds` is a parameter and the
    tests report the failure rate at 2.75n next to the exact setting.
    """

    def __init__(self, arith=None, rounds=None, c_rounds=2.75, neg=None):
        self.arith, self._rounds, self.c_rounds = arith, rounds, c_rounds
        self.neg = neg

    def L(self, n):
        return self._rounds or math.ceil(self.c_rounds * n)

    # -- the walk -------------------------------------------------------------------
    @staticmethod
    def _lift(m, R1, q):
        """rho1 <- x if x is odd else x - q (two's complement)."""
        ctx = m.ctx
        f = m.anc(1, "odd")
        ctx.cx(R1[0], f[0])
        ctx.x(f[0])                                   # f = [x even]
        cp, sc = m.anc(len(R1), "cp"), m.anc(len(R1), "sc")
        A.csub_const(ctx, f[0], R1, q, cp, sc)
        m.free(cp, sc)
        ctx.cx(R1[len(R1) - 1], f[0])                 # f was exactly the sign
        m.free(f)

    @staticmethod
    def _walk_body(m, R0, R1, recs):
        ctx, W = m.ctx, len(R0)
        for i, e in enumerate(recs):
            s, t = (R0, R1) if i % 2 == 0 else (R1, R0)
            ctx.cx(t[1], e[0])
            ctx.cx(s[1], e[0])                        # e = t[1] ^ s[1]
            anc = m.anc(W - 1, "pp")
            A.ci_add(ctx, e[0], s, t, anc)            # t += (-1)^e s
            m.free(anc)
            for j in range(W - 1):                    # t /= 2 (t even) ...
                ctx.swap(t[j], t[j + 1])
            ctx.cx(t[W - 2], t[W - 1])                # ... keeping the sign

    def _registers(self, m, x):
        W = len(x) + 2
        R0 = m.anc(W, "r0")
        hi = m.anc(2, "r1hi")
        return R0, hi, Reg(list(x) + list(hi), "r1")

    @staticmethod
    def _clear(m, R0, R1, eta):
        """(eta0, eta1) = (+-1, +-1) -> (0, 0); eta0/eta1 hold the signs."""
        for R, et in ((R0, eta[0]), (R1, eta[1])):
            m.ctx.x(R[0])
            for j in range(1, len(R)):
                m.ctx.cx(et[0], R[j])

    def record(self, m, x, q):
        n = len(x)
        R0, hi, R1 = self._registers(m, x)
        A.encode_const(m.ctx, R0, q)
        self._lift(m, R1, q)
        recs = [m.anc(1, "e") for _ in range(self.L(n))]
        self._walk_body(m, R0, R1, recs)
        eta = (m.anc(1, "eta0"), m.anc(1, "eta1"))
        m.ctx.cx(R0[-1], eta[0][0])
        m.ctx.cx(R1[-1], eta[1][0])                   # the terminal signs
        self._clear(m, R0, R1, eta)                   # not converged: not |0>
        m.free(R0, hi)
        return recs, eta

    def unrecord(self, m, x, q, recs, eta):
        R0, hi, R1 = self._registers(m, x)
        self._clear(m, R0, R1, eta)                   # back to (eta0, eta1)
        m.ctx.cx(R0[-1], eta[0][0])
        m.ctx.cx(R1[-1], eta[1][0])
        m.free(*eta)
        m.emit_inverse(self._walk_body, m, R0, R1, recs)
        m.emit_inverse(self._lift, m, R1, q)
        A.encode_const(m.ctx, R0, q)
        m.free(R0, hi)
        for e in recs:
            m.free(e)

    # -- division and multiplication ---------------------------------------------------
    def div(self, m, x, z, q):
        """|x, z> -> |x, z / x mod q>."""
        arith = self.arith or Exact(q)
        neg = self.neg or (lambda mm, c, v: MA.cmodneg(mm, c, v, q))
        n = len(x)
        recs, eta = self.record(m, x, q)
        c0 = m.anc(n, "c0")
        c1 = z
        for i, e in enumerate(recs):
            s, t = (c0, c1) if i % 2 == 0 else (c1, c0)
            arith.signadd(m, e[0], s, t)              # t += (-1)^e s
            arith.half(m, t)                          # t /= 2
        neg(m, eta[0][0], c0)                         # undo the terminal signs
        neg(m, eta[1][0], c1)
        MA.modsub(m, c1, c0, q)                       # two copies of z / x: clear one
        m.free(c0)
        self.unrecord(m, x, q, recs, eta)

    def mul(self, m, x, y, q):
        m.emit_inverse(self.div, m, x, y, q)


# =============================================================================
# ECDSA.Fail Sec 5.3.1-5.3.2: Jump-2 macro-steps and the base-5 transcript
# =============================================================================
class Jump2:
    """ECDSA.Fail Algorithm 4: up to two halvings per macro-step.

    Macro-step i (i > 0): v /= 2 (v is even at every boundary, so this is a
    relabelling), then s2 = [v even] and v /= 2 if s2, then b = v mod 2,
    s = b AND [v < u], swap if s, v -= u if b.  Step 0 replaces the first
    halving by a conditional one (t1 = [x even]) and takes s = b (x < q).
    Record sigma = (b, s, s2) -- only five of the eight patterns occur -- plus
    t1.  The replay, reverse order, is  if b: z += y;  if s: swap;
    z <- 2^k z  with k = 1 + s2 (k = t1 + s2 at step 0).

    Fewer, fatter steps: ~1.02n-1.3n macro-steps (measured worst case 1.3n
    at n = 16) instead of 1.41n + O(sqrt n) rounds, each with one extra
    conditional shift.  ECDSA.Fail measured the walk at 624k CCX against 729k
    for single steps; its replay needs a controlled doubling per step.
    """

    def __init__(self, arith=None, steps=None, c_steps=1.35):
        self.arith, self._steps, self.c_steps = arith, steps, c_steps

    def L(self, n):
        return self._steps or math.ceil(self.c_steps * n) + 2

    @staticmethod
    def _chalve(ctx, c, V):
        for j in range(len(V) - 1):
            ctx.cswap(c, V[j], V[j + 1])

    def _walk_body(self, m, U, V, recs, t1):
        ctx, w = m.ctx, len(U)
        for i, (b, s, s2) in enumerate(recs):
            if i == 0:
                ctx.cx(V[0], t1[0])
                ctx.x(t1[0])                             # t1 = [x even]
                self._chalve(ctx, t1[0], V)
            else:
                for j in range(w - 1):                   # v /= 2: relabel
                    ctx.swap(V[j], V[j + 1])
            ctx.cx(V[0], s2[0])
            ctx.x(s2[0])                                 # s2 = [v even]
            self._chalve(ctx, s2[0], V)
            ctx.cx(V[0], b[0])                           # b = v odd
            if i == 0:
                ctx.cx(b[0], s[0])
            else:
                sc = m.anc(w, "sc")
                A.cgt_fused(ctx, b[0], U, V, s[0], sc)   # s = b AND [u > v]
                m.free(sc)
            for j in range(w):
                ctx.cswap(s[0], U[j], V[j])
            cp, sc = m.anc(w, "cp"), m.anc(w, "sc")
            A.csub(ctx, b[0], U, V, cp, sc)              # if b: v -= u
            m.free(cp, sc)

    def _registers(self, m, x):
        hi = m.anc(1, "vhi")
        U = m.anc(len(x) + 1, "u")
        return U, hi, Reg(list(x) + list(hi), "v")

    def record(self, m, x, q):
        U, hi, V = self._registers(m, x)
        A.encode_const(m.ctx, U, q)
        recs = [tuple(m.anc(1, nm) for nm in ("b", "s", "s2")) for _ in range(self.L(len(x)))]
        t1 = m.anc(1, "t1")
        self._walk_body(m, U, V, recs, t1)
        m.ctx.x(U[0])                                    # (u, v) = (1, 0)
        m.free(U, hi)
        return recs, t1

    def unrecord(self, m, x, q, recs, t1):
        U, hi, V = self._registers(m, x)
        m.ctx.x(U[0])
        m.emit_inverse(self._walk_body, m, U, V, recs, t1)
        A.encode_const(m.ctx, U, q)
        m.free(U, hi, t1)
        for rec in recs:
            m.free(*rec)

    def mul(self, m, x, y, q):
        arith = self.arith or Exact(q)
        n = len(x)
        recs, t1 = self.record(m, x, q)
        z = m.anc(n, "z")
        for i in reversed(range(len(recs))):
            b, s, s2 = recs[i]
            arith.cadd(m, b[0], y, z)                    # if b: z += y
            for a_, c_ in zip(y, z):
                m.ctx.cswap(s[0], a_, c_)                # if s: swap
            if i == 0:
                MA.cmoddbl(m, t1[0], z, q)               # z <- 2^(t1 + s2) z
            else:
                arith.dbl(m, z)                          # z <- 2^(1 + s2) z
            MA.cmoddbl(m, s2[0], z, q)
        for a_, c_ in zip(y, z):
            m.ctx.swap(a_, c_)
        m.free(z)
        self.unrecord(m, x, q, recs, t1)

    def div(self, m, x, y, q):
        m.emit_inverse(self.mul, m, x, y, q)


JUMP2_SYMBOLS = [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]


def _base5_perm(k):
    """Permutation of 3k qubits packing k symbols (b, s, s2) into the low
    ceil(log2 5^k) qubits as a base-5 number; the rest come out |0>."""
    import itertools
    codes = {}
    for syms in itertools.product(range(5), repeat=k):
        src = 0
        for j, d in enumerate(syms):
            b, s, s2 = JUMP2_SYMBOLS[d]
            src |= (b | (s << 1) | (s2 << 2)) << (3 * j)
        codes[src] = sum(d * 5 ** j for j, d in enumerate(syms))
    perm = dict(codes)
    spare_in = [i for i in range(1 << (3 * k)) if i not in perm]
    spare_out = [i for i in range(1 << (3 * k)) if i not in set(perm.values())]
    perm.update(zip(spare_in, spare_out))
    return perm


def base5_pack(m, qubits):
    """ECDSA.Fail Sec 5.3.2: 3 symbols (9 qubits) -> 7, or 2 (6) -> 5.
    Returns the freed qubits (|0> on every reachable input)."""
    from ec_shor import permutation
    k = len(qubits) // 3
    assert len(qubits) == 3 * k and k in (2, 3)
    permutation(m.qc, list(qubits), _base5_perm(k))
    keep = math.ceil(math.log2(5 ** k))
    return list(qubits[keep:])


def base5_record_qubits(steps):
    """Transcript qubits with the codec: 7 per 3 symbols, 5 per 2."""
    q3, r = divmod(steps, 3)
    return 7 * q3 + (5 if r == 2 else 3 * r)
