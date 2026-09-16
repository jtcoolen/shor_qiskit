"""Modular inversion by Kaliski's algorithm -- the reference implementation.

This is [106] Algorithm 2, the swap-based reformulation [HJN+20] gave of the
binary extended Euclidean algorithm, which in turn is what [RNSL17] first put
on a quantum computer.  It is the expensive part of an affine point addition:
[106] Table 3 has division dominating everything else.

Why the algorithm is shaped the way it is
-----------------------------------------
Classical Kaliski branches four ways per step (u even / v even / both odd with
u > v / both odd with u <= v).  A quantum circuit cannot branch, so [HJN+20]
rewrite it so that *every* round runs the same straight-line sequence

    maybe swap (u,v) and (r,s);  maybe v -= u and s += r;  v /= 2;  r *= 2;
    maybe swap back

and the four classical cases fall out of the two decision bits.  Checking the
four cases against this template is the first thing `ec_classical.kaliski_round`
does, and the tests check the circuit against that.

What has to be kept
-------------------
Three bits per round, and they cannot be avoided in the conditional version:

  f  the round did something at all (v != 0 on entry).  Not recoverable
     afterwards -- an inert round and the final active round both leave v = 0.
  a  the swap happened.
  b  both were odd, so the subtract-and-add fired.  Not implied by (f, a):
     f=1, a=0 covers both "v was even" and "both odd with u <= v".

The comparison u > v is *not* kept: it is consumed while building `a` and then
uncomputed by running the comparator again, before u and v change.

Registers
---------
u, v are plain n-bit integers (they stay in [0, p]).  r, s are reduced mod p
every round -- they have to be, since r doubles each round and would otherwise
run past p; the tests check both facts.  The algorithm's own progress does not
depend on r and s, so reducing them is free of consequences.

The result is a *pseudo*-inverse: r = -x^{-1} 2^k mod p, where k counts the
active rounds.  `mod_inv` fixes the sign and the power of two.
"""

import ec_adders as A
import ec_modarith as MA
from ec_sim import Reg


def _shift_down(ctx, v):
    """v <- v/2 in place.  v's LSB must be 0.  n-1 SWAPs, no Toffoli."""
    for i in range(len(v) - 1):
        ctx.swap(v[i], v[i + 1])


def kaliski_round(m, u, v, r, s, p, rec, conditional=True):
    """One round of [106] Algorithm 2.  `rec` = (f, a, b), three clean qubits."""
    ctx, n = m.ctx, len(u)
    f, a, b = rec

    # --- f: is this round active? ---------------------------------------
    if conditional:
        MA.is_nonzero(m, v, f[0])
    else:
        ctx.x(f[0])                       # [106] Sec 3.3: never test v != 0

    # --- decision bits, from the pre-round u and v ------------------------
    t, sc = m.anc(1, "t"), m.anc(n, "sc")
    A.gt_uint(ctx, u, v, t[0], sc)        # t = [u > v]

    # a = f AND v0 AND (NOT u0 OR t)   -- swap when u is even and v odd, or
    #                                     both odd with u > v
    w = m.anc(1, "w")
    ctx.x(t[0])
    ctx.and_(u[0], t[0], w[0])            # w = u0 AND NOT t
    ctx.x(t[0])
    ctx.x(w[0])                           # w = NOT(u0 AND NOT t)
    fv = m.anc(1, "fv")
    ctx.and_(f[0], v[0], fv[0])           # fv = f AND v0
    ctx.and_(fv[0], w[0], a[0])
    ctx.and_(fv[0], u[0], b[0])           # b = f AND u0 AND v0
    ctx.and_dg(f[0], v[0], fv[0])
    m.free(fv)
    ctx.x(w[0])
    ctx.x(t[0])
    ctx.and_dg(u[0], t[0], w[0])
    ctx.x(t[0])
    m.free(w)
    A.gt_uint(ctx, u, v, t[0], sc)        # uncompute the comparison
    m.free(t, sc)

    # --- the straight-line body ------------------------------------------
    for i in range(n):
        ctx.cswap(a[0], u[i], v[i])
    for i in range(len(r)):
        ctx.cswap(a[0], r[i], s[i])

    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b[0], u, v, cp, sc)       # v -= u  (no borrow: u <= v here)
    m.free(cp, sc)
    MA.cmodadd(m, b[0], r, s, p)          # s += r  mod p

    _shift_down(ctx, v)                   # v /= 2  (v is even by now)
    MA.cmoddbl(m, f[0], r, p)             # r *= 2  mod p

    for i in range(n):
        ctx.cswap(a[0], u[i], v[i])
    for i in range(len(r)):
        ctx.cswap(a[0], r[i], s[i])


def kaliski_pseudo_inverse(m, x, r, p, rounds=None, conditional=True, ctrl=None):
    """r (|0>) <- -x^{-1} 2^k mod p.  Returns (records, u, v, s, counter).

    Everything returned is garbage that the caller must uncompute -- which is
    what `mod_div` does by running this backwards, exactly as [106] Fig. 7 and
    [RNSL17] before it.

    u and v come back clean on their own (u = 1 is undone by one X, v = 0), and
    so does s (it ends at p, which is 0 mod p).  The 3-bits-per-round record and
    the counter are the real garbage.

    `ctrl` makes the whole inversion conditional for the price of the *setup*.
    Load u <- p, v <- x, s <- 1 under the control and, when it is 0, every
    register starts at zero: v = 0 makes every round inert, so r stays 0, the
    counter stays 0, and the correction does nothing.  n ANDs and a handful of
    CNOTs buy a controlled inversion -- the rounds themselves are never
    controlled.

    This matters for correctness here, not just cost.  An uncontrolled inversion
    would invert whatever the register happens to hold on the q = 0 branch of a
    point addition -- which is x1 + 2 x2, and vanishes for some perfectly
    ordinary accumulators.  On input 0 this algorithm terminates with u = p and
    s = 1, not the u = 1, s == 0 mod p that `_clear_uvs` relies on to hand those
    registers back, so the cleanup would break.

    Note this is *not* what [106] does.  Its Fig. 7 runs the inversion and the
    multiplication unconditionally and gates only the copy-out of lambda, which
    is sound there because the inversion is undone by its literal inverse:
    whatever Inv did on a degenerate input, Inv-dagger undoes.  The gating here
    is the price of this implementation's cheaper cleanup, not a correction to
    the paper.
    """
    ctx, n = m.ctx, len(x)
    rounds = rounds or 2 * n
    u, v, s = m.anc(n, "u"), m.anc(n, "v"), m.anc(n, "s")
    A.encode_const(ctx, u, p, ctrl)       # u <- p
    for i in range(n):
        if ctrl is None:
            ctx.cx(x[i], v[i])            # v <- x
        else:
            ctx.and_(ctrl, x[i], v[i])
    if ctrl is None:
        ctx.x(s[0])                       # s <- 1
    else:
        ctx.cx(ctrl, s[0])

    cw = max(1, (rounds + 1).bit_length())
    cnt = m.anc(cw, "k") if conditional else None
    ccp, csc = (m.anc(cw, "ccp"), m.anc(cw, "csc")) if conditional else (None, None)

    recs = []
    for _ in range(rounds):
        rec = (m.anc(1, "f"), m.anc(1, "a"), m.anc(1, "b"))
        kaliski_round(m, u, v, r, s, p, rec, conditional)
        if conditional:
            A.cadd_const(ctx, rec[0][0], cnt, 1, ccp, csc)   # k += f
        recs.append(rec)

    if conditional:
        m.free(ccp, csc)
    return recs, u, v, s, cnt


def _clear_uvs(m, u, v, s, p, ctrl=None):
    """u = 1, v = 0, s = p == 0 mod p on termination: return them to |0>."""
    if ctrl is None:
        m.ctx.x(u[0])
    else:
        m.ctx.cx(ctrl, u[0])              # u = 1 only on the active branch
    m.free(u, v, s)


def mod_inv(m, x, out, p, conditional=True, ctrl=None):
    """out (|0>) <- x^{-1} mod p.  Returns the garbage to be uncomputed.

    Two corrections turn the pseudo-inverse into the real one:
      * 2^{-k}, applied as k modular halvings.  k lives in a counter, so each
        of the 2n halvings is controlled by [k > j] for a classical j -- a
        comparison against a constant, which is cheap.
      * the sign, one modular negation.
    """
    n = len(x)
    rounds = 2 * n
    recs, u, v, s, cnt = kaliski_pseudo_inverse(
        m, x, out, p, rounds, conditional, ctrl)
    _clear_uvs(m, u, v, s, p, ctrl)

    if conditional:
        cw = len(cnt)
        cp, sc = m.anc(cw, "cp"), m.anc(cw, "sc")
        for j in range(rounds):
            t = m.anc(1, "gt")
            A.geq_const(m.ctx, cnt, j + 1, t[0], cp, sc)     # t = [k > j]
            MA.cmodhalf(m, t[0], out, p)
            A.geq_const(m.ctx, cnt, j + 1, t[0], cp, sc)
            m.free(t)
        m.free(cp, sc)
    MA.modneg(m, out, p)
    return recs, cnt


def mod_div(m, ctrl, x, y, out, p):
    """out ^= ctrl * (y / x mod p).  [106] Fig. 7, the modular division.

    Invert x, multiply by y, copy the answer out, then undo the multiplication
    and the inversion.  Only the copy survives, so the inversion's per-round
    records and counter all come back to |0>.

    The control is pushed all the way down into the inversion's setup (see
    `kaliski_pseudo_inverse`), so on the q = 0 branch nothing runs at all --
    which both saves the copy's Toffolis and keeps the circuit away from
    inverting whatever junk the register happens to hold there.  [106] Fig. 7
    instead gates only the copy-out; see `kaliski_pseudo_inverse` for why both
    are correct in their own setting.
    """
    n = len(x)
    lam = m.anc(n, "lam")

    def build():
        inv = m.anc(n, "inv")
        recs, cnt = mod_inv(m, x, inv, p, ctrl=ctrl)
        import ec_mult as MU
        MU.modmul(m, y, inv, lam, p)
        return inv, recs, cnt

    # forward: lam = ctrl * y * x^{-1}
    start = len(m.qc.data)
    inv, recs, cnt = build()
    body = list(m.qc.data[start:])

    # lam is already 0 on the ctrl = 0 branch, so the copy needs no control
    for i in range(n):
        m.ctx.cx(lam[i], out[i])

    for ci in reversed(body):                         # undo everything else
        m.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)
    m.free(lam, inv)
    for rec in recs:
        m.free(*rec)
    if cnt is not None:
        m.free(cnt)
