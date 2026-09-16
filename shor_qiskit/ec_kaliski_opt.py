"""Kaliski inversion with the three optimizations of [106] Sec 3.3.

Against `ec_kaliski`, which is [HJN+20]:

1. Unconditional execution.  Drop the "is v still nonzero?" test and run all 2n
   rounds regardless.  An inert round does nothing except double r, so instead
   of stopping at -x^{-1} 2^k the register runs on to -x^{-1} 2^{2n} -- the same
   value for every input, with no counter to keep and no 2n-k corrective
   halvings to perform.  And 2^{2n} is exactly right: fed x = X 2^n in
   Montgomery form, the answer is X^{-1} 2^n, which is the Montgomery form of
   X^{-1}.  The correction [HJN+20] pays for is already done.

   It also removes a bit of per-round bookkeeping.  The reference keeps three
   bits per round (f, a, b); f only exists to record whether the round fired,
   so without the test there are two.

2. Postponed modular reduction.  Do not reduce r and s mod p each round.  Let
   them run into a 2n-bit register and reduce once, at the end.  This is the
   big one, because it changes what the round *is*: a modular doubling (compare,
   conditionally subtract, clear the flag) becomes a plain left shift, which is
   a rotation of wires and costs no Toffoli at all, and a modular addition
   becomes a plain addition.

   2n bits is not a comfortable margin, it is the exact requirement -- r reaches
   6 bits for p = 7, 32 for p = 65521.  The test measures it.

3. Low-depth round.  Fan the swap-decision bit out into several copies so the
   controlled swaps run in parallel instead of queueing on one control, and use
   [106] Fig. 4(b) copy-then-add for the controlled arithmetic.  This does not
   change the gate count -- it is bought with qubits, and [106] takes those from
   the scratch the following multiplication is not yet using.
"""

import ec_adders as A
from ec_montgomery import _reduce_below_p


def _rot_up(ctx, reg):
    """reg <- 2*reg in a fixed-width register.  The top bit must be 0.

    W-1 SWAPs, no Toffoli -- this is optimization 2 in one line.  The modular
    version has to compare against p and conditionally subtract; this does not.
    """
    for i in range(len(reg) - 1, 0, -1):
        ctx.swap(reg[i], reg[i - 1])


def _rot_down(ctx, reg):
    """reg <- reg/2.  The low bit must be 0."""
    for i in range(len(reg) - 1):
        ctx.swap(reg[i], reg[i + 1])


def kaliski_round_opt(m, u, v, r, s, rec, fanout=4):
    """One unconditional round.  `rec` = (a, b), two clean qubits kept."""
    ctx, n, W = m.ctx, len(u), len(r)
    a, b = rec

    # --- decision bits (no f: every round is active) ----------------------
    t, sc = m.anc(1, "t"), m.anc(n, "sc")
    A.gt_uint(ctx, u, v, t[0], sc)
    w_ = m.anc(1, "w")
    ctx.x(t[0])
    ctx.and_(u[0], t[0], w_[0])
    ctx.x(t[0])
    ctx.x(w_[0])                                  # w = NOT u0 OR (u > v)
    ctx.and_(v[0], w_[0], a[0])                   # a = v0 AND w
    ctx.and_(v[0], u[0], b[0])                    # b = u0 AND v0
    ctx.x(w_[0])
    ctx.x(t[0])
    ctx.and_dg(u[0], t[0], w_[0])
    ctx.x(t[0])
    m.free(w_)
    A.gt_uint(ctx, u, v, t[0], sc)
    m.free(t, sc)

    # --- optimization 3: fan the control out so the swaps parallelise -----
    k = max(1, fanout)
    ac = m.anc(k, "ac")
    for q in ac:
        ctx.cx(a[0], q)

    def swap_all(x, y):
        for i in range(len(x)):
            ctx.cswap(ac[i % k], x[i], y[i])

    swap_all(u, v)
    swap_all(r, s)

    # --- the body, all non-modular ---------------------------------------
    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b[0], u, v, cp, sc)               # v -= u
    m.free(cp, sc)
    cp, sc = m.anc(W, "cp"), m.anc(W, "sc")
    A.cadd(ctx, b[0], r, s, cp, sc)               # s += r   (plain, not mod p)
    m.free(cp, sc)

    _rot_down(ctx, v)                             # v /= 2
    _rot_up(ctx, r)                               # r *= 2   (free: no reduction)

    swap_all(u, v)
    swap_all(r, s)
    for q in ac:
        ctx.cx(a[0], q)
    m.free(ac)


def kaliski_pseudo_inverse_opt(m, x, p, rounds=None, fanout=4, ctrl=None):
    """Run 2n unconditional rounds.  Returns (r, records, u, v, s).

    r is a 2n-bit register holding the unreduced -x^{-1} 2^{2n} (mod p only
    after reduction).  Everything returned is garbage for the caller to undo.
    """
    ctx, n = m.ctx, len(x)
    rounds = rounds or 2 * n
    W = 2 * n
    u, v = m.anc(n, "u"), m.anc(n, "v")
    r, s = m.anc(W, "r"), m.anc(W, "s")
    A.encode_const(ctx, u, p, ctrl)
    for i in range(n):
        if ctrl is None:
            ctx.cx(x[i], v[i])
        else:
            ctx.and_(ctrl, x[i], v[i])
    if ctrl is None:
        ctx.x(s[0])
    else:
        ctx.cx(ctrl, s[0])

    recs = []
    for _ in range(rounds):
        rec = (m.anc(1, "a"), m.anc(1, "b"))
        kaliski_round_opt(m, u, v, r, s, rec, fanout)
        recs.append(rec)
    return r, recs, u, v, s


def mod_inv_mont(m, x, out, p, fanout=4, ctrl=None):
    """out (|0>) <- (x^{-1} * 2^{2n}) mod p.  Returns the garbage.

    Fed x in Montgomery form (x = X 2^n) this is X^{-1} 2^n, i.e. the Montgomery
    form of X^{-1} -- no counter, no corrective doublings.

    On the ctrl = 0 branch nothing happens at all: v starts at 0, so every round
    is inert, r stays 0, and the reduction of 0 is 0.
    """
    n = len(x)
    r, recs, u, v, s = kaliski_pseudo_inverse_opt(m, x, p, 2 * n, fanout, ctrl)

    # u = 1 and s = p on termination (both 0 on the inactive branch)
    if ctrl is None:
        m.ctx.x(u[0])
    else:
        m.ctx.cx(ctrl, u[0])
    m.free(u, v)
    A.encode_const(m.ctx, s, p, ctrl)
    m.free(s)

    flags = _reduce_below_p(m, r, p, len(r) - p.bit_length())   # the one reduction
    for i in range(len(out)):
        m.ctx.cx(r[i], out[i])
    import ec_modarith as MA
    MA.modneg(m, out, p)                                       # kill the sign
    return [r] + flags + [q for rec in recs for q in rec]


def mod_inv_mont_clean(m, x, out, p, fanout=4, ctrl=None):
    """`mod_inv_mont` with every scratch qubit returned to |0>."""
    n = len(x)
    tmp = m.anc(n, "it")
    start = len(m.qc.data)
    garbage = mod_inv_mont(m, x, tmp, p, fanout, ctrl)
    body = list(m.qc.data[start:])
    for i in range(len(out)):
        m.ctx.cx(tmp[i], out[i])
    for ci in reversed(body):
        m.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)
    m.free(tmp, *garbage)
