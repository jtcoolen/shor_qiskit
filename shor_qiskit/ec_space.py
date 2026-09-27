"""Space-optimised pseudo-Mersenne arithmetic: [1128]'s qubit-lean variant.

The circuits in `ec_approx` follow [1128]'s *gate*-optimised choices: a
controlled addition copies its operand into n clean qubits under the control
and then adds with Gidney's adder, which wants n - 1 more -- 2n + 1 qubits of
scratch for 2n Toffolis.  Inside the Bezout replay that scratch is live at the
moment the whole point addition peaks, next to the full GCD record (measured:
~2.8n of the ~6.8n ancillas at n = 64).  [1128] Sec 3.2's space-optimised
variant spends Toffolis to get it back, and so does this module:

  controlled addition   CDKM with the control on its UMA gates only
                        (`cdkm_cadd`): ONE carry qubit, 3n Toffolis instead
                        of 2n
  constant addition     the constant written into its lsbs-bit register, then
                        CDKM: one carry qubit instead of lsbs - 1

Everything else -- the top-bit comparisons, the x + y = q test, the
pseudo-Mersenne fold -- is [1128]'s Algorithm 7 / 11 unchanged, so the answers
are the same input for input (the tests check that); only the scratch and the
Toffoli count differ.  `Arith` wrapper: `ec_gcd.PMSpace`.

`lean=True` goes further, for the Bezout replay, where the whole point
addition peaks ([1128] Sec 3: record + two n-bit registers + scratch, and the
scratch is the only part left to squeeze):

  comparison            CDKM's MAJ chain, read and reversed: one ancilla
                        instead of msbs, 2 msbs Toffolis instead of msbs
  all-ones test         in chunks of ~sqrt(k), each chunk's AND parked in a
                        flag: ~2 sqrt(k) ancillas instead of k - 1, ~2k ANDs
  constant addition     CDKM on the low bitlen(f) bits only; the carry into
                        the rest is added as an *increment*, which borrows
                        dirty qubits ([Gid15]: w - g - NOT(g) = w + 1):
                        bitlen(f) + 3 clean qubits instead of lsbs + 1,
                        2 bitlen(f) + 4 (lsbs - bitlen(f) + 1) Toffolis

Same answers again, input for input (tested).
"""

import math

import ec_adders as A
import ec_approx as AX
from ec_sim import Reg
from rc_adder import _maj, _uma


# =============================================================================
# Lean building blocks
# =============================================================================
def _cuma(ctx, ctrl, c, b, a):
    """UMA that writes the sum bit only when ctrl.  On entry (after an
    *uncontrolled* MAJ) b = y ^ x, c = c_i ^ x, a = c_{i+1}."""
    ctx.ccx(c, b, a)                     # a <- x
    ctx.ccx(ctrl, c, b)                  # b ^= ctrl (c_i ^ x)
    ctx.cx(a, c)                         # c <- c_i
    ctx.cx(a, b)                         # b = y ^ ctrl (x ^ c_i)


def cdkm_cadd(ctx, ctrl, x, y, carry):
    """y += x (mod 2^n) when ctrl, one clean carry qubit: 3n Toffolis.

    The carries of x + y are computed unconditionally (plain MAJ chain) and
    unwound by a UMA that writes the sum only under the control -- the
    textbook 3n.  (`ec_adders.cdkm_add` with ctrls puts the control on both
    the MAJ and the UMA: 4n.)"""
    n = len(y)
    assert len(x) == n and n >= 1
    _maj(ctx, carry, y[0], x[0])
    for i in range(n - 1):
        _maj(ctx, x[i], y[i + 1], x[i + 1])
    for i in range(n - 1, 0, -1):
        _cuma(ctx, ctrl, x[i - 1], y[i], x[i])
    _cuma(ctx, ctrl, carry, y[0], x[0])


def _maj_dg(ctx, c, b, a):
    ctx.ccx(c, b, a)
    ctx.cx(a, c)
    ctx.cx(a, b)


def carry_out_cdkm(ctx, x, y, out, c):
    """out ^= carry-out of x + y; x, y restored; one clean ancilla c.
    The MAJ chain of [CDKM04] leaves the carry in x's top qubit: read it and
    run the chain back.  2n Toffolis."""
    n = len(x)
    assert len(y) == n and n >= 1
    _maj(ctx, c, y[0], x[0])
    for i in range(n - 1):
        _maj(ctx, x[i], y[i + 1], x[i + 1])
    ctx.cx(x[n - 1], out)
    for i in range(n - 2, -1, -1):
        _maj_dg(ctx, x[i], y[i + 1], x[i + 1])
    _maj_dg(ctx, c, y[0], x[0])


def lt_cdkm(m, x, y, out):
    """out ^= [x < y], one clean ancilla ([y > x] = carry of y + NOT x)."""
    ctx = m.ctx
    c = m.anc(1, "cmp")
    for b in x:
        ctx.x(b)
    carry_out_cdkm(ctx, y, x, out, c[0])
    for b in x:
        ctx.x(b)
    m.free(c)


def _chunk(k):
    """Chunk size minimising the peak of `all_ones_lean`."""
    best = None
    for s in range(2, k + 1):
        g = -(-k // s)
        peak = max(s - 1 + g, 2 * g - 1)
        if best is None or peak < best[0]:
            best = (peak, s)
    return best[1]


def all_ones_lean(m, qs, outs):
    """out ^= AND(qs) for every out, with ~2 sqrt(k) clean ancillas.

    Each chunk's AND is computed (temporary ANDs), copied into a flag and the
    chain uncomputed for free; the flags are ANDed into the outputs; each flag
    is cleared by computing its chunk again.  ~2k ANDs where `ec_approx.
    all_ones` spends k - 1 ANDs and k - 1 ancillas."""
    qs = list(qs)
    k = len(qs)
    s = _chunk(k) if k > 3 else k
    g = -(-k // s)
    if max(s - 1 + g, 2 * g - 1) >= k - 1:       # no saving at this size
        AX.all_ones(m, qs, outs)
        return
    chunks = [qs[i:i + s] for i in range(0, k, s)]
    multi = [c for c in chunks if len(c) > 1]
    flags = m.anc(len(multi), "fl")
    fl = iter(flags)
    terms, jobs = [], []
    for c in chunks:
        if len(c) == 1:
            terms.append(c[0])
        else:
            f = next(fl)
            terms.append(f)
            jobs.append((c, f))
    for c, f in jobs:
        AX.all_ones(m, c, [f])
    AX.all_ones(m, terms, outs)
    for c, f in jobs:
        AX.all_ones(m, c, [f])
    m.free(flags)


def eq_top_lean(m, reg, value, k, outs, also=()):
    """`ec_approx.eq_top` on `all_ones_lean`."""
    W = len(reg)
    k = min(k, W)
    top, c = AX.top_bits(reg, k), AX.top_const(value, W, k)
    flip = [top[i] for i in range(k) if not (c >> i) & 1]
    for qb in flip:
        m.ctx.x(qb)
    all_ones_lean(m, list(also) + list(top), outs)
    for qb in flip:
        m.ctx.x(qb)


def inc_borrowed(m, w, g):
    """w += 1 (mod 2^len(w)) with len(w) *borrowed* qubits g -- any state,
    returned unchanged -- and one clean qubit: w - g - NOT(g) = w + 1 - 2^k.
    Two CDKM subtractions, 4 len(w) Toffolis ([Gid15])."""
    ctx, k = m.ctx, len(w)
    g = list(g)[:k]
    assert len(g) == k and not set(g) & set(w), "need len(w) disjoint borrowed qubits"
    cy = m.anc(1, "icy")
    A.sub(ctx, Reg(g), Reg(list(w)), cy)
    for q in g:
        ctx.x(q)
    A.sub(ctx, Reg(g), Reg(list(w)), cy)
    for q in g:
        ctx.x(q)
    m.free(cy)


def _cadd_const_lean(m, ctrl, y, k, borrow=()):
    """y += ctrl * k (mod 2^len(y)) with bitlen(k) + 3 clean qubits.

    CDKM with k written into a bitlen(k)-qubit register, on y's low bitlen(k)
    bits; at the top of its MAJ chain the carry into bit bitlen(k) sits in the
    register's top qubit, and adding it to y's high bits is an increment of
    [carry, y_high] (then NOT the carry back) -- done on borrowed qubits: the
    idle half of the MAJ chain, and `borrow`.  Falls back to `_cadd_const`
    when there are too few."""
    ctx, L, b = m.ctx, len(y), k.bit_length()
    h = L - b
    if h <= 0:
        _cadd_const(m, ctrl, y, k)
        return
    creg, cy = m.anc(b, "kc"), m.anc(1, "kcy")
    w = [creg[b - 1]] + list(y[b:])
    dirty = [q for q in list(creg[:b - 1]) + list(y[:b]) + [cy[0]] + list(borrow)
             if q not in w and q is not ctrl]
    if len(dirty) < h + 1:
        m.free(creg, cy)
        _cadd_const(m, ctrl, y, k)
        return
    A.encode_const(ctx, creg, k, ctrl)
    _maj(ctx, cy[0], y[0], creg[0])
    for i in range(b - 1):
        _maj(ctx, creg[i], y[i + 1], creg[i + 1])
    inc_borrowed(m, w, dirty)                     # [c_b, y_high] += 1
    ctx.x(creg[b - 1])                            # ... = y_high += c_b
    for i in range(b - 1, 0, -1):
        _uma(ctx, creg[i - 1], y[i], creg[i])
    _uma(ctx, cy[0], y[0], creg[0])
    A.encode_const(ctx, creg, k, ctrl)
    m.free(creg, cy)


def _cadd_const(m, ctrl, y, k):
    """y += ctrl * k (mod 2^len(y)) with a len(y)-qubit constant register and
    one carry: CDKM instead of Gidney's len(y) - 1 ancillas."""
    ctx = m.ctx
    creg, carry = m.anc(len(y), "kc"), m.anc(1, "kcy")
    A.encode_const(ctx, creg, k, ctrl)
    A.cdkm_add(ctx, creg, y, carry[0])
    A.encode_const(ctx, creg, k, ctrl)
    m.free(creg, carry)


def moddbl_pm_space(m, x, q, lsbs=None, lean=False):
    """`ec_approx.moddbl_pm` with a CDKM constant adder."""
    u, f = AX.pseudo_mersenne(q)
    ctx, n = m.ctx, len(x)
    assert u == n
    lsbs = lsbs or min(n, 2 * max(1, f.bit_length()) + 8)
    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])
    if lean:
        _cadd_const_lean(m, xe[n], Reg(list(xe[:lsbs])), f, borrow=list(xe[lsbs:n]))
    else:
        _cadd_const(m, xe[n], Reg(list(xe[:lsbs])), f)
    ctx.cx(xe[0], xe[n])
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
    m.free(z)


def modhalf_pm_space(m, x, q, lsbs=None, lean=False):
    m.emit_inverse(moddbl_pm_space, m, x, q, lsbs, lean)


def cmodadd_pm_q_space(m, ctrl, x, y, q, lsbs=None, msbs=None, lean=False):
    """`ec_approx.cmodadd_pm_q` (Algorithm 11) with CDKM adders: the controlled
    addition needs one carry qubit instead of 2n + 1 qubits of scratch."""
    u, f = AX.pseudo_mersenne(q)
    ctx, n = m.ctx, len(y)
    assert u == n
    msbs = msbs or max(2, n // 2)
    lsbs = lsbs or min(n, 2 * max(1, f.bit_length()) + 8)
    eq = eq_top_lean if lean else AX.eq_top

    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    xe, ye = x + ax, y + ay
    carry = m.anc(1, "cy")
    if ctrl is None:
        A.cdkm_add(ctx, xe, ye, carry[0])
    else:
        cdkm_cadd(ctx, ctrl, xe, ye, carry[0])
    m.free(carry)

    e = m.anc(1, "e")
    on = [] if ctrl is None else [ctrl]
    eq(m, y, q, msbs, [e[0]], also=on)
    for i in range(n):
        if (q >> i) & 1:
            ctx.cx(e[0], y[i])
    if lean:
        _cadd_const_lean(m, ay[0], Reg(list(y[:lsbs])), f,
                         borrow=list(x) + list(y[lsbs:]))
    else:
        _cadd_const(m, ay[0], Reg(list(y[:lsbs])), f)

    t = m.anc(1, "t")
    ty, tx = AX.top_bits(y, msbs), AX.top_bits(x, msbs)
    if lean:
        lt_cdkm(m, ty, tx, t[0])
    else:
        sc = m.anc(msbs, "sc")
        A.lt_uint(ctx, ty, tx, t[0], sc)
    if ctrl is None:
        c = t
    else:
        c = m.anc(1, "c")
        ctx.and_(ctrl, t[0], c[0])
    ctx.cx(c[0], ay[0])
    eq(m, y, 0, msbs, [e[0], ay[0]], also=[c[0]])
    if ctrl is not None:
        ctx.and_dg(ctrl, t[0], c[0])
        m.free(c)
    if lean:
        lt_cdkm(m, ty, tx, t[0])
        m.free(t, e, ax, ay)
    else:
        A.lt_uint(ctx, ty, tx, t[0], sc)
        m.free(t, sc, e, ax, ay)


def _zero_q(m, ctrl, y, q, tau, first, lean):
    """`ec_approx._swap_zero_q` (first = 0) / `_fix_zero_q` (first = q)."""
    eq = eq_top_lean if lean else AX.eq_top
    fl = m.anc(1, "zq")
    on = [] if ctrl is None else [ctrl]
    eq(m, y, first, tau, [fl[0]], also=on)
    for i in range(len(y)):
        if (q >> i) & 1:
            m.ctx.cx(fl[0], y[i])
    eq(m, y, q - first, tau, [fl[0]], also=on)
    m.free(fl)


def csignadd_pm_space(m, e, x, y, q, lsbs=None, msbs=None, lean=False):
    """`ec_approx.csignadd_pm` on the space-optimised adder."""
    n = len(y)
    msbs = msbs or max(2, n // 2)
    if lean:
        _zero_q(m, e, y, q, msbs, 0, True)
    else:
        AX._swap_zero_q(m, e, y, q, msbs)
    for b in y:
        m.ctx.cx(e, b)
    cmodadd_pm_q_space(m, None, x, y, q, lsbs, msbs, lean)
    for b in y:
        m.ctx.cx(e, b)
    if lean:
        _zero_q(m, e, y, q, msbs, q, True)
    else:
        AX._fix_zero_q(m, e, y, q, msbs)


def csub_square_pm_space(m, ctrl, src, acc, q, lsbs=None, msbs=None, sqr_space=False,
                         lean=False):
    """`ec_square.csub_square_pm` on the CDKM cells: acc -= src^2 mod q.

    The square itself (2n + 1 qubits) is unavoidable in this form; what goes
    is the exact modular subtraction's 2n + 2 qubits of scratch, which sat on
    top of it at the addition's peak.  `sqr_space` also gives the squarer's own
    subtractions one ancilla each (CDKM) instead of up to n.

    The subtractions are the pseudo-Mersenne addition run backwards, so on top
    of the fold's f/2^n approximation they fail when acc < f on entry (see
    `ec_gcd.Exact.sub`): ~f/q, negligible at secp256k1, measured at toy q."""
    import ec_square as SQ
    u, f = AX.pseudo_mersenne(q)
    n = len(src)
    assert u == n
    z = m.anc(2 * n + 1, "z")
    SQ.sqr_int(m, src, z, space=sqr_space)
    lo, hi = Reg(z[:n], "zlo"), Reg(z[n:2 * n], "zhi")

    def sub(mm, c, v, a):
        mm.emit_inverse(cmodadd_pm_q_space, mm, c, v, a, q, lsbs, msbs, lean)

    sub(m, ctrl, lo, acc)
    SQ._csub_times_const(m, ctrl, hi, f, acc, q,
                         lambda mm, v: moddbl_pm_space(mm, v, q, lsbs, lean),
                         lambda mm, v: modhalf_pm_space(mm, v, q, lsbs, lean), sub)
    m.emit_inverse(SQ.sqr_int, m, src, z, sqr_space)
    m.free(z)
