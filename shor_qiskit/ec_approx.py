"""Approximate and pseudo-Mersenne modular arithmetic -- [1128] Sec 4.

Shor's algorithm only needs the point addition to work with large constant
probability, which buys a lot of freedom.  [1128] spends it in two places.

Approximate comparisons.  Every modular reduction asks "is this at least q?".
Comparing only the top few bits answers that correctly unless the top bits tie,
which happens with probability about 2^-msbs.  [1128] uses 40 to 50 bits at
n = 256.  This turns an n-bit comparison into an msbs-bit one, and -- more
importantly -- it lets the *exact* two-step reduction (subtract q, conditionally
add it back) collapse to a single controlled subtraction, because the flag can
now be produced before the subtraction rather than recovered after it.

Pseudo-Mersenne primes.  secp256k1 uses q = 2^u - f with f tiny (f = 2^32 + 977).
Then "is it at least q?" is almost "is bit u set?", and subtracting q is
"clear bit u, then add f" -- and since f is small, adding it only needs to touch
the bottom `lsbs` bits, because the carry cannot travel far.  An n-bit constant
subtraction becomes an lsbs-bit constant addition.  [1128] Table 3 attributes
essentially the whole gap between its secp256k1 and generic-prime gate counts to
this one substitution.

Both are *approximations*: the circuits below are wrong on a small, precisely
characterised set of inputs.  That is the design, not a defect -- but it means
they cannot be checked by "is it exact?".  tests/test_ec_opt1128.py measures the
actual error probability against the exact circuits in `ec_modarith` instead --
per operation, and composed into `ec_eea.inplace_mul` -- so the msbs and lsbs
budgets can be chosen from data rather than asserted.

One set of inputs is not small, though.  Algorithm 10 cannot reduce x + y = q
(the sum does not reach bit u), and inside the Bezout replay that is not a
random event: r must end at 0 and only a swap writes r, so the last swap moves
out a 0 that the addition before it produced -- as r + s = q exactly.  Every
multiplication with y != 0 hits it once, at the replay step mirroring the
dialog's first swap: v2(x)+1 steps from the *end* of the replay, the last step
whenever x is odd.  `ec_eea.bezout_replay` uses one `cadd` for every iteration,
so composed with Alg 10 the multiplier fails on every such input.  Algorithm 11
(`cmodadd_pm_q`) handles the case and is what the replay needs; Alg 10 is kept
because it is the paper's cheaper circuit and the per-operation baseline.
(An exact adder on only the last k replay steps would also work, but only with
probability 1 - 2^-k, since v2(x) is geometric.)
"""

import ec_adders as A
from ec_sim import Reg


# --- helpers -----------------------------------------------------------------
def top_bits(reg, k):
    """The top k bits of a register, as a register."""
    return Reg(list(reg[len(reg) - k:]), "top")


def top_const(value, width, k):
    """The top k bits of `value` seen as a `width`-bit number."""
    return (value % (1 << width)) >> (width - k)


def lt_approx(m, reg, q, k, out):
    """out ^= [reg < q], decided on the top k bits only.

    Correct whenever the top k bits differ; when they tie it answers "no".  So
    the error is one-sided: the circuit may treat a value below q as though it
    were above, and subtract q from it.
    """
    W = len(reg)
    k = min(k, W)
    cp, sc = m.anc(k, "cp"), m.anc(k, "sc")
    A.lt_const(m.ctx, top_bits(reg, k), top_const(q, W, k), out, cp, sc)
    m.free(cp, sc)


def pseudo_mersenne(q):
    """(u, f) with q == 2^u - f, or None if f would not be small."""
    u = q.bit_length()
    f = (1 << u) - q
    return (u, f) if f.bit_length() * 2 < u else None


def all_ones(m, qs, outs):
    """out ^= AND of the qubits `qs`, for every out in `outs`.

    A chain of len(qs)-1 temporary ANDs, read out by CNOT and uncomputed --
    `ec_modarith.is_zero` without the negation.  Extra targets are free.
    """
    ctx, k = m.ctx, len(qs)
    if k == 1:
        for o in outs:
            ctx.cx(qs[0], o)
        return
    a = m.anc(k - 1, "all")
    ctx.and_(qs[0], qs[1], a[0])
    for i in range(2, k):
        ctx.and_(a[i - 2], qs[i], a[i - 1])
    for o in outs:
        ctx.cx(a[k - 2], o)
    for i in range(k - 1, 1, -1):
        ctx.and_dg(a[i - 2], qs[i], a[i - 1])
    ctx.and_dg(qs[0], qs[1], a[0])
    m.free(a)


def eq_top(m, reg, value, k, outs, also=()):
    """out ^= [the top k bits of reg equal those of value] AND all of `also`.

    X gates turn "equals the constant" into "all ones".  For q = 2^u - f the top
    bits of q *are* all ones -- as long as k stays inside that leading run -- so
    `eq_top(m, y, q, k, ...)` is [1128]'s all(y_reg[-msbs:]) with no X gates at
    all.  With value = 0 it is the all-zeros test.
    """
    W = len(reg)
    k = min(k, W)
    top, c = top_bits(reg, k), top_const(value, W, k)
    flip = [top[i] for i in range(k) if not (c >> i) & 1]
    for qb in flip:
        m.ctx.x(qb)
    all_ones(m, list(also) + list(top), outs)
    for qb in flip:
        m.ctx.x(qb)


# --- Algorithm 6: approximate modular doubling ------------------------------
def moddbl_approx(m, x, q, msbs=None):
    """x <- 2x mod q, in place.  [1128] Algorithm 6.

    Against the exact Algorithm 5 -- subtract q, conditionally add it back --
    this decides first and subtracts once.  One controlled constant subtraction
    instead of a subtraction plus a controlled addition.
    """
    ctx, n = m.ctx, len(x)
    msbs = msbs or max(2, n // 2)
    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])                      # n+1 bits, reads as 2x
    a2 = m.anc(1, "a2")
    lt_approx(m, xe, q, msbs + 1, a2[0])          # a2 = [2x < q]  (approximately)
    ctx.x(a2[0])                                  # a2 = [2x >= q]
    cp, sc = m.anc(n + 1, "cp"), m.anc(n + 1, "sc")
    A.csub_const(ctx, a2[0], xe, q, cp, sc)
    m.free(cp, sc)
    ctx.cx(xe[0], a2[0])                     # answer odd <=> q was subtracted
    m.free(a2)
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
    m.free(z)


# --- Algorithm 7: pseudo-Mersenne modular doubling --------------------------
def moddbl_pm(m, x, q, lsbs=None):
    """x <- 2x mod q for q = 2^u - f.  [1128] Algorithm 7.

    The overflow bit of the shift *is* the comparison and *is* the flag: if it
    is set the value is at least 2^u, hence at least q.  Subtracting q is then
    "clear that bit and add f", and clearing it is what the final CNOT does.
    Nothing is left but an lsbs-bit constant addition.
    """
    pm = pseudo_mersenne(q)
    assert pm, f"q = {q} is not pseudo-Mersenne"
    u, f = pm
    ctx, n = m.ctx, len(x)
    assert u == n, "expected q just below 2^n"
    lsbs = lsbs or min(n, 2 * max(1, f.bit_length()) + 8)

    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])                      # n+1 bits; xe[n] is the overflow
    cp, sc = m.anc(lsbs, "cp"), m.anc(lsbs, "sc")
    A.cadd_const(ctx, xe[n], Reg(list(xe[:lsbs])), f, cp, sc)
    m.free(cp, sc)
    ctx.cx(xe[0], xe[n])                          # clears the bit == subtracts 2^u
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
    m.free(z)


# --- Algorithm 9: approximate controlled modular addition -------------------
def cmodadd_approx(m, ctrl, x, y, q, msbs=None):
    """y <- (y + x) mod q when ctrl.  [1128] Algorithm 9.

    Add without reducing, ask on the top bits whether the sum overflowed q, and
    subtract q once if so.  The flag clears against x exactly as in the exact
    circuit: a reduction is the only way the answer can end up below x.
    """
    ctx, n = m.ctx, len(y)
    msbs = msbs or max(2, n // 2)
    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    xe, ye = x + ax, y + ay
    cp, sc = m.anc(n + 1, "cp"), m.anc(n + 1, "sc")
    if ctrl is None:
        A.add(ctx, xe, ye, sc)
    else:
        A.cadd(ctx, ctrl, xe, ye, cp, sc)

    fl = m.anc(1, "fl")
    lt_approx(m, ye, q, msbs + 1, fl[0])
    ctx.x(fl[0])                                  # fl = [sum >= q]
    A.csub_const(ctx, fl[0], ye, q, cp, sc)
    # clear fl by comparing the answer against x, on the same top bits
    t = m.anc(1, "t")
    A.lt_uint(ctx, top_bits(y, msbs), top_bits(x, msbs), t[0], sc)
    if ctrl is None:
        ctx.cx(t[0], fl[0])
    else:
        ctx.ccx(ctrl, t[0], fl[0])
    A.lt_uint(ctx, top_bits(y, msbs), top_bits(x, msbs), t[0], sc)
    m.free(t, fl, cp, sc, ax, ay)


# --- Algorithm 10: pseudo-Mersenne controlled modular addition --------------
def cmodadd_pm(m, ctrl, x, y, q, lsbs=None, msbs=None):
    """y <- (y + x) mod q when ctrl, for q = 2^u - f.  [1128] Algorithm 10.

    Does not handle x + y == q, which the Bezout replay hits once in every
    multiplication; use `cmodadd_pm_q` there.  See the module docstring.
    """
    pm = pseudo_mersenne(q)
    assert pm, f"q = {q} is not pseudo-Mersenne"
    u, f = pm
    ctx, n = m.ctx, len(y)
    assert u == n
    msbs = msbs or max(2, n // 2)
    lsbs = lsbs or min(n, 2 * max(1, f.bit_length()) + 8)

    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    xe, ye = x + ax, y + ay
    cp, sc = m.anc(n + 1, "cp"), m.anc(n + 1, "sc")
    if ctrl is None:
        A.add(ctx, xe, ye, sc)
    else:
        A.cadd(ctx, ctrl, xe, ye, cp, sc)

    cp2, sc2 = m.anc(lsbs, "cp2"), m.anc(lsbs, "sc2")
    A.cadd_const(ctx, ay[0], Reg(list(y[:lsbs])), f, cp2, sc2)  # += f on overflow
    m.free(cp2, sc2)

    t = m.anc(1, "t")
    A.lt_uint(ctx, top_bits(y, msbs), top_bits(x, msbs), t[0], sc)
    if ctrl is None:
        ctx.cx(t[0], ay[0])
    else:
        ctx.ccx(ctrl, t[0], ay[0])
    A.lt_uint(ctx, top_bits(y, msbs), top_bits(x, msbs), t[0], sc)
    m.free(t, cp, sc, ax, ay)


# --- Algorithm 11: the same, also handling x + y == q ----------------------
def cmodadd_pm_q(m, ctrl, x, y, q, lsbs=None, msbs=None):
    """y <- (y + x) mod q when ctrl, for q = 2^u - f.  [1128] Algorithm 11.

    Algorithm 10 plus the one sum it gets wrong for certain.  x + y == q does
    not reach bit u, so Alg 10 never reduces it and leaves q where 0 belongs.
    Here it is caught by an all-ones test on the top bits -- q's top bits are
    all ones -- and q is XORed out: CNOTs, because y *is* q, so y becomes 0.

    Two departures from the paper's pseudocode, both gating a test on ctrl.
    The all-ones test is ANDed with ctrl, since without the addition y < q can
    never be q; this makes ctrl = 0 an exact identity at any msbs.  And the
    flag has to be cleared again from the output: [1128] uses "are the top
    bits of the answer all zero?", but the answer is also 0 whenever ctrl = 0
    and y = 0 -- which is the call the Bezout replay makes on each of its
    opening iterations (s = 0, b0 = 0), so the ungated form leaves the flag set
    in every multiplication.  Here the zero test is gated on ctrl AND
    [answer < x], the condition that already clears the overflow flag.  Both
    reductions make the answer drop below x; the zero test tells them apart,
    because the overflow one leaves the answer >= f > 0.

    Cost over Alg 10: the two AND chains over the top bits, about 2 msbs
    Toffoli-equivalents.  Accuracy: both top-bit tests can misfire, so at a
    given msbs the failure rate on random inputs is a few times Alg 10's --
    but still ~2^-msbs, and Alg 10 is certain to fail inside the replay.
    """
    pm = pseudo_mersenne(q)
    assert pm, f"q = {q} is not pseudo-Mersenne"
    u, f = pm
    ctx, n = m.ctx, len(y)
    assert u == n
    msbs = msbs or max(2, n // 2)
    lsbs = lsbs or min(n, 2 * max(1, f.bit_length()) + 8)

    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    xe, ye = x + ax, y + ay
    cp, sc = m.anc(n + 1, "cp"), m.anc(n + 1, "sc")
    if ctrl is None:
        A.add(ctx, xe, ye, sc)
    else:
        A.cadd(ctx, ctrl, xe, ye, cp, sc)

    e = m.anc(1, "e")
    on = [] if ctrl is None else [ctrl]           # y < q alone: never q
    eq_top(m, y, q, msbs, [e[0]], also=on)        # e = [sum == q]
    for i in range(n):
        if (q >> i) & 1:
            ctx.cx(e[0], y[i])                    # y ^= q: q becomes 0

    cp2, sc2 = m.anc(lsbs, "cp2"), m.anc(lsbs, "sc2")
    A.cadd_const(ctx, ay[0], Reg(list(y[:lsbs])), f, cp2, sc2)  # += f on overflow
    m.free(cp2, sc2)

    t = m.anc(1, "t")
    A.lt_uint(ctx, top_bits(y, msbs), top_bits(x, msbs), t[0], sc)
    if ctrl is None:
        c = t
    else:
        c = m.anc(1, "c")
        ctx.and_(ctrl, t[0], c[0])
    ctx.cx(c[0], ay[0])                           # either reduction: answer < x
    eq_top(m, y, 0, msbs, [e[0], ay[0]], also=[c[0]])   # ...and == 0: it was e
    if ctrl is not None:
        ctx.and_dg(ctrl, t[0], c[0])
        m.free(c)
    A.lt_uint(ctx, top_bits(y, msbs), top_bits(x, msbs), t[0], sc)
    m.free(t, e, cp, sc, ax, ay)


# --- measuring the approximation --------------------------------------------
# There is deliberately no generic failure-rate helper here.  The circuits in
# this module are the only ones in the package that are allowed to be wrong, so
# their error rates are measured explicitly in tests/test_ec_opt1128.py -- both
# per operation (where the rate should track 2^-msbs) and composed into
# `ec_eea.inplace_mul` (where it compounds over ~1.4n iterations, which is the
# thing that actually decides how large msbs has to be).


# --- IonQ Sec VI and X.D: complements instead of negations -------------------
def _fix_zero_q(m, ctrl, y, q, tau):
    """If (ctrl and) y == q, set y = 0: the one out-of-range value a
    complement-based circuit produces.  Approximate: both tests look at the top
    tau bits only.  y == 0 is otherwise impossible at this point, so the
    all-zeros test clears the flag again."""
    n = len(y)
    fl = m.anc(1, "zq")
    on = [] if ctrl is None else [ctrl]
    eq_top(m, y, q, tau, [fl[0]], also=on)        # fl = [y == q]
    for i in range(n):
        if (q >> i) & 1:
            m.ctx.cx(fl[0], y[i])                 # q -> 0
    eq_top(m, y, 0, tau, [fl[0]], also=on)        # fl = [y == 0] = old flag
    m.free(fl)


def csignadd_pm(m, e, x, y, q, lsbs=None, msbs=None):
    """y <- (y + (-1)^e x) mod q for q = 2^u - f.  IonQ Sec VI.

    NOT(y) = 2^u - 1 - y == (f - 1) - y  (mod q), so complementing y, adding x
    modulo q and complementing again gives
        (f - 1) - ((f - 1) - y + x) = y - x.
    The complements are CNOTs from e; the adder is Algorithm 11 (uncontrolled);
    the only out-of-range output, q in place of 0, is repaired by `_fix_zero_q`,
    and the only out-of-range *input*, NOT(0), is avoided by first mapping
    0 -> q (`_swap_zero_q`, as IonQ does).

    Approximate in two ways, both ~2^-msbs or ~f/2^u: the top-bit tests, and
    sums that land in [q, 2^u) without overflowing, which the pseudo-Mersenne
    adder does not reduce (Algorithm 11 shares that).  At secp256k1 sizes both
    are negligible; at toy primes f/q is not, and the tests measure it.
    """
    n = len(y)
    msbs = msbs or max(2, n // 2)
    _swap_zero_q(m, e, y, q, msbs)                # IonQ: 0 -> q before the adder
    for b in y:
        m.ctx.cx(e, b)
    cmodadd_pm_q(m, None, x, y, q, lsbs, msbs)
    for b in y:
        m.ctx.cx(e, b)
    _fix_zero_q(m, e, y, q, msbs)


def _swap_zero_q(m, ctrl, y, q, tau):
    """If (ctrl and) y == 0, set y = q.

    NOT(0) = 2^u - 1 is not a canonical residue, and the pseudo-Mersenne adder
    is only correct on canonical inputs; NOT(q) = f - 1 is.  IonQ Sec VI:
    "we conditionally swap the values 0 <-> p ... before invoking the
    conditionally-inverted adder".  y == q is impossible on entry, so the
    all-ones test clears the flag again.  Approximate on the top tau bits."""
    fl = m.anc(1, "z0")
    on = [] if ctrl is None else [ctrl]
    eq_top(m, y, 0, tau, [fl[0]], also=on)        # fl = [y == 0]
    for i in range(len(y)):
        if (q >> i) & 1:
            m.ctx.cx(fl[0], y[i])                 # 0 -> q
    eq_top(m, y, q, tau, [fl[0]], also=on)        # fl = [y == q] = old flag
    m.free(fl)


def cmodneg_approx(m, ctrl, x, q, kappa=None, tau=None):
    """x <- (-x) mod q (when ctrl), q = 2^u - f.  IonQ Sec X.D.

    q - x = NOT(x) - (f - 1): complement the bits, subtract the small constant
    f - 1 on the low kappa bits only (the borrow escapes them with probability
    ~f/2^kappa), and repair x = 0, which comes out as q.  kappa = tau = u is
    exact.  Cost ~kappa + 2 tau against ~4u for `ec_modarith.cmodneg`.
    """
    pm = pseudo_mersenne(q)
    assert pm, f"q = {q} is not pseudo-Mersenne"
    u, f = pm
    n = len(x)
    assert u == n
    kappa = min(n, kappa or n)
    tau = min(n, tau or n)
    ctx = m.ctx
    for b in x:
        ctx.cx(ctrl, b) if ctrl is not None else ctx.x(b)
    if f > 1:
        cp, sc = m.anc(kappa, "cp"), m.anc(kappa, "sc")
        low = Reg(list(x[:kappa]))
        if ctrl is None:
            A.sub_const(ctx, low, f - 1, cp, sc)
        else:
            A.csub_const(ctx, ctrl, low, f - 1, cp, sc)
        m.free(cp, sc)
    _fix_zero_q(m, ctrl, x, q, tau)


def modhalf_pm(m, x, q, lsbs=None):
    """x <- x/2 mod q: Algorithm 7 run backwards."""
    m.emit_inverse(moddbl_pm, m, x, q, lsbs)


# --- IonQ Alg 2: the flags uncomputed by measurement --------------------------
def _flag_predicates(ctx, x, y, k, anc):
    """Compute t1 = [top_k(y) < top_k(x)] and t2 = [top_k(y) == 0] into
    anc[0], anc[1] (the rest is scratch).  Undone by calling it again: both
    parts are carry chains / AND chains whose uncomputes are free."""
    t1, t2, sc = anc[0], anc[1], anc[2:]
    ty, tx = y[len(y) - k:], x[len(x) - k:]
    A.lt_uint(ctx, ty, tx, t1, sc)
    for b in ty:
        ctx.x(b)
    if k == 1:
        ctx.cx(ty[0], t2)
    else:                                            # AND chain over NOT(top y)
        ch = sc[:k - 1]
        ctx.and_(ty[0], ty[1], ch[0])
        for i in range(2, k):
            ctx.and_(ch[i - 2], ty[i], ch[i - 1])
        ctx.cx(ch[k - 2], t2)
        for i in range(k - 1, 1, -1):
            ctx.and_dg(ch[i - 2], ty[i], ch[i - 1])
        ctx.and_dg(ty[0], ty[1], ch[0])
    for b in ty:
        ctx.x(b)


def modadd_pm_mbu(m, x, y, q, lsbs=None, msbs=None):
    """y <- (y + x) mod q, q = 2^u - f: Algorithm 11 (uncontrolled) with both
    of its flags uncomputed by X-measurement.  IonQ Alg 2.

    After the reduction the overflow flag equals [y < x] AND [y != 0] and the
    x + y = q flag equals [y < x] AND [y == 0], both functions of the final
    registers.  Algorithm 11 clears them with comparisons; here each is
    measured, and only on outcome 1 -- half the time -- is its predicate
    applied as a *phase*.  Worst case the same comparisons; on average half.
    Approximate on the top msbs bits, exactly like Algorithm 11."""
    import ec_mbu as MB
    pm = pseudo_mersenne(q)
    assert pm, f"q = {q} is not pseudo-Mersenne"
    u, f = pm
    ctx, n = m.ctx, len(y)
    assert u == n
    msbs = min(n, msbs or max(2, n // 2))
    lsbs = lsbs or min(n, 2 * max(1, f.bit_length()) + 8)

    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    sc = m.anc(n + 1, "sc")
    A.add(ctx, x + ax, y + ay, sc)                   # ay = overflow
    m.free(sc)
    e = m.anc(1, "e")
    eq_top(m, y, q, msbs, [e[0]])                    # e = [sum == q]
    for i in range(n):
        if (q >> i) & 1:
            ctx.cx(e[0], y[i])
    cp2, sc2 = m.anc(lsbs, "cp2"), m.anc(lsbs, "sc2")
    A.cadd_const(ctx, ay[0], Reg(list(y[:lsbs])), f, cp2, sc2)
    m.free(cp2, sc2)

    data = list(x) + list(y)
    nanc = 2 + msbs + 1

    def make(which, as_phase):
        def build(qc, qs):
            from ec_gates import Ctx
            c = Ctx(qc, "and")
            xx, yy, fl, anc = qs[:n], qs[n:2 * n], qs[2 * n], qs[2 * n + 1:]
            _flag_predicates(c, xx, yy, msbs, anc)
            t1, t2 = anc[0], anc[1]
            if which == "ay":
                c.x(t2)                              # [y < x] AND NOT [y == 0]
            if as_phase:
                c.cz(t1, t2)
            else:
                c.ccx(t1, t2, fl)
            if which == "ay":
                c.x(t2)
            _flag_predicates(c, xx, yy, msbs, anc)
        return build

    MB.mbu_flag(m, ay[0], data, make("ay", False), make("ay", True), nanc=nanc)
    MB.mbu_flag(m, e[0], data, make("e", False), make("e", True), nanc=nanc)
    m.free(ax, ay, e)


def csignadd_pm_mbu(m, e, x, y, q, lsbs=None, msbs=None):
    """`csignadd_pm` on the measured-flag adder (IonQ Sec VI + Alg 2)."""
    n = len(y)
    msbs = msbs or max(2, n // 2)
    _swap_zero_q(m, e, y, q, msbs)
    for b in y:
        m.ctx.cx(e, b)
    modadd_pm_mbu(m, x, y, q, lsbs, msbs)
    for b in y:
        m.ctx.cx(e, b)
    _fix_zero_q(m, e, y, q, msbs)
