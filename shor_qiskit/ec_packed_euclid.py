"""Reversible packed Euclidean modular inversion (Luo et al. Alg. 1-3).

Luo et al. arXiv:2607.13816 Sec. 3 and Gu et al. SI II.2 store the
remainder/quotient pair and the Bezout coefficients in two opposed-end
words, Work1 and Work2, delimited by coherent length counters lt, lq, lrp
and a shift counter ls.  ``luo_step`` is one coherent walk step: it shifts
the boundary, subtracts/adds the shifted divisor in the remainder interval,
updates the quotient bit and the coefficient prefix, and exchanges the words
when a quotient finishes.  ``mod_inv`` runs 4*ceil(1.5795 n) steps, copies
the inverse and uncomputes the whole walk (Bennett), so every ancilla returns
to zero.  Selected by ``inversion="packed-euclid"``.

Scope: gates are plain ccx/cx/x/cswap with a borrowed-ancilla mcx ladder,
not the paper's cost-optimised 256-bit schedule.  Counter widths are toy
sized and justified by exhaustive checks against ``ec_euclid_reference``
(tests/test_ec_packed_euclid.py), not by a general proof; gate counts here
are not estimates of the paper's resource figures.
"""

import ec_adders as A
from ec_sim import Reg


def swap_at(m, word, index, flag):
    """Swap flag with the word bit addressed by a coherent binary index.

    Out-of-range indices leave both unchanged. The index and temporary
    selection bit are restored, even when the addressed word bit is occupied.
    """
    if not word or not index or 1 << len(index) < len(word):
        raise ValueError("address must cover a nonempty word")
    if len(set(word + index + [flag])) != len(word) + len(index) + 1:
        raise ValueError("word, index and flag must be disjoint")
    ctx = m.ctx
    selected = m.anc(1, "selected")
    for position, bit in enumerate(word):
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                ctx.x(address_bit)
        ctx.mcx(index, selected[0])
        ctx.cswap(selected[0], flag, bit)
        ctx.mcx(index, selected[0])
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                ctx.x(address_bit)
    m.free(selected)


def xor_at(m, word, index, target):
    """XOR an addressed packed-word bit into a separate target qubit.

    Unlike ``swap_at`` this does not change the work word, so it can
    temporarily select operands for a location-controlled comparison.
    The caller must reverse the selection before modifying its source.
    """
    if (not word or not index or 1 << len(index) < len(word)
            or len(set(word + index + [target]))
            != len(word) + len(index) + 1):
        raise ValueError("packed read needs distinct wires and a covering index")
    for position, bit in enumerate(word):
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                m.ctx.x(address_bit)
        m.ctx.mcx((*index, bit), target)
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                m.ctx.x(address_bit)


def rotate_word_if(m, word, control, direction):
    """Conditionally rotate a packed work word by one physical wire.

    Luo et al. Alg. 3 uses this reversible circular shift while the live
    coefficient and remainder boundaries move. A rotation alone does not
    update those boundaries or implement the full packed walk.
    """
    if (direction not in ("left", "right") or not word
            or control in word or len(set(word)) != len(word)):
        raise ValueError("rotation needs unique word wires and a disjoint control")
    if direction == "left":
        positions = range(len(word) - 2, -1, -1)
    else:
        positions = range(len(word) - 1)
    for i in positions:
        m.ctx.cswap(control, word[i], word[i + 1])


def shift_work2(m, word, shift, control, direction, steps=1):
    """Rotate Work2 and update its coherent shift counter together.

    Algorithms 2-3 use one-position pre/post shifts and two-position
    corrections. The caller must ensure the counter does not overflow its
    promised live range; modulo-register wrapping is not a valid boundary.
    """
    if (not word or not shift or type(steps) is not int
            or not 1 <= steps <= 2
            or len(set(word + shift + [control])) != len(word) + len(shift) + 1
            or direction not in ("left", "right")):
        raise ValueError("Work2 shift needs a nonempty word, counter, separate "
                         "control and a one- or two-position direction")
    cp = m.anc(len(shift), "constant")
    carry = m.anc(len(shift), "carry")
    for _ in range(steps):
        rotate_word_if(m, word, control, direction)
    A.cadd_const(m.ctx, control, shift,
                 steps if direction == "left" else -steps, cp, carry)
    m.free(cp, carry)


def binary_divmod(m, numerator, denominator, quotient, remainder):
    """XOR quotient and remainder of numerator / denominator into clean outputs.

    The input and denominator are n-bit unsigned integers; denominator must
    be nonzero. The scan holds a partial remainder in n+1 bits, obtains each
    quotient bit by comparing it with the divisor, conditionally subtracts,
    copies the output and reverses the scan to clear its work registers.
    """
    n = len(numerator)
    if not n or any(len(reg) != n for reg in (denominator, quotient, remainder)):
        raise ValueError("binary division needs four equal nonempty registers")
    if set(numerator) & set(denominator + quotient + remainder) or (
            set(denominator) & set(quotient + remainder) or
            set(quotient) & set(remainder)):
        raise ValueError("division registers must be disjoint")
    ctx = m.ctx
    rem = m.anc(n + 1, "partial")
    quot = m.anc(n, "quotient")
    hi = m.anc(1, "divisor_hi")
    divisor = Reg(list(denominator) + list(hi), "divisor")
    mark = m.begin()
    for bit in range(n - 1, -1, -1):
        for i in range(n, 0, -1):
            ctx.swap(rem[i], rem[i - 1])
        ctx.cx(numerator[bit], rem[0])
        comparator = m.anc(n + 1, "compare")
        A.lt_uint(ctx, rem, divisor, quot[bit], comparator)
        m.free(comparator)
        ctx.x(quot[bit])
        copy, carry = m.anc(n + 1, "copy"), m.anc(n + 1, "carry")
        A.csub(ctx, quot[bit], divisor, rem, copy, carry)
        m.free(copy, carry)
    body = m.since(mark)
    for a, b in zip(quot, quotient):
        ctx.cx(a, b)
    for a, b in zip(rem, remainder):
        ctx.cx(a, b)
    m.undo(body)
    m.free(rem, quot, hi)


# --- Luo et al. Algorithms 1-3: the packed Euclidean walk -------------------
#
# Gate-level counterpart of ``ec_euclid_reference.luo_step``.  Positions are
# one-based as in Luo Table 4 and live at index position-1 of ``w1``/``w2``.
# Every word update is addressed by the coherent counters lt, lq, lr', ls:
# interval ends and swap positions are decoded by equality tests into a
# running accumulator, and one carry qubit ripples through the addressed
# lanes (Luo Sec. 3.2's location-controlled MAJ/UMA cells).


def _mcx(ctx, ctrls, target):
    """Multi-controlled X as Toffolis on borrowed (dirty) qubits.

    Barenco et al. Lemma 7.2: k controls take 4(k - 2) Toffolis and k - 2
    idle qubits in any state, which are restored.  Matrix-product-state Aer
    keeps Toffolis native but expands wide MCX gates, so this keeps the
    walk sampleable without adding width.
    """
    ctrls = list(ctrls)
    if len(ctrls) < 3:
        ctx.mcx(ctrls, target)
        return
    qc = ctx.qc
    busy = set(ctrls) | {target}
    centre = qc.find_bit(target).index
    spare = sorted((q for q in qc.qubits if q not in busy),
                   key=lambda q: abs(qc.find_bit(q).index - centre))
    k = len(ctrls)
    if len(spare) < k - 2:
        ctx.mcx(ctrls, target)
        return
    a = spare[:k - 2]

    def ladder():
        for i in range(k - 3, 0, -1):
            ctx.ccx(ctrls[i + 1], a[i - 1], a[i])
        ctx.ccx(ctrls[0], ctrls[1], a[0])
        for i in range(1, k - 2):
            ctx.ccx(ctrls[i + 1], a[i - 1], a[i])

    for _ in range(2):
        ctx.ccx(ctrls[-1], a[-1], target)
        ladder()


class LuoRegs:
    """The walk's persistent registers (counters and flags little-endian)."""

    def __init__(self, m, p):
        n = p.bit_length()
        self.width = n + 3
        self.steps = luo_step_bound(n)
        cw = n.bit_length()
        self.w1 = m.anc(self.width, "work1")
        self.w2 = m.anc(self.width, "work2")
        self.lt, self.lq, self.lrp = (m.anc(cw, name)
                                      for name in ("lt", "lq", "lrp"))
        self.ls = m.anc(self.steps.bit_length(), "ls")
        self.p1, self.p2, self.iter, self.sign = (
            m.anc(1, name)[0] for name in ("phase1", "phase2", "iter", "sign"))

    def regs(self):
        return [self.w1, self.w2, self.lt, self.lq, self.lrp, self.ls,
                Reg([self.p1, self.p2, self.iter, self.sign], "flags")]


def luo_step_bound(n):
    import ec_euclid_reference as R
    return R.luo_step_bound(n)


def _eq(ctx, terms, ctrls, target, neg=()):
    """target ^= AND(ctrls) AND NOT(any neg) AND [reg == value for terms]."""
    flips = list(neg)
    qs = list(ctrls) + list(neg)
    for reg, value in terms:
        if not 0 <= value < 1 << len(reg):
            return
        flips += [q for k, q in enumerate(reg) if not (value >> k) & 1]
        qs += list(reg)
    for q in flips:
        ctx.x(q)
    _mcx(ctx, qs, target)
    for q in flips:
        ctx.x(q)


def _eq_sum(ctx, a, b, total, ctrls, target, neg=()):
    """target ^= AND(ctrls) AND [a + b == total] (disjoint value pairs)."""
    for va in range(1 << len(a)):
        _eq(ctx, [(a, va), (b, total - va)], ctrls, target, neg)


def _cinc(ctx, reg, ctrls, neg=(), down=False):
    """reg += 1 (or -= 1) modulo 2^len under positive/negative controls."""
    for q in neg:
        ctx.x(q)
    order = range(len(reg)) if down else reversed(range(len(reg)))
    for k in order:
        _mcx(ctx, list(ctrls) + list(neg) + list(reg[:k]), reg[k])
    for q in neg:
        ctx.x(q)


def _rotate(ctx, word, cond, left):
    """Cyclic rotation under cond; left means new[j] = old[j + 1]."""
    pairs = list(range(len(word) - 1))
    for i in (pairs if left else reversed(pairs)):
        ctx.cswap(cond, word[i], word[i + 1])


def _shift(m, r, live_or_none):
    """Rotate Work2 and move ls by one; direction is set by Phase2.

    Pre-shift (Phase1 = 0) is gated by the live flag; post-shift (Phase1 = 1)
    is not.  ls moves even when the rotation is gated off.
    """
    ctx = m.ctx
    pre = live_or_none is not None
    pol = [r.p1] if pre else []
    on = [] if pre else [r.p1]
    cond = m.anc(1, "rot")[0]
    for p2_set in (0, 1):
        ctrls = on + ([live_or_none] if pre else []) + ([r.p2] if p2_set else [])
        neg = pol + ([] if p2_set else [r.p2])
        _eq(ctx, [], ctrls, cond, neg)
        _rotate(ctx, r.w2, cond, left=not p2_set)
        _eq(ctx, [], ctrls, cond, neg)
        _cinc(ctx, r.ls, on + ([r.p2] if p2_set else []),
              pol + ([] if p2_set else [r.p2]), down=bool(p2_set))
    m.free(Reg([cond]))


def _maj(ctx, s, a, b, c, subtract):
    if subtract:
        ctx.cx(s, a)
    ctx.cx(c, a)
    ctx.cx(c, b)
    _mcx(ctx, [s, a, b], c)


def _uma(ctx, s, a, b, c, subtract):
    _mcx(ctx, [s, a, b], c)
    ctx.ccx(s, b, a)
    ctx.cx(c, b)
    ctx.cx(c, a)
    if subtract:
        ctx.cx(s, a)


def _interval_addsub(m, r, g, subtract, tap):
    """Big-endian Work1[lo..hi] +/-= Work2[lo..hi] under g.

    lo = lt + lq + 2 and hi = W - ls.  The lane flag s turns on at hi and off
    below lo while scanning from the least significant (right) end.
    """
    ctx, W = m.ctx, r.width
    s, c = (q[0] for q in (m.anc(1, "lane"), m.anc(1, "carry")))

    def toggle(j):
        _eq(ctx, [(r.ls, W - j)], [g], s)
        _eq_sum(ctx, r.lt, r.lq, j - 1, [g], s)

    for j in range(W, 0, -1):
        toggle(j)
        _maj(ctx, s, r.w1[j - 1], r.w2[j - 1], c, subtract)
    if tap:
        ctx.cx(c, r.sign)
    for j in range(1, W + 1):
        _uma(ctx, s, r.w1[j - 1], r.w2[j - 1], c, subtract)
        toggle(j)
    m.free(Reg([s, c]))


def _prefix_addsub(m, r, g, subtract, tap):
    """Little-endian Work2[1..E] +/-= Work1[1..E] under g.

    E = lt + 1 in Phase 3 and W - lr' - ls in Phase 4 (Luo Sec. 3.2).
    """
    ctx, W = m.ctx, r.width
    s, c = (q[0] for q in (m.anc(1, "lane"), m.anc(1, "carry")))

    def toggle(j):
        if j == 1:
            ctx.cx(g, s)
        _eq(ctx, [(r.lt, j - 2)], [g], s, neg=[r.p2])
        _eq_sum(ctx, r.lrp, r.ls, W - j + 1, [g, r.p2], s)

    for j in range(1, W + 1):
        toggle(j)
        _maj(ctx, s, r.w2[j - 1], r.w1[j - 1], c, subtract)
    if tap:
        ctx.cx(c, r.sign)
    for j in range(W, 0, -1):
        _uma(ctx, s, r.w2[j - 1], r.w1[j - 1], c, subtract)
        toggle(j)
    m.free(Reg([s, c]))


def _swap_sign(m, r):
    """If Phase1 xor Phase2, swap Sign with Work1 position lt + lq + 1."""
    ctx = m.ctx
    f, h = (m.anc(1, name)[0] for name in ("phase_xor", "at"))
    ctx.cx(r.p1, f)
    ctx.cx(r.p2, f)
    for j in range(1, r.width + 1):
        _eq_sum(ctx, r.lt, r.lq, j - 1, [f], h)
        ctx.cswap(h, r.sign, r.w1[j - 1])
        _eq_sum(ctx, r.lt, r.lq, j - 1, [f], h)
    ctx.cx(r.p2, f)
    ctx.cx(r.p1, f)
    m.free(Reg([f, h]))


def _xor_value(ctx, ctrls, neg, value, reg):
    for k, q in enumerate(reg):
        if (value >> k) & 1:
            _eq(ctx, [], ctrls, q, neg)


def _xor_highest_one(ctx, f, word, lrp, lt):
    """lt ^= f * (highest one of word at positions <= W - lr')."""
    W = len(word)
    for v in range(1 << len(lrp)):
        bound = W - v
        for j in range(1, bound + 1):
            ones = [f, word[j - 1]] + list(lrp)
            zeros = list(word[j:bound])
            flips = [q for k, q in enumerate(lrp) if not (v >> k) & 1]
            for q in flips:
                ctx.x(q)
            _xor_value(ctx, ones, zeros, j, lt)
            for q in flips:
                ctx.x(q)


def _xor_right_length(ctx, f, word, lt, lrp):
    """lrp ^= f * (W + 1 - first one at position >= lt + 2), 0 if none."""
    W = len(word)
    for v in range(1 << len(lt)):
        start = v + 2
        for j in range(start, W + 1):
            ones = [f, word[j - 1]] + list(lt)
            zeros = list(word[start - 1:j - 1])
            flips = [q for k, q in enumerate(lt) if not (v >> k) & 1]
            for q in flips:
                ctx.x(q)
            _xor_value(ctx, ones, zeros, W + 1 - j, lrp)
            for q in flips:
                ctx.x(q)


def luo_step(m, r):
    """One step of Luo Algorithm 3 on LuoRegs ``r``; all scratch is cleaned."""
    ctx = m.ctx
    live = m.anc(1, "live")[0]
    ctx.x(live)
    _eq(ctx, [(r.lrp, 0)], [], live)

    # Pre-shift, then the r-side subtract/compare and conditional restore.
    _shift(m, r, live)
    g, g2 = (m.anc(1, name)[0] for name in ("r_sub", "r_add"))
    _eq(ctx, [], [live], g, neg=[r.p1])
    _interval_addsub(m, r, g, subtract=True, tap=True)
    _eq(ctx, [], [r.p2], r.sign, neg=[r.p1])
    ctx.cx(g, g2)
    _mcx(ctx, [g, r.p2, r.sign], g2)
    _interval_addsub(m, r, g2, subtract=False, tap=False)
    _mcx(ctx, [g, r.p2, r.sign], g2)
    ctx.cx(g, g2)
    _eq(ctx, [], [live], g, neg=[r.p1])
    m.free(Reg([g, g2]))

    # Quotient-window growth, Sign <-> quotient bit, window shrink.
    _cinc(ctx, r.lq, [r.p2], [r.p1])
    _swap_sign(m, r)
    _cinc(ctx, r.lq, [r.p1], [r.p2], down=True)

    # t-side subtract/compare and conditional restore with carry into Sign.
    g = m.anc(1, "t_sub")[0]
    ctx.cx(r.p1, g)
    _eq(ctx, [], [r.p1, r.sign], g, neg=[r.p2])
    _prefix_addsub(m, r, g, subtract=True, tap=False)
    _eq(ctx, [], [r.p1, r.sign], g, neg=[r.p2])
    ctx.cx(r.p1, g)
    m.free(Reg([g]))
    ctx.cx(r.p1, r.sign)
    _prefix_addsub(m, r, r.p1, subtract=False, tap=True)
    _shift(m, r, None)

    # Phase update.
    f = m.anc(1, "phase_gate")[0]
    _eq(ctx, [(r.lq, 0)], [live], f)
    ctx.ccx(f, r.sign, r.p2)
    ctx.ccx(f, r.p1, r.p2)
    ctx.ccx(f, r.p2, r.sign)
    _eq(ctx, [(r.lq, 0)], [live], f)
    _eq(ctx, [(r.lrp, 0)], [], live)
    ctx.x(live)
    m.free(Reg([f, live]))
    _eq(ctx, [(r.ls, 0)], [], r.p1)
    _eq(ctx, [(r.ls, 0)], [], r.p2)

    # Iteration end: exchange the words and re-measure the moving boundaries.
    f = m.anc(1, "exchange")[0]
    _eq(ctx, [(r.lq, 0), (r.ls, 0)], [], f)
    for a, b in zip(r.w1, r.w2):
        ctx.cswap(f, a, b)
    _xor_highest_one(ctx, f, r.w2, r.lrp, r.lt)
    _xor_highest_one(ctx, f, r.w1, r.lrp, r.lt)
    _xor_right_length(ctx, f, r.w1, r.lt, r.lrp)
    _xor_right_length(ctx, f, r.w2, r.lt, r.lrp)
    ctx.cx(f, r.iter)
    _eq(ctx, [(r.lq, 0), (r.ls, 0)], [], f)
    m.free(Reg([f]))


def luo_init(m, r, x, p, ctrl=None):
    """Algorithm 1 preparation: Work1 = (1, p), Work2 = (0, x or p - x).

    x (0 <= x < p) is copied right-aligned into Work2 under ctrl; Iter marks
    x > p // 2, which is then replaced by p - x.  lr' is the length of that
    value, so a ctrl = 0 branch walks the all-zero padding schedule.
    """
    import ec_modarith as MA
    ctx, W, n = m.ctx, r.width, p.bit_length()
    if len(x) != n:
        raise ValueError("x must have p.bit_length() qubits")
    ctx.x(r.w1[0])
    ctx.x(r.lt[0])
    for k in range(n):
        if (p >> k) & 1:
            ctx.x(r.w1[W - 1 - k])
    low = Reg([r.w2[W - 1 - k] for k in range(n)], "x_slot")
    for src, dst in zip(x, low):
        if ctrl is None:
            ctx.cx(src, dst)
        else:
            ctx.ccx(ctrl, src, dst)
    cp, sc = m.anc(n, "constant"), m.anc(n, "compare")
    A.geq_const(ctx, low, p // 2 + 1, r.iter, cp, sc)
    m.free(cp, sc)
    MA.cmodneg(m, r.iter, low, p)
    for k in range(n):
        _xor_value(ctx, [low[k]], list(low[k + 1:]), k + 1, r.lrp)


def mod_inv(m, x, out, p, ctrl=None):
    """out ^= ctrl * x^-1 mod p (0 maps to 0) by the packed Euclidean walk.

    Prepare, run Luo's fixed 4 ceil(1.5795 n) steps, copy t' (negated when
    Iter = 0), then run preparation and walk backwards: every work bit,
    counter and flag returns to |0>.  Returns no garbage.
    """
    import ec_modarith as MA
    n = p.bit_length()
    if (p < 5 or p % 2 == 0 or len(out) != n
            or any(p % d == 0 for d in range(3, int(p ** 0.5) + 1, 2))):
        raise ValueError("packed Euclid needs an odd prime p >= 5")
    r = LuoRegs(m, p)
    mark = m.begin()
    luo_init(m, r, x, p, ctrl)
    for _ in range(r.steps):
        luo_step(m, r)
    walk = m.since(mark)
    value = m.anc(n, "inverse")
    tprime = Reg(r.w2[:n], "tprime")
    for a, b in zip(tprime, value):
        m.ctx.cx(a, b)
    m.ctx.x(r.iter)
    MA.cmodneg(m, r.iter, value, p)
    m.ctx.x(r.iter)
    for a, b in zip(value, out):
        m.ctx.cx(a, b)
    m.ctx.x(r.iter)
    MA.cmodneg(m, r.iter, value, p)
    m.ctx.x(r.iter)
    for a, b in zip(tprime, value):
        m.ctx.cx(a, b)
    m.free(value)
    m.undo(walk)
    m.free(*r.regs())
    return []
