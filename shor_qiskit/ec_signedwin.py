"""Signed odd-digit windows: half the table, and no point at infinity.

[106] Sec 6.1 prices each windowed lookup at 2^(w-1) - 1 ANDs, not 2^w - 1:
following [HJN+20] Sec 5.1 / Fig. 10, it uses "signed point addition to handle
negative points", -P = (x, -y).  One of the w window bits is a sign, the other
w-1 address the table, and the sign costs a modular negation -- cheap next to
the half of every lookup it saves.  Luo et al. (Sec 6.4) use the same
construction for their secp256k1 count.  This module is that construction on
top of `ec_window.windowed_point_add_cfg`, which is not modified; the lookup
term 3 * 2^w of [1128] eq. (1) becomes 3 * 2^(w-1).

The recoding, and why the quantum register needs none
----------------------------------------------------
Read the w-bit window value i in [0, 2^w) as the *odd* digit

    d = 2i + 1 - 2^w   in  {+-1, +-3, ..., +-(2^w - 1)}.

Split i = s 2^(w-1) + l, with s the top bit.  Then

    s = 1:   d =   2l + 1
    s = 0:   d = -(2l' + 1),   l' = 2^(w-1) - 1 - l = NOT l  (w-1 bits)

so |d| = 2 l~ + 1, where l~ is l with its bits complemented when s = 0 --
w-1 CNOTs from NOT s -- and d < 0 exactly when s = 0.  Nothing is computed
from the scalar: d is affine in i, and the affine part is paid classically.
Over W windows the digits telescope,

    sum_J d_J 2^(wJ) = 2k + sum_J (1 - 2^w) 2^(wJ) = 2k + 1 - 2^(wW),

and the factor 2 is cancelled by building the table on the half-point:
T[j] = [(2j+1) h] B with h = 2^-1 mod r, r = ord(B).  Then window J adds
[i_J 2^(wJ)] P + Delta_J, and the whole run adds

    [k] P + Delta,   Delta = [(1 - 2^(wW)) h] P,

a known constant that post-processing subtracts (or the start point absorbs),
exactly like the sum of IonQ's per-table masks.  h exists iff r is odd, which
is the one requirement: every curve Shor's ECDLP is run against (secp256k1,
P-256) has prime order.  Since [2h] B = B, the table is simply
T[j] = H + [j] B with H = [h] B -- the IonQ masked table with mu = [1/2] B,
over 2^(w-1) entries.

Why the table never holds O
---------------------------
T[j] = O iff r | (2j+1) h iff r | 2j+1, impossible for odd r > 2^w - 1.  So
the O-free table comes for free, without a random mask, and
`PointAddCfg.offsets` applies: no [i != 0] flag, no controls on the square and
the final negation, and no x_R = 0 failure of the (0, 0) encoding of O.

The sign
--------
-T = (x_T, -y_T), and negation is a group automorphism, so

    R - T = -((-R) + T).

The addition itself is therefore untouched -- `windowed_point_add_cfg` with
the half table on the (w-1)-bit address -- and the sign is a controlled
negation of the accumulator's y on each side.  Along a run of windows the
negation closing window J and the one opening window J+1 are both controlled
negations of the same register, so they merge into one controlled by the XOR
of the two sign bits (two CNOTs): W windows cost W + 1 negations, one per
window as [HJN+20] counts it, plus one.

The negation is cheaper than `ec_modarith.cmodneg`, for the same reason the
digits work at all.  y = 0 only on points of order 2, and the accumulator
lives in the odd-order group <P>, so its y is never 0 and cmodneg's zero
guard (two n-bit zero tests) is dead weight: `cneg_y` is the complement and
the +(p+1) alone, about a third of the price.

Exceptional cases are those of the inner addition, on (+-R, T): x_R = x_T and
+-R = -2T.  O(1/p), as before.

Sources: [HJN+20] Haener et al., PQCrypto 2020, Sec 5.1 and Fig. 10;
[106] Kim et al., eprint 2026/106, Sec 6.1; IonQ Alg 4 for the masked table.
"""

import ec_adders as A
import ec_window as W
from ec_sim import Reg

SIGNED = W.PointAddCfg(offsets=True)        # the table is O-free by construction
SIGNED_IONQ = W.PointAddCfg(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True)


# =============================================================================
# Classical side: the table, the offset, and the accumulation it implies
# =============================================================================
def _half(curve, base, order):
    r = order or curve.point_order(base)
    assert r % 2 == 1, "signed odd digits need ord(base) odd (2 invertible mod r)"
    return r, (r + 1) // 2


def signed_window_points(curve, base, w, order=None):
    """The 2^(w-1) entries T[j] = [(2j+1)/2] base, and the offset Delta.

    Returns (table, delta).  The signed addition at address i adds
    [i] base + delta, with delta = [(1 - 2^w)/2] base.  `order` is ord(base);
    at toy sizes it is found by brute force.
    """
    from ec_classical import Point
    assert w >= 2, "w = 1 leaves no address bits; add +-[1/2] base directly"
    r, h = _half(curve, base, order)
    H = curve.mul(h, base)
    table, acc = [], H
    for _ in range(1 << (w - 1)):              # T[j] = H + [j] base, since [2h] = 1
        table.append(Point(acc.x, acc.y, acc.inf))
        acc = curve.add(acc, base)
    assert not any(P.inf for P in table), "ord(base) must exceed 2^w - 1"
    delta = curve.mul(((1 - (1 << w)) * h) % r, base)
    return table, delta


def signed_window_tables(curve, base, w, nwin, order=None):
    """Tables for nwin windows over `base` (window J on [2^(wJ)] base) and the
    total offset Delta = [(1 - 2^(w nwin))/2] base.  Returns (tables, Delta)."""
    r, h = _half(curve, base, order)
    tables = []
    for J in range(nwin):
        B = curve.mul(1 << (J * w), base)
        tables.append(signed_window_points(curve, B, w, r)[0])
    return tables, curve.mul(((1 - (1 << (w * nwin))) * h) % r, base)


def signed_digit(i, w):
    """(negative, l~): the sign and table index the circuit derives from i."""
    s, lo = i >> (w - 1), i & ((1 << (w - 1)) - 1)
    return (s == 0), (lo if s else (~lo) & ((1 << (w - 1)) - 1))


def signed_add_ref(curve, R, i, table, w):
    """What `windowed_point_add_signed` does, step by step: R -+ T[l~],
    computed as -((-R) + T) when the digit is negative."""
    neg, j = signed_digit(i, w)
    T = table[j]
    if not neg:
        return curve.add(R, T)
    return curve.neg(curve.add(curve.neg(R), T))


def signed_accumulate(curve, base, k, w, nwin, start=None, order=None, trace=None):
    """Run nwin signed windows of the scalar k classically, as the circuit does.

    Returns the final accumulator, which equals start + [k] base + Delta for
    Delta from `signed_window_tables`.  With `trace` (a list), each window
    appends (R_in, negative, T) -- the inner addition is then (+-R_in) + T,
    which is what `ec_classical.point_add_exceptional` must be asked about.
    """
    from ec_classical import O
    tables, _ = signed_window_tables(curve, base, w, nwin, order)
    R = start if start is not None else O
    for J, T in enumerate(tables):
        i = (k >> (J * w)) & ((1 << w) - 1)
        if trace is not None:
            neg, j = signed_digit(i, w)
            trace.append((R, neg, T[j]))
        R = signed_add_ref(curve, R, i, T, w)
    return R


# =============================================================================
# Quantum side
# =============================================================================
def cneg_y(m, ctrl, y, p):
    """y <- p - y when ctrl, for y in [1, p): a y-coordinate of odd order.

    `ec_modarith.cmodneg` without the zero guard: complement (CNOTs) and add
    p + 1 mod 2^n under the control, since NOT y + p + 1 = 2^n + p - y.
    Wrong only on y = 0, which no point of odd order has.
    """
    n = len(y)
    for q in y:
        m.ctx.cx(ctrl, q)
    cp, a = m.anc(n, "cp"), m.anc(n, "a")
    A.cadd_const(m.ctx, ctrl, y, p + 1, cp, a)
    m.free(cp, a)


def signed_windows(m, addrs, x2, y2, tables, p, cfg=SIGNED, neg=None):
    """A run of signed windowed additions on one accumulator.

        |i_J>... |R>  ->  |i_J>... |R + sum_J +-T_J[l~_J]>

    `addrs[J]` is window J's w bits, little-endian, sign in the top bit;
    `tables[J]` its 2^(w-1)-entry table.  The addition is
    `windowed_point_add_cfg` with `cfg`, which must have offsets=True (the
    table has no O).  `neg(m, ctrl, y, p)` negates the accumulator's y for
    the sign: `cneg_y` by default, or `ec_modarith.cmodneg` (exact on y = 0
    too).  W windows, W + 1 negations.
    """
    ctx = m.ctx
    assert cfg.offsets, "signed tables never contain O: use cfg.offsets"
    neg = neg or cneg_y

    prev = None
    for addr, table in zip(addrs, tables):
        w = len(addr)
        assert w >= 2 and len(table) == 1 << (w - 1), "table must have 2^(w-1) entries"
        s, low = addr[w - 1], Reg(list(addr[:w - 1]), "l")
        ctx.x(s)                                    # s <- [d < 0]
        for q in low:
            ctx.cx(s, q)                            # low <- l~, the |d| index
        if prev is None:
            neg(m, s, y2, p)                        # R <- -R when d < 0
        else:
            ps, plow = prev                         # close the last window and
            ctx.cx(ps, s)                           # open this one: one negation,
            neg(m, s, y2, p)                        # controlled by the XOR
            ctx.cx(ps, s)
            for q in plow:
                ctx.cx(ps, q)
            ctx.x(ps)
        W.windowed_point_add_cfg(m, low, x2, y2, table, p, cfg)
        prev = (s, low)

    ps, plow = prev
    neg(m, ps, y2, p)                               # R <- -R when the last d < 0
    for q in plow:
        ctx.cx(ps, q)
    ctx.x(ps)


def windowed_point_add_signed(m, addr, x2, y2, table, p, cfg=SIGNED, neg=None):
    """|i>|R> -> |i>|R + [i] B + Delta>: one signed window.

    `table` and Delta from `signed_window_points(curve, B, w)`.  The lookups
    run over 2^(w-1) entries on addr[:w-1]; the sign costs two controlled
    negations of y (merged across windows by `signed_windows`).
    """
    signed_windows(m, [addr], x2, y2, [table], p, cfg, neg)
