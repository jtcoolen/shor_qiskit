"""In-place modular multiplication via a Euclidean dialog -- [1128] Sec 3.

The expensive operation in a point addition is |x, y> -> |x, x y mod q>.
Everyone before [1128] built it out of a modular inversion plus an out-of-place
multiplication, ran the inversion forwards to get x^{-1}, and ran it backwards
to erase the garbage.  [1128], following [14], does something better.

The two ideas
-------------
1. Split the extended Euclidean algorithm in half.  Run the binary GCD on
   (u, v) alone and do *not* maintain the Bezout coefficients; just record the
   two decision bits per iteration.  That record -- the "dialog" -- is 1.5 bits
   per iteration on average, against the 2n qubits carrying (r, s) would cost.

2. Replay the record onto any pair you like.  The updates to (r, s) are linear
   and depend only on those two bits, so starting the replay at (y, 0) instead
   of (1, 0) ends at (0, y x mod q).  The inversion and the multiplication come
   out of one pass -- the multiplication is *free*.

And because the replay is a circuit, running it backwards multiplies by x^{-1}
instead, which is the other in-place multiplication the point addition needs.
One circuit, both steps.

Direction matters
-----------------
The record is replayed in *reverse* order.  Read forwards, the same update rules
compute a different linear map -- not the inverse.  `ec_classical` has both, and
the tests pin down which is which.

What is and is not implemented here
-----------------------------------
Implemented: the dialog (Alg. 2), the 3-pairs-into-5-bits compression (Fig. 1),
the Bezout replay (Alg. 3), and in-place multiplication and division (Alg. 4).

Not implemented: the register *sharing* of Sec 3.1, where u and v shrink as the
algorithm proceeds and the freed qubits absorb the record, taking the space from
2.83n to 2.12n.  That optimization is probabilistic -- it works "with high
probability" for a padding constant c_pad ~ 2.3 -- and everything else in this
package is exact.  `shrink_schedule` computes the widths it would use, so
`ec_cost` can price it, and `dialog_width` reports what this implementation
actually spends.

The compression circuit here is built from a generic permutation, so it costs
57 Toffoli-equivalents per triple where [1128] Fig. 1 uses 5.  It compresses
correctly -- all 27 valid patterns, verified -- and the paper calls this cost
negligible against the arithmetic, so the gap moves no headline number.  But it
is a real gap, and this is the routine to replace if the paper's constant
matters.
"""

import ec_adders as A
import ec_modarith as MA
from ec_classical import compress_triple, eea_iterations
from ec_sim import Reg


# --- Algorithm 2: the dialog ------------------------------------------------
def dialog_round(m, u, v, rec):
    """One iteration of [1128] Algorithm 2.  `rec` = (b0, b0b1), kept.

    b0 is v's parity and b0b1 is "swap"; both have to be recorded because v is
    halved and u and v are swapped, so neither is recoverable afterwards.
    """
    ctx, n = m.ctx, len(u)
    b0, b0b1 = rec
    ctx.cx(v[0], b0[0])                              # b0 = v mod 2

    t, sc = m.anc(1, "t"), m.anc(n, "sc")
    A.gt_uint(ctx, u, v, t[0], sc)                   # t = [u > v]
    ctx.and_(b0[0], t[0], b0b1[0])                   # b0b1 = b0 AND b1
    A.gt_uint(ctx, u, v, t[0], sc)
    m.free(t, sc)

    for i in range(n):
        ctx.cswap(b0b1[0], u[i], v[i])               # if b0b1: swap u, v

    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b0[0], u, v, cp, sc)                 # if b0: v -= u
    m.free(cp, sc)

    for i in range(n - 1):                           # v /= 2 (v is even now)
        ctx.swap(v[i], v[i + 1])


def dialog(m, x, q, iters=None, ctrl=None):
    """Consume x, produce the dialog.  Returns (records, u, v).

    On termination u = 1 and v = 0, so x's own register comes back clean -- the
    value has been traded for the record.  That is [1128] Alg. 4 step 1.
    """
    ctx, n = m.ctx, len(x)
    iters = iters or eea_iterations(n)
    assert ctrl is None, (
        "a controlled dialog is not implemented -- gate the caller instead. "
        "Unlike Kaliski, loading v <- 0 does not make the rounds inert here: "
        "the dialog would record a different bit string and the replay would "
        "no longer invert it.")
    hi = m.anc(1, "vhi")
    u = m.anc(n + 1, "u")
    v = Reg(list(x) + list(hi), "v")
    A.encode_const(ctx, u, q)                        # u <- q (odd)

    recs = []
    for _ in range(iters):
        rec = (m.anc(1, "b0"), m.anc(1, "b1"))
        dialog_round(m, u, v, rec)
        recs.append(rec)
    return recs, u, v


def dialog_undo(m, x, q, recs, u, v, iters=None):
    """Rebuild x: Algorithm 2 run backwards ([1128] Alg. 4 step 3)."""
    n = len(x)
    iters = iters or eea_iterations(n)
    assert len(recs) == iters
    for rec in reversed(recs):
        m.emit_inverse(dialog_round, m, u, v, rec)
    m.emit_inverse(A.encode_const, m.ctx, u, q)


def clear_dialog_end(m, u, v):
    """After a dialog, u = 1 and v = 0: return both registers to |0>."""
    m.ctx.x(u[0])
    m.free(u)
    m.free(Reg([v[-1]]))              # the borrowed high bit of v


# --- Algorithm 3: the Bezout replay -----------------------------------------
def bezout_replay(m, r, s, recs, q, dbl=None, cadd=None):
    """(r, s) updated by [1128] Algorithm 3, reading the record in reverse.

    Started at (y, 0) this ends at (0, y x mod q).  `dbl` and `cadd` default to
    the exact modular doubling and controlled modular addition; `ec_approx`
    supplies the cheaper approximate ones.
    """
    dbl = dbl or (lambda mm, reg: MA.moddbl(mm, reg, q))
    cadd = cadd or (lambda mm, c, a, b: MA.cmodadd(mm, c, a, b, q))
    for b0, b0b1 in reversed(recs):
        dbl(m, s)                                        # s <- 2s mod q
        cadd(m, b0[0], r, s)                             # if b0: s += r mod q
        for i in range(len(r)):
            m.ctx.cswap(b0b1[0], r[i], s[i])             # if b0b1: swap r, s


# --- Algorithm 4: in-place modular multiplication ---------------------------
def inplace_mul(m, x, y, q, iters=None, dbl=None, cadd=None):
    """|x, y> -> |x, y*x mod q>.  [1128] Algorithm 4.

    Three phases: x becomes a dialog, the dialog is replayed onto (y, 0), the
    dialog becomes x again.  The answer lands in the replay's scratch half, so n
    SWAPs put it back in y's own qubits and the caller's handle stays valid --
    pure CNOT, no Toffoli.

    The rebuild also consumes the record: running a dialog round backwards
    un-copies b0 and un-ANDs b0b1, so every record qubit returns to |0> and the
    whole multiplication is garbage-free.
    """
    n = len(x)
    iters = iters or eea_iterations(n)

    recs, u, v = dialog(m, x, q, iters)                  # 1: x -> dialog
    clear_dialog_end(m, u, v)

    s = m.anc(n, "bz")                                   # 2: replay onto (y, 0)
    bezout_replay(m, y, s, recs, q, dbl, cadd)
    for a, b in zip(y, s):                               # r is 0, s is the answer
        m.ctx.swap(a, b)
    m.free(s)

    hi = m.anc(1, "vhi")                                 # 3: dialog -> x
    u2 = m.anc(n + 1, "u")
    v2 = Reg(list(x) + list(hi), "v")
    m.ctx.x(u2[0])                                       # the dialog ends at u = 1
    for rec in reversed(recs):
        m.emit_inverse(dialog_round, m, u2, v2, rec)
    A.encode_const(m.ctx, u2, q)                         # and starts at u = q
    m.free(u2, hi)
    for rec in recs:
        m.free(*rec)


def inplace_div(m, x, y, q, iters=None, dbl=None, cadd=None):
    """|x, y> -> |x, y*x^{-1} mod q>: `inplace_mul` run backwards."""
    m.emit_inverse(inplace_mul, m, x, y, q, iters, dbl, cadd)


# --- Fig. 1: compressing three (b0, b0&b1) pairs into five bits -------------
def compression_permutation():
    """The 6-bit permutation of [1128] Fig. 1, as an explicit map.

    Only 27 of the 64 input patterns can occur -- (b0, b0&b1) is never (0, 1) --
    so three pairs fit in 5 bits with one to spare.  Over 1.413n iterations that
    is 2.355n record qubits instead of 2.83n, and unlike the shift-based
    encoding it leaves the record a *fixed* size, which is what lets the rest of
    the circuit be laid out statically.
    """
    valid, targets = [], []
    for a in ((0, 0), (1, 0), (1, 1)):
        for b in ((0, 0), (1, 0), (1, 1)):
            for c in ((0, 0), (1, 0), (1, 1)):
                code = a[0] | (a[1] << 1) | (b[0] << 2) | (b[1] << 3) | \
                       (c[0] << 4) | (c[1] << 5)
                valid.append(code)
                targets.append(compress_triple([a, b, c]))   # < 32, so bit 5 is 0
    perm = dict(zip(valid, targets))
    spare_in = [i for i in range(64) if i not in perm]
    spare_out = [i for i in range(64) if i not in set(perm.values())]
    perm.update(dict(zip(spare_in, spare_out)))
    return perm


def compress_records(m, recs):
    """Compress the dialog in groups of three, freeing one qubit per group."""
    from ec_shor import permutation
    perm = compression_permutation()
    freed = []
    for i in range(0, len(recs) - 2, 3):
        qs = [recs[i][0][0], recs[i][1][0],
              recs[i + 1][0][0], recs[i + 1][1][0],
              recs[i + 2][0][0], recs[i + 2][1][0]]
        permutation(m.qc, qs, perm)
        freed.append(qs[5])                    # bit 5 is now 0
    return freed


# --- space accounting -------------------------------------------------------
def dialog_width(n, c_iter=2.4):
    """Record qubits this implementation actually uses: 2 per iteration."""
    return 2 * eea_iterations(n, c_iter)


def dialog_width_compressed(n, c_iter=2.4):
    """With Fig. 1 compression: 5 per 3 iterations, ~2.355n."""
    it = eea_iterations(n, c_iter)
    return 2 * it - it // 3


def shrink_schedule(n, c_pad=2.3):
    """[1128] Sec 3.1: the width u and v would occupy at each iteration.

    n - 0.5 log2(8/3) i + c_pad sqrt(n).  Not used by the circuits here -- it is
    a probabilistic optimization and this package is exact -- but it is what the
    paper's 2.12n rests on, so `ec_cost` can quote both.
    """
    import math
    step = 0.5 * math.log2(8 / 3)
    out = []
    for i in range(eea_iterations(n)):
        w = n - step * i + c_pad * math.sqrt(n)
        out.append(max(1, int(math.ceil(w))))
    return out
