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

Algorithm 11 of [1128] -- the pseudo-Mersenne controlled addition that also
handles x + y = q -- is not built here.  That case cannot arise from random
inputs; it arises only in the first few iterations of the Bezout replay, and
`ec_eea.bezout_replay` handles it by using the exact adder there, which is the
same fix at negligible cost.
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

    Does not handle x + y == q; see the module docstring.
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

# --- measuring the approximation --------------------------------------------
# There is deliberately no generic failure-rate helper here.  The circuits in
# this module are the only ones in the package that are allowed to be wrong, so
# their error rates are measured explicitly in tests/test_ec_opt1128.py -- both
# per operation (where the rate should track 2^-msbs) and composed into
# `ec_eea.inplace_mul` (where it compounds over ~1.4n iterations, which is the
# thing that actually decides how large msbs has to be).
