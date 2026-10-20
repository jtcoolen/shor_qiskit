"""Comparison-free ping-pong dialog-GCD [ECDSA.Fail, arXiv:2609.09582 Sec 5.3.3].

The binary-GCD dialog of `ec_eea` ([1128] Alg. 3) compares |u| against |v| every
round, and the comparison's result controls a full-width swap and subtraction.
The ping-pong walk removes both.  It keeps two *odd signed* integers (rho0,
rho1), starting from (p, x*) where x* = x if x is odd and x - p otherwise, and
updates them in a fixed alternating order: even rounds rewrite rho1, odd rounds
rho0.  With t the target and s the source,

    e  = t[1] XOR s[1]          (the bit that separates 1 and 3 mod 4)
    t <- (t + (-1)^e s) / 2

If t and s agree mod 4 their sum is 2 mod 4, otherwise their difference is, so
the numerator is divisible by 2 but not 4 and t stays odd.  gcd is preserved,
|t'| <= max(|t|, |s|), and the round is reversible given e
(t = 2t' - (-1)^e s).  A signed-unit pair (+-1, +-1) is a fixed point, so a
fixed round budget L can over-run safely.

In the circuit a round is: two CNOTs into a fresh transcript qubit e, a
conditional complement (CNOTs), one plain (n+2)-bit addition, a wire rotation
for the halving and one CNOT for the sign extension -- n+1 Toffolis and no
comparator, no swap.  The transcript is L one-bit symbols.

Replay.  Every update is linear, so the same transcript applied mod p to the
coefficient pair (0, y) gives D_x(0, y) = y x^{-1} (eta0, eta1).  Running it
backwards from (eta0 y, eta1 y) gives (0, x y): multiplication.  This module
builds the multiplication (`pp_mul`) and gets division as its exact inverse
(`pp_div`), the same "one dialog, both directions" structure as `ec_eea`.
"""

import math

import ec_adders as A
import ec_modarith as MA
from ec_sim import Reg


# --- classical model ---------------------------------------------------------
def pp_walk(x, p, L):
    """The value walk on integers.  Returns ((rho0, rho1), transcript)."""
    r = [p, x if x & 1 else x - p]
    tau = []
    for i in range(L):
        ti = 1 if i % 2 == 0 else 0
        t, s = r[ti], r[1 - ti]
        e = ((t >> 1) ^ (s >> 1)) & 1
        r[ti] = (t - s) // 2 if e else (t + s) // 2
        tau.append(e)
    return tuple(r), tau


def pp_replay_div(tau, y, p):
    """D_x(0, y) mod p from a transcript: returns the coefficient pair."""
    c = [0, y % p]
    half = pow(2, -1, p)
    for i, e in enumerate(tau):
        ti = 1 if i % 2 == 0 else 0
        c[ti] = (c[ti] + (-1) ** e * c[1 - ti]) * half % p
    return tuple(c)


def pp_converged(x, p, L):
    (r0, r1), _ = pp_walk(x, p, L)
    return abs(r0) == 1 and abs(r1) == 1


_ROUNDS = {}
EXHAUSTIVE_LIMIT = 1 << 17


def pp_rounds_needed(x, p):
    """Rounds until the walk from x reaches a signed-unit pair (a fixed point)."""
    if math.gcd(x, p) != 1:
        raise ValueError(f"{x} is not a unit mod {p}")
    r = [p, x if x & 1 else x - p]
    i = 0
    while abs(r[0]) != 1 or abs(r[1]) != 1:
        ti = 1 if i % 2 == 0 else 0
        t, s = r[ti], r[1 - ti]
        r[ti] = (t - s) // 2 if ((t >> 1) ^ (s >> 1)) & 1 else (t + s) // 2
        i += 1
    return i


def pp_rounds(p):
    """A round budget L that converges for every nonzero x mod p.

    Exact (the worst case over every x) for p < 2^17, which covers every prime
    this package simulates.  Beyond that it returns ceil(4.5n): the exhaustive
    worst case is 3.0n-4.0n for n <= 17 (4.0n at p = 8191, 66 = 3.9n at
    p = 131063) and sampled worst cases fall towards 2.7n at n = 256, so 4.5n
    is a safe *empirical* budget with margin, not a proven one.  The paper's L = 704 = 2.75n at n = 256 is an average-case
    figure and is too small for some inputs already at n = 15 (p = 16411,
    x = 9 needs 43 > 42 rounds).  Pass L explicitly to `pp_mul` to override.
    """
    if p in _ROUNDS:
        return _ROUNDS[p]
    if p >= EXHAUSTIVE_LIMIT:
        return math.ceil(4.5 * p.bit_length())
    worst = max((pp_rounds_needed(x, p) for x in range(1, p)), default=0)
    _ROUNDS[p] = worst
    return worst


def pp_inverse_classical(x, p, L=None):
    """x^{-1} mod p through the walk and the replay -- the circuit's recipe."""
    L = L or pp_rounds(p)
    (r0, r1), tau = pp_walk(x, p, L)
    assert abs(r0) == 1 and abs(r1) == 1, "walk did not converge"
    c0, c1 = pp_replay_div(tau, 1, p)
    return c0 * r0 % p, c1 * r1 % p


# --- circuit -----------------------------------------------------------------
def _halve(ctx, t):
    """t <- t/2 for an even two's-complement t: rotate wires, sign-extend.

    The low bit is 0, so it becomes the new top; one CNOT copies the sign into
    it.  Zero Toffoli.
    """
    h = Reg(list(t[1:]) + [t[0]], getattr(t, "name", "t"))
    ctx.cx(h[-2], h[-1])
    return h


def pp_round(m, t, s, e):
    """One ping-pong round on signed registers t (target) and s (source).

    `e` is a clean qubit that keeps the transcript bit.  Returns the handle
    that holds the updated target.
    """
    ctx, w = m.ctx, len(t)
    ctx.cx(t[1], e[0])
    ctx.cx(s[1], e[0])
    for q in t:                                  # e: t <- ~t, so ~(~t + s) = t - s
        ctx.cx(e[0], q)
    anc = m.anc(w - 1, "ppc")
    A.add(ctx, s, t, anc)
    m.free(anc)
    for q in t:
        ctx.cx(e[0], q)
    return _halve(ctx, t)


def pp_walk_circuit(m, x, p, L):
    """(p, x) -> (eta0, eta1) with a fresh L-qubit transcript.

    Returns (steps, r0, r1, lift): `steps` holds what each round needs to be
    undone (target handle before the round, source handle, transcript qubit).
    """
    ctx, n = m.ctx, len(x)
    w = n + 2
    r0 = m.anc(w, "rho0")
    A.encode_const(ctx, r0, p)
    hi = m.anc(2, "xhi")
    r1 = Reg(list(x) + list(hi), "rho1")

    # odd lift: x even -> x - p.  The flag equals the sign of the result.
    f = m.anc(1, "lift")
    ctx.cx(x[0], f[0])
    ctx.x(f[0])
    cp, a = m.anc(w, "cp"), m.anc(w - 1, "a")
    A.csub_const(ctx, f[0], r1, p, cp, a)
    m.free(cp, a)
    ctx.cx(r1[w - 1], f[0])
    m.free(f)

    regs = [r0, r1]
    steps = []
    for i in range(L):
        ti = 1 if i % 2 == 0 else 0
        e = m.anc(1, "tau")
        t_before, s = regs[ti], regs[1 - ti]
        regs[ti] = pp_round(m, t_before, s, e)
        steps.append((t_before, s, e))
    return steps, regs[0], regs[1], hi


def pp_unwalk_circuit(m, x, p, steps, hi):
    """Exact inverse of `pp_walk_circuit`: restores x and clears everything."""
    ctx, n = m.ctx, len(x)
    w = n + 2
    for t_before, s, e in reversed(steps):
        m.emit_inverse(pp_round, m, t_before, s, e)
        m.free(e)
    r0 = steps[0][1] if steps else None
    r1 = Reg(list(x) + list(hi), "rho1")
    f = m.anc(1, "lift")
    ctx.cx(r1[w - 1], f[0])
    cp, a = m.anc(w, "cp"), m.anc(w - 1, "a")
    A.cadd_const(ctx, f[0], r1, p, cp, a)
    m.free(cp, a)
    ctx.x(f[0])
    ctx.cx(x[0], f[0])
    m.free(f)
    A.encode_const(ctx, r0, p)
    m.free(r0, hi)


def pp_mul(m, x, y, p, L=None):
    """|x, y> -> |x, x*y mod p> for nonzero x, through the ping-pong dialog.

    1. walk x to (eta0, eta1) with transcript tau;
    2. copy y into c0, so (c0, c1) = (y, y) = (eta0, eta1) * (eta0 y, eta1 y);
    3. replay tau backwards, c_t <- 2 c_t - (-1)^e c_s mod p, which ends at
       (0, x y) because D_x(0, xy) = y (eta0, eta1);
    4. unwalk, restoring x and clearing the transcript.

    Lazy signs.  The registers hold a_j = sigma_j c_j for a sign frame sigma
    that is never stored.  Writing sigma_j = (-1)^b_j, the update can always be
    done as "a_t <- (-1)^k a_t; a_t <- 2 a_t + a_s", provided the frame moves
    to b_t' = 1 + e_i + b_s (mod 2).  That value does not depend on b_t, so
    b_t was itself fixed one round earlier to 1 + e_{i+1} + b_s, which makes
    the negation control

        k_i = b_t + b_t' = e_i + e_{i+1}            (mod 2)

    -- two CNOTs on the transcript.  A round is one controlled negation, one
    doubling and one modular addition, against two negations and a
    subtraction for the eager form.  Only the first round (b from the walk's
    terminal signs) and the final output (one cmodneg by the closing frame,
    a parity of the transcript) see the frame explicitly.
    """
    ctx, n = m.ctx, len(x)
    L = L or pp_rounds(p)
    steps, r0, r1, hi = pp_walk_circuit(m, x, p, L)
    signs = [r0[-1], r1[-1]]                     # b_j at the start of replay

    c0 = m.anc(n, "c0")
    for a_, b_ in zip(y, c0):
        ctx.cx(a_, b_)
    c = [c0, y]
    for i in reversed(range(L)):
        ti = 1 if i % 2 == 0 else 0
        e = steps[i][2][0]
        if i == L - 1:                           # k = b_t + 1 + e + b_s
            ctrl = [signs[ti], signs[1 - ti], e]
            flip = True
        else:                                    # k = e_i + e_{i+1}
            ctrl = [steps[i + 1][2][0], e]
            flip = False
        k = m.anc(1, "k")
        for q in ctrl:
            ctx.cx(q, k[0])
        if flip:
            ctx.x(k[0])
        MA.cmodneg(m, k[0], c[ti], p)
        if flip:
            ctx.x(k[0])
        for q in ctrl:
            ctx.cx(q, k[0])
        m.free(k)
        MA.moddbl(m, c[ti], p)
        MA.modadd(m, c[1 - ti], c[ti], p)
    m.free(c0)

    # closing frame of c1: b_1 = (L mod 2) + sum(e) + b_{s, L-1} at the start
    s_first = signs[0 if (L - 1) % 2 == 0 else 1]
    if L:
        bit = [s_first] + [st[2][0] for st in steps]
        k = m.anc(1, "k")
        for q in bit:
            ctx.cx(q, k[0])
        if L % 2:
            ctx.x(k[0])
        MA.cmodneg(m, k[0], y, p)
        if L % 2:
            ctx.x(k[0])
        for q in bit:
            ctx.cx(q, k[0])
        m.free(k)

    pp_unwalk_circuit(m, x, p, steps, hi)


def pp_div(m, x, y, p, L=None):
    """|x, y> -> |x, y/x mod p>: `pp_mul` run backwards."""
    m.emit_inverse(pp_mul, m, x, y, p, L)
