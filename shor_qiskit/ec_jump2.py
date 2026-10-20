"""Jump-2 Euclidean dialog [ECDSA.Fail, arXiv:2609.09582 Sec 5.3.1-5.3.2].

[1128] Alg. 2 spends one round -- a comparison, a swap, a subtraction and a
halving -- per binary-GCD divstep.  Jump-2 retimes the same divsteps into
macro-steps.  After a subtraction of two odd operands v is even, so the
halving that must follow is moved to the start of the next macro-step, where
it is unconditional (a wire relabelling, free).  If v/2 is still even a second
halving is done before the next subtraction can happen:

    i = 0:  t1 <- [v even]; v <- v/2 if t1       (x's parity is unknown)
    i > 0:  v <- v/2                             (always even here)
            s2 <- [v even]; v <- v/2 if s2
            b  <- [v odd]                        (subtract?)
            s  <- b if i = 0 else b AND [v < u]  (swap first?)
            (u, v) <- (v, u) if s;  v <- v - u if b

One symbol (b, s, s2) per macro-step, and the macro-steps converge in about
1.3n against [1128]'s 1.41n + 2.4 sqrt(n) rounds, so the comparisons, swaps and
subtractions are all cut.  The induced map on (u, v) is unchanged, so the same
"one dialog, both directions" replay works.  Reversed, macro-step i is

    if b: z <- z + y;  if s: (y, z) <- (z, y);  z <- 2^k z,
    k = t1 + s2 (i = 0) or 1 + s2 (i > 0)

and starting the reverse replay at (y, 0) ends at (0, x y).

Base-5 codec (Sec 5.3.2, opt-in `codec=True`).  Only five of the eight
(b, s, s2) triples occur -- s2 = 0 forces b = 1, s = 1 forces b = 1 -- so
three symbols (125 states) fit in 7 qubits instead of 9.  The encoder here is
(i) a CNOT that makes a b = 0 symbol read 000, after which the digit is
d = b + 2s + s2 in {0..4}, a *linear* function of the bits; (ii) z = d0 + 5d1 +
25d2 accumulated into seven fresh qubits by controlled constant additions;
(iii) the nine symbol qubits cleared by a 125-entry table lookup on z.  The
decoder is the exact inverse.  It costs O(1) adder work per three symbols
on 7-bit registers, plus a 128-entry lookup, so at toy widths it is more
expensive than the macro-steps it compresses (p = 61: 3471 vs 1007 Toffolis)
and the scratch it needs while encoding outweighs the two qubits it saves per
triple until n ~ 12 (p = 4093: 114 vs 115 qubits; p = 65521: 147 vs 150).
Without register sharing (Sec 5.3.5) the freed qubits only lower the peak
when transcript storage dominates, which is the regime the paper targets.
"""

import ec_adders as A
import ec_modarith as MA
from ec_sim import Reg
from qrom import lookup_ui


# --- classical model ---------------------------------------------------------
def j2_walk(x, p, L):
    """The macro-step walk.  Returns ((u, v), [(b, s, s2, t1)])."""
    u, v = p, x
    tau = []
    for i in range(L):
        t1 = 0
        if i == 0:
            t1 = int(v % 2 == 0)
            if t1:
                v //= 2
        else:
            v //= 2
        s2 = int(v % 2 == 0)
        if s2:
            v //= 2
        b = v & 1
        s = b if i == 0 else int(b and v < u)
        if s:
            u, v = v, u
        if b:
            v -= u
        tau.append((b, s, s2, t1))
    return (u, v), tau


def j2_replay_mul(tau, y, p):
    """The reverse replay from (y, 0); ends at (0, x y) mod p."""
    Y, Z = y % p, 0
    for i in reversed(range(len(tau))):
        b, s, s2, t1 = tau[i]
        if b:
            Z = (Z + Y) % p
        if s:
            Y, Z = Z, Y
        Z = Z * pow(2, (t1 + s2) if i == 0 else 1 + s2, p) % p
    return Y, Z


def j2_rounds_needed(x, p):
    L = 1
    while j2_walk(x, p, L)[0] != (1, 0):
        L += 1
    return L


_ROUNDS = {}


def j2_rounds(p):
    """A macro-step budget that converges for every unit x mod p.

    (1, 0) is a fixed point -- v = 0 halves to 0 and records (0, 0, 1) -- so a
    budget larger than an input needs is harmless.  Exact (worst case over all
    x, cached) for p < 2^17.  Beyond that it returns ceil(1.5n) + 8: the
    exhaustive worst case is at most 1.36n for n <= 16, and sampled worst
    cases fall to 1.02n at n = 256 (the paper's budget is 261), so this is an
    empirical budget with margin, not a proven bound.
    """
    if p not in _ROUNDS:
        if p >= 1 << 17:
            return -(-3 * p.bit_length() // 2) + 8
        _ROUNDS[p] = max(j2_rounds_needed(x, p) for x in range(1, p))
    return _ROUNDS[p]


# --- base-5 codec ------------------------------------------------------------
SYMBOLS = ((0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1))


def digit(b, s, s2):
    """The bijection phi: S -> {0..4} that the encoder realises."""
    return 0 if b == 0 else 1 + 2 * s + s2


def _normalise(ctx, sym):
    """b = 0 forces s2 = 1; flip it so a b = 0 symbol reads 000.  Self-inverse."""
    b, _, s2 = sym
    ctx.x(b)
    ctx.cx(b, s2)
    ctx.x(b)


def _codec_table():
    """z -> the nine normalised symbol bits, for z < 125."""
    def bits(d):
        if d == 0:
            return (0, 0, 0)
        d -= 1
        return (1, d >> 1, d & 1)
    table = []
    for z in range(128):
        if z >= 125:
            table.append(0)
            continue
        word = 0
        for k, d in enumerate((z % 5, z // 5 % 5, z // 25)):
            for j, bit in enumerate(bits(d)):
                word |= bit << (3 * k + j)
        table.append(word)
    return table


def encode_triple(m, syms, z):
    """Three raw symbols -> z = d0 + 5 d1 + 25 d2 in the clean 7-qubit `z`.

    The symbol qubits come back to |0>.  (The b = 0 normalisation makes the
    digit linear, d = b + 2 s + s2; the plain `b + s + s2` count would not be.)
    """
    ctx = m.ctx
    flat = []
    for sym in syms:
        _normalise(ctx, sym)
        flat.extend(sym)
    for k, (b, s, s2) in enumerate(syms):
        for bit, w in ((b, 1), (s, 2), (s2, 1)):
            cp, sc = m.anc(len(z), "cp"), m.anc(len(z), "sc")
            A.cadd_const(ctx, bit, z, w * 5 ** k, cp, sc)
            m.free(cp, sc)
    one = m.anc(1, "one")
    ctx.x(one[0])
    anc = m.anc(len(z), "lk")
    lookup_ui(ctx, one[0], z, Reg(flat, "sym"), _codec_table(), anc)
    m.free(anc)
    ctx.x(one[0])
    m.free(one)


# --- circuit -----------------------------------------------------------------
def _chalve(m, ctrl, v):
    """v <- v/2 if ctrl, for v even when ctrl: walk the bits down one place."""
    for i in range(len(v) - 1):
        m.ctx.cswap(ctrl, v[i], v[i + 1])


def j2_step(m, u, v, sym, first):
    """One macro-step on (u, v); `sym` = (b, s, s2[, t1]) clean qubits.

    Returns the handle now holding v (the unconditional halving relabels it).
    """
    ctx, n = m.ctx, len(u)
    b, s, s2 = sym[:3]
    if first:
        t1 = sym[3]
        ctx.cx(v[0], t1)
        ctx.x(t1)
        _chalve(m, t1, v)
    else:
        v = Reg(list(v[1:]) + [v[0]], "v")
    ctx.cx(v[0], s2)
    ctx.x(s2)
    _chalve(m, s2, v)
    ctx.cx(v[0], b)
    if first:
        ctx.cx(b, s)
    else:
        t, sc = m.anc(1, "t"), m.anc(n, "sc")
        A.gt_uint(ctx, u, v, t[0], sc)
        ctx.and_(b, t[0], s)
        A.gt_uint(ctx, u, v, t[0], sc)
        m.free(t, sc)
    for i in range(n):
        ctx.cswap(s, u[i], v[i])
    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b, u, v, cp, sc)
    m.free(cp, sc)
    return v


def j2_walk_circuit(m, x, p, L, codec=False):
    """Consume x into a transcript.  Ends at (u, v) = (1, 0); u is cleared.

    Returns (steps, groups): `steps[i]` = (v handle before step i, symbol
    register or None if packed), `groups` = [(first index, z register)].  With
    `codec`, each complete triple of ordinary symbols (indices 1-3, 4-6, ...;
    symbol 0 carries t1 and stays raw) is packed into 7 qubits as soon as it
    is finished and its nine qubits go back to the pool.
    """
    ctx, n = m.ctx, len(x)
    u = m.anc(n, "u")
    A.encode_const(ctx, u, p)
    v = Reg(list(x), "v")
    steps, groups = [], []
    for i in range(L):
        sym = m.anc(4 if i == 0 else 3, f"sym{i}")
        steps.append([v, sym])
        v = j2_step(m, u, v, sym, i == 0)
        start = i - 2
        if codec and i >= 3 and start % 3 == 1:
            z = m.anc(7, f"z{start}")
            trip = [steps[start + k][1] for k in range(3)]
            encode_triple(m, trip, z)
            m.free(*trip)
            for k in range(3):
                steps[start + k][1] = None
            groups.append((start, z))
    ctx.x(u[0])                                   # u = 1 -> 0
    m.free(u)
    return steps, groups


def _decoded(m, steps, groups, consume):
    """Yield (index, symbol register) in reverse order, decoding packed
    triples on demand.  Without `consume` (replay: symbols only read) the
    triple is re-packed afterwards; with it (unwalk: the caller clears the
    symbols) the emptied z is released instead."""
    gstart = {s: z for s, z in groups}
    i = len(steps) - 1
    while i >= 0:
        start = i - 2
        if start in gstart:
            trip = [m.anc(3, f"dec{start + k}") for k in range(3)]
            m.emit_inverse(encode_triple, m, trip, gstart[start])
            for k in reversed(range(3)):
                yield start + k, trip[k]
            if consume:
                m.free(gstart[start])
            else:
                encode_triple(m, trip, gstart[start])
                m.free(*trip)
            i = start - 1
        else:
            yield i, steps[i][1]
            i -= 1


def j2_mul(m, x, y, p, L=None, codec=False):
    """|x, y> -> |x, x*y mod p> for nonzero x, through the Jump-2 dialog.

    walk (x -> transcript), reverse replay onto (y, 0), unwalk.  Exact for
    every nonzero x when L >= `j2_rounds(p)`.
    """
    ctx, n = m.ctx, len(x)
    L = L or j2_rounds(p)
    steps, groups = j2_walk_circuit(m, x, p, L, codec)

    st = {"Y": y, "Z": m.anc(n, "bz")}
    for i, sym in _decoded(m, steps, groups, False):
        b, s, s2 = sym[:3]
        MA.cmodadd(m, b, st["Y"], st["Z"], p)            # if b: z += y
        for a_, b_ in zip(st["Y"], st["Z"]):
            ctx.cswap(s, a_, b_)                           # if s: swap
        if i == 0:
            MA.cmoddbl(m, sym[3], st["Z"], p)              # z <- 2^(t1+s2) z
        else:
            st["Z"] = MA.moddbl_rotate(m, st["Z"], p)      # z <- 2^(1+s2) z
        MA.cmoddbl(m, s2, st["Z"], p)
    for a_, b_ in zip(y, st["Z"]):                         # (0, xy) -> y
        ctx.swap(a_, b_)
    m.free(st["Z"])

    u = m.anc(n, "u")                                      # unwalk
    ctx.x(u[0])
    for i, sym in _decoded(m, steps, groups, True):
        m.emit_inverse(j2_step, m, u, steps[i][0], sym, i == 0)
        m.free(sym)
    m.emit_inverse(A.encode_const, ctx, u, p)
    m.free(u)


def j2_div(m, x, y, p, L=None, codec=False):
    """|x, y> -> |x, y/x mod p>: `j2_mul` run backwards."""
    m.emit_inverse(j2_mul, m, x, y, p, L, codec)
