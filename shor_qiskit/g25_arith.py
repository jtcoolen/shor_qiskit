"""Gidney 2025's modular addition of a looked-up value (Sec 2.4).

The windowed multiplications of the factoring half add a *looked-up* value T[b]
to an accumulator modulo N.  `windowed.add_quantum_mod` does it with Beauregard's
five-block adder over the Fourier adder; `rc_adder.rc_c_add_mod` with five
ripple-carry additions, ~10n Toffolis.  Gidney 2025 needs two:

    1. look up T2[b] = p - T[b] instead of T[b], and subtract it into y with one
       extra top qubit Q:  Q = [y < T2]  (the borrow);
    2. if Q, add p back.   Now y = (y + T[b]) mod p;
    3. uncompute Q.  Q = [result + T2 >= p] -- on its own an n-bit comparison,
       but with measurement-based uncomputation it is an X-measurement and, on
       outcome 1 (half the time), a comparison *phase* oracle.

So n (Gidney subtraction) + n (controlled constant addition) + n/2 expected
(half a comparison) = 2.5n Toffolis, against ~10n -- *if* the comparison is a
plain n-bit one.  Here it is not: the comparison is against p - T2, which is not
in a register.  The comparison is done here
by adding the constant 2^(n+1) - p to T2 in n + 2 bits (T2 = p reaches 2^(n+1)
exactly, so n + 1 bits would wrap), reading bit n+1 of result + T2', and
subtracting the constant again: about 3n.  So the static form costs ~5n and
the measured one ~2n + 3n/2 expected -- both well under the five-block
adder's ~10n, but short of Gidney's 2.5n, which needs the comparison value in
a register (a second, cheap lookup of the unflipped T).

Built on `ec_adders` through a `Machine`, so it is checked exhaustively by
`ec_sim` like every EC circuit, and its measured form by `ec_mbu.run_live`.
"""

import ec_adders as A
import ec_mbu as MB
from ec_gates import Ctx
from ec_sim import Reg


def _carry_ge(ctx, r, t2, p, out, cp, sc):
    """out ^= [r + t2 >= p], r and t2 of n bits, t2 <= p.

    Add c = 2^(n+1) - p to t2 in n + 2 bits (no overflow: t2 + c <= 2^(n+1)).
    Then r + t2 >= p  <=>  r + t2 + c >= 2^(n+1)  <=>  bit n+1 of r + (t2 + c),
    which is the carry out of the low n + 1 bits XOR bit n+1 of t2 + c (that
    bit is set only for t2 = p).  cp: n + 2 qubits; sc: n + 4.
    """
    n = len(r)
    c = (1 << (n + 1)) - p
    t2e = Reg(list(t2) + [sc[0], sc[1]])
    re_ = Reg(list(r) + [sc[2]])
    work = sc[3:3 + n + 1]
    A.add_const(ctx, t2e, c, cp[:n + 2], work)
    A.carry_out(ctx, re_, Reg(t2e[:n + 1]), out, work)
    ctx.cx(t2e[n + 1], out)
    A.add_const(ctx, t2e, -c, cp[:n + 2], work)


def _scratch(m, n):
    return m.anc(n + 2, "cp"), m.anc(n + 5, "sc")


def sub_mod_underflow(m, y, t2, p, mode="static"):
    """y <- (y - t2) mod p, for y < p and 0 < t2 <= p (so y + (p - t2)).

    mode="static": the borrow is uncomputed by comparison (exact, ~5n).
    mode="mbu":    by X-measurement and a comparison phase oracle (2.5n expected).
    """
    ctx, n = m.ctx, len(y)
    q = m.anc(1, "Q")
    zero = m.anc(1, "z")
    sc = m.anc(n, "sub")
    A.sub(ctx, Reg(list(t2) + list(zero)), Reg(list(y) + list(q)), sc)   # Q = [y < t2]
    m.free(sc, zero)
    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.cadd_const(ctx, q[0], y, p, cp, sc)                                 # if Q: y += p
    m.free(cp, sc)
    if mode == "static":
        cp, sc = _scratch(m, n)
        _carry_ge(ctx, y, t2, p, q[0], cp, sc)                            # Q ^= [y + t2 >= p]
        m.free(cp, sc)
    else:
        nd = 2 * n

        def recompute(qc, qs):
            yy, tt, f, anc = qs[:n], qs[n:nd], qs[nd], qs[nd + 1:]
            _carry_ge(Ctx(qc, "and"), yy, tt, p, f, anc[:n + 2], anc[n + 2:])

        def phase(qc, qs):
            yy, tt, anc = qs[:n], qs[n:nd], qs[nd + 1:]
            t, rest = anc[0], anc[1:]
            _carry_ge(Ctx(qc, "and"), yy, tt, p, t, rest[:n + 2], rest[n + 2:])
            qc.z(t)
            _carry_ge(Ctx(qc, "and"), yy, tt, p, t, rest[:n + 2], rest[n + 2:])

        MB.mbu_flag(m, q[0], list(y) + list(t2), recompute, phase, nanc=2 * n + 8)
    m.free(q)


def add_mod_lookup(m, y, t, p, mode="static"):
    """y <- (y + t) mod p with t in a register (0 <= t < p): negate t into
    p - t in place (complement + constant), use `sub_mod_underflow`, undo.
    In the windowed circuits the table is stored flipped instead, which makes
    the negation free; this wrapper exists for testing against y + t."""
    ctx, n = m.ctx, len(y)
    te = Reg(list(t) + [m.anc(1, "te")[0]])

    def negate(mm):
        for b in te:
            mm.ctx.x(b)
        cp, sc = mm.anc(n + 1, "cp"), mm.anc(n + 1, "sc")
        A.add_const(mm.ctx, te, p + 1, cp, sc)      # NOT(t) + p + 1 = p - t (n+1 bits)
        mm.free(cp, sc)
    negate(m)
    sub_mod_underflow(m, y, Reg(te[:n]), p, mode)   # p - t <= p fits n bits
    m.emit_inverse(negate, m)
    m.free(Reg([te[n]]))


# =============================================================================
# Windowed multiplication and order finding on these parts ([G25] Sec 2.4)
# =============================================================================
def windowed_mult_add(m, ctrl, x, y, k, p, w, mode="static"):
    """y <- y + ctrl * k * x  (mod p), x windowed w bits at a time.

    Each window's contribution k * a * 2^(jw) mod p is looked up -- *flipped*,
    as p - T, so the subtract-and-underflow adder applies it -- with the
    control folded into the address (Gidney 2019 Fig 5: a zero control reads
    an entry that adds nothing, T2 = p).  The lookup is unloaded by
    measurement (`ec_mbu`), so each window costs one lookup of 2^(w+1) entries,
    one ~2.5n-5n adder and a sqrt-sized repair."""
    n = len(y)
    t = m.anc(n, "T2")
    for j in range(0, len(x), w):
        win = list(x[j:j + w])
        addr = win + ([ctrl] if ctrl is not None else [])
        table = []
        for a in range(1 << len(addr)):
            c = 1 if ctrl is None else (a >> len(win)) & 1
            v = (k * (a & ((1 << len(win)) - 1)) << j) % p if c else 0
            table.append(p - v)                         # flipped: v = 0 -> p
        MB.lookup(m, Reg(addr), t, table)
        sub_mod_underflow(m, y, t, p, mode)             # y -= p - v  ==  y += v
        MB.unlookup(m, Reg(addr), t, table)
    m.free(t)


def c_mult_inplace(m, ctrl, x, k, p, w, mode="static"):
    """x <- k^ctrl x mod p: multiply out of place, swap, clear with k^-1."""
    n = len(x)
    y = m.anc(n, "prod")
    windowed_mult_add(m, ctrl, x, y, k, p, w, mode)     # y = ctrl k x (+0)
    for a, b in zip(x, y):
        m.ctx.cswap(ctrl, a, b)                          # x <-> y when ctrl
    kinv = pow(k, -1, p)
    windowed_mult_add(m, ctrl, x, y, (-kinv) % p, p, w, mode)   # y -= k^-1 (k x) = x
    m.free(y)


def order_circuit_g25(A, N, t=None, w=2, mode="static"):
    """Order finding with the G25 multiplier.  Returns (Machine, info); the
    Machine's circuit ends with the inverse QFT and measurement."""
    import math
    from qiskit.circuit import ClassicalRegister
    from ec_sim import Machine
    from shor_essentials import qft
    n = math.ceil(math.log2(N))
    t = t or 2 * n
    m = Machine("and", "order-g25")
    ctr = m.alloc(t, "ctr")
    x = m.alloc(n, "x")
    out = ClassicalRegister(t, "out")
    m.qc.add_register(out)
    m.ctx.x(x[0])
    for q in ctr:
        m.qc.h(q)
    for i in range(t):
        c_mult_inplace(m, ctr[i], x, pow(A, 1 << i, N), N, w, mode)
    m.qc.append(qft(t).inverse(), list(ctr))
    m.qc.measure(list(ctr), out)
    return m, {"t": t, "n": n, "qubits": m.qc.num_qubits}
