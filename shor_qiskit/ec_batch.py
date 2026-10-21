"""Batch inversion: k modular inverses for the price of one, plus multiplications.

Montgomery's trick [Mon87], put on a quantum computer by Litinski [Lit23]
Fig. 9b / 13.  With the prefix products a_i = x_0 x_1 ... x_i, one inversion
b_{k-1} = a_{k-1}^-1 and the back-substitution b_{i-1} = b_i x_i give every
b_i = a_i^-1, and then

    x_0^-1 = b_0,        x_i^-1 = b_i a_{i-1}   (i >= 1).

That is (k-1) + (k-1) + (k-1) = 3(k-1) multiplications and ONE inversion, where
k separate inversions cost k.  Litinski's use is to run several ECDLP instances
side by side and let their point additions share each inversion; the same
applies to any k independent affine additions in one circuit -- which is what
`point_add_ctrl_batch` does.  Since a clean inversion is ~11 multiplications
here (`ec_kaliski` against `ec_mult`; [Lit23] Fig. 8 has 26n^2 against
2.25n^2 for the garbage-keeping one), it is the largest single saving left in
the affine addition.

Clean, not garbage-keeping
--------------------------
[Lit23] keeps every intermediate (6k - 4 garbage registers, 2k + 2 log k with
Fig. 14's sequencing) and pays for it later.  Here the batch is a closed
operation in this package's style: the forward part

    F = prefix products,  Kaliski inversion,  back-substitution

is recorded, the outputs are written, and F is undone, so every intermediate
-- the a_i, the b_i, the inversion's round records and counter -- is back to
|0> and checked there.  The outputs x_i^-1 = b_i a_{i-1} are multiplied
*directly* into the caller's clean registers, never undone, so the whole thing
costs

    2 Inv + 5(k-1) Mul        against  2k Inv  for k clean inversions

(every clean inversion here is Inv, copy, Inv-dagger).  The break-even is
2 Inv > 5 Mul; measured, 2 Inv = 11.5-12 Mul at n = 4..256, so batching wins
from k = 2 on and each extra element costs ~5/11.5 = 0.43 of an inversion.

A division y/x (`batch_mod_div`) is the same forward part, then per element
o_i = x_i^-1 into scratch, out_i = y_i o_i straight into the output register
(`ec_mult.cmodmul` when controlled), o_i undone: 2 Inv + (7k-6) Mul, plus
(5k-3) n ANDs of gating, against k (2 Inv + 2 Mul) for `ec_kaliski.mod_div`
-- which is also where the direct write helps a single division (k = 1: one
Mul fewer than mod_div).  tests/test_ec_batch.py checks both formulas exactly.
Clearing lambda at step 8 of the point addition runs the output step backwards:
the inverse of out <- y o is out <- (out - y o) / 2^n, which is 0 exactly when
out held y/x -- the same identity [106] Alg. 3 is built around.

Controls, and why x has to be gated
-----------------------------------
A single inversion of 0 would not merely give a wrong answer for one element:
Kaliski's cleanup assumes gcd(x, p) = 1 (see `ec_kaliski._clear_uvs`), so a
zero anywhere in the product leaves the whole batch dirty.  On the q = 0
branch of a controlled point addition the x register holds x1 - x2 at step 3
and x1 + 2 x2 at step 8, and the latter vanishes for perfectly ordinary
accumulators.  `ec_kaliski` avoids this by gating the inversion's setup; the
batch gates each factor instead:

    t_i = x_i  if ctrl_i  else  1,

n ANDs, built where it is used and unbuilt at once (and_dg is free), so an
idle element contributes 1 to the product and 1 to every inverse, and its
output is gated to 0 by the controlled multiplication.

What batching costs is failure probability and space.  One exceptional
addition among the active ones (x_i = 0) zeroes the product and fails the
batch: the failure rate is ~k times one addition's O(1/p), still what Shor's
algorithm tolerates.  And the forward part holds 2(k-1) extra registers (the
a_i and b_i) plus one gated factor while the outputs are written.

Sources: [Mon87] P. L. Montgomery, Speeding the Pollard and elliptic curve
methods of factorization, Math. Comp. 48 (1987), Sec 10.3.1; [Lit23]
D. Litinski, arXiv:2306.08585, Fig. 9 and Figs. 13-14; [106] Alg. 3.
"""

import ec_kaliski as K
import ec_modarith as MA
import ec_mult as MU
from ec_pointadd import _csub_2x2_or_x2


# =============================================================================
# Gated factors: x if ctrl else 1
# =============================================================================
def _gate(m, ctrl, x):
    """t (fresh) <- x if ctrl else 1.  n ANDs.  With no control, x itself."""
    if ctrl is None:
        return x
    ctx, n = m.ctx, len(x)
    t = m.anc(n, "t")
    for j in range(n):
        ctx.and_(ctrl, x[j], t[j])
    ctx.x(ctrl)
    ctx.cx(ctrl, t[0])                        # idle: t = 1
    ctx.x(ctrl)
    return t


def _ungate(m, ctrl, x, t):
    if ctrl is None:
        return
    ctx, n = m.ctx, len(x)
    ctx.x(ctrl)
    ctx.cx(ctrl, t[0])
    ctx.x(ctrl)
    for j in range(n):
        ctx.and_dg(ctrl, x[j], t[j])
    m.free(t)


# =============================================================================
# The forward part: prefix products, one inversion, back-substitution
# =============================================================================
def _forward(m, xs, p, ctrls):
    """a[i] = t_0 ... t_i and b[i] = a[i]^-1, all live.  Returns (a, b, live).

    t_0 stays built (it is a[0], needed for x_1^-1 = b_1 a_0); every other
    t_i is built for its prefix step and again for its back-substitution step.
    """
    k, n = len(xs), len(xs[0])
    cs = ctrls or [None] * k
    a = [_gate(m, cs[0], xs[0])]
    for i in range(1, k):
        t = _gate(m, cs[i], xs[i])
        ai = m.anc(n, f"a{i}")
        MU.modmul(m, a[i - 1], t, ai, p)          # a_i = a_{i-1} t_i
        _ungate(m, cs[i], xs[i], t)
        a.append(ai)

    inv = m.anc(n, "inv")
    recs, cnt = K.mod_inv(m, a[-1], inv, p)      # b_{k-1} = a_{k-1}^-1
    b = [None] * (k - 1) + [inv]
    for i in range(k - 1, 0, -1):
        t = _gate(m, cs[i], xs[i])
        bi = m.anc(n, f"b{i - 1}")
        MU.modmul(m, t, b[i], bi, p)              # b_{i-1} = b_i t_i
        _ungate(m, cs[i], xs[i], t)
        b[i - 1] = bi

    live = ([a[0]] if cs[0] is not None else []) + a[1:] + b
    live += [q for rec in recs for q in rec]
    if cnt is not None:
        live.append(cnt)
    return a, b, live


def _batched(m, xs, p, ctrls, outputs):
    """Run F, then `outputs(a, b)`, then undo F and free everything."""
    mark = m.begin()
    a, b, live = _forward(m, xs, p, ctrls)
    fwd = m.since(mark)
    outputs(a, b)
    m.undo(fwd)
    m.free(*live)


def _inverse_into(m, a, b, i, out, p):
    """out (|0>) <- x_i^-1 = b_i a_{i-1} (i >= 1)."""
    MU.modmul(m, b[i], a[i - 1], out, p)


# =============================================================================
# Public: inverses and quotients
# =============================================================================
def batch_mod_inv(m, xs, outs, p, ctrls=None):
    """outs[i] (|0>) <- x_i^-1 mod p for every i: one inversion, 5(k-1) Mul.

    xs are preserved; every x_i must be nonzero (with `ctrls`, only where
    ctrl_i = 1, and outs[i] <- ctrl_i x_i^-1).  All intermediates uncomputed.
    """
    k = len(xs)
    assert len(outs) == k and k >= 1
    cs = ctrls or [None] * k

    def outputs(a, b):
        for j in range(len(outs[0])):              # x_0^-1 = b_0: a copy
            if cs[0] is None:
                m.ctx.cx(b[0][j], outs[0][j])
            else:
                m.ctx.and_(cs[0], b[0][j], outs[0][j])
        for i in range(1, k):
            if cs[i] is None:
                _inverse_into(m, a, b, i, outs[i], p)
            else:
                MU.cmodmul(m, cs[i], b[i], a[i - 1], outs[i], p)

    _batched(m, xs, p, ctrls, outputs)


def batch_mod_div(m, ctrls, xs, ys, outs, p, clear=False):
    """outs[i] (|0>) <- ctrl_i * y_i / x_i mod p, one inversion for all k.

    clear=True runs the output step backwards: outs[i] holding exactly that
    quotient is returned to |0> (step 8 of the point addition).  `ctrls` may
    be None (all active).  x_i must be nonzero where ctrl_i = 1; idle x_i may
    hold anything, zero included.
    """
    k = len(xs)
    assert len(ys) == len(outs) == k and k >= 1
    cs = ctrls or [None] * k
    n = len(xs[0])

    def write(i, o):
        fn, args = (MU.modmul, (m, ys[i], o, outs[i], p)) if cs[i] is None else \
            (MU.cmodmul, (m, cs[i], ys[i], o, outs[i], p))
        if clear:
            m.emit_inverse(fn, *args)
        else:
            fn(*args)

    def outputs(a, b):
        write(0, b[0])
        for i in range(1, k):
            o = m.anc(n, "o")
            _inverse_into(m, a, b, i, o, p)
            write(i, o)
            m.emit_inverse(_inverse_into, m, a, b, i, o, p)
            m.free(o)

    _batched(m, xs, p, ctrls, outputs)


# =============================================================================
# k controlled point additions sharing each inversion
# =============================================================================
def point_add_ctrl_batch(m, qs, x1s, y1s, pts, p):
    """(x1_i, y1_i) <- (x1_i, y1_i) + pts[i] when qs[i], for i < k.

    k independent `ec_pointadd.point_add_ctrl`s, [106] Alg. 3 step for step,
    except that the two divisions (steps 3 and 8) are each one
    `batch_mod_div` over all k: 2 inversions in total instead of 2k.
    pts[i] = (x2, y2) is classical.  Every active addition must be
    non-exceptional (ec_classical.point_add_exceptional); idle ones may hold
    any accumulator.
    """
    k, n = len(qs), len(x1s[0])
    assert len(x1s) == len(y1s) == len(pts) == k
    lams = [m.anc(n, f"lam{i}") for i in range(k)]

    for q, x, y, (x2, y2) in zip(qs, x1s, y1s, pts):
        MA.modsub_const(m, x, x2, p)                   # 1  x <- x1 - x2
        MA.cmodsub_const(m, q, y, y2, p)               # 2  y <- y1 - q y2
    batch_mod_div(m, qs, x1s, y1s, lams, p)            # 3  lambda_i <- q y/x
    for q, x, y, lam, (x2, y2) in zip(qs, x1s, y1s, lams, pts):
        MU.modmul_xor(m, x, lam, y, p)                 # 4  y <- 0
        MA.modadd_const(m, x, 3 * x2 % p, p)           # 5  x <- x + 3 x2
        MU.modsqr_sub(m, lam, x, p)                    # 6  x <- x - lambda^2
        MU.modmul_add(m, x, lam, y, p)                 # 7  y <- x lambda
    batch_mod_div(m, qs, x1s, y1s, lams, p, clear=True)  # 8  lambda_i <- 0
    for q, x, y, (x2, y2) in zip(qs, x1s, y1s, pts):
        _csub_2x2_or_x2(m, q, x, x2, p)                # 9  x <- -x3
        MA.cmodsub_const(m, q, y, y2, p)               # 10 y <- y3
        MA.cmodneg(m, q, x, p)                         # 11 x <- x3

    m.free(*lams)


# =============================================================================
# Costs: k separate operations against one batch, built (not simulated)
# =============================================================================
def mod_inv_clean(m, x, out, p):
    """out (|0>) <- x^-1: `ec_kaliski.mod_inv`, copy, undo.  The unbatched
    reference -- exactly `batch_mod_inv` with k = 1."""
    batch_mod_inv(m, [x], [out], p)


def prime_below(bound):
    """The largest prime below `bound` (Miller-Rabin, deterministic < 3.3e24;
    for the pseudo-Mersenne sizes pass p explicitly)."""
    def is_prime(v):
        if v < 2:
            return False
        for f in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if v % f == 0:
                return v == f
        d, r = v - 1, 0
        while d % 2 == 0:
            d, r = d // 2, r + 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a, d, v)
            if x in (1, v - 1):
                continue
            for _ in range(r - 1):
                x = x * x % v
                if x == v - 1:
                    break
            else:
                return False
        return True
    v = bound - 1
    while not is_prime(v):
        v -= 1
    return v


def _machine_for(kind, n, k, p, batched, mk=None):
    from ec_sim import Machine
    import ec_pointadd as PA
    m = (mk or Machine)("and")
    qs = m.alloc(k, "q") if kind != "inv" else None
    xs = [m.alloc(n, f"x{i}") for i in range(k)]
    if kind == "inv":
        outs = [m.alloc(n, f"o{i}") for i in range(k)]
        if batched:
            batch_mod_inv(m, xs, outs, p)
        else:
            for x, o in zip(xs, outs):
                mod_inv_clean(m, x, o, p)
    elif kind == "div":
        ys = [m.alloc(n, f"y{i}") for i in range(k)]
        outs = [m.alloc(n, f"o{i}") for i in range(k)]
        if batched:
            batch_mod_div(m, list(qs), xs, ys, outs, p)
        else:
            for q, x, y, o in zip(qs, xs, ys, outs):
                K.mod_div(m, q, x, y, o, p)
    else:
        assert kind == "padd", kind
        ys = [m.alloc(n, f"y{i}") for i in range(k)]
        pts = [((3 + 2 * i) % p, (5 + 7 * i) % p) for i in range(k)]
        if batched:
            point_add_ctrl_batch(m, list(qs), xs, ys, pts, p)
        else:
            for q, x, y, (x2, y2) in zip(qs, xs, ys, pts):
                PA.point_add_ctrl(m, q, x, y, x2, y2, p)
    return m


HIER_TARGETS = [("ec_kaliski", ["kaliski_round"])]


def batch_costs(n, k, kind="inv", p=None, separate="scale", hier=False):
    """Toffoli-eq and qubits of k separate operations against one batch.

    kind: "inv" (clean inversions / batch_mod_inv), "div" (ec_kaliski.mod_div
    / batch_mod_div, controlled), "padd" (ec_pointadd.point_add_ctrl /
    point_add_ctrl_batch).  Gate counts do not depend on the values, only on
    n, k and p.  separate="scale" builds ONE separate operation and scales it
    (the k are sequential, share the ancilla pool, and cost exactly k times
    one; qubits add k-1 sets of the operation's own registers), "build" builds
    all k.  hier=True builds through `hier` (exact, and fast enough for
    n = 256).  Returns {"separate": {...}, "batched": {...}}, each with
    toffoli and qubits.
    """
    import contextlib
    import ec_cost as CO
    p = p or prime_below(1 << n)
    mk, counter, scope = None, CO.count, contextlib.nullcontext()
    if hier:
        import hier as H
        mk, counter = H.HierMachine, H.count
        scope = H.tracing(H.DEFAULT_TARGETS + HIER_TARGETS)
    with scope:
        b = counter(_machine_for(kind, n, k, p, True, mk))
        if separate == "build":
            s = counter(_machine_for(kind, n, k, p, False, mk))
            sep = {"toffoli": s["toffoli_paper"], "qubits": s["qubits"]}
        else:
            one = counter(_machine_for(kind, n, 1, p, False, mk))
            regs = {"inv": 2 * n, "div": 3 * n + 1, "padd": 2 * n + 1}[kind]
            sep = {"toffoli": k * one["toffoli_paper"],
                   "qubits": one["qubits"] + (k - 1) * regs}
    return {"separate": sep,
            "batched": {"toffoli": b["toffoli_paper"], "qubits": b["qubits"]}}
