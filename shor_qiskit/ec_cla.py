"""Logarithmic-depth carry-lookahead addition [DKRS04], from temporary ANDs.

Every adder in `ec_adders` is a ripple: the carry into bit i waits for the
carry into bit i - 1, so Gidney's adder has Toffoli depth n - 1 and CDKM 2n.
[106] Sec 3.1 is depth-optimised and uses the Draper-Kutin-Rains-Svore
carry-lookahead adder instead; this module is that adder, built from this
package's gates so its cost is counted the same way.

The carries of x + y are a prefix computation over (generate, propagate)
pairs, g_i = x_i AND y_i, p_i = x_i XOR y_i, with

    (g, p)[i, k] = (g[j, k] XOR (g[i, j] AND p[j, k]),  p[i, j] AND p[j, k])

(g and p of one block are never both 1, so the OR of the textbook rule is an
XOR).  [DKRS04] Sec 3 evaluates it in place on k bit positions, holding the
generates in a register Z (Z_j = g_{j-1} on entry, the carry c_j on exit):

    P rounds   P_t[m] = P_{t-1}[2m] AND P_{t-1}[2m+1]       block propagates
    G rounds   Z[2^t (m+1)] ^= Z[2^t m + 2^(t-1)] AND P_{t-1}[2m+1]
    C rounds   Z[2^t m + 2^(t-1)] ^= Z[2^t m] AND P_{t-1}[2m]      (t down)
    P^-1       the P rounds backwards

The P rounds write clean ancillas, so here they are temporary ANDs and the
P^-1 rounds cost nothing.  The G and C rounds XOR onto a *dirty* Z and stay
Toffolis.  Toffoli depth ~ log k (G) + log(2k/3) (C).

In-place addition ([DKRS04] Sec 4.2) computes the carries, writes the sum
s_i = p_i XOR c_i onto y, and must then erase Z without the inputs it was
computed from.  The trick: the carries of x + NOT(s) are the carries of x + y
(induction on MAJ(x_i, NOT s_i, c_i) = MAJ(x_i, y_i, c_i)).  So complement s,
run the carry network *backwards* on (x, NOT s) -- which turns carries back
into generates -- and uncompute those generates with free AND-daggers.  The
cost is two networks and one set of generates: 7k - 4w(k) - 4 floor(log k) - 2
Toffoli-equivalents for k positions (k = n - 1 mod 2^n, k = n with a carry
out), against Gidney's n - 1, at Toffoli depth ~ 4 log n against n - 1.
That trade -- ~7x the Toffolis and ~n more qubits for exponentially less
depth -- is exactly [106]'s choice.

Comparison needs only the top carry, so it gets its own shape: a balanced
(g, p) tree with *every* node on a fresh qubit.  All of its ANDs then have
clean targets, and the whole tree unwinds for free after the carry is copied
out -- the `ec_adders.carry_out` pattern, at depth ceil(log n) + 1 instead of
n: 3n - 2 - ceil(log n) ANDs and as many ancillas.

A classical operand needs no generate ANDs at all: g_i is y_i or 0 and p_i is
y_i or NOT y_i.  Every routine takes x as a register or a classical int, and
internally as a list mixing qubits and 0/1 -- which is how `cla_modadd` adds a
controlled constant without spending ANDs on its zero bits.

Modular addition is [106] Fig. 3's structure in the compare-then-subtract
order: y += x with the carry into a spare qubit, flag [x + y >= p] with the
constant comparator, subtract p under the flag (the carry register is its
borrow and comes back clean), and clear the flag with [answer < x] -- exactly
the flag, because a reduced sum is < x and an unreduced one is >= x
(`ec_modarith.cmodadd`'s argument).

[DKRS04] T. G. Draper, S. A. Kutin, E. M. Rains, K. M. Svore, "A
         logarithmic-depth quantum carry-lookahead adder", quant-ph/0406142.
[106]    H. Kim et al., "New quantum circuits for ECDLP: breaking prime
         elliptic curve cryptography", eprint 2026/106, Sec 3.1, Figs. 3-4.
[Gid18]  C. Gidney, "Halving the cost of quantum addition", Quantum 2, 74.
"""

from ec_adders import _bits


# --- operands: a register, a classical int, or a mix -------------------------
def _operand(x, n):
    """x as n per-bit literals: each a qubit or a classical 0/1, zero-extended."""
    if isinstance(x, int):
        return _bits(x, n)
    xs = list(x)
    assert len(xs) <= n, "operand wider than the target"
    return xs + [0] * (n - len(xs))


def _gen(ctx, a, b, z, undo=False):
    """z ^= a AND b onto a clean z (undo: z holds a AND b, and is cleared)."""
    if isinstance(a, int):
        if a:
            ctx.cx(b, z)
    elif undo:
        ctx.and_dg(a, b, z)
    else:
        ctx.and_(a, b, z)


def _prop(ctx, a, b):
    """b ^= a."""
    if isinstance(a, int):
        if a:
            ctx.x(b)
    else:
        ctx.cx(a, b)


# --- the [DKRS04] carry network, in place on Z --------------------------------
def _log2(k):
    return k.bit_length() - 1


def _p_rounds(ctx, P, k, undo=False):
    """Block propagates P_t[m] = p[2^t m, 2^t (m+1)], m >= 1, onto clean ancillas."""
    ts = [t for t in range(1, _log2(k) + 1) if k >> t >= 2]
    for t in (reversed(ts) if undo else ts):
        for mm in range(1, k >> t):
            (ctx.and_dg if undo else ctx.and_)(P[t - 1, 2 * mm], P[t - 1, 2 * mm + 1],
                                               P[t, mm])


def _g_rounds(ctx, Z, P, k, undo=False):
    """Z[2^t (m+1)] becomes the generate of the whole aligned block."""
    ts = range(1, _log2(k) + 1)
    for t in (reversed(ts) if undo else ts):
        h = 1 << (t - 1)
        for mm in range(k >> t):
            ctx.ccx(Z[(mm << t) + h], P[t - 1, 2 * mm + 1], Z[(mm + 1) << t])


def _c_rounds(ctx, Z, P, k, undo=False):
    """Carries into the positions the G rounds left half-finished, t descending."""
    ts = [t for t in range(1, _log2(k) + 1) if 3 << (t - 1) <= k]
    for t in (ts if undo else reversed(ts)):
        h = 1 << (t - 1)
        for mm in range(1, (k - h) // (1 << t) + 1):
            ctx.ccx(Z[mm << t], P[t - 1, 2 * mm], Z[(mm << t) + h])


def _n_prop(k):
    """P ancillas for k positions: k - w(k) - floor(log k)."""
    return sum((k >> t) - 1 for t in range(1, _log2(k) + 1) if k >> t >= 2)


def _carries(m, Z, p0, k, undo=False):
    """Z_j: generate g_{j-1} -> carry c_j, j = 1..k (undo: the way back).

    p0[i] holds p_i.  Both directions recompute the P tree from p0 and
    uncompute it with AND-daggers, so only the G and C rounds cost twice.
    """
    ctx = m.ctx
    pa = m.anc(_n_prop(k), "cla_p")
    P = {(0, i): p0[i] for i in range(k)}
    it = iter(pa)
    for t in range(1, _log2(k) + 1):
        for mm in range(1, k >> t):
            P[t, mm] = next(it)
    # P before G/C: the hi half's propagate feeds the next P round first, so
    # the G rounds trail the P rounds by one layer instead of alternating.
    _p_rounds(ctx, P, k)
    if undo:
        _c_rounds(ctx, Z, P, k, undo=True)
        _g_rounds(ctx, Z, P, k, undo=True)
    else:
        _g_rounds(ctx, Z, P, k)
        _c_rounds(ctx, Z, P, k)
    _p_rounds(ctx, P, k, undo=True)
    m.free(pa)


def _add_inplace(m, xs, y, carry):
    """y <- y + xs (mod 2^n); carry ^= the carry-out if carry is given."""
    ctx, n = m.ctx, len(y)
    k = n if carry is not None else n - 1
    if k == 0:
        _prop(ctx, xs[0], y[0])
        return
    z = m.anc(k, "cla_z")
    Z = {j: z[j - 1] for j in range(1, k + 1)}
    for i in range(k):
        _gen(ctx, xs[i], y[i], Z[i + 1])
    for i in range(n):
        _prop(ctx, xs[i], y[i])                    # y_i = p_i
    _carries(m, Z, y, k)                           # Z_j = c_j
    if carry is not None:
        ctx.cx(Z[k], carry)
    for i in range(1, n):
        ctx.cx(Z[i], y[i])                         # y = s
    # erase Z: the carries of x + NOT(s) are the carries of x + y
    for i in range(k):
        ctx.x(y[i])
        _prop(ctx, xs[i], y[i])                    # p'_i = x_i XOR NOT s_i
    _carries(m, Z, y, k, undo=True)                # Z_{i+1} = x_i AND NOT s_i
    for i in range(k):
        _prop(ctx, xs[i], y[i])
        _gen(ctx, xs[i], y[i], Z[i + 1], undo=True)
        ctx.x(y[i])
    m.free(z)


# --- public: addition and subtraction -----------------------------------------
def cla_add(m, x, y, carry=None):
    """y <- (y + x) mod 2^n, in place, at Toffoli depth O(log n).  [DKRS04].

    x: a register no wider than y (zero-extended), or a classical int.  With
    `carry` (a qubit), also carry ^= the carry-out, so y + [carry] is the
    (n+1)-bit sum when carry starts at 0.  x is restored; ancillas come back
    clean.  7k - 4w(k) - 4 floor(log k) - 2 Toffolis for k = n - 1 positions
    (k = n with a carry); k fewer for a classical x.
    """
    assert len(y) >= 1
    _add_inplace(m, _operand(x, len(y)), y, carry)


def cla_sub(m, x, y, borrow=None):
    """y <- (y - x) mod 2^n; borrow ^= [x > y].  y - x = NOT(NOT(y) + x)."""
    for q in y:
        m.ctx.x(q)
    cla_add(m, x, y, borrow)
    for q in y:
        m.ctx.x(q)


# --- the comparator: a (g, p) tree on fresh qubits, unwound for free ---------
# A literal is a classical 0/1 or (qubit, negated).
def _and(ctx, a, b, fresh):
    """(a AND b as a literal, whether it sits on a qubit of its own)."""
    if a == 0 or b == 0:
        return 0, False
    if a == 1:
        return b, False
    if b == 1:
        return a, False
    t = fresh()
    for q, neg in (a, b):
        if neg:
            ctx.x(q)
    ctx.and_(a[0], b[0], t)
    for q, neg in (a, b):
        if neg:
            ctx.x(q)
    return (t, False), True


def _load(ctx, a, t):
    """t ^= literal a."""
    if a == 1:
        ctx.x(t)
    elif a != 0:
        ctx.cx(a[0], t)
        if a[1]:
            ctx.x(t)


def _xor(ctx, a, b, b_own, fresh):
    """a XOR b; b is overwritten when it sits on a qubit of its own."""
    if a == 0:
        return b
    if b == 0:
        return a
    if not b_own:
        t = fresh()
        _load(ctx, b, t)
        b = (t, False)
    _load(ctx, a, b[0])
    return b


def _generate(m, xs, y, fresh):
    """The block generate of all n positions, i.e. the carry-out of x + y.

    Leaves the tree standing (y holds p where x is quantum); the caller copies
    the result and unwinds everything with `Machine.undo`.
    """
    ctx, n = m.ctx, len(y)
    lit = [a if isinstance(a, int) else (a, False) for a in xs]
    g = [_and(ctx, lit[i], (y[i], False), fresh)[0] for i in range(n)]
    p = []
    for i in range(n):
        if isinstance(xs[i], int):
            p.append((y[i], bool(xs[i])))          # y or NOT y: no gate
        else:
            ctx.cx(xs[i], y[i])
            p.append((y[i], False))
    nodes = list(zip(g, p))
    while len(nodes) > 1:
        pairs = [(nodes[j], nodes[j + 1]) for j in range(0, len(nodes) - 1, 2)]
        # the leftmost block's propagate is never read (carry-in is 0); P first,
        # so a hi child's P is free again when its parent's G wants it
        ps = [None] + [_and(ctx, hi[1], lo[1], fresh)[0] for lo, hi in pairs[1:]]
        gs = []
        for lo, hi in pairs:
            t, own = _and(ctx, hi[1], lo[0], fresh)
            gs.append(_xor(ctx, hi[0], t, own, fresh))
        nxt = list(zip(gs, ps))
        if len(nodes) % 2:
            nxt.append(nodes[-1])
        nodes = nxt
    return nodes[0][0]


def cla_carry_out(m, x, y, out):
    """out ^= carry-out of (x + y), at Toffoli depth ceil(log n) + 1.

    x: a register (zero-extended) or a classical int.  x and y restored, all
    ancillas clean.  3n - 2 - ceil(log n) ANDs for a quantum x, about 2n for a
    classical one; every uncompute is a free AND-dagger.
    """
    n = len(y)
    assert n >= 1
    xs = _operand(x, n)
    regs = []

    def fresh():
        regs.append(m.anc(1, "cla_t"))
        return regs[-1][0]

    mark = m.begin()
    root = _generate(m, xs, y, fresh)
    body = m.since(mark)
    _load(m.ctx, root, out)
    m.undo(body)
    if regs:
        m.free(*regs)


def cla_lt(m, x, y, out):
    """out ^= [x < y].  y + NOT(x) carries out exactly when y > x."""
    for q in x:
        m.ctx.x(q)
    cla_carry_out(m, x, y, out)
    for q in x:
        m.ctx.x(q)


def cla_geq_const(m, y, k, out):
    """out ^= [y >= k] for classical k: the carry of y + (2^n - k).

    No generate ANDs (the operand is classical); the ends are decided
    classically, as in `ec_adders.geq_const`.
    """
    n = len(y)
    if k <= 0:
        m.ctx.x(out)
        return
    if k >= 1 << n:
        return
    cla_carry_out(m, (1 << n) - k, y, out)


# --- modular addition ---------------------------------------------------------
def cla_modadd(m, x, y, p):
    """y <- (y + x) mod p for classical odd p and x, y in [0, p), exactly.

    [106] Fig. 3 in the compare-then-subtract order, every step logarithmic:

        (y, hi) += x            cla_add, carry-out into hi
        f ^= [(y, hi) >= p]     constant comparator, no generate ANDs
        (y, hi) -= f ? p : 0    p copied under f ([106] Fig. 4(b)) onto
                                popcount(p) qubits, then cla_sub; hi is the
                                borrow, which clears it
        f ^= [y < x]            a reduced sum is < x, an unreduced one >= x
    """
    ctx, n = m.ctx, len(y)
    assert len(x) == n and p & 1 and p < 1 << n
    hi, f = m.anc(1, "cla_hi"), m.anc(1, "cla_f")
    cla_add(m, x, y, hi[0])
    cla_geq_const(m, y + hi, p, f[0])
    ones = [i for i in range(n) if p >> i & 1]
    cp = m.anc(len(ones), "cla_cp")
    xs = [0] * n
    for j, i in enumerate(ones):
        ctx.cx(f[0], cp[j])
        xs[i] = cp[j]
    cla_sub(m, xs, y, hi[0])
    for j in range(len(ones)):
        ctx.cx(f[0], cp[j])
    m.free(cp, hi)
    cla_lt(m, y, x, f[0])
    m.free(f)


def cla_modsub(m, x, y, p):
    """y <- (y - x) mod p: `cla_modadd` run backwards."""
    m.emit_inverse(cla_modadd, m, x, y, p)


# --- closed forms -------------------------------------------------------------
def _w(k):
    return bin(k).count("1")


def cla_cost(n):
    """Closed forms at width n (quantum operands), checked by the tests.

    toffoli   Toffoli-equivalents as the papers count them (AND + CCX)
    ancillas  clean qubits drawn beyond x, y (and the carry / out qubit)
    depth     Toffoli depth under `depth.toffoli_depth`'s ASAP schedule

    "add" is y += x mod 2^n (k = n - 1 positions), "add_carry" the same with
    the carry-out (k = n), "carry_out" the comparator.  In-place depth is
    1 + 2 floor(log k) + 2 floor(log(2k/3)) + 2[k >= 4]: the generates, the G
    and C rounds forwards and backwards, and -- once P rounds exist -- one
    layer each way where P round 1 and G round 1 share the odd p_i.
    """
    def inplace(k):
        if k == 0:
            return {"toffoli": 0, "ancillas": 0, "depth": 0}
        L, C = _log2(k), _log2(2 * k // 3) if k >= 2 else 0
        return {"toffoli": 7 * k - 4 * _w(k) - 4 * L - 2,
                "ancillas": k + _n_prop(k),
                "depth": 1 + 2 * L + 2 * C + 2 * (k >= 4)}

    Lc = (n - 1).bit_length()                      # ceil(log2 n)
    return {"add": inplace(n - 1),
            "add_carry": inplace(n),
            "carry_out": {"toffoli": 3 * n - 2 - Lc,
                          "ancillas": 3 * n - 2 - Lc,
                          "depth": Lc + 1}}
