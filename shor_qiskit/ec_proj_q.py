"""Projective point addition with a *quantum* addend, its zig-zag chain, and a
projective ECDLP circuit -- [106] Sec 4.2, the parts `ec_proj` leaves out.

`ec_proj.jacobian_add` is [106] Algorithm 4 with the addend a compile-time
constant: steps 2 and 7 (U2 = x2 Z1^2, S2 = y2 Z1^3) are `modmul_const`,
bitlen + popcount modular additions each.  That is not how [106] uses it.  Its
Algorithm 4 takes "point P2 (x2, y2) from lookup": in a windowed Shor circuit
the addend is T + [i]P for a *quantum* window value i, so it arrives in a
register and steps 2 and 7 are full quantum multiplications.  This module is
that circuit, and the pieces around it that make a whole ECDLP circuit.

Three addition forms, one builder
---------------------------------
    jacobian_add_q(m, P1, P2, out, p)

  P2 = (x2, y2)       mixed Jacobian-affine, Z2 = 1: [106] Alg. 4 verbatim,
                      11 multiplications.  The form a lookup produces.
  P2 = (X2, Y2, Z2)   full Jacobian-Jacobian (add-1998-cmo-2): U1 = X1 Z2^2
                      and S1 = Y1 Z2^3 stop being free, Z3 picks up a factor
                      Z2 -- 16 multiplications (12M + 4S).  Needed when the
                      addend is itself a projective accumulator, e.g. to join
                      two independent chains with one addition.

Both write a fresh (X3 : Y3 : Z3), keep both inputs, and unwind every
intermediate, exactly as `ec_proj` does and for the same reason: each step is
out of place and every step's inputs are still live when it is undone.

The control
-----------
Shor's circuit needs "+R if q".  There are two ways to get it here.

  lookup   [106] Sec 4.2 / IonQ Alg. 4: add T_J + [i]B_J for the window value
           i, with a per-window offset T_J chosen so no entry is O.  Every
           addition then runs unconditionally -- the control qubits are only
           the lookup's address -- and the answer is shifted by the known
           constant sum(T_J).  For w = 1 the "lookup" is X and CNOT gates:
           no Toffoli at all.  This is `ecdlp_projective(form="lookup")`.
  ctrl     `jacobian_add_q_ctrl`: out <- P1 + P2 if q else P1 (a bit copy).
           Out of place, a control is nearly free: compute everything
           unconditionally into temporaries, then *select* into `out` with
           one AND per bit (dst = q AND (t XOR src1), then dst ^= src1).  Z3
           folds the select into its multiplication instead, Z3 = Z1 *
           (q ? W : 1), so the only extra cost over the uncontrolled addition
           is 3n ANDs and two modular subtractions (the X3, Y3 temporaries
           must now be unwound too).  Compare the affine in-place addition,
           where the control has to reach inside both divisions.

The zig-zag chain
-----------------
`zigzag_chain_q` is `ec_proj.zigzag_chain` for quantum addends, on the tape of
`ec_classical.zigzag_schedule` ([106] Fig. 10).  The one new thing is that
PA-dagger needs the addend again: [106] Fig. 9's Lookup / Lookup-dagger pairs
around every PA become a `load` callback run around every PA *and* PA-dagger.

The ECDLP circuit
-----------------
`ecdlp_projective` loads the offset S as (S.x : S.y : 1), runs the chain, and
converts to affine *once*:

    x = X Z^-2,  y = Y Z^-3        (one Kaliski inversion, `to_affine`)

[106] writes the conversion as (X Z^-1, Y Z^-1), which is the homogeneous-
projective map; for the Jacobian coordinates Algorithm 4 uses it is Z^-2 and
Z^-3.  Then the whole chain runs backwards (Bennett), returning the m live
register triples -- the zig-zag residue *and* the final projective point -- to
|0>.  That unwinding is not optional.  A projective triple depends on the path
(k, l) took, not only on the point it represents, so leaving any of them
entangled with the control registers destroys the interference Shor relies on
(`ec_classical.zigzag_schedule` NOTE).  It doubles the additions; [106]'s mG
garbage figure is the register budget of the forward half.

Exceptional cases
-----------------
The Jacobian law, like the affine one, assumes a generic addition.  What the
circuit computes -- exactly, on every input, with every ancilla clean -- is
the formula; where the formula is not the group law:

  P1 == P2            H = R = 0: out = (0 : 0 : 0), not a point
  P1 == -P2           H = 0, R != 0: out = (R^2 : -R^3 : 0), a *valid* O --
                      which the next addition cannot consume
  P1 or P2 == O       Z1 or Z2 = 0: out = (0 : 0 : 0)

All three give Z3 = 0 (Z3 = Z1 Z2 H), and Z stays 0 through every later
addition, so a whole chain fails exactly when its final Z is 0.  The one
inversion at the end is where that surfaces: Kaliski on 0 does not terminate
in the state `ec_kaliski._clear_uvs` expects, and the simulator reports dirty
ancillas.  A zero test on Z would flag the run for a few ANDs.  The rate is
O(additions / p): nothing at cryptographic p, most of the grid at p = 7.

Cost (measured by `cost_rows`; flat to n = 32, `hier` at 256)
--------------------------------------------------------------
Per addition, Alg. 4 with a quantum addend costs 0.64-0.68x the Toffolis of
`ec_pointadd.point_add_ctrl` from n = 3 to n = 256 (11.0M against 16.3M at
256), for about 1.5x the qubits (6,406 against 4,133).  The constant addend of
`ec_proj` is ~10% cheaper again; the full Jacobian law costs as much as the
affine addition.  But the oracle runs about 2(2N - m) projective additions
against N affine ones -- the zig-zag's PA-daggers, then the Bennett mirror --
and with this package's reference arithmetic an inversion costs only a handful
of multiplications, so the whole projective oracle comes out about twice the
affine one (p = 7: 33,264 against 16,488 Toffolis; 17,334 for the forward half
alone).  The projective route pays where inversion dominates, as in [106]'s
Table 3, not here.

Sources: [106] Kim et al., eprint 2026/106, Sec 4.2, Alg. 4, Figs. 9-10;
[CMO98] Cohen, Miyaji, Ono (the Jacobian addition law).
"""

import ec_adders as A
import ec_kaliski as K
import ec_mbu as MB
import ec_modarith as MA
import ec_mult as MU
import ec_shor as SH
from ec_classical import O, Point, zigzag_schedule
from ec_sim import Machine, Reg, SimError, run


# =============================================================================
# The addition
# =============================================================================
def _sub_into(m, a, b, out, p):
    """out (|0>) <- a - b mod p."""
    for q, r in zip(a, out):
        m.ctx.cx(q, r)
    MA.modsub(m, b, out, p)


def _double_into(m, a, out, p):
    """out (|0>) <- 2a mod p."""
    for q, r in zip(a, out):
        m.ctx.cx(q, r)
    MA.modadd(m, a, out, p)


def _select(m, ctrl, src, t, dst):
    """dst (|0>) <- t if ctrl else src.  src and t preserved.  n ANDs.

    dst = ctrl AND (t XOR src), then dst ^= src.  The AND target is clean, so
    it is a temporary AND; the addition run backwards clears it with AND-dg.
    """
    ctx = m.ctx
    for a, b in zip(src, t):
        ctx.cx(a, b)
    for b, d in zip(t, dst):
        ctx.and_(ctrl, b, d)
    for a, b in zip(src, t):
        ctx.cx(a, b)
    for a, d in zip(src, dst):
        ctx.cx(a, d)


def _one_or(m, ctrl, t, dst):
    """dst (|0>) <- t if ctrl else 1.  n ANDs."""
    ctx = m.ctx
    for b, d in zip(t, dst):
        ctx.and_(ctrl, b, d)
    ctx.x(ctrl)
    ctx.cx(ctrl, dst[0])
    ctx.x(ctrl)


def _add(m, P1, P2, out, p, ctrl):
    """The shared body of `jacobian_add_q` and `jacobian_add_q_ctrl`.

    Step numbers are [106] Alg. 4's; 1a-1d exist only when Z2 is quantum.
    """
    X1, Y1, Z1 = P1
    X2, Y2 = P2[0], P2[1]
    Z2 = P2[2] if len(P2) == 3 else None
    X3, Y3, Z3 = out
    n = len(X1)
    temps, bodies = [], []

    def new(name):
        r = m.anc(n, name)
        temps.append(r)
        return r

    def step(fn, *args):
        _, b = m.step(fn, m, *args)
        bodies.append(b)

    if Z2 is None:
        U1, S1 = X1, Y1                          # Z2 = 1: both free
    else:
        Z2sq = new("Z2sq"); step(MU.modsqr, Z2, Z2sq, p)              # 1a
        U1 = new("U1");     step(MU.modmul, X1, Z2sq, U1, p)          # 1b
        Z2cu = new("Z2cu"); step(MU.modmul, Z2sq, Z2, Z2cu, p)        # 1c
        S1 = new("S1");     step(MU.modmul, Y1, Z2cu, S1, p)          # 1d

    Z1sq = new("Z1sq"); step(MU.modsqr, Z1, Z1sq, p)                  # 1
    U2 = new("U2");     step(MU.modmul, X2, Z1sq, U2, p)              # 2  quantum x2
    Z1cu = new("Z1cu"); step(MU.modmul, Z1sq, Z1, Z1cu, p)            # 3
    H = new("H");       step(_sub_into, U2, U1, H, p)                 # 4
    Hsq = new("Hsq");   step(MU.modsqr, H, Hsq, p)                    # 5
    Hcu = new("Hcu");   step(MU.modmul, Hsq, H, Hcu, p)               # 6
    S2 = new("S2");     step(MU.modmul, Y2, Z1cu, S2, p)              # 7  quantum y2
    R = new("R");       step(_sub_into, S2, S1, R, p)                 # 8
    Rsq = new("Rsq");   step(MU.modsqr, R, Rsq, p)                    # 9
    V = new("V");       step(MU.modmul, U1, Hsq, V, p)                # 10
    twoV = new("2V");   step(_double_into, V, twoV, p)                # 11
    T0 = new("T0");     step(_sub_into, Rsq, Hcu, T0, p)              # 12

    if ctrl is None:
        _sub_into(m, T0, twoV, X3, p)                                 # 13 OUTPUT
        x3 = X3
    else:
        x3 = new("X3t");    step(_sub_into, T0, twoV, x3, p)          # 13 temp
        _select(m, ctrl, X1, x3, X3)                                  #    OUTPUT
    T1 = new("T1");     step(_sub_into, V, x3, T1, p)                 # 14
    T2 = new("T2");     step(MU.modmul, R, T1, T2, p)                 # 15
    T3 = new("T3");     step(MU.modmul, S1, Hcu, T3, p)               # 16
    if ctrl is None:
        _sub_into(m, T2, T3, Y3, p)                                   # 17 OUTPUT
    else:
        y3 = new("Y3t");    step(_sub_into, T2, T3, y3, p)            # 17 temp
        _select(m, ctrl, Y1, y3, Y3)                                  #    OUTPUT

    W = H                                                             # 18: Z3 = Z1 (Z2) H
    if Z2 is not None:
        W = new("Z2H");     step(MU.modmul, Z2, H, W, p)
    if ctrl is not None:
        Wc = new("Wc")
        _, b = m.step(_one_or, m, ctrl, W, Wc)                        #  q ? W : 1
        bodies.append(b)
        W = Wc
    MU.modmul(m, Z1, W, Z3, p)                                        # 18 OUTPUT

    # unwind every temporary in reverse order of creation; each one's inputs
    # are still live at that point, and the three outputs are never touched
    for b in reversed(bodies):
        m.undo(b)
    m.free(*reversed(temps))


def jacobian_add_q(m, P1, P2, out, p):
    """out = (X3 : Y3 : Z3) <- P1 + P2, both points quantum.  [106] Alg. 4.

    P1 = (X1, Y1, Z1); P2 = (x2, y2) (affine, the lookup's output: 11
    multiplications) or (X2, Y2, Z2) (Jacobian: 16).  `out` must be |0>.
    Both inputs survive unchanged -- P1 is the garbage of [106] Sec 4.2 -- and
    every other scratch register is returned to |0>.
    """
    _add(m, P1, P2, out, p, None)


def jacobian_add_q_ctrl(m, ctrl, P1, P2, out, p):
    """out <- P1 + P2 if ctrl else P1 (a bitwise copy).  Otherwise as
    `jacobian_add_q`, for 3n more ANDs and two more modular subtractions.

    On the ctrl = 0 branch the sum is still computed, then discarded and
    unwound exactly, so exceptional (P1, P2) are harmless there.
    """
    _add(m, P1, P2, out, p, ctrl)


def jacobian_add_q_inv(m, P1, P2, out, p, ctrl=None):
    """PA-dagger: clears `out` given the inputs it was made from."""
    if ctrl is None:
        m.emit_inverse(jacobian_add_q, m, P1, P2, out, p)
    else:
        m.emit_inverse(jacobian_add_q_ctrl, m, ctrl, P1, P2, out, p)


# =============================================================================
# The zig-zag chain
# =============================================================================
def zigzag_chain_q(m, start, addends, p, ctrls=None, loads=None, m_regs=None):
    """A chain of `jacobian_add_q` under [106] Fig. 10's schedule.

    start    (X0, Y0, Z0), preserved
    addends  one register tuple per addition, (x, y) or (X, Y, Z); entries may
             share registers when `loads` fills them
    ctrls    optional, one qubit (or None) per addition: `jacobian_add_q_ctrl`
    loads    optional, one builder per addition that writes its addend into
             its (clean) registers.  Run before every PA and PA-dagger of that
             addition and inverted after -- [106] Fig. 9's Lookup / Lookup-dagger.

    Returns (final triple, residual triples, all triples allocated).  The
    triples number m = `zigzag_registers(len(addends))`; the residual ones
    hold the intermediate points the schedule cannot clear.
    """
    n = len(start[0])
    N = len(addends)
    ctrls = ctrls or [None] * N
    tape, m_regs, _ = zigzag_schedule(N, m_regs)
    pool, live, trips = [], {0: tuple(start)}, []

    def pa(j, invert):
        if loads:
            loads[j - 1]()
        c = ctrls[j - 1]
        if c is None:
            m.emit_maybe_inverse(invert, jacobian_add_q,
                                 m, live[j - 1], addends[j - 1], live[j], p)
        else:
            m.emit_maybe_inverse(invert, jacobian_add_q_ctrl,
                                 m, c, live[j - 1], addends[j - 1], live[j], p)
        if loads:
            m.emit_inverse(loads[j - 1])

    for op, j in tape:
        if op == "add":
            if not pool:
                trips.append((m.anc(n, "Xz"), m.anc(n, "Yz"), m.anc(n, "Zz")))
                pool.append(trips[-1])
            live[j] = pool.pop()
            pa(j, False)
        else:
            pa(j, True)
            pool.append(live.pop(j))
    assert len(trips) <= m_regs
    final = max(live)
    return live[final], [live[k] for k in sorted(live) if k not in (0, final)], trips


# =============================================================================
# Back to affine: the one inversion
# =============================================================================
def to_affine(m, P, out, p):
    """out = (x, y) (|0>) <- (X Z^-2, Y Z^-3).  One Kaliski inversion.

    Z must be nonzero; on Z = 0 (an exceptional chain) the inversion's
    registers do not come back clean.  x and y are written directly by the
    last two multiplications, so only Z^-1, Z^-2, Z^-3 are unwound.
    """
    X, Y, Z = P
    xo, yo = out
    n = len(X)
    zi = m.anc(n, "zi")
    (recs, cnt), b_inv = m.step(K.mod_inv, m, Z, zi, p)
    zi2 = m.anc(n, "zi2"); _, b_sq = m.step(MU.modsqr, m, zi, zi2, p)
    zi3 = m.anc(n, "zi3"); _, b_cu = m.step(MU.modmul, m, zi2, zi, zi3, p)
    MU.modmul(m, X, zi2, xo, p)
    MU.modmul(m, Y, zi3, yo, p)
    for b in (b_cu, b_sq, b_inv):
        m.undo(b)
    m.free(zi3, zi2, zi)
    for rec in recs:
        m.free(*rec)
    if cnt is not None:
        m.free(cnt)


# =============================================================================
# The ECDLP circuit
# =============================================================================
def _load_const_point(m, x, y, R):
    A.encode_const(m.ctx, x, R.x)
    A.encode_const(m.ctx, y, R.y)


def ecdlp_plan(curve, P, Q, order, w=1, form="lookup", m_bits=None,
               offset=None, seed=0):
    """The classical half of `ecdlp_projective`: layout, addends and offset.

    form="lookup"  w-bit windows over the masked tables T_J + [i]B_J of
                   `ec_shor.window_tables` (masks on, so no entry is O).
    form="ctrl"    w = 1; the rungs [2^i]P, [2^i]Q, one per control bit.

    `offset` None picks the start point S with the fewest exceptional (k, l)
    over the whole control grid (first on ties; toy curves only, like
    `ec_shor._pick_offset`, which it falls back to on a big grid).  For
    form="ctrl" every addend lies in <P>, so when the curve's order is not
    prime an S outside <P> keeps the accumulator in a coset no addend or its
    negative is in, and nothing is exceptional -- the `ec_shor` strict-offset
    argument again.  The lookup masks move addends between cosets, so there
    `pick_plan` also searches the mask seed.
    """
    p = curve.p
    assert form in ("lookup", "ctrl"), form
    assert form == "lookup" or w == 1, "form='ctrl' is one bit per addition"
    mb, nwk, nwl, bk, bl = SH._windowed_layout(order, w, m_bits, 0)
    plan = {"n": p.bit_length(), "w": w, "form": form, "m_bits": mb,
            "bits_k": bk, "bits_l": bl, "order": order, "seed": seed}
    if form == "lookup":
        tP, muP = SH.window_tables(curve, P, nwk, w, True, seed)
        tQ, muQ = SH.window_tables(curve, Q, nwl, w, True, seed + 1)
        plan.update(tables=(tP, tQ), shift=curve.add(muP, muQ))
    else:
        Ps, Qs = SH._rungs(curve, P, Q, mb)
        plan.update(rungs=Ps + Qs, shift=O)
    if offset is None and bk + bl > 12:
        offset = SH._pick_offset(curve, P, Q, order, mb, strict=False)
    if offset is None:
        cands = [S for S in curve.points() if not S.inf]
        offset = min(cands, key=lambda S: exceptional_count(
            curve, dict(plan, offset=S))[0])
    plan.update(offset=offset, target=curve.add(offset, plan["shift"]))
    return plan


def exceptional_count(curve, plan):
    """(exceptional (k, l) on the whole control grid, {reason: count})."""
    why = {}
    for k in range(1 << plan["bits_k"]):
        for l in range(1 << plan["bits_l"]):
            r = ecdlp_projective_ref(curve, plan, k, l)[1]
            if r:
                why[r] = why.get(r, 0) + 1
    return sum(why.values()), why


def pick_plan(curve, P, Q, order, seeds=range(64), **kw):
    """The `ecdlp_plan` with the fewest exceptional (k, l) over `seeds` (the
    lookup masks; irrelevant for form="ctrl").  Toy curves only."""
    best = None
    for s in seeds:
        plan = ecdlp_plan(curve, P, Q, order, seed=s, **kw)
        c = exceptional_count(curve, plan)[0]
        if best is None or c < best[0]:
            best = (c, plan)
        if c == 0 or plan["form"] == "ctrl":
            break
    return best[1]


def ecdlp_projective(curve, P, Q, order, w=1, form="lookup", m_bits=None,
                     offset=None, seed=0, m_regs=None, mode="and",
                     uncompute=True, plan=None):
    """Shor's ECDLP oracle with projective arithmetic throughout.  Returns
    (Machine, info); the style of `ec_shor.ecdlp_circuit(oracle="arith")`.

    form="lookup"  per window, the masked table entry is looked up into an
                   affine addend register (`ec_mbu.lookup`; X/CNOT only for
                   w = 1) and added unconditionally by `jacobian_add_q`.  The
                   answer is S + shift + [k]P + [l]Q, shift = sum of the masks.
    form="ctrl"    w = 1; the constant [2^i]P (or Q) is loaded with X gates
                   and added by `jacobian_add_q_ctrl` under bit i.  The answer
                   is S + [k]P + [l]Q.

    Then ONE inversion (`to_affine`) writes the affine point to (px, py) and,
    with `uncompute`, the chain runs backwards so that only k, l, px, py are
    left holding anything.  uncompute=False builds the forward half alone --
    the figure [106] counts -- and leaves the m triples as garbage.

    `plan` (from `ecdlp_plan` / `pick_plan`) overrides the layout arguments.
    info["target"] is the constant S + shift the answer is offset by.
    """
    plan = plan or ecdlp_plan(curve, P, Q, order, w, form, m_bits, offset, seed)
    p, n, w = curve.p, plan["n"], plan["w"]
    S, bk, bl = plan["offset"], plan["bits_k"], plan["bits_l"]

    m = Machine(mode, "ecdlp-proj")
    kr, lr = m.alloc(bk, "k"), m.alloc(bl, "l")
    px, py = m.alloc(n, "px"), m.alloc(n, "py")
    X0, Y0, Z0 = m.anc(n, "X0"), m.anc(n, "Y0"), m.anc(n, "Z0")
    ax, ay = m.anc(n, "ax"), m.anc(n, "ay")

    def load_start():                                 # (S.x : S.y : 1)
        _load_const_point(m, X0, Y0, S)
        m.ctx.x(Z0[0])
    load_start()

    if plan["form"] == "lookup":
        tP, tQ = plan["tables"]
        jobs = [(kr[J * w:(J + 1) * w], T) for J, T in enumerate(tP)] + \
               [(lr[J * w:(J + 1) * w], T) for J, T in enumerate(tQ)]
        out = Reg(list(ax) + list(ay), "a")

        def lookup(addr, T):
            data = [E.x | (E.y << n) for E in T]
            return lambda: MB.lookup(m, Reg(list(addr)), out, data)
        loads = [lookup(addr, T) for addr, T in jobs]
        ctrls = None
    else:
        loads = [(lambda R=R: _load_const_point(m, ax, ay, R)) for R in plan["rungs"]]
        ctrls = list(kr) + list(lr)

    mark = m.begin()
    final, resid, trips = zigzag_chain_q(
        m, (X0, Y0, Z0), [(ax, ay)] * len(loads), p, ctrls, loads, m_regs)
    chain = m.since(mark)

    to_affine(m, final, (px, py), p)                  # the ONE inversion

    if uncompute:                                     # Bennett: run the chain back
        m.undo(chain)
        m.free(*[r for t in trips for r in t])
        load_start()
        m.free(X0, Y0, Z0, ax, ay)
    tape = zigzag_schedule(len(loads), m_regs)[0]
    info = dict(plan, additions=len(loads), m_regs=len(trips), residual=len(resid),
                pa_calls=len(tape) * (2 if uncompute else 1),
                uncompute=uncompute, regs=(kr, lr, px, py), qubits=m.qc.num_qubits)
    return m, info


# =============================================================================
# Classical reference and the basis-state checker
# =============================================================================
def jacobian_add_q_ref(curve, P1, P2):
    """The map `jacobian_add_q` computes, on raw triples: the formula, not the
    group law (so it is also the circuit's answer on exceptional inputs)."""
    p = curve.p
    X1, Y1, Z1 = P1
    X2, Y2, Z2 = (P2[0], P2[1], 1) if len(P2) == 2 else P2
    Z1sq, Z2sq = Z1 * Z1 % p, Z2 * Z2 % p
    U1, U2 = X1 * Z2sq % p, X2 * Z1sq % p
    S1, S2 = Y1 * Z2sq * Z2 % p, Y2 * Z1sq * Z1 % p
    H, R = (U2 - U1) % p, (S2 - S1) % p
    Hsq = H * H % p
    Hcu, V = Hsq * H % p, U1 * Hsq % p
    X3 = (R * R - Hcu - 2 * V) % p
    Y3 = (R * (V - X3) - S1 * Hcu) % p
    return X3, Y3, Z1 * Z2 * H % p


def jacobian_rep(curve, P, lam):
    """The Jacobian triple (lam^2 x : lam^3 y : lam) of P; O is (lam^2 : lam^3 : 0)."""
    p = curve.p
    if P.inf:
        return lam * lam % p, lam ** 3 % p, 0
    return P.x * lam * lam % p, P.y * lam ** 3 % p, lam % p


def from_jacobian(curve, X, Y, Z):
    """The point a triple represents, O for a valid (t^2 : t^3 : 0), or None
    for a triple that represents nothing (e.g. (0 : 0 : 0))."""
    p = curve.p
    if Z % p:
        zi = pow(Z, -1, p)
        P = Point(X * zi * zi % p, Y * zi ** 3 % p)
        return P if curve.on_curve(P) else None
    if (X, Y) != (0, 0) and (X ** 3 - Y * Y) % p == 0:
        return O
    return None


def jacobian_exceptional(curve, P1, P2):
    """Why the Jacobian law fails on the points (P1, P2), or None."""
    if P1.inf or P2.inf:
        return "O"
    if P1 == P2:
        return "P1 = P2"
    if P1.x == P2.x:
        return "P1 = -P2"
    return None


def _addends(curve, info, k, l):
    """The classical points the circuit adds for control values (k, l), in
    order; None for a ctrl-form rung whose bit is 0 (computed, discarded)."""
    if info["form"] == "ctrl":
        mb = len(info["rungs"]) // 2
        bits = [(k >> i) & 1 for i in range(mb)] + [(l >> i) & 1 for i in range(mb)]
        return [R if b else None for R, b in zip(info["rungs"], bits)]
    w = info["w"]
    tP, tQ = info["tables"]
    mask = (1 << w) - 1
    return [T[(k >> (J * w)) & mask] for J, T in enumerate(tP)] + \
           [T[(l >> (J * w)) & mask] for J, T in enumerate(tQ)]


def ecdlp_projective_ref(curve, info, k, l):
    """(the affine point the circuit leaves in (px, py), None) for (k, l), or
    (None, reason) when some addition on the path is exceptional."""
    acc = info["offset"]
    for R in _addends(curve, info, k, l):
        if R is None:
            continue
        why = jacobian_exceptional(curve, acc, R)
        if why:
            return None, why
        acc = curve.add(acc, R)
    return acc, None


def check_ecdlp_projective(curve, P, Q, m, info, pairs, loud=True):
    """Basis-state check: for each (k, l), (px, py) == target + [k]P + [l]Q
    after the one inversion, k and l untouched, every ancilla clean.

    Exceptional (k, l) are counted by reason, not skipped silently; with
    `loud` each is also run and must fail visibly (Z = 0 reaches the
    inversion, whose registers do not come back clean).
    Returns (verified, {reason: count}).
    """
    kr, lr, px, py = info["regs"]
    good, bad = 0, {}
    for k, l in pairs:
        want, why = ecdlp_projective_ref(curve, info, k, l)
        if why:
            bad[why] = bad.get(why, 0) + 1
            if loud:
                try:
                    run(m, {kr: k, lr: l})
                except SimError:
                    continue
                raise AssertionError(f"exceptional ({k}, {l}) ran clean")
            continue
        law = curve.add(info["target"], curve.add(curve.mul(k, P), curve.mul(l, Q)))
        assert want == law, (k, l, want, law)
        rd = run(m, {kr: k, lr: l})
        assert (rd(px), rd(py)) == (want.x, want.y), (k, l, rd(px), rd(py), want)
        assert (rd(kr), rd(lr)) == (k, l)
        good += 1
    return good, bad


# =============================================================================
# Cost: the projective additions against the affine in-place one
# =============================================================================
def cost_rows(p, x2=None, y2=None, count=None):
    """[(label, n, qubits, Toffolis)] for one addition of each kind at prime p.

    Toffolis are `ec_cost.count`'s `toffoli_paper` (Toffoli + CCZ + AND, the
    papers' metric).  The classical addend (x2, y2) only matters for the two
    circuits that take it as a constant; defaults are mid-sized values.
    Inside `hier.tracing()` pass count=hier.count.
    """
    import ec_cost as CO
    import ec_pointadd as PA
    import ec_proj as PJ
    count = count or CO.count
    n = p.bit_length()
    x2 = p // 3 if x2 is None else x2
    y2 = p // 5 if y2 is None else y2
    rows = []

    def build(label, fn):
        m = Machine("and")
        fn(m)
        c = count(m)
        rows.append((label, n, c["qubits"], c["toffoli_paper"]))

    def trip(m, nm):
        return tuple(m.alloc(n, f"{c}{nm}") for c in "XYZ")

    def affine(m):
        q, x, y = m.alloc(1, "q"), m.alloc(n, "x"), m.alloc(n, "y")
        PA.point_add_ctrl(m, q[0], x, y, x2, y2, p)

    def classical(m):
        P1, P3 = trip(m, "1"), trip(m, "3")
        PJ.jacobian_add(m, *P1, x2, y2, *P3, p)

    def quantum(ctrl, full):
        def f(m):
            P1, P3 = trip(m, "1"), trip(m, "3")
            P2 = trip(m, "2") if full else (m.alloc(n, "x2"), m.alloc(n, "y2"))
            if ctrl:
                jacobian_add_q_ctrl(m, m.alloc(1, "q")[0], P1, P2, P3, p)
            else:
                jacobian_add_q(m, P1, P2, P3, p)
        return f

    def conv(m):
        to_affine(m, trip(m, ""), (m.alloc(n, "x"), m.alloc(n, "y")), p)

    build("affine in place, ctrl (ec_pointadd)", affine)
    build("Jacobian + classical affine (ec_proj)", classical)
    build("Jacobian + quantum affine", quantum(False, False))
    build("Jacobian + quantum affine, ctrl", quantum(True, False))
    build("Jacobian + quantum Jacobian", quantum(False, True))
    build("Jacobian + quantum Jacobian, ctrl", quantum(True, True))
    build("to_affine (once per run)", conv)
    return rows
