"""Kim et al., ePrint 2026/106 Sec 4.4: out-of-place mixed Edwards addition
with a windowed QROM that loads the three addend constants -- toy scale.

Scope
-----
This is the *dataflow* of Sec 4.4 / Fig. 11 on a tiny complete a = -1 twisted
Edwards curve (p = 5 or 13), small enough to verify on every basis state and to
sample the whole Shor circuit on Aer CPU.  It is not Ed25519 (p = 2^255 - 19),
it does not use the paper's QCSA/Montgomery multipliers or its depth-optimised
layout, and none of its counts are the paper's Table 4 numbers.  The modular
arithmetic underneath is this package's exact schoolbook reference
(`ec_modarith`, `ec_mult`).

What Sec 4.4 specifies, and how it maps here
--------------------------------------------
Accumulator P1 = (X1 : Y1 : Z1 : T1) in extended coordinates, classically known
addend P2 = (x2, y2), k = 2d:

    A = (Y1 - X1)(y2 - x2)   B = (Y1 + X1)(y2 + x2)   C = T1 * (k x2 y2)
    D = 2 Z1   E = B - A   F = D - C   G = D + C   H = B + A
    X3 = E F   Y3 = G H   Z3 = F G   T3 = E H

"For windowing, we load the values (y2 - x2), (y2 + x2) and 2d x2 y2 from the
quantum lookup table."  So with a window address |a> the addend is
P2 = W[a] (W[0] = identity, whose constants are (1, 1, 0)), and

* `qrom_load` writes all three constants of W[a] into three n-bit registers
  L1, L2, L3 with ONE unary-iteration walk over the address (2^w - 2 ANDs plus
  CNOT fan-out at each leaf); running it again unloads them.
* `mixed_add` then performs the seven *quantum x quantum* multiplications --
  three against the loaded constants and four for the outputs -- out of place
  into fresh (X3, Y3, Z3, T3), retaining P1, and uncomputes every intermediate.

That is the distinction from `ec_edwards.ext_add_ctrl` (kept as the baseline):
there the addend is (ctrl ? R : identity) for a single control bit and its
constants never enter a qubit -- `_mul_sel` folds them into a Horner chain of
(controlled) constant additions.  That trick does not extend to a multi-bit
window: it would need one controlled pass per window entry.  The QROM path
makes the addend a *quantum* value and pays three general multiplications for
it, which is what lets one addition cover w scalar bits.

Two register layouts:

* layout="paper"    Fig. 11's wires: Y1 - X1 into a fresh register, D, E, F in
                    fresh registers, H overwrites A and G overwrites C.
* layout="compact"  the same products, but Y1 - X1, Y1 + X1, D, F rewrite the
                    input registers in place and E, H overwrite B, A -- 4n
                    fewer qubits, which is what fits the Aer toy.

Shor's oracle
-------------
The scalar registers |u>, |v> are cut into w-bit windows; window j of u
addresses W_j[a] = [a 2^{jw}] P (likewise for v and Q).  The first window of u
is not added but *looked up* straight into the accumulator (it starts at the
identity).  The retained-input problem of an out-of-place projective adder is
handled in one of the two ways `ec_edwards` uses for its baseline:

* style="inplace"  affine (x, y) -> embed (x : y : 1 : xy) -> QROM mixed add ->
                   Fermat normalise -> clear the old (x, y) with the inverse of
                   "+ (-W[a])" (the constants of -W[a] are just another QROM
                   table).  Width independent of the number of windows.
* style="bennett"  chain extended accumulators, normalise the last one once,
                   then run the chain backwards (the paper's Sec 4.2 "return
                   to unique representation" done once at the end).  Retains
                   4n qubits per addition, so only the w = mb case (one mixed
                   addition) fits on Aer.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit

import ec_edwards_toy as E
import ec_modarith as MA
import ec_mult as MU
from ec_sim import Machine

LAYOUTS = ("compact", "paper")
STYLES = ("inplace", "bennett")


# =============================================================================
# Windowed QROM: three constants per address, one unary iteration
# =============================================================================
def _walk(ctx, ctrl, addr, outs, tables, offset, anc):
    if not addr:
        for reg, tab in zip(outs, tables):
            val = tab[offset] if offset < len(tab) else 0
            for i, q in enumerate(reg):
                if (val >> i) & 1:
                    ctx.cx(ctrl, q)
        return
    a, rest, node = addr[-1], addr[:-1], anc[0]
    ctx.and_(ctrl, a, node)                       # node = ctrl AND a
    _walk(ctx, node, rest, outs, tables, offset + (1 << len(rest)), anc[1:])
    ctx.cx(ctrl, node)                            # node = ctrl AND NOT a
    _walk(ctx, node, rest, outs, tables, offset, anc[1:])
    ctx.cx(ctrl, node)
    ctx.and_dg(ctrl, a, node)


def qrom_load(m, addr, outs, tables):
    """outs[j] ^= tables[j][addr] for every j, in one walk; self-inverse.

    Unary iteration (Babbush et al. 2018) without an always-on control: the
    most significant address bit itself selects the upper half, and its
    negation the lower half.  len(addr) - 1 clean ancillas, 2^w - 2 ANDs (and
    as many AND-daggers), every table shares the same address decoding.
    """
    ctx, addr = m.ctx, list(addr)
    if not addr:
        for reg, tab in zip(outs, tables):
            E._xor_const(ctx, reg, tab[0])
        return
    top, rest = addr[-1], addr[:-1]
    anc = m.anc(len(rest), "qrom") if rest else []
    _walk(ctx, top, rest, outs, tables, 1 << len(rest), list(anc))
    ctx.x(top)
    _walk(ctx, top, rest, outs, tables, 0, list(anc))
    ctx.x(top)
    if rest:
        m.free(anc)


def mixed_tables(curve, window):
    """The three Sec 4.4 lookup tables (y-x, y+x, 2d x y) of a window."""
    cs = [curve.addend_constants(R) for R in window]
    return tuple([c[i] for c in cs] for i in range(3))


def window_points(curve, base, shift, width):
    """W[a] = [a 2^shift] base for a in [0, 2^width); W[0] = identity."""
    step = curve.mul(1 << shift, base)
    pts, acc = [], curve.identity
    for _ in range(1 << width):
        pts.append(acc)
        acc = curve.add(acc, step)
    return pts


# =============================================================================
# The out-of-place mixed addition, constants in quantum registers
# =============================================================================
def mixed_add(m, P1, L, curve, P3, layout="compact"):
    """P3 (|0>) <- P1 + P2, P2's constants (y2-x2, y2+x2, 2d x2 y2) held in L.

    Seven general modular multiplications (A, B, C, then X3, Y3, Z3, T3); T3
    may be None.  P1 and L are restored, every other scratch register is freed
    clean.  The addend is whatever L holds -- a single basis value or a
    superposition loaded by `qrom_load` -- so the same circuit serves every
    window entry.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    p, n = curve.p, len(P1[0])
    X1, Y1, Z1, T1 = P1
    L1, L2, L3 = L
    X3, Y3, Z3, T3 = P3
    mark = m.begin()
    if layout == "paper":
        U = m.anc(n, "U")
        MU.copy_reg(m.ctx, Y1, U)
        MA.modsub(m, X1, U, p)                          # U  = Y1 - X1
        MA.modadd(m, X1, Y1, p)                         # Y1 = Y1 + X1
        A, B, C = m.anc(n, "A"), m.anc(n, "B"), m.anc(n, "C")
        MU.modmul(m, U, L1, A, p)
        MU.modmul(m, Y1, L2, B, p)
        MU.modmul(m, T1, L3, C, p)
        D = m.anc(n, "D")
        MU.copy_reg(m.ctx, Z1, D)
        MA.moddbl(m, D, p)                              # D = 2 Z1
        Er, F = m.anc(n, "E"), m.anc(n, "F")
        MU.copy_reg(m.ctx, B, Er)
        MA.modsub(m, A, Er, p)                          # E = B - A
        MU.copy_reg(m.ctx, D, F)
        MA.modsub(m, C, F, p)                           # F = D - C
        MA.modadd(m, B, A, p)                           # A -> H = B + A
        MA.modadd(m, D, C, p)                           # C -> G = D + C
        scratch = (F, Er, D, C, B, A, U)
        H, G = A, C
    else:
        MA.modsub(m, X1, Y1, p)                         # Y1 = Y1 - X1
        MA.moddbl(m, X1, p)
        MA.modadd(m, Y1, X1, p)                         # X1 = Y1 + X1
        A, B, C = m.anc(n, "A"), m.anc(n, "B"), m.anc(n, "C")
        MU.modmul(m, Y1, L1, A, p)
        MU.modmul(m, X1, L2, B, p)
        MU.modmul(m, T1, L3, C, p)
        MA.moddbl(m, Z1, p)                             # Z1 = D
        MA.modsub(m, A, B, p)                           # B = E
        MA.moddbl(m, A, p)
        MA.modadd(m, B, A, p)                           # A = H
        MA.modadd(m, Z1, C, p)                          # C = G
        MA.moddbl(m, Z1, p)
        MA.modsub(m, C, Z1, p)                          # Z1 = F
        scratch = (C, B, A)
        Er, F, G, H = B, Z1, C, A
    body = m.since(mark)
    MU.modmul(m, Er, F, X3, p)
    MU.modmul(m, G, H, Y3, p)
    MU.modmul(m, F, G, Z3, p)
    if T3 is not None:
        MU.modmul(m, Er, H, T3, p)
    m.undo(body)
    m.free(*scratch)


def windowed_mixed_add(m, addr, P1, window, curve, P3, layout="compact"):
    """P3 (|0>) <- P1 + window[addr]: QROM load, `mixed_add`, QROM unload."""
    n = len(P1[0])
    L = tuple(m.anc(n, nm) for nm in ("L1", "L2", "L3"))
    tabs = mixed_tables(curve, window)
    qrom_load(m, addr, L, tabs)
    mixed_add(m, P1, L, curve, P3, layout)
    qrom_load(m, addr, L, tabs)
    m.free(*reversed(L))


# =============================================================================
# Affine in -> affine out (no retained input)
# =============================================================================
def affine_add_out(m, addr, x, y, window, curve, ox, oy, layout="compact"):
    """ox, oy (|0>) <- affine((x, y) + window[addr]); x, y preserved."""
    p, n = curve.p, len(x)
    mark = m.begin()
    Z1, T1 = m.anc(n, "Z1"), m.anc(n, "T1")
    m.ctx.x(Z1[0])
    MU.modmul(m, x, y, T1, p)
    P3 = (m.anc(n, "X3"), m.anc(n, "Y3"), m.anc(n, "Z3"), None)
    windowed_mixed_add(m, addr, (x, y, Z1, T1), window, curve, P3, layout)
    body = m.since(mark)
    E.ext_to_affine(m, P3, ox, oy, p)
    m.undo(body)
    m.free(*P3[2::-1], T1, Z1)


def affine_add_window(m, addr, x, y, window, curve, layout="compact"):
    """(x, y) <- (x, y) + window[addr]; returns the new registers.

    The old point is cleared by the inverse of "+ (-window[addr])", whose QROM
    tables are those of the negated window.
    """
    n = len(x)
    nx, ny = m.anc(n, "ax"), m.anc(n, "ay")
    affine_add_out(m, addr, x, y, window, curve, nx, ny, layout)
    neg = [curve.neg(R) for R in window]
    m.emit_inverse(affine_add_out, m, addr, nx, ny, neg, curve, x, y, layout)
    m.free(x, y)
    return nx, ny


# =============================================================================
# Shor's ECDLP oracle with windowed QROM mixed additions
# =============================================================================
def windows(curve, P, Q, mb, w):
    """[(address bits of u or v, window points)], u's windows first."""
    out = []
    for base, which in ((P, 0), (Q, 1)):
        for lo in range(0, mb, w):
            width = min(w, mb - lo)
            out.append((which, lo, width,
                        window_points(curve, base, lo, width)))
    return out


def ecdlp_oracle(curve, P, Q, mb, w, mode="toffoli", style="inplace",
                 layout="compact", first_lookup=True):
    """|u>|v>|0> -> |u>|v>|affine([u]P + [v]Q)>, every other qubit clean.

    w is the window width (1 <= w <= mb).  With first_lookup the first window
    of u is loaded by QROM (affine for "inplace", extended (x : y : 1 : xy)
    for "bennett") and each remaining window is one QROM mixed addition;
    without it the accumulator starts at the identity and every window is a
    mixed addition (the like-for-like count against `ec_edwards`, whose
    accumulator also starts at the identity).  Returns (Machine, info).
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    if not 1 <= w <= mb:
        raise ValueError("need 1 <= w <= m_bits")
    p = curve.p
    n = p.bit_length()
    m = Machine(mode, "edwards_kim_ecdlp")
    kr, lr = m.alloc(mb, "k"), m.alloc(mb, "l")
    wins = [((kr if which == 0 else lr)[lo:lo + width], pts)
            for which, lo, width, pts in windows(curve, P, Q, mb, w)]
    if first_lookup:
        (a0, W0), rest = wins[0], wins[1:]
    else:
        a0, W0, rest = [], [curve.identity], wins

    if style == "inplace":
        x, y = m.anc(n, "ax"), m.anc(n, "ay")
        qrom_load(m, a0, (x, y), ([R[0] for R in W0], [R[1] for R in W0]))
        for addr, Wj in rest:
            x, y = affine_add_window(m, addr, x, y, Wj, curve, layout)
        ox, oy, retained = x, y, 0
    else:
        ext0 = [curve.to_ext(R) for R in W0]
        start = tuple(m.anc(n, nm) for nm in ("X0", "Y0", "Z0", "T0"))
        ox, oy = m.alloc(n, "ox"), m.alloc(n, "oy")
        mark = m.begin()
        m.ctx.x(start[2][0])                              # Z0 = 1
        qrom_load(m, a0, (start[0], start[1], start[3]),
                  tuple([e[i] for e in ext0] for i in (0, 1, 3)))
        cur, kept = start, []
        for i, (addr, Wj) in enumerate(rest):
            # the last sum is only normalised, so its T3 is never needed
            names = ("Xa", "Ya", "Za") + (("Ta",) if i < len(rest) - 1 else ())
            regs = tuple(m.anc(n, nm) for nm in names)
            nxt = regs + (None,) * (4 - len(regs))
            windowed_mixed_add(m, addr, cur, Wj, curve, nxt, layout)
            kept.append(regs)
            cur = nxt
        # every extended accumulator is live at the normalisation
        retained = 4 * n + sum(len(r) * n for r in kept)
        forward = m.since(mark)
        E.ext_to_affine(m, cur, ox, oy, p)
        m.undo(forward)
        for regs in reversed(kept):
            m.free(*regs)
        m.free(*start)
    return m, {"n": n, "m_bits": mb, "w": w, "style": style, "layout": layout,
               "regs": (kr, lr, ox, oy), "retained_qubits_peak": retained,
               "lookups_into_accumulator": int(first_lookup),
               "mixed_additions": len(rest),
               "windows": [len(a) for a, _ in wins]}


def ecdlp_circuit(curve, P, Q, order, w, m_bits=None, mode="toffoli",
                  measure_scratch=False, style="inplace", layout="compact",
                  first_lookup=True):
    """Hadamards, the QROM-windowed oracle, inverse QFTs and measurement."""
    from shor_essentials import qft
    mb = max(1, math.ceil(math.log2(order))) if m_bits is None else m_bits
    m, info = ecdlp_oracle(curve, P, Q, mb, w, mode, style, layout,
                           first_lookup)
    kr, lr, ox, oy = info["regs"]
    ck, cl = ClassicalRegister(mb, "ok"), ClassicalRegister(mb, "ol")
    cregs = [ck, cl]
    keep = set(list(kr) + list(lr) + list(ox) + list(oy))
    scratch = [q for q in m.qc.qubits if q not in keep]
    if measure_scratch:
        cregs.append(ClassicalRegister(len(scratch), "scr"))
    qc = QuantumCircuit(*m.qc.qregs, *cregs, name="edwards_kim_ecdlp")
    qc.h(kr)
    qc.h(lr)
    qc.compose(m.qc, inplace=True)
    qc.append(qft(mb).inverse(), list(kr))
    qc.append(qft(mb).inverse(), list(lr))
    qc.measure(kr, ck)
    qc.measure(lr, cl)
    if measure_scratch:
        qc.measure(scratch, cregs[2])
    info.update(order=order, machine=m, qubits=qc.num_qubits,
                scratch_qubits=len(scratch))
    return qc, info


def run_ecdlp(curve, P, Q, order, w, shots=2048, seed=7, measure_scratch=True,
              **kw):
    """Sample the full circuit on Aer; returns (pairs, scr, cands, info)."""
    import aer_device as AD
    import ec_classical as C
    qc, info = ecdlp_circuit(curve, P, Q, order, w,
                             measure_scratch=measure_scratch, **kw)
    counts = AD.run(qc, shots=shots, seed=seed)
    info["aer"] = AD.describe(qc.num_qubits)
    pairs, scr = E.split_counts(counts, measure_scratch)
    cands = C.ecdlp_postprocess(pairs, order, info["m_bits"])
    return pairs, scr, cands, info


def build_oracle(adder, curve, P, Q, mb, mode="toffoli", style="inplace",
                 w=None, layout="compact", first_lookup=True):
    """Select the Edwards oracle: adder="horner" is the `ec_edwards` baseline
    (one control bit per addition, constants folded into constant additions);
    adder="kim-qrom" is this module (w-bit windows, constants QROM-loaded)."""
    if adder == "horner":
        return E.ecdlp_oracle(curve, P, Q, mb, mode, style=style)
    if adder == "kim-qrom":
        return ecdlp_oracle(curve, P, Q, mb, w or mb, mode, style, layout,
                            first_lookup)
    raise ValueError(f"unknown adder {adder!r}")
