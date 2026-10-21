"""Toy complete twisted-Edwards arithmetic in extended coordinates.

Scope, stated first because it is easy to over-read
---------------------------------------------------
This is a *toy* construction: the field is tiny (p = 5 or 13) so that every
basis state can be simulated and the whole Shor circuit can be sampled on Aer.
Its mixed addition uses the same seven products as Kim et al., ePrint 2026/106
Sec 4.4, but selects each classical addend with controlled constant arithmetic,
not the paper's three-register windowed QROM. It does not implement Ed25519-scale
arithmetic, the paper's QCSA multiplier or its resource schedule.

Sources for the formulas
------------------------
* H. Hisil, K. K.-H. Wong, G. Carter, E. Dawson, "Twisted Edwards Curves
  Revisited", ASIACRYPT 2008 -- extended coordinates (X : Y : Z : T) with
  x = X/Z, y = Y/Z, T = XY/Z, and the a = -1 addition law below (the EFD names
  it add-2008-hwcd-3).  It is complete when a is a square and d is not.
* RFC 8032 Sec 5.1.4 states the same a = -1 extended addition for Ed25519
  (p = 2^255 - 19, which is 1 mod 4 so a = -1 is a square, and d non-square).
* Kim et al., ePrint 2026/106 Sec 4.4 and Fig. 11 use the same mixed
  fixed-affine-addend formula and count three constant and four variable
  products. Their windowed lookup selects (y-x, y+x, 2dxy).

The law, for a = -1 and k = 2d:

    A = (Y1 - X1)(Y2 - X2)    B = (Y1 + X1)(Y2 + X2)
    C = k T1 T2               D = 2 Z1 Z2
    E = B - A   F = D - C   G = D + C   H = B + A
    X3 = E F    Y3 = G H    T3 = E H    Z3 = F G

Why Edwards is interesting for Shor
-----------------------------------
The affine Weierstrass adders in `ec_pointadd`/`ec_window` exclude doubling,
inverses and the point at infinity, so `ec_shor` must search for a "safe"
offset and refuses instances where the control registers cover the whole
subgroup.  The law above has *no* exceptional inputs (identity (0, 1) is an
ordinary point), so this circuit runs the full-group toy instance with no
offset at all.

What the reversible circuit does, and what it costs
---------------------------------------------------
`ext_add_ctrl` is an *out-of-place* controlled addition of a classical point
R.  The addend is (ctrl ? R : identity); its affine constants (y-x, y+x, kxy)
are never loaded into qubits -- `_mul_sel` multiplies by "ctrl ? c1 : c0"
directly (Horner doubling plus (controlled) constant additions).  The input
registers are rewritten in place (Y1 - X1, Y1 + X1, then 2 Z1 = D), three
n-bit products A, B, C are formed by `_mul_sel` and turned in place into E, H,
G and F, the outputs X3, Y3, Z3 (and optionally T3) are multiplied out, and
the whole body is then undone, so the input comes back unchanged and every scratch qubit is clean.
The input (X1 : Y1 : Z1 : T1) itself cannot be cleared by this call:
projective coordinates are not unique, so the input is *retained* -- 4n qubits
per addition (the same "G" accounting as `ec_proj`).

`ecdlp_oracle` offers two ways of dealing with that retained input:

* style="bennett" runs every extended addition forward, normalises the last
  accumulator to canonical affine (x, y) = (X/Z, Y/Z) with a Fermat inversion
  Z^(p-2), and runs the additions backwards.  It is cheaper in gates, but
  2m * 4n retained qubits are live at the peak (117 qubits for the p = 5 toy,
  above the 63-qubit Aer limit).
* style="inplace" (the default) does each addition affine -> affine:
  `affine_add_ctrl` computes affine(P + R) out of place via one extended
  addition and normalisation, then runs the inverse of the "+(-R)" map on the
  new point to clear the old (x, y).  Nothing is retained, so the p = 5 toy
  needs 54 qubits and the whole Shor circuit is sampled on Aer, at roughly
  2.5x the gate count.

Both styles are reported in bench/ec_edwards_toy.json; neither dominates.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit

import ec_modarith as MA
import ec_mult as MU
from ec_sim import Machine


# =============================================================================
# Classical curve
# =============================================================================
def _is_square(a, p):
    a %= p
    return a == 0 or pow(a, (p - 1) // 2, p) == 1


class EdwardsCurve:
    """-x^2 + y^2 = 1 + d x^2 y^2 over GF(p), complete: p = 1 mod 4, d non-square.

    Points are affine tuples (x, y); the identity is (0, 1).
    """

    a = -1

    def __init__(self, p, d, name=""):
        if p < 5 or any(p % q == 0 for q in range(2, int(p**0.5) + 1)):
            raise ValueError("p must be an odd prime >= 5")
        if p % 4 != 1:
            raise ValueError("a = -1 must be a square: need p = 1 mod 4")
        if _is_square(d, p):
            raise ValueError("d must be a non-square for a complete law")
        self.p, self.d, self.name = p, d % p, name
        self.k = 2 * self.d % p
        self.identity = (0, 1)

    # --- affine group law ----------------------------------------------------
    def on_curve(self, P):
        x, y = P
        p = self.p
        return (-x * x + y * y - 1 - self.d * x * x * y * y) % p == 0

    def add(self, P, Q):
        """The unified affine law.  Denominators never vanish (completeness)."""
        (x1, y1), (x2, y2) = P, Q
        p, t = self.p, self.d * x1 * x2 * y1 * y2
        x3 = (x1 * y2 + y1 * x2) * pow(1 + t, -1, p) % p
        y3 = (y1 * y2 + x1 * x2) * pow(1 - t, -1, p) % p
        return (x3, y3)

    def neg(self, P):
        return ((-P[0]) % self.p, P[1])

    def mul(self, k, P):
        R, Q = self.identity, P
        k %= self.point_order(P)
        while k:
            if k & 1:
                R = self.add(R, Q)
            Q = self.add(Q, Q)
            k >>= 1
        return R

    def points(self):
        p = self.p
        return [(x, y) for x in range(p) for y in range(p)
                if self.on_curve((x, y))]

    def point_order(self, P):
        n, R = 1, P
        while R != self.identity:
            R, n = self.add(R, P), n + 1
        return n

    # --- extended coordinates ------------------------------------------------
    def to_ext(self, P, lam=1):
        """(lam x : lam y : lam : lam x y), a representative of P."""
        x, y = P
        p = self.p
        lam %= p
        if lam == 0:
            raise ValueError("lambda must be non-zero")
        return (lam * x % p, lam * y % p, lam, lam * x * y % p)

    def ext_valid(self, E):
        X, Y, Z, T = E
        p = self.p
        return (Z % p != 0 and (X * Y - Z * T) % p == 0
                and self.on_curve(self.from_ext(E)))

    def from_ext(self, E):
        X, Y, Z, _ = E
        zi = pow(Z, -1, self.p)
        return (X * zi % self.p, Y * zi % self.p)

    def ext_add(self, E1, E2):
        """HWCD 2008 / RFC 8032 Sec 5.1.4 extended addition for a = -1."""
        X1, Y1, Z1, T1 = E1
        X2, Y2, Z2, T2 = E2
        p = self.p
        A = (Y1 - X1) * (Y2 - X2) % p
        B = (Y1 + X1) * (Y2 + X2) % p
        C = self.k * T1 * T2 % p
        D = 2 * Z1 * Z2 % p
        E, F, G, H = (B - A) % p, (D - C) % p, (D + C) % p, (B + A) % p
        return (E * F % p, G * H % p, F * G % p, E * H % p)

    def addend_constants(self, R):
        """(Y2 - X2, Y2 + X2, k T2) for the affine point R with Z2 = 1."""
        x, y = R
        p = self.p
        return ((y - x) % p, (y + x) % p, self.k * x * y % p)

    # --- the birationally equivalent short Weierstrass curve -----------------
    def weierstrass(self):
        """(a_W, b_W, map) for y^2 = x^3 + a_W x + b_W, via Montgomery form.

        Edwards (x, y) -> Montgomery u = (1 + y)/(1 - y), v = u / x with
        A = 2(a + d)/(a - d), B = 4/(a - d); then X = u/B + A/(3B), Y = v/B.
        Identity -> infinity (returned as None), (0, -1) -> the 2-torsion point.
        """
        p, a, d = self.p, self.a % self.p, self.d
        inv = lambda z: pow(z % p, -1, p)
        A = 2 * (a + d) * inv(a - d) % p
        B = 4 * inv(a - d) % p
        aw = (3 - A * A) * inv(3 * B * B) % p
        bw = (2 * A**3 - 9 * A) * inv(27 * B**3) % p

        def to_w(P):
            x, y = P
            if P == self.identity:
                return None
            if x == 0:                       # (0, -1): Montgomery (0, 0)
                u, v = 0, 0
            else:
                u = (1 + y) * inv(1 - y) % p
                v = u * inv(x) % p
            return ((u * inv(B) + A * inv(3 * B)) % p, v * inv(B) % p)

        return aw, bw, to_w


# Toy instances: both have 8 points and a cyclic group ((1, 2) and (6, 4)
# generate).  EDW5 is the one small enough (n = 3) for the Aer run.
EDW5 = EdwardsCurve(5, 3, "edwards-p5-d3")
EDW13 = EdwardsCurve(13, 11, "edwards-p13-d11")


def generator(curve):
    """A point of maximal order; ties go to the lexicographically lowest."""
    return max(curve.points(),
               key=lambda P: (curve.point_order(P), -P[0], -P[1]))


# =============================================================================
# Reversible arithmetic
# =============================================================================
def _xor_const(ctx, reg, value, ctrl=None):
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            if ctrl is None:
                ctx.x(q)
            else:
                ctx.cx(ctrl, q)


def _mul_sel(m, ctrl, x, k1, k0, z, p):
    """z (|0>) <- (k1 if ctrl else k0) * x mod p, for classical k1, k0.

    Horner over the bits of the two constants: one modular doubling per bit,
    then x is added unconditionally where both constants have a 1, under ctrl
    where only k1 does, and under NOT ctrl where only k0 does.  This is how the
    classical addend (ctrl ? R : identity) enters the circuit -- no register is
    loaded with it, and it costs bitlen + popcount(k1 | k0) modular additions
    instead of a full n-bit multiplication.  ctrl=None means k1 always.
    """
    k1 %= p
    k0 = k1 if ctrl is None else k0 % p
    for i in reversed(range(max(k1, k0).bit_length())):
        MA.moddbl(m, z, p)
        b1, b0 = (k1 >> i) & 1, (k0 >> i) & 1
        if b1 and b0:
            MA.modadd(m, x, z, p)
        elif b1:
            MA.cmodadd(m, ctrl, x, z, p)
        elif b0:
            m.ctx.x(ctrl)
            MA.cmodadd(m, ctrl, x, z, p)
            m.ctx.x(ctrl)


def ext_add_ctrl(m, ctrl, P1, R, curve, P3):
    """P3 (|0>) <- P1 + (R if ctrl else identity), extended coordinates.

    P1 = (X1, Y1, Z1, T1) is the retained input: it is transformed in place to
    hold the HWCD intermediates and restored before returning.  P3 is
    (X3, Y3, Z3, T3); T3 may be None when the caller only needs (X:Y:Z).
    `ctrl` may be None for an unconditional addition.  Scratch (A, B, C and the
    adders' carries) is returned to |0> and freed with a cleanliness check.

    Register life, all in place:
        Y1 -> Y1 - X1,  X1 -> Y1 + X1                    (u, v)
        A = u a2,  B = v b2,  C = T1 c2                   (selected constants)
        Z1 -> 2 Z1 = D
        B -> B - A = E,  A -> 2A + E = H
        C -> C + D = G,  Z1 -> 2D - G = F
    then X3 = E F, Y3 = G H, Z3 = F G, T3 = E H, and the block is undone.
    """
    p = curve.p
    n = len(P1[0])
    X1, Y1, Z1, T1 = P1
    X3, Y3, Z3, T3 = P3
    on = curve.addend_constants(R)
    off = curve.addend_constants(curve.identity)           # (1, 1, 0)
    mark = m.begin()

    MA.modsub(m, X1, Y1, p)                                # Y1 = u
    MA.moddbl(m, X1, p)
    MA.modadd(m, Y1, X1, p)                                # X1 = v
    A, B, C = m.anc(n, "A"), m.anc(n, "B"), m.anc(n, "C")
    _mul_sel(m, ctrl, Y1, on[0], off[0], A, p)
    _mul_sel(m, ctrl, X1, on[1], off[1], B, p)
    _mul_sel(m, ctrl, T1, on[2], off[2], C, p)
    MA.moddbl(m, Z1, p)                                    # Z1 = D
    MA.modsub(m, A, B, p)                                  # B = E
    MA.moddbl(m, A, p)
    MA.modadd(m, B, A, p)                                  # A = H
    MA.modadd(m, Z1, C, p)                                 # C = G
    MA.moddbl(m, Z1, p)
    MA.modsub(m, C, Z1, p)                                 # Z1 = F
    body = m.since(mark)

    E, F, G, H = B, Z1, C, A
    MU.modmul(m, E, F, X3, p)
    MU.modmul(m, G, H, Y3, p)
    MU.modmul(m, F, G, Z3, p)
    if T3 is not None:
        MU.modmul(m, E, H, T3, p)

    m.undo(body)
    m.free(C, B, A)


def fermat_inverse_chain(m, Z, p):
    """Registers computing Z^(p-2) by square-and-multiply; returns (zinv, regs).

    All returned registers are scratch the caller must uncompute (the chain is
    garbage-producing by design; `ext_to_affine` undoes it).
    """
    e = p - 2
    acc, regs = Z, []
    for bit in bin(e)[3:]:
        sq = m.anc(len(Z), "zsq")
        MU.modsqr(m, acc, sq, p)
        regs.append(sq)
        acc = sq
        if bit == "1":
            pr = m.anc(len(Z), "zmul")
            MU.modmul(m, acc, Z, pr, p)
            regs.append(pr)
            acc = pr
    return acc, regs


def ext_to_affine(m, P, ox, oy, p):
    """ox, oy (|0>) <- X/Z, Y/Z for P = (X, Y, Z, ...).

    Z^-1 = Z^(p-2) (Fermat); Z is never 0 because the law is complete.  P and
    every scratch register are restored.
    """
    X, Y, Z = P[:3]
    mark = m.begin()
    zinv, regs = fermat_inverse_chain(m, Z, p)
    body = m.since(mark)
    MU.modmul(m, X, zinv, ox, p)
    MU.modmul(m, Y, zinv, oy, p)
    m.undo(body)
    m.free(*reversed(regs))


def affine_add_out(m, ctrl, x, y, R, curve, ox, oy):
    """ox, oy (|0>) <- affine((x, y) + (ctrl ? R : identity)); x, y preserved.

    Embed (x, y) as (x : y : 1 : xy), add in extended coordinates, normalise,
    then unwind the embedding and the projective sum.  Nothing is retained.
    """
    p, n = curve.p, len(x)
    mark = m.begin()
    Z1, T1 = m.anc(n, "Z1"), m.anc(n, "T1")
    m.ctx.x(Z1[0])
    MU.modmul(m, x, y, T1, p)
    P3 = (m.anc(n, "X3"), m.anc(n, "Y3"), m.anc(n, "Z3"), None)
    ext_add_ctrl(m, ctrl, (x, y, Z1, T1), R, curve, P3)
    body = m.since(mark)
    ext_to_affine(m, P3, ox, oy, p)
    m.undo(body)
    m.free(*P3[2::-1], T1, Z1)


def affine_add_ctrl(m, ctrl, x, y, R, curve):
    """(x, y) <- (x, y) + (ctrl ? R : identity), in place up to relabelling.

    Forward: (x, y, 0, 0) -> (x, y, x', y') with x', y' the sum.  Backward:
    the *inverse* of "add -R" maps (x', y', x, y) -> (x', y', 0, 0), because
    x' - R = x exactly.  The complete law makes both steps exception-free, so
    no input is retained and the old registers are freed.  Returns (x', y').
    """
    n = len(x)
    nx, ny = m.anc(n, "ax"), m.anc(n, "ay")
    affine_add_out(m, ctrl, x, y, R, curve, nx, ny)
    m.emit_inverse(affine_add_out, m, ctrl, nx, ny, curve.neg(R), curve,
                   x, y)
    m.free(x, y)
    return nx, ny


# =============================================================================
# Shor's ECDLP with the complete Edwards oracle
# =============================================================================
def rungs(curve, P, Q, mb):
    return ([curve.mul(1 << i, P) for i in range(mb)],
            [curve.mul(1 << i, Q) for i in range(mb)])


def ecdlp_oracle(curve, P, Q, mb, mode="toffoli", offset=None,
                 style="inplace"):
    """|u>|v>|0> -> |u>|v>|affine(S + [u]P + [v]Q)>, every other qubit clean.

    `offset` S defaults to the identity: completeness means no safe-offset
    search is needed.

    style="inplace"  each controlled addition is `affine_add_ctrl` (forward
                     extended sum + inverse of the -R sum).  Width does not
                     grow with m.
    style="bennett"  extended accumulators chained with `ext_add_ctrl`; every
                     retained projective input (4n qubits per addition) stays
                     live until one normalisation, then all additions run
                     backwards.  Fewer arithmetic passes, width O(m n).

    Returns (Machine, info).
    """
    if style not in ("inplace", "bennett"):
        raise ValueError(f"unknown style {style!r}")
    p = curve.p
    n = p.bit_length()
    S = curve.identity if offset is None else offset
    if not curve.on_curve(S):
        raise ValueError("offset is not on the curve")
    m = Machine(mode, "edwards_ecdlp")
    kr, lr = m.alloc(mb, "k"), m.alloc(mb, "l")
    Ps, Qs = rungs(curve, P, Q, mb)
    ctrls = list(kr) + list(lr)

    if style == "inplace":
        x, y = m.anc(n, "ax"), m.anc(n, "ay")
        _xor_const(m.ctx, x, S[0])
        _xor_const(m.ctx, y, S[1])
        for c, R in zip(ctrls, Ps + Qs):
            x, y = affine_add_ctrl(m, c, x, y, R, curve)
        ox, oy, retained = x, y, 0
    else:
        start = tuple(m.anc(n, nm) for nm in ("X0", "Y0", "Z0", "T0"))
        ox, oy = m.alloc(n, "ox"), m.alloc(n, "oy")
        S_ext = curve.to_ext(S)
        for reg, val in zip(start, S_ext):
            _xor_const(m.ctx, reg, val)
        mark = m.begin()
        cur, kept = start, []
        for c, R in zip(ctrls, Ps + Qs):
            nxt = tuple(m.anc(n, nm) for nm in ("Xa", "Ya", "Za", "Ta"))
            ext_add_ctrl(m, c, cur, R, curve, nxt)
            kept.append(nxt)
            cur = nxt
        retained = len(kept) * 4 * n
        forward = m.since(mark)
        ext_to_affine(m, cur, ox, oy, p)
        m.undo(forward)
        for trip in reversed(kept):
            m.free(*trip)
        for reg, val in zip(start, S_ext):                # constant start -> |0>
            _xor_const(m.ctx, reg, val)
        m.free(*start)
    return m, {"n": n, "m_bits": mb, "offset": S, "style": style,
               "regs": (kr, lr, ox, oy), "retained_qubits_peak": retained,
               "additions": len(Ps) + len(Qs)}


def ecdlp_circuit(curve, P, Q, order, m_bits=None, mode="toffoli",
                  measure_scratch=False, offset=None, style="inplace"):
    """Hadamards, the Edwards oracle, inverse QFTs and measurement.

    measure_scratch: also measure every qubit other than the controls and the
    affine output into a register "scr" -- in a correct circuit every shot
    reads all zeros there, which checks uncomputation *in superposition*.
    """
    from shor_essentials import qft
    mb = max(1, math.ceil(math.log2(order))) if m_bits is None else m_bits
    m, info = ecdlp_oracle(curve, P, Q, mb, mode, offset, style)
    kr, lr, ox, oy = info["regs"]
    ck, cl = ClassicalRegister(mb, "ok"), ClassicalRegister(mb, "ol")
    cregs = [ck, cl]
    keep = set(list(kr) + list(lr) + list(ox) + list(oy))
    scratch = [q for q in m.qc.qubits if q not in keep]
    if measure_scratch:
        cregs.append(ClassicalRegister(len(scratch), "scr"))
    qc = QuantumCircuit(*m.qc.qregs, *cregs, name="edwards_ecdlp")
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


def split_counts(counts, measure_scratch=False):
    """Qiskit counts -> ({(j1, j2): shots}, {scratch value: shots})."""
    pairs, scr = {}, {}
    for bits, c in counts.items():
        parts = bits.split()
        if measure_scratch:
            s, lo, ko = parts
            scr[int(s, 2)] = scr.get(int(s, 2), 0) + c
        else:
            lo, ko = parts
        key = (int(ko, 2), int(lo, 2))
        pairs[key] = pairs.get(key, 0) + c
    return pairs, scr


def run_ecdlp(curve, P, Q, order, shots=2048, seed=7, measure_scratch=True,
              **kw):
    """Sample the full Edwards ECDLP circuit on Aer; returns (pairs, scr, cands, info)."""
    import aer_device as AD
    import ec_classical as C
    qc, info = ecdlp_circuit(curve, P, Q, order,
                             measure_scratch=measure_scratch, **kw)
    counts = AD.run(qc, shots=shots, seed=seed)
    info["aer"] = AD.describe(qc.num_qubits)
    pairs, scr = split_counts(counts, measure_scratch)
    cands = C.ecdlp_postprocess(pairs, order, info["m_bits"])
    return pairs, scr, cands, info
