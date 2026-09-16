"""Shor's algorithm for the elliptic-curve discrete logarithm.

Given P and Q = [k]P on a curve over GF(p), recover k.  The circuit is
[106] Fig. 1 / [1128] Sec 2:

    prepare  sum_{u,v} |u>|v>|S>          (S a fixed non-identity offset)
    compute  sum_{u,v} |u>|v>|S + [u]P + [v]Q>
    QFT both control registers, measure, and solve classically.

The offset S is not decoration.  Affine addition formulas cannot represent the
point at infinity, so the accumulator must never pass through it; starting at a
fixed S and reading the answer relative to S avoids that, and it is the same
trick as the Classiq tutorial's P0 and as the window offset T of [106] Sec 4.2.
S is a constant, so it shifts the computed coset without touching the
interference that Shor relies on.

Two oracles, for two different jobs
-----------------------------------
`oracle="arith"` builds the real thing out of `ec_pointadd.point_add_ctrl`:
Kaliski inversion, modular multiplication, the lot.  It is a permutation, so
checking it on every basis state *is* checking it -- which the tests do -- but
at ~50 qubits and ~10^5 gates per addition no simulator will run it in
superposition, so it cannot be taken end to end.  This is the circuit to cost.

`oracle="table"` keeps the identical circuit shape -- same registers, same
controlled additions, same QFT, same post-processing -- and replaces only the
point addition by an explicit permutation over the toy curve's points.  That
fits in a dozen qubits, so Aer runs it with genuine superposition and real
measurement statistics, and the discrete logarithm actually comes out.  It is
the same move the Classiq tutorial makes with its lookup-table inverse, for the
same reason.

Between the two, everything is covered: the arithmetic is verified exactly at
full width, and the algorithm around it is verified end to end.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

import ec_classical as C
import ec_pointadd as PA
from ec_sim import Machine, Reg
from shor_essentials import qft


# =============================================================================
# A controlled permutation over a toy curve's point registers
# =============================================================================
def transposition(qc, qubits, a, b, ctrls=()):
    """Swap the basis states |a> and |b> of `qubits`; leave all others alone.

    Pick a bit j where a and b differ.  CNOTs from j fold every other differing
    bit away, so the two states end up differing in j alone; one multi-controlled
    X on j swaps them; undo the CNOTs.  Conjugating a transposition by a
    permutation gives a transposition, so the net effect is exactly (a b).
    """
    n = len(qubits)
    d = a ^ b
    assert d, "a and b must differ"
    j = (d & -d).bit_length() - 1                 # lowest differing bit
    if not (a >> j) & 1:
        a, b = b, a                               # ensure a_j = 1
    fold = [i for i in range(n) if i != j and ((d >> i) & 1)]
    for i in fold:
        qc.cx(qubits[j], qubits[i])
    # now a and b agree everywhere but j; control on b's other bits
    others = [i for i in range(n) if i != j]
    flips = [i for i in others if not ((b >> i) & 1)]
    for i in flips:
        qc.x(qubits[i])
    qc.mcx([*ctrls] + [qubits[i] for i in others], qubits[j])
    for i in flips:
        qc.x(qubits[i])
    for i in fold:
        qc.cx(qubits[j], qubits[i])


def permutation(qc, qubits, perm, ctrls=()):
    """Apply the permutation `perm` (a dict old -> new) to `qubits`.

    Decomposed by cycles: the cycle (c0 c1 ... cm-1) is the product of
    transpositions (c0 c1), (c0 c2), ..., (c0 cm-1) applied in that order.
    """
    seen, cycles = set(), []
    for st in perm:
        if st in seen or perm[st] == st:
            continue
        cyc, cur = [], st
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = perm[cur]
        if len(cyc) > 1:
            cycles.append(cyc)
    for cyc in cycles:
        for j in range(1, len(cyc)):
            transposition(qc, qubits, cyc[0], cyc[j], ctrls)


def point_perm(curve, R, n):
    """The permutation of the packed (x, y) register induced by "+R".

    Points pack as x + 2^n * y.  The point at infinity needs a code of its own
    or the map is not a bijection -- P -> P + R sends exactly one point to
    infinity and brings exactly one back, so dropping it breaks closure.  Code 0
    is used, which is legal precisely when (0, 0) is off the curve, i.e. when
    b is not a square; that is asserted rather than assumed.

    Codes that are neither points nor the infinity code are left fixed.  They
    are unreachable from a valid start, and leaving them alone keeps the gate
    count down.
    """
    p = curve.p
    assert not any(P.x == 0 and P.y == 0 for P in curve.points() if not P.inf), (
        "(0,0) is on this curve, so it cannot double as the infinity code")

    def code(P):
        return 0 if P.inf else P.x + (P.y << n)

    perm = {code(P): code(curve.add(P, R)) for P in curve.points()}
    assert len(set(perm.values())) == len(perm), "point addition is not injective"
    assert set(perm.values()) == set(perm.keys()), "orbit is not closed"
    return perm


# =============================================================================
# The circuit
# =============================================================================
def ecdlp_circuit(curve, P, Q, order, oracle="table", offset=None, m_bits=None,
                  mode="and"):
    """Shor's ECDLP circuit.  Returns (QuantumCircuit, info dict).

    `order`  the order of P (the group the logarithm lives in)
    `oracle` "table" (simulable end to end) or "arith" (the real arithmetic)
    `offset` the fixed non-identity starting point S
    """
    p = curve.p
    n = p.bit_length()
    mb = m_bits or max(1, math.ceil(math.log2(order)))
    S = offset or _pick_offset(curve, P, Q, order, mb, strict=(oracle == "arith"))

    if oracle == "table":
        return _ecdlp_table(curve, P, Q, order, S, n, mb)
    if oracle == "arith":
        return _ecdlp_arith(curve, P, Q, order, S, n, mb, mode)
    raise ValueError(f"unknown oracle {oracle!r}")


def _pick_offset(curve, P, Q, order, mb, strict):
    """A start point S for the accumulator.

    `strict` (the arithmetic oracle) demands that no reachable S + [u]P + [v]Q
    is the point at infinity, because the affine formulas cannot represent it.
    That is possible only when the control registers do not cover the whole
    group -- once 2^mb >= order, every offset's orbit contains infinity, and
    there is nothing to pick.  Failing loudly there is the point: silently
    returning a bad offset would produce a circuit that is wrong on inputs the
    caller never thinks to check.

    The table oracle is not strict: it gives infinity its own code and handles
    it, so any non-identity point will do.
    """
    best = None
    for S in curve.points():
        if S.inf:
            continue
        if best is None:
            best = S
        if not strict:
            return S
        if all(not curve.add(curve.add(S, curve.mul(u, P)), curve.mul(v, Q)).inf
               for u in range(1 << mb) for v in range(1 << mb)):
            return S
    if strict:
        raise ValueError(
            f"no offset keeps the orbit off infinity: 2^{mb} control values "
            f"cover the whole group of order {order}, so the affine circuit "
            f"cannot avoid the point at infinity here")
    return best


def _rungs(curve, P, Q, mb):
    """The 2*mb constant points added, one per control qubit."""
    return ([curve.mul(1 << i, P) for i in range(mb)],
            [curve.mul(1 << i, Q) for i in range(mb)])


def _ecdlp_table(curve, P, Q, order, S, n, mb):
    kr, lr = QuantumRegister(mb, "k"), QuantumRegister(mb, "l")
    px, py = QuantumRegister(n, "px"), QuantumRegister(n, "py")
    ck, cl = ClassicalRegister(mb, "ok"), ClassicalRegister(mb, "ol")
    qc = QuantumCircuit(kr, lr, px, py, ck, cl)

    for i in range(n):                                  # accumulator <- S
        if (S.x >> i) & 1:
            qc.x(px[i])
        if (S.y >> i) & 1:
            qc.x(py[i])
    qc.h(kr)
    qc.h(lr)

    pack = list(px) + list(py)
    Ps, Qs = _rungs(curve, P, Q, mb)
    for i, R in enumerate(Ps):
        permutation(qc, pack, point_perm(curve, R, n), ctrls=[kr[i]])
    for i, R in enumerate(Qs):
        permutation(qc, pack, point_perm(curve, R, n), ctrls=[lr[i]])

    qc.append(qft(mb).inverse(), list(kr))
    qc.append(qft(mb).inverse(), list(lr))
    qc.measure(kr, ck)
    qc.measure(lr, cl)
    return qc, {"n": n, "m_bits": mb, "offset": S, "order": order,
                "oracle": "table", "qubits": qc.num_qubits}


def _ecdlp_arith(curve, P, Q, order, S, n, mb, mode):
    """The same circuit with real modular arithmetic. For costing and for
    basis-state verification -- too wide to simulate in superposition."""
    p = curve.p
    m = Machine(mode, "ecdlp")
    kr = m.alloc(mb, "k")
    lr = m.alloc(mb, "l")
    px, py = m.alloc(n, "px"), m.alloc(n, "py")
    for i in range(n):
        if (S.x >> i) & 1:
            m.ctx.x(px[i])
        if (S.y >> i) & 1:
            m.ctx.x(py[i])

    Ps, Qs = _rungs(curve, P, Q, mb)
    for i, R in enumerate(Ps):
        PA.point_add_ctrl(m, kr[i], px, py, R.x, R.y, p)
    for i, R in enumerate(Qs):
        PA.point_add_ctrl(m, lr[i], px, py, R.x, R.y, p)

    return m, {"n": n, "m_bits": mb, "offset": S, "order": order,
               "oracle": "arith", "regs": (kr, lr, px, py),
               "qubits": m.qc.num_qubits}


# =============================================================================
# The semiclassical (one control qubit) variant
# =============================================================================
def ecdlp_circuit_1c(curve, P, Q, order, offset=None, m_bits=None):
    """Shor's ECDLP with ONE control qubit, recycled -- Griffiths-Niu.

    Both papers assume this.  [106] Appendix D and [1128] Sec 2 both note that
    the two n-bit control registers "do not count towards the total qubit count
    thanks to the semiclassical Fourier transform with qubit recycling", which
    is why their headline figures are the cost of the point arithmetic alone.

    Every controlled rotation in the inverse QFT is controlled by a qubit that
    is about to be measured, so measure it first and make the control classical.
    One qubit is Hadamarded, drives its rung, receives phase corrections
    conditioned on every previously measured bit of *its own* register, is
    measured, and is reset.

    The two registers are independent transforms, so the corrections run within
    a register and not across the pair.  Rungs go most-significant-exponent
    first, which puts the first measured bit at the least significant end --
    Qiskit's own bit order, so no reversal is needed on readout.  (`onectrl.py`
    makes the same choice for factoring, and for the same reason.)

    Only the table oracle: with the arithmetic oracle this is a mid-circuit
    measurement on ~60 qubits, which Aer must simulate shot by shot.
    """
    p = curve.p
    n = p.bit_length()
    mb = m_bits or max(1, math.ceil(math.log2(order)))
    S = offset or _pick_offset(curve, P, Q, order, mb, strict=False)

    ctr = QuantumRegister(1, "ctr")
    px, py = QuantumRegister(n, "px"), QuantumRegister(n, "py")
    ck, cl = ClassicalRegister(mb, "ok"), ClassicalRegister(mb, "ol")
    qc = QuantumCircuit(ctr, px, py, ck, cl)

    for i in range(n):
        if (S.x >> i) & 1:
            qc.x(px[i])
        if (S.y >> i) & 1:
            qc.x(py[i])

    pack = list(px) + list(py)
    Ps, Qs = _rungs(curve, P, Q, mb)

    for base, creg in ((Ps, ck), (Qs, cl)):
        for i in range(mb):
            R = base[mb - 1 - i]                     # most significant first
            qc.h(ctr[0])
            permutation(qc, pack, point_perm(curve, R, n), ctrls=[ctr[0]])
            for j in range(i):                       # semiclassical inverse QFT
                with qc.if_test((creg[j], 1)):
                    qc.p(-math.pi / 2 ** (i - j), ctr[0])
            qc.h(ctr[0])
            qc.measure(ctr[0], creg[i])
            with qc.if_test((creg[i], 1)):           # reset for the next rung
                qc.x(ctr[0])

    return qc, {"n": n, "m_bits": mb, "offset": S, "order": order,
                "oracle": "table-1c", "qubits": qc.num_qubits}


# =============================================================================
# Running it
# =============================================================================
def run_ecdlp(curve, P, Q, order, shots=4096, seed=7, one_control=False, **kw):
    """Build the table-oracle circuit, sample it, and solve for k."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    if one_control:
        qc, info = ecdlp_circuit_1c(curve, P, Q, order, **kw)
    else:
        qc, info = ecdlp_circuit(curve, P, Q, order, oracle="table", **kw)
    sim = AerSimulator(seed_simulator=seed)
    res = sim.run(transpile(qc, sim, optimization_level=0), shots=shots).result()
    counts = res.get_counts()

    tally = {}
    for bits, c in counts.items():
        lo, ko = bits.split()                   # Qiskit prints last register first
        tally[(int(ko, 2), int(lo, 2))] = tally.get((int(ko, 2), int(lo, 2)), 0) + c
    cands = C.ecdlp_postprocess(tally, order, info["m_bits"])
    return counts, cands, info


def solve(curve, P, Q, order, **kw):
    """The discrete logarithm, or None. Verified against the curve before use."""
    counts, cands, info = run_ecdlp(curve, P, Q, order, **kw)
    for k, _ in cands:
        if C.verify_dlog(curve, P, Q, k):
            return k, counts, info
    return None, counts, info
