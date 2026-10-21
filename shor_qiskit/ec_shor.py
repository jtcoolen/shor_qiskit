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
`oracle="arith"` now builds the complete circuit: Hadamards, reversible point
arithmetic, inverse QFTs and measurement.  `info["machine"]` retains the
arithmetic-only `Machine` for exact basis-state verification and cost
inspection. The circuit is still too wide for statevector simulation at useful
field sizes, and generic arithmetic requires a safe offset that avoids its
exceptional addition inputs.

`arithmetic="affine"` uses `ec_pointadd.point_add_ctrl` and can select the
reference or Montgomery/QCSA/lookup multiplier and reference or optimized
Kaliski inversion. `arithmetic="windowed"` composes
`ec_window.windowed_point_add`, including its Euclidean-dialog multiplication,
into the same ECDLP circuit. The projective zig-zag implementation remains a
lower-level component: its out-of-place projective result is not a canonical
affine point and does not yet implement this path's controlled point-addition
contract.

`oracle="table"` remains the exact, simulable path for toy curves. It assigns
the point at infinity a dedicated code and therefore supports the complete
subgroup, including the exceptional states excluded by the arithmetic
adders.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

import ec_classical as C
import ec_pointadd as PA
import ec_window as W
from ec_sim import Machine
from semiclassical import semiclassical_iqft
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
                  mode="and", arithmetic="affine", window_size=None,
                  multiplier="reference", inversion="kaliski",
                  word_size=None, gcd="dialog", square="reference"):
    """Shor's ECDLP circuit.  Returns (QuantumCircuit, info dict).

    `order`  the order of P (the group the logarithm lives in)
    `oracle` "table" (simulable end to end) or "arith" (the arithmetic circuit)
    `offset` the fixed non-identity starting point S
    `arithmetic` for `oracle="arith"`, "affine" or "windowed"
    `window_size` address width for windowed point additions
    `multiplier` for affine arithmetic: reference, Montgomery/QCSA, or lookup
    `inversion` for affine arithmetic: Kaliski, optimized Kaliski, or toy divsteps
    `word_size` for a Montgomery multiplier; defaults to a divisor of the field width
    `gcd` for windowed arithmetic: "dialog" ([1128] Alg. 3), "pingpong"
          (the comparison-free walk of ECDSA.Fail Sec 5.3.3), "jump2" or
          "jump2-codec" (its Jump-2 dialog, Sec 5.3.1-5.3.2)
    `square` for windowed arithmetic: "reference", "symmetric" or "karatsuba"
          (the square-subtract of ECDSA.Fail Sec 5.3.4)

    The arithmetic oracle is now returned as a complete quantum circuit.  Its
    underlying reversible arithmetic-only `Machine` is exposed in
    `info["machine"]` for basis-state checks and cost inspection.
    """
    p = curve.p
    n = p.bit_length()
    mb = max(1, math.ceil(math.log2(order))) if m_bits is None else m_bits
    if mb < 1:
        raise ValueError("m_bits must be positive")
    if arithmetic not in ("affine", "windowed"):
        raise ValueError(f"unknown arithmetic backend {arithmetic!r}")
    if window_size is not None and window_size < 1:
        raise ValueError("window_size must be positive")
    if arithmetic == "windowed" and oracle == "arith" and window_size is None:
        raise ValueError("windowed arithmetic requires window_size")
    if multiplier not in ("reference", "montgomery-qcsa", "montgomery-lookup"):
        raise ValueError(f"unknown multiplier backend {multiplier!r}")
    if inversion not in ("kaliski", "kaliski-opt", "divsteps"):
        raise ValueError(f"unknown inversion backend {inversion!r}")
    if arithmetic == "windowed" and (
            multiplier != "reference" or inversion != "kaliski"):
        raise ValueError(
            "windowed arithmetic uses the Euclidean dialog internally; "
            "multiplier and inversion selections apply only to affine arithmetic")
    if gcd not in ("dialog", "pingpong", "jump2", "jump2-codec"):
        raise ValueError(f"unknown gcd backend {gcd!r}")
    if gcd != "dialog" and arithmetic != "windowed":
        raise ValueError("gcd selection applies only to windowed arithmetic")
    if square not in ("reference", "symmetric", "karatsuba"):
        raise ValueError(f"unknown square backend {square!r}")
    if square != "reference" and arithmetic != "windowed":
        raise ValueError("square selection applies only to windowed arithmetic")
    if oracle not in ("table", "arith"):
        raise ValueError(f"unknown oracle {oracle!r}")
    if offset is None:
        S = _pick_offset(
            curve, P, Q, order, mb, strict=(oracle == "arith"),
            arithmetic=arithmetic, window_size=window_size)
    else:
        S = offset
        if oracle == "arith" and not _offset_is_safe(
                curve, P, Q, mb, S, arithmetic, window_size):
            raise ValueError(
                f"offset {S!r} reaches an exceptional input for the "
                f"{arithmetic} arithmetic backend")

    if oracle == "table":
        return _ecdlp_table(curve, P, Q, order, S, n, mb)
    if oracle == "arith":
        return _ecdlp_arith(
            curve, P, Q, order, S, n, mb, mode, arithmetic, window_size,
            multiplier, inversion, word_size, gcd, square)
    raise AssertionError("validated oracle")


def _pick_offset(curve, P, Q, order, mb, strict, arithmetic="affine",
                 window_size=None):
    """A start point S for the accumulator.

    `strict` (the arithmetic oracle) demands that every reachable addition is
    in the generic domain of the selected point adder.  The arithmetic adders
    do not handle infinity or their exceptional division inputs. If the control
    registers cover the subgroup, there may be no such offset; failing loudly
    is preferable to building a circuit that is wrong on reachable inputs.

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
        if _offset_is_safe(curve, P, Q, mb, S, arithmetic, window_size):
            return S
    if strict:
        raise ValueError(
            f"no offset keeps all reachable additions in the generic domain "
            f"for the {arithmetic} backend with {mb}-bit controls and "
            f"subgroup order {order}; use the table oracle or reduce m_bits")
    return best


def _rungs(curve, P, Q, mb):
    """The 2*mb constant points added, one per control qubit."""
    return ([curve.mul(1 << i, P) for i in range(mb)],
            [curve.mul(1 << i, Q) for i in range(mb)])


def _offset_is_safe(curve, P, Q, mb, S, arithmetic, window_size):
    if S.inf or not curve.on_curve(S):
        return False
    Ps, Qs = _rungs(curve, P, Q, mb)
    for u in range(1 << mb):
        for v in range(1 << mb):
            acc = S
            if arithmetic == "windowed":
                assert window_size is not None
                for base, scalar in ((P, u), (Q, v)):
                    for shift in range(0, mb, window_size):
                        w = min(window_size, mb - shift)
                        addr = (scalar >> shift) & ((1 << w) - 1)
                        addend = curve.mul(addr << shift, base)
                        if addend.inf:
                            if acc.x == 0:
                                return False
                            continue
                        if C.point_add_exceptional(curve, acc, addend):
                            return False
                        acc = curve.add(acc, addend)
                continue
            for i, R in enumerate(Ps):
                if (u >> i) & 1:
                    if R.inf or C.point_add_exceptional(curve, acc, R):
                        return False
                    acc = curve.add(acc, R)
            for i, R in enumerate(Qs):
                if (v >> i) & 1:
                    if R.inf or C.point_add_exceptional(curve, acc, R):
                        return False
                    acc = curve.add(acc, R)
    return True


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


def _ecdlp_arith(curve, P, Q, order, S, n, mb, mode, arithmetic,
                 window_size, multiplier, inversion, word_size, gcd="dialog",
                 square="reference"):
    """Build a complete QFT circuit around the reversible arithmetic oracle."""
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
    if arithmetic == "affine":
        for i, R in enumerate(Ps):
            PA.point_add_ctrl(
                m, kr[i], px, py, R.x, R.y, p, multiplier, inversion,
                word_size)
        for i, R in enumerate(Qs):
            PA.point_add_ctrl(
                m, lr[i], px, py, R.x, R.y, p, multiplier, inversion,
                word_size)
    else:
        assert window_size is not None
        for base, ctrls in ((P, kr), (Q, lr)):
            for shift in range(0, mb, window_size):
                addr = ctrls[shift:min(shift + window_size, mb)]
                points = W.window_points(
                    curve, curve.mul(1 << shift, base), len(addr))
                W.windowed_point_add(m, addr, px, py, points, p, gcd=gcd,
                                       square=square)

    ck, cl = ClassicalRegister(mb, "ok"), ClassicalRegister(mb, "ol")
    qc = QuantumCircuit(*m.qc.qregs, ck, cl, name="ecdlp_arith")
    qc.h(kr)
    qc.h(lr)
    qc.compose(m.qc, inplace=True)
    qc.append(qft(mb).inverse(), list(kr))
    qc.append(qft(mb).inverse(), list(lr))
    qc.measure(kr, ck)
    qc.measure(lr, cl)

    return qc, {"n": n, "m_bits": mb, "offset": S, "order": order,
                "oracle": "arith", "arithmetic": arithmetic,
                "window_size": window_size, "gcd": gcd, "square": square, "multiplier": multiplier,
                "inversion": inversion, "word_size": word_size,
                "regs": (kr, lr, px, py),
                "machine": m, "qubits": qc.num_qubits}


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
    measured, and is reset.  That loop is `semiclassical.semiclassical_iqft`,
    shared with `onectrl.py`.

    The two registers are independent transforms, so the corrections run within
    a register and not across the pair: the loop runs twice, once per register.
    It puts bit i of each measured integer in bit i of its classical register,
    as the full circuit does, so the post-processing reads both unchanged.

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

    def rung(R):                                     # controlled "+R"
        return lambda qc, c: permutation(qc, pack, point_perm(curve, R, n),
                                         ctrls=[c])
    for base, creg in ((Ps, ck), (Qs, cl)):
        semiclassical_iqft(qc, ctr[0], creg, [rung(R) for R in base])

    return qc, {"n": n, "m_bits": mb, "offset": S, "order": order,
                "oracle": "table-1c", "qubits": qc.num_qubits}


# =============================================================================
# Running it
# =============================================================================
def run_ecdlp(curve, P, Q, order, shots=4096, seed=7, one_control=False, **kw):
    """Build a circuit, sample it, and solve for k.

    `oracle="table"` is the practical Aer path. The full arithmetic circuit is
    intended for resource inspection or hardware; its width is too large for
    statevector simulation.
    """
    oracle = kw.pop("oracle", "table")
    if one_control:
        if oracle != "table":
            raise ValueError("the one-control runner currently supports only the table oracle")
        qc, info = ecdlp_circuit_1c(curve, P, Q, order, **kw)
    else:
        if oracle != "table":
            raise ValueError(
                "run_ecdlp is the Aer table-oracle runner; build the arithmetic "
                "circuit with ecdlp_circuit and run it on suitable hardware")
        qc, info = ecdlp_circuit(curve, P, Q, order, oracle=oracle, **kw)
    import aer_device as AD
    counts = AD.run(qc, shots=shots, seed=seed)
    cands = C.ecdlp_postprocess(pair_counts(counts), order, info["m_bits"])
    return counts, cands, info


def run_ecdlp_arith(curve, P, Q, order, shots=2048, seed=7, **kw):
    """Sample the *arithmetic* oracle circuit end to end on Aer.

    The circuit is too wide for a statevector, but only the 2m control qubits
    are ever in superposition, so `aer_device` runs it on matrix-product
    states (CPU) or tensor networks (CUDA GPU).  Returns (counts, candidates,
    info) with info["aer"] naming the device and method used.
    """
    import aer_device as AD
    qc, info = ecdlp_circuit(curve, P, Q, order, oracle="arith", **kw)
    counts = AD.run(qc, shots=shots, seed=seed)
    info["aer"] = AD.describe(qc.num_qubits)
    cands = C.ecdlp_postprocess(pair_counts(counts), order, info["m_bits"])
    return counts, cands, info


def pair_counts(counts):
    """Qiskit counts over the two registers -> {(j1, j2): shots}."""
    tally = {}
    for bits, c in counts.items():
        lo, ko = bits.split()                   # Qiskit prints last register first
        tally[(int(ko, 2), int(lo, 2))] = tally.get((int(ko, 2), int(lo, 2)), 0) + c
    return tally


def solve(curve, P, Q, order, **kw):
    """The discrete logarithm, or None. Verified against the curve before use."""
    counts, cands, info = run_ecdlp(curve, P, Q, order, **kw)
    for k, _ in cands:
        if C.verify_dlog(curve, P, Q, k):
            return k, counts, info
    return None, counts, info
