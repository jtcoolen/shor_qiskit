"""Measurement-based uncomputation (MBU) for the ECDLP circuits.

Every 2026 point-addition circuit uncomputes table lookups the way the temporary
AND uncomputes its target: measure the register in the X basis and repair the
phase the measurement kicks back.  For a lookup |a>|T[a]> the outcome m leaves
(-1)^{m . T[a]} on the address, so the repair is a *phase* lookup of the
one-bit table F[a] = parity(m AND T[a]) -- which costs about sqrt(L), not L.
[1128] Sec 2, Litinski Fig 4b and IonQ Alg 6 all price a windowed addition's
lookups this way: three loads of 2^w Toffolis each, and the unloads nearly free.
IonQ merges the repairs further: every unload in a window shares one address,
so their tables are XORed and ONE phase lookup repairs them all.

The obstacle in this package is that MBU is not unitary, and the whole EC stack
is built on unitary machinery -- `emit_inverse` runs builders backwards,
`ec_sim` pushes basis states, `Machine.free` asserts ancillas clean.  So MBU is
represented the way `ec_gates` already represents the AND-dagger: as named
*logical* gates whose `definition` is the exact unitary uncompute (an XOR of the
table), and which carry their measurement-based price in an `ec_cost`
attribute that `ec_cost.count` adds in.

  LookupGate    "ec_lookup"    out ^= T[addr] by temporary-AND unary
                               iteration: L-1 ANDs controlled, L-2 not.
                               Its inverse is an UnlookupGate.
  UnlookupGate  "ec_unlookup"  out ^= T[addr], *by measurement*: len(out)
                               X-measurements plus a sqrt(L) phase lookup, or,
                               in a group, no repair of its own.
                               Its inverse is a LookupGate (a recompute).
  PhaseFixGate  "ec_phasefix"  logically the identity: the one merged repair
                               for a group of unlookups on the same address.
  MbuFlagGate   "ec_mbuflag"   a one-qubit flag f(data) uncomputed by
                               X-measurement, repaired by a caller-supplied
                               phase oracle (-1)^f(data) -- the carry and
                               comparator uncomputes of IonQ Alg 2 and Alg 8.

Those definitions make every existing tool keep working unchanged.  What they
cannot show is that the *measurement-based* realisation is phase-correct -- that
the repair really cancels the kickback.  `run_live` does that: it executes the
circuit on `sparse_sim.Session`, and at each of these gates it performs the real
X-basis measurement, builds the repair from the actual outcome, and applies it.
`live_coherent` then runs every basis input under fixed outcome sequences and
asserts they all come back with the same phase.  An approximate repair
(IonQ's PhaseGE on the top delta bits) fails that on some inputs, and
`live_faults` measures how many.
"""

import math

from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister

from ec_gates import Ctx


# =============================================================================
# Unary iteration with temporary ANDs, and its costs
# =============================================================================
def walk_ands(w, controlled=True):
    """ANDs in a unary iteration over w address bits (compute half; the
    uncompute is measurement-based and free)."""
    L = 1 << w
    if w == 0:
        return 0
    return L - 1 if controlled else L - 2


def walk_ancillas(w, controlled=True):
    return w if controlled else max(0, w - 1)


def _walk(ctx, ctrl, addr, leaf, offset, anc):
    """Visit every address value; leaf(ctx, c, value) runs under control c,
    which is a qubit that is 1 exactly on that value (c None: always)."""
    if not addr:
        leaf(ctx, ctrl, offset)
        return
    a, rest = addr[-1], addr[:-1]
    half = 1 << len(rest)
    if ctrl is None:                              # root: a itself is the control
        _walk(ctx, a, rest, leaf, offset + half, anc)
        ctx.x(a)
        _walk(ctx, a, rest, leaf, offset, anc)
        ctx.x(a)
        return
    node = anc[0]
    ctx.and_(ctrl, a, node)                       # node = ctrl AND a
    _walk(ctx, node, rest, leaf, offset + half, anc[1:])
    ctx.cx(ctrl, node)                            # node = ctrl AND NOT a
    _walk(ctx, node, rest, leaf, offset, anc[1:])
    ctx.cx(ctrl, node)
    ctx.and_dg(ctrl, a, node)


def xor_leaf(out, table):
    def leaf(ctx, c, v):
        T = table[v] if v < len(table) else 0
        for b in range(len(out)):
            if (T >> b) & 1:
                ctx.cx(c, out[b]) if c is not None else ctx.x(out[b])
    return leaf


def lookup_walk(ctx, ctrl, addr, out, table, anc):
    """out ^= table[addr] (when ctrl), temporary-AND unary iteration."""
    _walk(ctx, ctrl, list(addr), xor_leaf(list(out), list(table)), 0, list(anc))


def _mcx_xor(qc, ctrl, addr, out, table):
    """out ^= table[addr], ancilla-free: one open/closed-control MCX per bit.
    Exponentially many gates -- used only as a *definition* when a logical gate
    has no ancillas to walk with, never costed."""
    w = len(addr)
    for v in range(1 << w):
        T = table[v] if v < len(table) else 0
        cs = ([ctrl] if ctrl is not None else []) + list(addr)
        st = (v << 1 | 1) if ctrl is not None else v
        for b in range(len(out)):
            if (T >> b) & 1:
                if cs:
                    qc.mcx(cs, out[b], ctrl_state=st)
                else:
                    qc.x(out[b])


# --- the phase lookup: (-1)^F[addr], about sqrt(L) ---------------------------
def phase_cost(w, l, controlled=True):
    """ANDs for `phase_lookup` with l low bits one-hot encoded.

    l = 0 is a plain unary iteration with a Z at each marked leaf: L-1 ANDs.
    l > 0 one-hot encodes the low l bits (a lookup of 1<<j, 2^l - 1 ANDs, and
    the same again to unload it), then walks the high h = w-l bits applying
    CZ(node, onehot[j]) where F is set: 2(2^l - 1) + (2^h - 1).  With
    l = h = w/2 that is about 3 sqrt(L) -- Litinski's figure.
    """
    if l == 0:
        return walk_ands(w, controlled)
    h = w - l
    return 2 * walk_ands(l, controlled) + walk_ands(h, controlled)


def phase_ancillas(w, l, controlled=True):
    if l == 0:
        return walk_ancillas(w, controlled)
    h = w - l
    return (1 << l) + max(walk_ancillas(l, controlled), walk_ancillas(h, controlled))


def best_split(w, k, controlled=True):
    """The cheapest l whose ancilla need fits in k."""
    best = None
    for l in range(0, w + 1):
        if phase_ancillas(w, l, controlled) <= k:
            c = phase_cost(w, l, controlled)
            if best is None or c < best[1]:
                best = (l, c)
    return best                                   # None: not even l = 0 fits


def phase_lookup(ctx, ctrl, addr, F, anc, l=0):
    """Multiply |addr> by (-1)^F[addr] (when ctrl).  Ancillas returned clean."""
    addr, anc = list(addr), list(anc)
    w = len(addr)
    if l == 0:
        def leaf(cx, c, v):
            if v < len(F) and F[v]:
                if c is None:
                    raise AssertionError("uncontrolled l=0 leaf needs w >= 1")
                cx.qc.z(c)
        if w == 0:
            if F and F[0] and ctrl is not None:
                ctx.qc.z(ctrl)
            return
        _walk(ctx, ctrl, addr, leaf, 0, anc)
        return
    lo, hi = addr[:l], addr[l:]
    onehot, wanc = anc[:1 << l], anc[1 << l:]
    one = [1 << j for j in range(1 << l)]
    lookup_walk(ctx, ctrl, lo, onehot, one, wanc)            # onehot[j] = [lo == j]

    def leaf(cx, c, v):                                       # high value v
        for j in range(1 << l):
            idx = j + (v << l)
            if idx < len(F) and F[idx]:
                if c is None:
                    cx.qc.z(onehot[j])
                else:
                    cx.cz(c, onehot[j])
    if hi:
        # the high walk needs a control; with none, a leaf's control is the
        # address bit itself (the root trick), which _walk provides
        _walk(ctx, None if ctrl is None else ctrl, hi, leaf, 0, wanc)
    else:
        leaf(ctx, ctrl, 0)
    lookup_walk(ctx, ctrl, lo, onehot, one, wanc)            # unload the one-hot


def repair_plan(w, nanc, controlled, repair):
    """(split, ANDs) of the phase repair a gate with nanc ancillas can afford."""
    if repair == "phaseup":
        wa = w + int(controlled)                  # the control is one more bit
        l = wa // 2
        assert phaseup_ancillas(wa, l) <= nanc, (
            f"phaseup of {wa} bits needs {phaseup_ancillas(wa, l)} ancillas")
        return l, phaseup_ands(wa, l)
    assert repair == "onehot", repair
    split = best_split(w, nanc, controlled)
    assert split is not None, (
        f"a repair of {w} address bits needs at least "
        f"{phase_ancillas(w, 0, controlled)} ancillas")
    return split


def repair_ancillas(w, controlled, repair, l=None):
    if repair == "phaseup":
        wa = w + int(controlled)
        return phaseup_ancillas(wa, wa // 2 if l is None else l)
    return phase_ancillas(w, (w // 2) if l is None else l, controlled)


def apply_repair(ctx, ctrl, addr, F, anc, repair, split):
    """(-1)^F[addr] (when ctrl) with the chosen repair circuit."""
    if repair == "phaseup":
        if ctrl is None:
            phaseup(ctx, list(addr), F, anc, split)
        else:                                      # ctrl as address bit 0
            F2 = [0] * (2 * len(F))
            for a, f in enumerate(F):
                F2[2 * a + 1] = f
            phaseup(ctx, [ctrl] + list(addr), F2, anc, split)
    else:
        phase_lookup(ctx, ctrl, addr, F, anc, split)


# =============================================================================
# The logical gates
# =============================================================================
class _TableGate(Gate):
    """Common layout: [ctrl] + addr + out + anc.  Table kept as an attribute."""

    def __init__(self, name, w, nout, table, controlled, nanc, label=None):
        self.w, self.nout, self.table = w, nout, list(table)
        self.controlled, self.nanc = controlled, nanc
        super().__init__(name, int(controlled) + w + nout + nanc, [], label=label)

    def _split(self, qs):
        c = int(self.controlled)
        ctrl = qs[0] if self.controlled else None
        addr = qs[c:c + self.w]
        out = qs[c + self.w:c + self.w + self.nout]
        anc = qs[c + self.w + self.nout:]
        return ctrl, addr, out, anc

    def _xor_definition(self):
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        ctrl, addr, out, anc = self._split(list(q))
        if len(anc) >= walk_ancillas(self.w, self.controlled):
            lookup_walk(Ctx(qc, "and"), ctrl, addr, out, self.table, anc)
        else:
            _mcx_xor(qc, ctrl, addr, out, self.table)
        return qc

    def _define(self):
        self.definition = self._xor_definition()


class LookupGate(_TableGate):
    """out ^= T[addr]: unary iteration with temporary ANDs, or (anf=True,
    uncontrolled) the algebraic-normal-form lookup of [G25], 2^w - w - 1 ANDs."""

    def __init__(self, w, nout, table, controlled=True, nanc=None, anf=False):
        assert not (anf and controlled), "the ANF lookup is uncontrolled"
        self.anf = anf
        if nanc is None:
            nanc = max(w - 1, 1) if anf else walk_ancillas(w, controlled)
        super().__init__("ec_lookup", w, nout, table, controlled, nanc)
        cost = anf_ands(w) if anf else walk_ands(w, controlled)
        self.ec_cost = {"toffoli": cost, "measure": 0}

    def _define(self):
        if not self.anf:
            return super()._define()
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        _, addr, out, anc = self._split(list(q))
        lookup_anf_ctx(Ctx(qc, "and"), addr, out, self.table, anc)
        self.definition = qc

    def inverse(self, annotated=False):
        return UnlookupGate(self.w, self.nout, self.table, self.controlled, self.nanc)


class UnlookupGate(_TableGate):
    """out ^= T[addr] where out is known to hold T[addr]: by X-measurement.

    group=None: repaired here by a phase lookup on this gate's own ancillas
    (the best split they allow).  group=g: repaired later, once, by the
    PhaseFixGate of group g; costs only its measurements.
    """

    def __init__(self, w, nout, table, controlled=True, nanc=0, group=None,
                 repair="onehot"):
        super().__init__("ec_unlookup", w, nout, table, controlled, nanc)
        self.group, self.repair = group, repair
        if group is None:
            self.split, fix = repair_plan(w, nanc, controlled, repair)
        else:
            self.split, fix = None, 0
        self.ec_cost = {"toffoli": fix, "measure": nout}

    def inverse(self, annotated=False):
        return LookupGate(self.w, self.nout, self.table, self.controlled, self.nanc)


class PhaseFixGate(Gate):
    """The merged repair of a group of unlookups: [ctrl] + addr + anc.
    Logically the identity (each unlookup's definition is already exact)."""

    def __init__(self, w, group, controlled=True, nanc=0, repair="onehot"):
        self.w, self.group, self.controlled, self.nanc = w, group, controlled, nanc
        self.repair = repair
        super().__init__("ec_phasefix", int(controlled) + w + nanc, [])
        self.split, fix = repair_plan(w, nanc, controlled, repair)
        self.ec_cost = {"toffoli": fix, "measure": 0}

    def _define(self):
        self.definition = QuantumCircuit(QuantumRegister(self.num_qubits, "q"),
                                         name=self.name)

    def inverse(self, annotated=False):
        return PhaseFixGate(self.w, self.group, self.controlled, self.nanc, self.repair)


class MbuFlagGate(Gate):
    """flag ^= f(data) where flag holds f(data): by X-measurement.

    `recompute(qc, qubits)` appends a circuit XORing f(data) into the flag;
    `phase(qc, qubits)` appends one applying (-1)^f(data).  Both act on the
    gate's qubits in order: data + [flag] + anc.  `fix_cost` is the phase
    oracle's Toffoli count (it fires on outcome 1, i.e. half the time).
    """

    def __init__(self, ndata, nanc, recompute, phase, label="ec_mbuflag"):
        self.ndata, self.nanc = ndata, nanc
        self.recompute, self.phase = recompute, phase
        super().__init__("ec_mbuflag", ndata + 1 + nanc, [], label=label)
        self.fix_cost = _toffolis_of(phase, self.num_qubits)
        self.recompute_cost = _toffolis_of(recompute, self.num_qubits)
        self.ec_cost = {"toffoli": self.fix_cost, "measure": 1, "p_fire": 0.5}

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        self.recompute(qc, list(q))
        self.definition = qc

    def inverse(self, annotated=False):
        g = _RecomputeFlagGate(self)
        return g


class VentGate(Gate):
    """reg ^= f(data) where reg holds f(data): cleared by X-measurement, and
    the phase (-1)^(b . f) of the outcome b is left for a later `ZFixGate`
    with the same key -- [Luo26] Sec 5's recycling of a register whose value
    can be recomputed later from the others, but not cheaply now.

    Qubits: the data registers (sizes `groups`), then reg.  `f(*values)` is a
    classical function of the data values; `ec_sim` applies it directly
    (`ec_basis`), and `run_live` performs the measurement.  No Toffolis: the
    cost is the recomputation the caller builds around the ZFixGate."""

    def __init__(self, groups, nreg, f, key, label="ec_vent"):
        self.groups, self.nreg, self.f, self.key = list(groups), nreg, f, key
        super().__init__("ec_vent", sum(groups) + nreg, [], label=label)
        self.ec_cost = {"toffoli": 0, "measure": nreg}

    def _split(self, qs):
        vals, i = [], 0
        for g in self.groups:
            vals.append(qs[i:i + g])
            i += g
        return vals, qs[i:]

    def ec_basis(self, bits, w):
        """The unitary this gate stands for: reg ^= f(data)."""
        data, reg = self._split(w)
        vals = [sum(bits[q] << i for i, q in enumerate(g)) for g in data]
        v = self.f(*vals)
        for i, q in enumerate(reg):
            bits[q] ^= (v >> i) & 1

    def inverse(self, annotated=False):
        # Run backwards, a vent would be a *recomputation* of f(data) -- which
        # costs whatever computing f costs, not zero -- and its Z fix would
        # come before its measurement.  Builders that must be undone build
        # their inverse forwards (ec_cqadd.GidneyArith.csub / half).
        raise NotImplementedError(
            "a vent cannot be run backwards; build the inverse forwards")


class ZFixGate(Gate):
    """Z on the qubits of reg where the outcome of `VentGate(key)` has a 1.
    Placed where reg again holds the vented value, it cancels the vent's
    phase.  `key` may also be a list, one vent per qubit of reg (Z on
    reg[i] when vent key[i] read 1): [Gid25b]'s classically controlled Zs.
    An outcome can be used more than once.  Diagonal: the identity on basis
    states."""

    def __init__(self, nreg, key):
        self.nreg, self.key = nreg, key
        super().__init__("ec_zfix", nreg, [])
        self.ec_cost = {"toffoli": 0, "measure": 0}

    def ec_basis(self, bits, w):
        return

    def inverse(self, annotated=False):
        return self


def vent(m, data, reg, f, key):
    """Clear `reg`, which holds f(*data values), by X-measurement ([Luo26]
    Sec 5).  Its phase must be cancelled by `zfix(m, reg, key)` at a point
    where reg holds the same value again."""
    g = VentGate([len(d) for d in data], len(reg), f, key)
    m.qc.append(g, [q for d in data for q in d] + list(reg))


def zfix(m, reg, key):
    m.qc.append(ZFixGate(len(reg), key), list(reg))


class _RecomputeFlagGate(Gate):
    def __init__(self, fwd):
        self.fwd = fwd
        super().__init__("ec_reflag", fwd.num_qubits, [])
        self.ec_cost = {"toffoli": fwd.recompute_cost, "measure": 0}

    def _define(self):
        self.definition = self.fwd.definition

    def inverse(self, annotated=False):
        return self.fwd


def _toffolis_of(build, nq):
    from depth import profile
    qc = QuantumCircuit(QuantumRegister(nq, "q"))
    build(qc, list(qc.qubits))
    return profile(qc)["toffoli"]


# =============================================================================
# Machine helpers
# =============================================================================
def lookup(m, addr, out, table, ctrl=None, anf=False):
    """out ^= table[addr] (under ctrl), L-1 ANDs (L-2 uncontrolled; with
    anf=True, uncontrolled, 2^w - w - 1)."""
    w = len(addr)
    k = max(w - 1, 1) if anf else walk_ancillas(w, ctrl is not None)
    anc = m.anc(k, "lk")
    g = LookupGate(w, len(out), table, ctrl is not None, len(anc), anf=anf)
    m.qc.append(g, ([ctrl] if ctrl is not None else []) + list(addr) + list(out) + list(anc))
    m.free(anc)


def unlookup(m, addr, out, table, ctrl=None, group=None, l=None, repair="onehot"):
    """Clear out = table[addr] by measurement.  With `group`, the repair is
    deferred to `phase_fix(m, group)`; otherwise it is done here, on
    ancillas drawn for the chosen repair ("onehot" sqrt(L) split, or
    "phaseup": [G25]'s power products)."""
    w = len(addr)
    ctl = ctrl is not None
    k = repair_ancillas(w, ctl, repair, l) if group is None else 0
    anc = m.anc(k, "ulk") if k else []
    g = UnlookupGate(w, len(out), table, ctl, k, group, repair)
    m.qc.append(g, ([ctrl] if ctl else []) + list(addr) + list(out) + list(anc))
    if k:
        m.free(anc)
    if group is not None:
        _groups(m).setdefault(group, []).append((tuple(addr), ctrl))


def swap_network_order(k, a_lo):
    """Which of the 2^k loaded words each block holds after `select_swap_lookup`'s
    swap network on low address bits a_lo: block 0 holds word a_lo."""
    idx = list(range(1 << k))
    for i in reversed(range(k)):
        if (a_lo >> i) & 1:
            s = 1 << i
            for j in range(s):
                idx[j], idx[j + s] = idx[j + s], idx[j]
    return idx


def select_swap_ands(w, k, b):
    """Toffolis of a SELECT-SWAP load of b-bit words, uncontrolled: the lookup
    over w - k high bits plus (2^k - 1) b Fredkins.  The junk is cleared by
    measurement."""
    return walk_ands(w - k, False) + ((1 << k) - 1) * b


def select_swap_lookup(m, addr, out, table, k, group=None):
    """out ^= table[addr], uncontrolled, by SELECT-SWAP (Low, Kliuchnikov and
    Schaeffer 2018).

    A lookup over the high w - k address bits loads lam = 2^k consecutive words
    at once -- `out` and lam - 1 junk blocks -- and k layers of swaps controlled
    by the low bits move word a_lo into `out`: 2^(w-k) - 2 ANDs and (lam - 1) b
    Fredkins instead of 2^w - 2 ANDs.  After the swaps every junk block holds a
    word that is a known function of the whole address, so the junk is cleared
    at once by X-measurement and its phase kickback joins `group`'s merged
    repair (or is repaired here without one): the (lam - 1) b extra qubits are
    live only during the load, and the unload of `out` is unchanged."""
    w, b = len(addr), len(out)
    k = min(k, w)
    if k == 0:
        lookup(m, addr, out, table)
        return
    lam = 1 << k
    lo, hi = list(addr[:k]), list(addr[k:])

    def word(v):
        return table[v] if v < len(table) else 0
    junk = m.anc((lam - 1) * b, "ssw")
    blocks = [list(out)] + [list(junk[j * b:(j + 1) * b]) for j in range(lam - 1)]
    wide = [sum(word((v << k) | j) << (j * b) for j in range(lam)) for v in range(1 << (w - k))]
    lookup(m, hi, [q for blk in blocks for q in blk], wide)
    for i in reversed(range(k)):                  # block 0 <- word a_lo
        s = 1 << i
        for j in range(s):
            for qa, qb in zip(blocks[j], blocks[j + s]):
                m.ctx.cswap(lo[i], qa, qb)
    G = []                                        # the junk, as a function of addr
    for a in range(1 << w):
        idx, base = swap_network_order(k, a & (lam - 1)), (a >> k) << k
        G.append(sum(word(base | idx[j]) << ((j - 1) * b) for j in range(1, lam)))
    unlookup(m, addr, junk, G, group=group)
    m.free(junk)


def _groups(m):
    if not hasattr(m, "_mbu_groups"):
        m._mbu_groups = {}
    return m._mbu_groups


def phase_fix(m, group, l=None, repair="onehot"):
    """The one merged repair for every unlookup of `group` so far."""
    members = _groups(m).pop(group, [])
    assert members, f"group {group!r} has no unlookups"
    addr, ctrl = members[0]
    assert all(a == addr and c == ctrl for a, c in members), (
        "a merged phase fix needs every member on the same address and control")
    w, ctl = len(addr), ctrl is not None
    k = repair_ancillas(w, ctl, repair, l)
    anc = m.anc(k, "pfx") if k else []
    g = PhaseFixGate(w, group, ctl, k, repair)
    m.qc.append(g, ([ctrl] if ctl else []) + list(addr) + list(anc))
    if k:
        m.free(anc)


def mbu_flag(m, flag, data, recompute, phase, nanc=0):
    """Clear a one-qubit flag holding f(data) by X-measurement.

    recompute(qc, qs) / phase(qc, qs) act on qs = data + [flag] + anc; their
    Toffoli counts are measured off the circuits they build."""
    anc = m.anc(nanc, "mf") if nanc else []
    g = MbuFlagGate(len(data), nanc, recompute, phase)
    m.qc.append(g, list(data) + [flag] + list(anc))
    if nanc:
        m.free(anc)


# =============================================================================
# Live execution: the real measurements, the real repairs
# =============================================================================
def _parity(v):
    return bin(v).count("1") & 1


def run_live(qc, init=None, state=None, outcomes=None, seed=0, checks=(),
             and_literal=True):
    """Execute qc on a sparse Session, realising every MBU literally.

    ec_unlookup  X-measure out; F[a] = parity(m & T[a]); repair now (or
                 accumulate into its group)
    ec_phasefix  apply the group's accumulated F with the split its ancillas allow
    ec_mbuflag   X-measure the flag; on 1, apply the phase oracle
    ec_vent      X-measure the register; keep the outcome for its ec_zfix
    ec_zfix      Z on the register's qubits where that outcome has a 1
    ecand_dg     (and_literal) X-measure the target; on 1, CZ(a, b)
    anything else runs as a unitary.  Returns the Session.
    """
    from sparse_sim import Session
    S = Session(qc, init=init, state=state, seed=seed, outcomes=outcomes)
    regs = list(qc.qregs)
    pending, vented, seg, seg_start = {}, {}, [], 0
    by_pos = {}
    for pos, qs, val in checks:
        by_pos.setdefault(pos, []).append((qs, val))

    def new_seg():
        return QuantumCircuit(*regs)

    def flush(upto):
        nonlocal seg, seg_start
        if seg:
            c = new_seg()
            for ci in seg:
                c.append(ci.operation, ci.qubits)
            chk = [(p - seg_start, qs, v) for p in range(seg_start, upto + 1)
                   for qs, v in by_pos.get(p, ())]
            S.run(c, chk)
        else:
            for qs, v in by_pos.get(upto, ()):
                vals = S.branch.values(qs)
                assert all(int(x) == v for x in vals), f"ancilla check at {upto}"
        seg, seg_start = [], upto + 1

    def run_ops(build, qubits):
        c = new_seg()
        build(c, list(qubits))
        S.run(c)

    for pos, ci in enumerate(qc.data):
        op, name = ci.operation, ci.operation.name
        if name not in ("ec_unlookup", "ec_phasefix", "ec_mbuflag", "ec_vent",
                        "ec_zfix") and not (
                and_literal and name == "ecand_dg"):
            if not seg:
                seg_start = pos
            seg.append(ci)
            continue
        flush(pos)
        qs = list(ci.qubits)
        if name == "ecand_dg":
            mm = S.mx([qs[2]])
            if mm:
                run_ops(lambda c, q: c.cz(q[0], q[1]), qs[:2])
        elif name == "ec_unlookup":
            ctrl, addr, out, anc = op._split(qs)
            mm = S.mx(out)
            F = [_parity(mm & (op.table[v] if v < len(op.table) else 0))
                 for v in range(1 << op.w)]
            if op.group is None:
                run_ops(lambda c, q: apply_repair(Ctx(c, "and"), ctrl, addr, F, anc,
                                                  op.repair, op.split), [])
            else:
                key = op.group
                acc = pending.setdefault(key, [0] * (1 << op.w))
                pending[key] = [a ^ b for a, b in zip(acc, F)]
        elif name == "ec_phasefix":
            c0 = int(op.controlled)
            ctrl = qs[0] if op.controlled else None
            addr, anc = qs[c0:c0 + op.w], qs[c0 + op.w:]
            F = pending.pop(op.group, [0] * (1 << op.w))
            if any(F):
                run_ops(lambda c, q: apply_repair(Ctx(c, "and"), ctrl, addr, F, anc,
                                                  op.repair, op.split), [])
        elif name == "ec_mbuflag":
            flag = qs[op.ndata]
            mm = S.mx([flag])
            if mm:
                run_ops(lambda c, q: op.phase(c, qs), [])
        elif name == "ec_vent":
            vented[op.key] = S.mx(op._split(qs)[1])
        elif name == "ec_zfix":
            if isinstance(op.key, list):                # one vent per qubit
                b = sum((vented[k] & 1) << i for i, k in enumerate(op.key))
            else:
                b = vented[op.key]
            if b:
                run_ops(lambda c, q: [c.z(qs[i]) for i in range(op.nreg) if (b >> i) & 1], [])
        seg_start = pos + 1
    flush(len(qc.data))
    return S


def live_phases(qc, inputs, outcome_seq, checks=(), expect=None):
    """Run every basis input under one outcome sequence; return the phases."""
    out = []
    for inp in inputs:
        S = run_live(qc, init=inp, outcomes=outcome_seq, checks=checks)
        br = S.branch
        br.assert_basis()
        if expect is not None:
            expect(inp, br)
        out.append(br.phase())
    return out


def live_coherent(qc, inputs, outcomes, checks=(), expect=None, tol=1e-9):
    """Every input ends with the same phase, for each outcome sequence."""
    from sparse_sim import SparseSimError
    for oi, seq in enumerate(outcomes):
        ph = live_phases(qc, inputs, seq, checks, expect)
        bad = [i for i, p in enumerate(ph) if abs(p - ph[0]) > tol]
        if bad:
            raise SparseSimError(
                f"outcome sequence {oi}: {len(bad)}/{len(ph)} inputs end with a "
                f"different phase -- the measurement-based uncompute is not repaired")


def live_faults(qc, inputs, outcome_seq, checks=(), tol=1e-9):
    """Fraction of inputs whose phase differs from the most common phase."""
    ph = live_phases(qc, inputs, outcome_seq, checks)
    buckets = []
    for p in ph:
        for b in buckets:
            if abs(b[0] - p) < tol:
                b[1] += 1
                break
        else:
            buckets.append([p, 1])
    return 1 - max(b[1] for b in buckets) / len(ph)


def all_ones_outcome(n=10_000):
    return lambda i, c: 1


def random_outcomes(seed):
    import random
    r = random.Random(seed)
    cache = {}

    def pick(i, c):
        if i not in cache:
            cache[i] = r.getrandbits(1)
        return cache[i]
    return pick


def fix_cost_table(w):
    """{l: (ANDs, ancillas)} for an address of w bits, controlled."""
    return {l: (phase_cost(w, l), phase_ancillas(w, l)) for l in range(w + 1)}


def sqrt_split(w):
    return max(0, min(w, round(w / 2)))


_ = math  # keep the import for callers that use ec_mbu.math


# =============================================================================
# Algebraic normal form: Gidney 2025's uncontrolled lookup and phaseup
# =============================================================================
def _mobius(table, w, width):
    """ANF coefficients: c[S] with table[a] = XOR over S subset of a of c[S]
    (bitwise, `width` bits).  The GF(2) Moebius transform."""
    c = [table[a] if a < len(table) else 0 for a in range(1 << w)]
    for i in range(w):
        for a in range(1 << w):
            if (a >> i) & 1:
                c[a] ^= c[a ^ (1 << i)]
    return c


def _monomials(ctx, bits, anc, visit):
    """Depth-first over the monomials of `bits`: each monomial of degree >= 2
    is one AND of its parent with a new bit, uncomputed on the way back (free).
    visit(mask, qubit) is called for every monomial (qubit None for the empty
    one).  Uses len(bits) - 1 ancillas, 2^w - w - 1 ANDs."""
    visit(0, None)

    def rec(mask, q, start, depth):
        for i in range(start, len(bits)):
            if q is None:
                visit(1 << i, bits[i])
                rec(1 << i, bits[i], i + 1, depth)
            else:
                t = anc[depth]
                ctx.and_(q, bits[i], t)
                visit(mask | (1 << i), t)
                rec(mask | (1 << i), t, i + 1, depth + 1)
                ctx.and_dg(q, bits[i], t)
    rec(0, None, 0, 0)


def anf_ands(w):
    return max(0, (1 << w) - w - 1)


def lookup_anf_ctx(ctx, addr, out, table, anc):
    w = len(addr)
    coeff = _mobius(table, w, len(out))

    def visit(mask, q):
        c = coeff[mask]
        for b in range(len(out)):
            if (c >> b) & 1:
                ctx.x(out[b]) if q is None else ctx.cx(q, out[b])
    _monomials(ctx, list(addr), list(anc), visit)


def lookup_anf(m, addr, out, table):
    """out ^= table[addr], uncontrolled, in 2^w - w - 1 ANDs ([G25] App A).

    Unary iteration pays one AND per internal node of the address tree
    (2^w - 1, or 2^w - 2 uncontrolled).  Writing the table in algebraic
    normal form, table[a] = XOR over monomials of a, needs one AND per
    monomial of degree >= 2 -- the degree-1 monomials are the address bits
    themselves -- and the data load is CNOTs from each monomial."""
    w = len(addr)
    coeff = _mobius(table, w, len(out))
    anc = m.anc(max(w - 1, 1), "anf")

    def visit(mask, q):
        c = coeff[mask]
        for b in range(len(out)):
            if (c >> b) & 1:
                m.ctx.x(out[b]) if q is None else m.ctx.cx(q, out[b])
    _monomials(m.ctx, list(addr), anc, visit)
    m.free(anc)


def _build_monomials(ctx, bits, regs):
    """Every monomial of `bits` in its own register: degree 0 -> None,
    degree 1 -> the bit itself, degree >= 2 -> one AND of its parent (the
    monomial without its top bit) with that bit.  Returns (store, order)."""
    store = {0: None}
    for i, b in enumerate(bits):
        store[1 << i] = b
    it, order = iter(regs), []
    for mask in sorted(range(1, 1 << len(bits)), key=lambda v: (bin(v).count("1"), v)):
        if bin(mask).count("1") < 2:
            continue
        top = mask.bit_length() - 1
        parent = mask ^ (1 << top)
        t = next(it)
        ctx.and_(store[parent], bits[top], t)
        store[mask] = t
        order.append((parent, top, mask))
    return store, order


def _unbuild_monomials(ctx, bits, store, order):
    for parent, top, mask in reversed(order):
        ctx.and_dg(store[parent], bits[top], store[mask])


def phaseup(ctx, addr, F, anc, l=None):
    """Multiply |addr> by (-1)^F[addr] ([G25] App A.3, "power products").

    Split the address into l low and h high bits and materialise *every*
    monomial of each half, one AND each: (2^l - l - 1) + (2^h - h - 1) ANDs,
    uncomputed for free.  Over GF(2), F(lo, hi) = sum of c[S, T] m_S(lo) m_T(hi),
    so the phase is one CZ (or Z) per nonzero coefficient: Clifford.  About
    2 sqrt(L) ANDs, against ~3 sqrt(L) for the one-hot split and L for a
    phase walk.  `anc`: phaseup_ancillas(w, l) monomial registers."""
    addr = list(addr)
    w = len(addr)
    l = (w // 2) if l is None else l
    h = w - l
    lo, hi = addr[:l], addr[l:]
    C = _mobius([int(bool(v)) for v in F], w, 1)
    nlo = anf_ands(l)
    slo, olo = _build_monomials(ctx, lo, anc[:nlo])
    shi, ohi = _build_monomials(ctx, hi, anc[nlo:])
    for S in range(1 << l):
        for T in range(1 << h):
            if not C[S | (T << l)]:
                continue
            a, b = slo[S], shi[T]
            if a is None and b is None:
                continue                                 # a global phase
            if a is None:
                ctx.qc.z(b)
            elif b is None:
                ctx.qc.z(a)
            else:
                ctx.cz(a, b)
    _unbuild_monomials(ctx, hi, shi, ohi)
    _unbuild_monomials(ctx, lo, slo, olo)


def phaseup_ands(w, l=None):
    l = (w // 2) if l is None else l
    return anf_ands(l) + anf_ands(w - l)


def phaseup_ancillas(w, l=None):
    return phaseup_ands(w, l)
