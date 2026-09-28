"""Hierarchical, memoised circuit building: the same builders at n = 256.

Every builder in this package emits a flat circuit.  That is what the exact
simulators need, and it is fine at toy sizes, but a 256-bit point addition is
~10^7 gates -- too many Python objects to hold.  Yet almost all of those gates
come from a few builders called over and over *on registers of the same
shape*: 400 GCD rounds of one width, 800 modular doublings of 256 bits...

So, without touching a line of the existing code:

  boundary(fn)       a decorator.  On an ordinary `Machine` it is invisible
                     (it calls fn).  On a `HierMachine` it builds fn once per
                     *shape* -- register sizes, which arguments share qubits,
                     and the classical parameters -- on a fresh child machine,
                     caches the result, and emits it as ONE opaque gate that
                     carries its cost.  Inverting that gate inverts the cost.
  tracing(...)       applies `boundary` to a list of existing builders for the
                     duration of a `with` block and restores them afterwards
                     -- the decorator, applied at runtime, so the modules and
                     every listing cut from them stay byte-identical.
  count(m)           the same schema as `ec_cost.count`, summed through the
                     cache.

Exactness: a builder's gate counts depend only on its shape (every builder here
is deterministic in its sizes and classical arguments), so the hierarchical
count equals the flat one -- which `tests/test_hier.py` checks at toy sizes on
the full windowed point addition.  Anything a boundary cannot safely capture
(a callable argument, a returned register, qubits left allocated or freed that
belong to the caller) is run flat instead, silently and correctly.

`to_qualtran(m)` exposes the cached tree as Qualtran bloqs (optional).
"""

import contextlib
import functools

from qiskit.circuit import Gate, QuantumCircuit, Qubit

from ec_sim import Machine, Reg

_CACHE = {}          # key -> Node
_NOBOUND = set()     # functions found unsafe to capture: always run flat


class Node:
    """One cached builder call: its cost (leaf tally, children expanded), its
    own structure (leaf gates here, and which cached calls it makes), and the
    qubits it needs beyond its arguments."""

    def __init__(self, name, key, n_args, n_anc, tally, local, children):
        self.name, self.key = name, key
        self.n_args, self.n_anc = n_args, n_anc
        self.tally = tally            # {gate name: count}, fully expanded
        self.local = local            # {gate name: count}, this level only
        self.children = children      # {(child Node, inverted): multiplicity}


class NodeGate(Gate):
    def __init__(self, node, inverted=False):
        self.node, self.inverted = node, inverted
        super().__init__(f"hier:{node.name}{'^-1' if inverted else ''}",
                         node.n_args + node.n_anc, [])

    def inverse(self, annotated=False):
        return NodeGate(self.node, not self.inverted)

    def _define(self):
        raise RuntimeError("hierarchical gates are for counting; build flat to simulate")


_INV_NAME = {"ecand": "ecand_dg", "ecand_dg": "ecand",
             "ec_lookup": "ec_unlookup", "ec_unlookup": "ec_lookup",
             "ec_mbuflag": "ec_reflag", "ec_reflag": "ec_mbuflag"}


def _invert_tally(t):
    """The tally of the inverse circuit.  Gate names map to their inverses;
    a logical gate's carried cost ('cost:...') and its inverse's ('inv:...')
    trade places, because a lookup and its inverse (an unlookup) cost
    different amounts."""
    out = {}
    swap = {"cost:": "inv:", "inv:": "cost:", "exp:": "iexp:", "iexp:": "exp:"}
    for k, v in t.items():
        pre = next((p for p in swap if k.startswith(p)), None)
        if pre:
            k2 = swap[pre] + k[len(pre):]
        else:
            k2 = _INV_NAME.get(k, k)
        out[k2] = out.get(k2, 0) + v
    return out


def node_tally(gate):
    t = gate.node.tally
    return _invert_tally(t) if gate.inverted else t


class HierMachine(Machine):
    """A Machine on which `boundary` builders become cached opaque gates."""
    hier = True

    def __init__(self, mode="and", name="ec"):
        super().__init__(mode, name)
        self.ctx._machine = self             # so ctx-only builders find it


# =============================================================================
# The decorator
# =============================================================================
def _is_qubits(x):
    return isinstance(x, (list, tuple)) and x and all(isinstance(q, Qubit) for q in x)


def _shape(args, kwargs):
    """(key, qubit order, rebuild) for a call, or None if not capturable."""
    qubits, index = [], {}

    def qid(q):
        if q not in index:
            index[q] = len(qubits)
            qubits.append(q)
        return index[q]

    def enc(x):
        if isinstance(x, Machine):
            return ("M",)
        if type(x).__name__ == "Ctx":
            return ("C", x.mode)
        if isinstance(x, Qubit):
            return ("q", qid(x))
        if isinstance(x, Reg) or _is_qubits(x):
            return ("R", tuple(qid(q) for q in x))
        if isinstance(x, (int, str, bool, float, type(None))):
            return ("v", x)
        if isinstance(x, (list, tuple)) and x and type(x[0]).__name__ == "Point":
            # a window table: it only enters through lookups, whose cost depends
            # on the number of entries, not their values -- so key it by size
            # and let every window of an ECDLP circuit share one node.  (The
            # cached gates carry the first table's data: counting only.)
            return ("Pts", len(x), any(getattr(P, "inf", False) for P in x))
        if isinstance(x, (list, tuple)):
            parts = tuple(enc(e) for e in x)
            if any(p is None for p in parts):
                return None
            return ("L", type(x).__name__, parts)
        if isinstance(x, functools.partial) or type(x).__name__ in (
                "function", "method", "builtin_function_or_method"):
            # a callable passed as configuration: keyed by identity (correct;
            # a lambda made afresh on every call simply never hits the cache)
            return ("F", id(x))
        if hasattr(x, "__dict__"):                   # a config object (arith, backend)
            items = tuple(sorted((k, enc(v)) for k, v in vars(x).items()
                                 if not k.startswith("_cache")))
            if any(v is None for _, v in items):
                return None
            return ("O", type(x).__name__, items)
        return None

    ea = tuple(enc(a) for a in args)
    ek = tuple(sorted((k, enc(v)) for k, v in kwargs.items()))
    if any(e is None for e in ea) or any(e is None for _, e in ek):
        return None
    return (ea, ek), qubits


def _rebuild(x, qmap, child):
    if isinstance(x, Machine):
        return child
    if type(x).__name__ == "Ctx":
        return child.ctx
    if isinstance(x, Qubit):
        return qmap[x]
    if isinstance(x, Reg):
        return Reg([qmap[q] for q in x], getattr(x, "name", ""))
    if _is_qubits(x):
        return type(x)(qmap[q] for q in x)
    if isinstance(x, (list, tuple)):
        return type(x)(_rebuild(e, qmap, child) for e in x)
    return x


def _find_machine(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, Machine):
            return a, None
        if type(a).__name__ == "Ctx":
            return getattr(a, "_machine", None), a
    return None, None


def boundary(fn, name=None):
    label = name or f"{fn.__module__}.{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        m, ctx = _find_machine(args, kwargs)
        if m is None or not getattr(m, "hier", False) or label in _NOBOUND:
            return fn(*args, **kwargs)
        sh = _shape(args, kwargs)
        if sh is None:
            return fn(*args, **kwargs)
        (ea, ek), qubits = sh
        key = (label, ea, ek)
        node = _CACHE.get(key)
        if node is None:
            node = _build(fn, label, key, args, kwargs, qubits, m)
            if node is None:
                _NOBOUND.add(label)
                return fn(*args, **kwargs)
            _CACHE[key] = node
        anc = m.anc(node.n_anc, "h") if node.n_anc else []
        m.qc.append(NodeGate(node), list(qubits) + list(anc))
        if node.n_anc:
            m.free(anc)
        return None

    wrapper.__boundary__ = True
    return wrapper


def _build(fn, label, key, args, kwargs, qubits, parent):
    child = HierMachine(parent.mode, label.split(".")[-1])
    from qiskit.circuit import QuantumRegister
    reg = QuantumRegister(len(qubits), "args")
    child.qc.add_register(reg)
    qmap = dict(zip(qubits, reg))
    cargs = [_rebuild(a, qmap, child) for a in args]
    ckw = {k: _rebuild(v, qmap, child) for k, v in kwargs.items()}
    ret = fn(*cargs, **ckw)
    if ret is not None:
        return None                                   # returns handles: run flat
    if child._live != 0 or any(q in qmap.values() for q in child._pool):
        return None                                   # leaves or frees caller qubits
    local, kids = structure(child)
    node = Node(label.split(".")[-1], key, len(qubits), child._nanc, count_tally(child),
                local, kids)
    node._st, node._fi = _node_timing(child, len(qubits) + child._nanc)
    node.depth = max([f for f in node._fi if f is not None], default=0)
    idx = {q: i for i, q in enumerate(child.qc.qubits)}
    node.ops = [(ci.operation, tuple(idx[q] for q in ci.qubits)) for ci in child.qc.data]
    return node


# =============================================================================
# Counting
# =============================================================================
def count_tally(m):
    """{gate name: count} over m's circuit, NodeGates expanded; logical gates'
    carried costs recorded as 'cost:<name>|toffoli' / '|measure'."""
    qc = getattr(m, "qc", m)
    t = {}
    for ci in qc.data:
        op = ci.operation
        if isinstance(op, NodeGate):
            for k, v in node_tally(op).items():
                t[k] = t.get(k, 0) + v
            continue
        t[op.name] = t.get(op.name, 0) + 1
        c = getattr(op, "ec_cost", None)
        if c:
            _carried(t, op, c)
    return t


def _carried(t, op, c):
    """A logical gate's carried costs, and its inverse's, into tally t."""
    ci_ = getattr(op.inverse(), "ec_cost", None) or {}
    for what in ("toffoli", "measure"):
        kk = f"cost:{op.name}|{what}"
        t[kk] = t.get(kk, 0) + c.get(what, 0)
        ki = f"inv:{op.name}|{what}"
        t[ki] = t.get(ki, 0) + ci_.get(what, 0)
    ke, kie = f"exp:{op.name}|toffoli", f"iexp:{op.name}|toffoli"
    t[ke] = t.get(ke, 0) + c.get("toffoli", 0) * c.get("p_fire", 1)
    t[kie] = t.get(kie, 0) + ci_.get("toffoli", 0) * ci_.get("p_fire", 1)


def structure(m):
    """(leaf tally of this level, {(child node, inverted): multiplicity})."""
    qc = getattr(m, "qc", m)
    local, kids = {}, {}
    for ci in qc.data:
        op = ci.operation
        if isinstance(op, NodeGate):
            k = (op.node, op.inverted)
            kids[k] = kids.get(k, 0) + 1
            continue
        local[op.name] = local.get(op.name, 0) + 1
        c = getattr(op, "ec_cost", None)
        if c:
            _carried(local, op, c)
    return local, kids


def _leaf_depth(op):
    """Toffoli-class depth of one leaf: 1 per Toffoli/AND/Fredkin, the ladder
    of a multi-controlled X, a logical gate's carried cost taken as serial."""
    c = getattr(op, "ec_cost", None)
    if c:
        return c.get("toffoli", 0)
    name = op.name
    if name in ("ecand", "ccx", "cswap", "ccz"):
        return 1
    if name == "mcx":
        k = op.num_qubits - 1
        return 1 if k == 2 else max(0, 2 * k - 3)
    return 0


def _schedule(qc):
    """As-soon-as-possible Toffoli depth with every input ready at 0.
    Returns ({qubit: time its last operation ends}, {qubit: time its first
    operation starts}).  A cached call is placed by its per-qubit timing:
    it starts as late as its latest-arriving input allows (each input is
    first needed `start[i]` into the call) and hands each qubit back
    `finish[i]` after that -- so consecutive calls overlap where the first
    one's late qubits are the second one's late qubits (an adder's carry
    chain feeding the next adder).  Cliffords take no time but order their
    qubits.  An upper bound on the true depth."""
    ready, first = {}, {}
    for ci in qc.data:
        op, qs = ci.operation, ci.qubits
        if isinstance(op, NodeGate):
            st, fi = op.node.timing(op.inverted)
            used = [i for i in range(len(qs)) if st[i] is not None]
            S = max((ready.get(qs[i], 0) - st[i] for i in used), default=0)
            S = max(S, 0)
            for i in used:
                first.setdefault(qs[i], S + st[i])
                ready[qs[i]] = S + fi[i]
            continue
        d = _leaf_depth(op)
        t0 = max((ready.get(q, 0) for q in qs), default=0)
        for q in qs:
            first.setdefault(q, t0)
            ready[q] = t0 + d
    return ready, first


def _node_timing(child, nq):
    ready, first = _schedule(child.qc)
    qubits = child.qc.qubits
    st = [first.get(q) for q in qubits][:nq]
    fi = [ready.get(q) for q in qubits][:nq]
    return st, fi


def _timing(self, inverted=False):
    """(start, finish) per qubit, relative to the call's start; run backwards,
    what finished last starts first."""
    if not inverted:
        return self._st, self._fi
    D = self.depth
    return ([None if f is None else D - f for f in self._fi],
            [None if s is None else D - s for s in self._st])


Node.timing = _timing


def block_depth(m):
    """A fast estimate of the Toffoli depth: see `_schedule`.  Not a bound
    either way -- `exact_depth` is the number."""
    ready, _ = _schedule(getattr(m, "qc", m))
    return max(ready.values(), default=0)


def exact_depth(m):
    """The Toffoli depth of the *flat* circuit, without building it: every
    cached call is expanded on the fly from its node's gate list (run
    backwards, gate by gate inverted, for an inverted call) and scheduled
    as soon as possible per qubit, as `depth.toffoli_depth` does on a flat
    circuit -- a Toffoli, AND or Fredkin takes one step, an AND-dagger none,
    a logical gate its carried cost; qubits shared by two gates order them.
    Time linear in the flat gate count; memory that of the cache."""
    qc = getattr(m, "qc", m)
    idx = {q: i for i, q in enumerate(qc.qubits)}
    ready = [0] * qc.num_qubits
    inv_depth = {}

    def leaf(op, inverted):
        if not inverted:
            return _leaf_depth(op)
        key = id(op)
        if key not in inv_depth:
            inv_depth[key] = (op, _leaf_depth(op.inverse()))
        return inv_depth[key][1]

    def run(ops, qmap, inverted):
        for op, qs in (reversed(ops) if inverted else ops):
            w = [qmap[i] for i in qs]
            if isinstance(op, NodeGate):
                run(op.node.ops, w, inverted ^ op.inverted)
                continue
            d = leaf(op, inverted)
            t = max(ready[j] for j in w) + d if w else 0
            for j in w:
                ready[j] = t

    top = [(ci.operation, tuple(idx[q] for q in ci.qubits)) for ci in qc.data]
    run(top, list(range(qc.num_qubits)), False)
    return max(ready, default=0)


def count(m):
    """`ec_cost.count`'s headline fields, through the cache."""
    from ec_cost import AND_T, TOFFOLI_T
    t = count_tally(m)
    ands, dgs = t.get("ecand", 0), t.get("ecand_dg", 0)
    toffs = t.get("ccx", 0) + t.get("mcx", 0)
    cswaps = t.get("cswap", 0)
    mbu = sum(v for k, v in t.items() if k.startswith("cost:") and k.endswith("|toffoli"))
    meas = sum(v for k, v in t.items() if k.startswith("cost:") and k.endswith("|measure"))
    mbu_exp = sum(v for k, v in t.items() if k.startswith("exp:"))
    qc = getattr(m, "qc", m)
    return {"qubits": qc.num_qubits, "and": ands, "and_dg": dgs, "toffoli": toffs,
            "cswap": cswaps, "mbu_toffoli": mbu, "measure": meas,
            "toffoli_paper": ands + toffs + cswaps + mbu,
            "toffoli_expected": ands + toffs + cswaps + mbu_exp,
            "toffoli_equiv": ands + dgs + toffs + cswaps + mbu,
            "t": ands * AND_T + (toffs + cswaps) * TOFFOLI_T + mbu * AND_T,
            "cached_nodes": len(_CACHE), "top_level_ops": len(qc.data)}


# =============================================================================
# Applying the decorator to existing builders, for the duration of a block
# =============================================================================
DEFAULT_TARGETS = [
    ("ec_adders", ["gidney_add", "cdkm_add", "add", "sub", "cadd", "csub", "add_const",
                   "sub_const", "cadd_const", "csub_const", "carry_out", "gt_uint",
                   "lt_uint", "geq_const", "lt_const", "clt_uint", "cgt_fused",
                   "gt_top", "ci_add"]),
    ("ec_modarith", ["is_zero", "is_nonzero", "cmodadd", "modadd", "modsub", "cmodsub",
                     "cmodadd_const", "modadd_const", "modsub_const", "cmodsub_const",
                     "moddbl", "cmoddbl", "cmodhalf", "modhalf", "cmodneg", "modneg",
                     "csignadd"]),
    ("ec_approx", ["all_ones", "eq_top", "lt_approx", "moddbl_approx", "moddbl_pm",
                   "cmodadd_approx", "cmodadd_pm", "cmodadd_pm_q", "csignadd_pm",
                   "cmodneg_approx", "modhalf_pm", "_fix_zero_q", "_swap_zero_q",
                   "modadd_pm_phase", "csignadd_pm_phase"]),
    ("ec_mult", ["modmul", "modsqr", "modmul_add", "modmul_sub", "modmul_xor",
                 "modsqr_sub", "cmodmul", "modmul_const"]),
    ("ec_square", ["sqr_int", "sqr_int_budget", "csub_square_pm", "csub_square_generic",
                   "csub_square_karatsuba_pm"]),
    ("ec_space", ["moddbl_pm_space", "modhalf_pm_space", "cmodadd_pm_q_space",
                  "csignadd_pm_space", "csub_square_pm_space", "_cadd_const",
                  "carry_out_cdkm", "lt_cdkm", "all_ones_lean", "eq_top_lean",
                  "inc_borrowed", "_cadd_const_lean", "_zero_q"]),
    ("ec_window", ["_lookup", "_csub_square", "windowed_point_add_cfg"]),
    ("ec_cla", ["cla_add", "cla_sub", "cla_carry_out", "cla_lt", "cla_geq_const",
                "cla_modadd", "cla_modsub"]),
    ("ec_proj_q", ["jacobian_add_q", "jacobian_add_q_ctrl", "to_affine"]),
    ("ec_signedwin", ["cneg_y", "windowed_point_add_signed", "signed_windows"]),
    ("ec_luo", ["eea_step", "eea_init", "eea_finish", "r_compare", "r_subtract", "t_add",
                "t_compare", "loc_swap", "update_lengths", "mul_acc_reg", "moddbl_reg",
                "cmodadd_reg", ("Luo", "div")]),
    ("ec_batch", ["batch_mod_inv", "batch_mod_div", "point_add_ctrl_batch"]),
    ("ec_kaliski", ["kaliski_round"]),
    ("ec_depth", ["moddbl_cla", "modhalf_cla", "cla_csub", "cla_gt_ctrl", "sqr_int_cla",
                  "csub_square_cla", "fan_cswap", "fan_and"]),
    ("ec_cqadd", ["carry_xor", "cq_add_dirty", "cq_add", ("GidneyArith", "_cadd"),
                  ("GidneyArith", "_dbl")]),
    ("ec_luo3", ["mul_acc", "mul_unacc"]),
    ("ec_gcd", [("Dialog", "_round"), ("CondInv", "_round"), ("CondInv", "_first"),
                ("Jump2", "_walk_body"), ("Jump2Packed", "_step")]),
]


@contextlib.contextmanager
def tracing(targets=None):
    """Apply `boundary` to `targets` (module, [names or (class, method)]) and
    clear the cache; restore everything on exit.

    Drivers that build their own machine (`ec_shor.ecdlp_windowed`,
    `g25_arith.order_circuit_g25`, ...) call `Machine(...)`; inside the block
    that name means `HierMachine` in every loaded module, so they trace too.
    `HierMachine` subclasses `Machine`, so isinstance checks still hold."""
    import importlib
    import sys
    saved = []
    _CACHE.clear()
    _NOBOUND.clear()
    try:
        for mod in list(sys.modules.values()):
            if getattr(mod, "Machine", None) is Machine and mod is not sys.modules[__name__]:
                saved.append((mod, "Machine", Machine))
                mod.Machine = HierMachine
        for modname, names in (targets or DEFAULT_TARGETS):
            mod = importlib.import_module(modname)
            for nm in names:
                if isinstance(nm, tuple):
                    cls = getattr(mod, nm[0])
                    orig = cls.__dict__[nm[1]]
                    saved.append((cls, nm[1], orig))
                    setattr(cls, nm[1], boundary(orig, f"{modname}.{nm[0]}.{nm[1]}"))
                else:
                    orig = getattr(mod, nm)
                    saved.append((mod, nm, orig))
                    setattr(mod, nm, boundary(orig, f"{modname}.{nm}"))
        yield
    finally:
        for owner, nm, orig in reversed(saved):
            setattr(owner, nm, orig)


# =============================================================================
# Qualtran export (optional dependency)
# =============================================================================
def to_qualtran(m, name="top"):
    """The cached tree as Qualtran bloqs: one `HierBloq` per (node, inverted),
    whose call graph is its own leaf gates plus its children.  Then
    `qualtran.resource_counting.get_cost_value(bloq, QECGatesCost())` counts
    it, and Qualtran's call-graph tools can draw it.

    Leaf mapping: ecand -> And, ecand_dg -> And(uncompute), ccx/mcx -> Toffoli
    (mcx as one, as `ec_cost` does), cswap -> TwoBitCSwap, cx -> CNOT, x ->
    XGate, swap -> TwoBitSwap, cz -> CZ; the carried Toffoli cost of a
    measurement-based lookup / repair -> that many And, its measurements ->
    MeasureX.  Diagonal and bookkeeping gates are dropped."""
    import attrs
    from qualtran import Bloq, QAny, Register, Signature
    from qualtran.bloqs.basic_gates import (CNOT, CZ, MeasureX, Toffoli,
                                            TwoBitCSwap, TwoBitSwap, XGate)
    from qualtran.bloqs.mcmt import And

    ids = {}

    def nid(node):
        return ids.setdefault(id(node), len(ids))

    registry = {}

    def leaf_counts(local, inverted):
        t = _invert_tally(local) if inverted else local
        out = {}

        def add(b, k):
            if k:
                out[b] = out.get(b, 0) + k
        add(And(), t.get("ecand", 0))
        add(And(uncompute=True), t.get("ecand_dg", 0))
        add(Toffoli(), t.get("ccx", 0) + t.get("mcx", 0))
        add(TwoBitCSwap(), t.get("cswap", 0))
        add(CNOT(), t.get("cx", 0))
        add(XGate(), t.get("x", 0))
        add(TwoBitSwap(), t.get("swap", 0))
        add(CZ(), t.get("cz", 0))
        add(And(), sum(v for k, v in t.items() if k.startswith("cost:") and k.endswith("|toffoli")))
        add(MeasureX(), sum(v for k, v in t.items()
                                if k.startswith("cost:") and k.endswith("|measure")))
        return out

    @attrs.frozen
    class HierBloq(Bloq):
        label: str
        node_id: int
        inverted: bool
        nq: int

        @property
        def signature(self):
            return Signature([Register("q", QAny(max(self.nq, 1)))])

        def build_call_graph(self, ssa):
            local, kids = registry[(self.node_id, self.inverted)]
            out = leaf_counts(local, self.inverted)
            for (child, inv), k in kids.items():
                b = bloq_for(child, inv ^ self.inverted)
                out[b] = out.get(b, 0) + k
            return out

        def __str__(self):
            return self.label + ("^-1" if self.inverted else "")

    def bloq_for(node, inverted):
        i = nid(node)
        registry[(i, inverted)] = (node.local, node.children)
        return HierBloq(node.name, i, inverted, node.n_args + node.n_anc)

    qc = getattr(m, "qc", m)
    top = Node(name, ("top",), qc.num_qubits, 0, count_tally(m), *structure(m))
    return bloq_for(top, False)
