"""Depth metrics that price time the way a fault-tolerant machine does.

`QuantumCircuit.depth()` counts every gate as one time step.  On an
error-corrected machine that is the wrong clock: Cliffords are nearly free and
can be tracked in software, while each Toffoli consumes a magic state and each
measurement whose outcome steers a later gate costs a classical round trip --
the *reaction time*.  Runtime is then set by how many of those must happen one
after another, not by the gate count.

Three numbers, all walking into composite gates and control flow exactly as
`resources.count` does, and classifying gates the same way:

  toffoli_depth    the longest chain of Toffoli-class operations: CCX, CCZ,
                   CSWAP, temporary AND (compute), and the ladders inside
                   multi-controlled gates.  A measurement-based AND-dagger
                   costs nothing, as in `resources` and `ec_cost`.
  reaction_depth   the longest chain of *reactions*: every Toffoli-class
                   operation is one (its injection needs a correction chosen by
                   a measurement), and every measurement whose outcome a later
                   classically-controlled block reads is one.  This is the
                   quantity the 2025-2026 resource estimates multiply by the
                   reaction time (e.g. 10 us) to get wall-clock runtime.
  expected_toffoli the Toffoli count with each classically-controlled block
                   weighted by the probability it fires (default 1/2, the
                   right value for a measurement-based fix-up).  This is the
                   "average executed" convention of Babbush et al. and
                   ECDSA.Fail; `resources.count` instead counts every block as
                   firing, which is the worst case.  Reported, never headlined.

Scheduling is as-soon-as-possible per qubit and per classical bit.  Both
branches of an if/else are scheduled from the same start and the later finish
is kept.
"""

from qiskit.circuit import ControlledGate

_SKIP = {"barrier", "delay", "global_phase"}
_FREE_1Q = {"x", "y", "z", "s", "sdg", "sx", "sxdg", "id", "h",
            "t", "tdg", "p", "u1", "rz"}


def _toffolis(op):
    """Toffoli-class cost of one primitive, and whether it is primitive.

    Returns (cost, primitive).  The ladder conventions are those of
    `resources._op_controlled_x`: X with k >= 3 controls is 2k - 3 Toffolis.
    """
    name = op.name
    if name == "ecand":
        return 1, True
    if name == "ecand_dg":
        return 0, True
    if isinstance(op, ControlledGate):
        k, base = op.num_ctrl_qubits, op.base_gate.name
        if base in ("x", "z"):
            return (0 if k == 1 else 1 if k == 2 else 2 * k - 3), True
        if base == "swap":
            return (1 if k == 1 else 2 * k - 1), True
        if base in ("p", "u1", "rz"):
            return (2 * (k - 1) if k >= 2 else 0), True
    if name in ("ccx", "ccz", "cswap", "fredkin", "toffoli"):
        return 1, True
    if name == "mcx":
        k = op.num_qubits - 1
        return (0 if k == 1 else 1 if k == 2 else 2 * k - 3), True
    if name in _FREE_1Q or name in ("cx", "cz", "swap", "cp", "cy", "ch"):
        return 0, True
    if name in ("measure", "reset", "if_else") or name in _SKIP:
        return 0, True
    if name.startswith("save_") or name in ("initialize", "state_preparation",
                                            "unitary", "set_statevector"):
        return 0, True
    return 0, op.definition is None


def _cond_clbits(op, clbit_index):
    cond = getattr(op, "condition", None)
    if cond is None:
        return []
    target = cond[0] if isinstance(cond, tuple) else None
    if target is None:                          # classical expression
        from qiskit.circuit.classical import expr
        return [clbit_index[v.var] for v in expr.iter_vars(cond)
                if getattr(v, "var", None) in clbit_index]
    if hasattr(target, "__len__"):              # a whole ClassicalRegister
        return [clbit_index[b] for b in target]
    return [clbit_index[target]]


class _Clock:
    def __init__(self, and_dg_reacts):
        self.and_dg_reacts = and_dg_reacts
        self.tq = {}            # qubit -> (toffoli time, reaction time)
        self.tc = {}            # clbit -> reaction time at which it is known

    def snapshot(self):
        return dict(self.tq), dict(self.tc)

    def restore(self, snap):
        self.tq, self.tc = dict(snap[0]), dict(snap[1])

    def merge(self, other):
        for q, (a, b) in other[0].items():
            x, y = self.tq.get(q, (0, 0))
            self.tq[q] = (max(a, x), max(b, y))
        for c, v in other[1].items():
            self.tc[c] = max(v, self.tc.get(c, 0))


def _walk(circ, qmap, cmap, clock, weight, tally, cond_ready=0):
    qidx = {q: i for i, q in enumerate(circ.qubits)}
    cidx = {c: i for i, c in enumerate(circ.clbits)}
    for ci in circ.data:
        op = ci.operation
        name = op.name
        if name in _SKIP or name.startswith("save_"):
            continue
        qs = [qmap[qidx[q]] for q in ci.qubits]
        cs = [cmap[cidx[c]] for c in ci.clbits]

        if name == "if_else":
            ready = max([clock.tc.get(cmap[i], 0)
                         for i in _cond_clbits(op, cidx)] + [cond_ready])
            start = clock.snapshot()
            ends = []
            for block in op.blocks:
                if block is None:
                    continue
                clock.restore(start)
                bq = {i: qs[i] for i in range(len(block.qubits))}
                bc = {i: cs[i] for i in range(len(block.clbits))}
                _walk(block, bq, bc, clock, weight * tally["p_fire"], tally, ready)
                ends.append(clock.snapshot())
            clock.restore(start)
            for e in ends:
                clock.merge(e)
            continue

        cost, primitive = _toffolis(op)
        if not primitive:
            _walk(op.definition, {i: q for i, q in enumerate(qs)},
                  {i: c for i, c in enumerate(cs)}, clock, weight, tally, cond_ready)
            continue

        tally["toffoli"] += cost
        tally["expected"] += cost * weight
        t0 = max([clock.tq.get(q, (0, 0))[0] for q in qs] + [0])
        r0 = max([clock.tq.get(q, (0, 0))[1] for q in qs] + [cond_ready])
        react = cost
        if name == "ecand_dg" and clock.and_dg_reacts:
            react = 1
        t1, r1 = t0 + cost, r0 + react
        for q in qs:
            clock.tq[q] = (t1, r1)
        if name == "measure":
            for c in cs:
                clock.tc[c] = r1 + 1          # the outcome must travel back


def profile(obj, and_dg_reacts=False, p_fire=0.5):
    """All three metrics at once: {toffoli, toffoli_depth, reaction_depth,
    expected_toffoli}.  Accepts a QuantumCircuit or anything with `.qc`."""
    qc = getattr(obj, "qc", obj)
    clock = _Clock(and_dg_reacts)
    tally = {"toffoli": 0, "expected": 0.0, "p_fire": p_fire}
    _walk(qc, dict(enumerate(qc.qubits)), dict(enumerate(qc.clbits)),
          clock, 1.0, tally)
    tq = clock.tq.values()
    return {
        "toffoli": tally["toffoli"],
        "toffoli_depth": max((a for a, _ in tq), default=0),
        "reaction_depth": max([b for _, b in tq] + list(clock.tc.values()) + [0]),
        "expected_toffoli": tally["expected"],
    }


def toffoli_depth(obj):
    return profile(obj)["toffoli_depth"]


def reaction_depth(obj, and_dg_reacts=False):
    return profile(obj, and_dg_reacts=and_dg_reacts)["reaction_depth"]


def expected_toffoli(obj, p_fire=0.5):
    return profile(obj, p_fire=p_fire)["expected_toffoli"]
