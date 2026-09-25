"""Logical resource counts: qubits, Toffolis, T gates and small-angle rotations.

A fault-tolerant machine does not pay for gates in general; it pays for
*non-Clifford* gates.  Clifford gates (H, S, CNOT, Paulis) are cheap on an
error-correcting code, while the T gate is injected through magic-state
distillation and dominates the space-time cost.  Every other non-Clifford gate
is paid for in T gates:

  * a Toffoli is 7 T -- or 4 as Gidney's temporary AND, whose uncompute is a
    measurement and costs none;
  * a phase shift P(theta) is exact in Clifford+T only when theta is a
    multiple of pi/4 (Z, S, T).  Any other angle must be *approximated*, at
    about 3 log2(1/eps) T gates per rotation with Ross-Selinger synthesis, or
    O(log^3.97 (1/eps)) with the generic Solovay-Kitaev construction.

So `count` walks a circuit -- descending into composite gates such as `qft` --
and prices every primitive by the standard decompositions:

  gate                          Toffoli    T                       CNOT
  H, X, S, Z, Sdg ...           -          -                       -
  CX, CZ / SWAP                 -          -                       1 / 3
  T, Tdg                        -          1                       -
  P(theta), RZ(theta)           -          1 if theta is an odd multiple of
                                           pi/4; a *rotation* if not a
                                           multiple of pi/4 at all
  CP(theta)                     -          three P(+-theta/2)      2
  CCX, CSWAP                    1          7                       0 / 2
  MCX, k >= 3 controls          2k - 3     7 each                  (k-2 ancillas)
  MCP(theta), k >= 2 controls   2(k - 1)   7 each, + CP(theta)     (k-1 ancillas)
  AND / AND-dagger (ec_gates)   1 / 0      4 / 0                   -

The multi-controlled gates assume the textbook Toffoli ladder on clean
ancillas; the peak number assumed is reported as `ancillas`, not hidden in
`qubits`.  Half of each ladder is an uncompute, which is why `T_and` -- the
same circuit with the ladders built from temporary ANDs -- is cheaper than `T`.

Counts are *logical*: no routing, no error-correction overhead.  Classically
controlled blocks (`if_test`) are counted as if they fire, which is the worst
case, and tallied in `conditional`.
"""

import math
from dataclasses import dataclass, fields

from qiskit.circuit import ControlledGate

_CLIFFORD_1Q = {"x", "y", "z", "s", "sdg", "sx", "sxdg", "id"}
_PHASE = {"p", "u1", "rz"}
_SKIP = {"barrier", "delay", "global_phase"}


@dataclass
class Resources:
    qubits: int = 0
    ancillas: int = 0       # peak clean ancillas the multi-controlled ladders assume
    toffoli: int = 0        # CCX, CSWAP, and the ladders inside MCX / MCP
    ladder: int = 0         # compute/uncompute pairs inside those ladders
    and_: int = 0           # temporary AND, compute (ec_gates)
    and_dg: int = 0         # temporary AND, measurement-based uncompute
    cx: int = 0
    h: int = 0
    clifford1: int = 0      # single-qubit Cliffords other than H
    cphase: int = 0         # controlled phases, including the one inside each MCP
    phase: int = 0          # single-qubit phase gates as written (P, RZ, T, S, Z)
    t_gates: int = 0        # T gates from T/Tdg and from pi/4-angle phases
    rotations: int = 0      # phases that are not multiples of pi/4: need synthesis
    measure: int = 0
    conditional: int = 0    # classically controlled blocks
    other: int = 0          # opaque: state preparation, arbitrary unitaries

    @property
    def T(self):
        """T-count with every Toffoli at 7 T (rotations excluded)."""
        return self.t_gates + 7 * self.toffoli + 4 * self.and_

    @property
    def T_and(self):
        """T-count with the ladders built from temporary ANDs (4 T, then 0)."""
        return self.T - 10 * self.ladder

    def T_with_rotations(self, eps_total=1e-3):
        """T-count including the synthesised rotations, for a total error budget.

        Errors of the approximations add at most linearly, so each of the R
        rotations gets eps_total / R, costing `rotation_t(eps)` T gates."""
        if not self.rotations:
            return self.T
        return self.T + self.rotations * rotation_t(eps_total / self.rotations)

    def as_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["and"] = d.pop("and_")
        d["T"], d["T_and"] = self.T, self.T_and
        return d


def count(qc):
    """The logical resources of `qc` (a QuantumCircuit, or anything with .qc)."""
    qc = getattr(qc, "qc", qc)
    r = Resources(qubits=qc.num_qubits)
    _walk(qc, r)
    return r


def _walk(circ, r):
    for inst in circ.data:
        _op(inst.operation, r)


def _angle(theta, r):
    """Classify one z-rotation by what it costs in Clifford+T."""
    x = float(theta) / (math.pi / 4)
    k = round(x)
    if abs(x - k) > 1e-9:
        r.rotations += 1
    elif k % 2:
        r.t_gates += 1


def _cphase(theta, r):
    theta = float(theta)
    r.cphase += 1
    if abs(math.remainder(theta - math.pi, 2 * math.pi)) < 1e-9:
        r.cx += 1                                   # CP(pi) is CZ: Clifford
        return
    r.cx += 2                                       # P(t/2) . CX . P(-t/2) . CX . P(t/2)
    for a in (theta / 2, -theta / 2, theta / 2):
        _angle(a, r)


def _op(op, r):
    name = op.name
    if name in _SKIP or name.startswith("save_"):
        return
    if name == "measure":
        r.measure += 1
        return
    if name == "reset":
        r.measure += 1
        r.clifford1 += 1
        return
    if name == "if_else":
        r.conditional += 1
        for block in op.blocks:
            _walk(block, r)
        return
    if name == "ecand":
        r.and_ += 1
        return
    if name == "ecand_dg":
        r.and_dg += 1
        return

    if isinstance(op, ControlledGate):
        k, base = op.num_ctrl_qubits, op.base_gate.name
        open_ctrls = k - bin(op.ctrl_state).count("1")
        r.clifford1 += 2 * open_ctrls                   # X before and after
        if base in ("x", "z"):
            if k == 1:
                r.cx += 1                               # CX, or CZ = H.CX.H
            else:
                _op_controlled_x(k, r)                  # CCX / CCZ, or a ladder
            return
        if base in _PHASE:
            if k >= 2:                                  # AND the controls into one
                r.toffoli += 2 * (k - 1)
                r.ladder += k - 1
                r.ancillas = max(r.ancillas, k - 1)
            _cphase(op.params[0], r)
            return
        if base == "swap":                              # CX . C^k-X . CX
            r.cx += 2
            _op_controlled_x(k + 1, r)
            return

    if name == "h":
        r.h += 1
    elif name in _CLIFFORD_1Q:
        r.clifford1 += 1
        if name in ("z", "s", "sdg"):
            r.phase += 1
    elif name in ("t", "tdg"):
        r.t_gates += 1
        r.phase += 1
    elif name in _PHASE:
        r.phase += 1
        _angle(op.params[0], r)
    elif name == "swap":
        r.cx += 3
    elif name in ("initialize", "state_preparation", "unitary"):
        r.other += 1
    elif op.definition is not None:
        _walk(op.definition, r)                         # composite: qft, unary, ...
    else:
        r.other += 1


def _op_controlled_x(k, r):
    """An X with k >= 2 controls: one CCX, or k-2 ANDs + CCX + k-2 undone."""
    if k == 2:
        r.toffoli += 1
    else:
        r.toffoli += 2 * k - 3
        r.ladder += k - 2
        r.ancillas = max(r.ancillas, k - 2)


# =============================================================================
# Rotation synthesis: what one small-angle phase shift costs in T gates
# =============================================================================
_ROT_T = {}


def rotation_t(eps, angle=0.7853981633974483 / 3.1):
    """T gates to approximate one generic z-rotation to operator-norm error eps.

    Measured with Qiskit's Ross-Selinger synthesis (`gridsynth_rz`) on a generic
    angle, and cached; falls back to the asymptotic 3 log2(1/eps) + 4 when the
    synthesis is unavailable.
    """
    key = float(f"{eps:.3g}")
    if key not in _ROT_T:
        try:
            from qiskit.synthesis import gridsynth_rz
            ops = gridsynth_rz(angle, epsilon=key).count_ops()
            _ROT_T[key] = ops.get("t", 0) + ops.get("tdg", 0)
        except ImportError:                                  # older Qiskit
            _ROT_T[key] = math.ceil(3 * math.log2(1 / key)) + 4
    return _ROT_T[key]
