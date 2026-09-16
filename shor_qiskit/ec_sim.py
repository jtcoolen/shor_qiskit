"""Registers, ancilla pooling, and exact basis-state simulation.

Why this exists
---------------
The circuits in this package are sized for cryptographic primes: the reference
Kaliski inversion at n bits already needs ~7n qubits, and the projective point
addition of 2026/106 needs tens of thousands.  A statevector simulator dies at
~30.  But every one of these circuits is a *permutation* of basis states -- the
gate set is X / CX / CCX / SWAP / CSWAP plus the two AND gadgets, all of which
map one basis state to exactly one basis state.  So a single basis state can be
pushed through in O(gates) time and O(qubits) memory, exactly, at any width.

That is what `simulate` does.  It is not a substitute for a quantum simulator --
it says nothing about superposition inputs -- so the test suite *also* runs the
small circuits through Aer on genuine superpositions, and checks ancillas return
to |0> there.  On a reversible circuit built from these gates the two agree by
construction: linearity means correctness on every basis state is correctness,
full stop.  The fast path buys the width; Aer keeps it honest.
"""

from qiskit.circuit import ControlledGate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SwapGate, XGate


# --- registers ---------------------------------------------------------------
class Reg(list):
    """A list of qubits that concatenates with `+`, as in the papers' pseudocode.

    Paper 2026/1128 writes register concatenation as `x_reg + anc1 + anc2`
    (a value plus its overflow bits); keeping that spelling makes Algorithms
    5-11 transcribe line for line.
    """

    def __init__(self, qubits, name=""):
        super().__init__(qubits)
        self.name = name

    def __add__(self, other):
        nm = getattr(other, "name", "")
        return Reg(list(self) + list(other), f"{self.name}+{nm}")

    def __getitem__(self, k):
        r = super().__getitem__(k)
        return Reg(r, self.name) if isinstance(k, slice) else r

    def __repr__(self):
        return f"Reg({self.name}, n={len(self)})"

    # A Reg is a *handle* on some qubits, not a value: identity semantics, so
    # it can key a dict of register assignments.
    __hash__ = object.__hash__
    __eq__ = object.__eq__


class Machine:
    """A circuit under construction, with a pool of clean ancillas.

    `anc()` hands out qubits guaranteed to be |0>; `free()` returns them and
    records a check that the simulator enforces at exactly that point in the
    instruction stream.  Reuse is what keeps the qubit counts near the papers':
    the inversion's scratch becomes the multiplication's scratch, and so on.
    """

    def __init__(self, mode="and", name="ec"):
        from ec_gates import Ctx
        self.qc = QuantumCircuit(name=name)
        self.ctx = Ctx(self.qc, mode)
        self.mode = mode
        self._pool = []          # clean qubits available for reuse
        self._nanc = 0           # total ancillas ever created (peak width)
        self.checks = []         # (instruction index, qubits, expected value)
        self._live = 0
        self.peak_live = 0

    # -- allocation ---------------------------------------------------------
    def alloc(self, size, name):
        """A fresh, persistent register (an input or an output). Never pooled."""
        r = QuantumRegister(size, name)
        self.qc.add_register(r)
        return Reg(list(r), name)

    def anc(self, size, name="anc"):
        """`size` clean ancillas, drawn from the pool where possible."""
        out = []
        while len(out) < size and self._pool:
            out.append(self._pool.pop())
        need = size - len(out)
        if need:
            r = QuantumRegister(need, f"{name}_{self._nanc}")
            self.qc.add_register(r)
            self._nanc += need
            out.extend(list(r))
        self._live += size
        self.peak_live = max(self.peak_live, self._live)
        return Reg(out, name)

    def free(self, *regs):
        """Return ancillas to the pool, asserting they are |0> at this point."""
        for reg in regs:
            qs = list(reg)
            self.checks.append((len(self.qc.data), qs, 0))
            self._pool.extend(qs)
            self._live -= len(qs)

    # -- running a builder backwards ----------------------------------------
    def emit_inverse(self, fn, *args, **kw):
        """Append the *inverse* of whatever `fn` would have appended.

        Both papers lean on this constantly -- Inv-dagger, Mul-dagger, PA-dagger
        in [106] Figs. 7-10, and [1128]'s observation that the in-place
        multiplier run backwards divides.  Building forwards and reversing is
        the only way to be sure the two agree.

        Ancilla bookkeeping: a builder takes ancillas from |0> and returns them
        to |0>, so reversing preserves that.  Cleanliness checks recorded during
        the forward pass index into the forward ordering and are meaningless
        after reversal, so they are dropped and one check on the whole ancilla
        set is left in their place by the caller's `free`.
        """
        start, nchk = len(self.qc.data), len(self.checks)
        fn(*args, **kw)
        body = list(self.qc.data[start:])
        del self.qc.data[start:]
        del self.checks[nchk:]
        for ci in reversed(body):
            self.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)

    # -- recording a step so it can be undone later, out of order -----------
    def begin(self):
        """A mark in the instruction stream."""
        return len(self.qc.data)

    def since(self, mark):
        """The instructions appended since `mark`, as an undoable body."""
        return list(self.qc.data[mark:])

    def undo(self, body):
        """Append the inverse of a recorded body.

        Needed where a computation must be unwound in an order that is not the
        reverse of how it was built -- the projective point addition keeps X3
        and so cannot simply run backwards from the end.
        """
        for ci in reversed(body):
            self.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)

    def step(self, fn, *args, **kw):
        """Run a builder and return (result, undoable body)."""
        mark = self.begin()
        out = fn(*args, **kw)
        return out, self.since(mark)

    def emit_maybe_inverse(self, invert, fn, *args, **kw):
        run = self.emit_inverse if invert else (lambda f, *a, **k: f(*a, **k))
        run(fn, *args, **kw)

    @property
    def width(self):
        return self.qc.num_qubits


# --- exact basis-state simulation --------------------------------------------
_DIAGONAL = {
    "z", "s", "sdg", "t", "tdg", "p", "rz", "u1", "cz", "ccz", "cp", "crz",
    "id", "delay", "global_phase", "s_adj", "t_adj",
}
_SKIP = {"barrier", "measure"}


class SimError(AssertionError):
    pass


def simulate(qc, init=None, checks=(), strict=True):
    """Push one basis state through `qc` and return the final bit assignment.

    init   : {qubit-or-index: 0/1}, everything unlisted starts at 0
    checks : [(instruction index, qubits, expected int value)] -- enforced when
             the walk reaches that index (this is how `Machine.free` asserts
             that an ancilla really came back clean, *at the point it is freed*,
             not merely at the end)
    strict : raise on any gate that is not a basis-state permutation

    Diagonal gates (Z, S, T, CZ, ...) are skipped: they multiply a basis state
    by a phase and cannot change which state it is.  That is exactly why the
    AND gadget's *internal* T gates are invisible here while its H gates are
    not -- so AND is applied atomically by name instead.

    The safety condition, stated plainly: this is only sound for circuits whose
    phases carry no information.  A phase oracle would be silently ignored.
    Every arithmetic circuit in this package satisfies that -- the only phase
    and Hadamard gates outside the AND gadget are in `ec_shor`'s state
    preparation and Fourier transform, which are run on Aer instead.  If that
    ever stops being true, `tests/test_ec_quantum.py` is the thing that catches
    it: it evolves the same circuits on real superpositions and would see a
    phase this path cannot.
    """
    idx = {q: i for i, q in enumerate(qc.qubits)}
    bits = [0] * qc.num_qubits
    if init:
        for k, v in init.items():
            bits[idx[k] if k in idx else k] = int(v) & 1

    by_pos = {}
    for pos, qs, val in checks:
        by_pos.setdefault(pos, []).append((qs, val))

    def check(pos):
        for qs, val in by_pos.get(pos, ()):
            got = sum(bits[idx[q]] << i for i, q in enumerate(qs))
            if got != val:
                raise SimError(
                    f"ancilla check failed at instruction {pos}: "
                    f"expected {val}, got {got} on {len(qs)} qubit(s)")

    def apply(op, qargs):
        name = op.name
        w = [idx[q] for q in qargs]

        if name in _SKIP or name in _DIAGONAL:
            return
        if name == "x":
            bits[w[0]] ^= 1
            return
        if name == "swap":
            bits[w[0]], bits[w[1]] = bits[w[1]], bits[w[0]]
            return
        if name == "ecand":
            if strict and bits[w[2]] != 0:
                raise SimError("AND target was not |0>: temporary AND invalid")
            bits[w[2]] = bits[w[0]] & bits[w[1]]
            return
        if name == "ecand_dg":
            if strict and bits[w[2]] != (bits[w[0]] & bits[w[1]]):
                raise SimError("AND-dagger target did not hold a AND b")
            bits[w[2]] = 0
            return

        # any controlled-X / controlled-SWAP, whatever its arity or ctrl_state
        base, nctrl, cstate = None, 0, None
        if isinstance(op, ControlledGate):
            base, nctrl, cstate = op.base_gate, op.num_ctrl_qubits, op.ctrl_state
        elif name in ("cx", "ccx", "mcx", "c3x", "c4x", "mcx_gray"):
            base, nctrl, cstate = XGate(), len(w) - 1, (1 << (len(w) - 1)) - 1
        elif name in ("cswap", "fredkin"):
            base, nctrl, cstate = SwapGate(), 1, 1

        if base is not None and base.name in ("x", "swap"):
            cs = w[:nctrl]
            if cstate is None:
                cstate = (1 << nctrl) - 1
            fire = all(bits[c] == ((cstate >> i) & 1) for i, c in enumerate(cs))
            if fire:
                t = w[nctrl:]
                if base.name == "x":
                    bits[t[0]] ^= 1
                else:
                    bits[t[0]], bits[t[1]] = bits[t[1]], bits[t[0]]
            return

        # unknown: descend into the definition
        d = getattr(op, "definition", None)
        if d is None:
            if strict:
                raise SimError(f"gate {name!r} is not a basis-state permutation")
            return
        sub = {q: qargs[i] for i, q in enumerate(d.qubits)}
        for ci in d.data:
            apply(ci.operation, [sub[q] for q in ci.qubits])

    for pos, ci in enumerate(qc.data):
        check(pos)
        apply(ci.operation, list(ci.qubits))
    check(len(qc.data))
    return bits


def getv(bits, qc, reg):
    """Read a register out of a `simulate` result, little-endian."""
    idx = {q: i for i, q in enumerate(qc.qubits)}
    return sum(bits[idx[q]] << i for i, q in enumerate(reg))


def setv(init, reg, value):
    """Write `value` into `reg` in an init dict, little-endian."""
    for i, q in enumerate(reg):
        init[q] = (value >> i) & 1
    return init


def run(m, inputs):
    """Simulate a Machine. `inputs`: {Reg: int}. Returns a reader closure."""
    init = {}
    for reg, val in inputs.items():
        setv(init, reg, val)
    bits = simulate(m.qc, init, m.checks)
    return lambda reg: getv(bits, m.qc, reg)
