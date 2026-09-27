"""Sparse state-vector simulation with phases, measurement and feed-forward.

The package already has two simulators, and each is blind to something:

  ec_sim.simulate      one basis state, any width -- but it *skips* diagonal
                       gates and measurements, so a phase error is invisible;
  tests/_dynsim        exact measurement branching and phases -- but dense,
                       so nothing above ~20 qubits.

Measurement-based uncomputation needs both at once.  Uncomputing a register by
measuring it in the X basis is only correct if the phase the measurement kicks
back is repaired, and the circuits that do it are 50-500 qubits wide.  So this
keeps the state as a *sparse* map {basis state: amplitude}, stored as two numpy
arrays.  Permutation gates (X, CX, Toffoli, SWAP, the temporary AND) move keys;
diagonal gates (Z, S, T, P, CZ, CCZ ...) multiply amplitudes; a Hadamard or any
other one-qubit gate branches a key in two and merges duplicates; measurement
collapses, and `if_else` runs on the branches whose classical bits satisfy it.
Reversible arithmetic on a basis input keeps exactly one key alive, so a
300-qubit point addition simulates as fast as `ec_sim` does -- and now the
phase comes out too.

Measurements can be enumerated (every branch, weighted by its Born
probability, as `_dynsim` does), sampled, or *supplied*.  Supplied outcomes are
what make exact MBU checks cheap: fix the outcome sequence, run every basis
input, and the uncomputation is correct iff each input comes back to its
expected basis state with the same phase (`assert_coherent`).  A difference in
phase between two inputs is precisely the garbage that would destroy Shor's
interference.

Keys are uint64 up to 63 qubits and Python ints above.  Bit i of a key is
qc.qubits[i]; bit i of a branch's classical value is qc.clbits[i], matching
`_dynsim.outcome_distribution`.
"""

import cmath
import math
import random

import numpy as np
from qiskit.circuit import ClassicalRegister, Clbit, ControlledGate, QuantumRegister


class SparseSimError(AssertionError):
    pass


_SKIP = {"barrier", "delay", "id"}
_PHASE1 = {  # single-qubit diagonal gates: phase on |1>, as a function of params
    "z": lambda p: -1.0, "s": lambda p: 1j, "sdg": lambda p: -1j,
    "t": lambda p: cmath.exp(1j * math.pi / 4), "tdg": lambda p: cmath.exp(-1j * math.pi / 4),
    "p": lambda p: cmath.exp(1j * float(p[0])), "u1": lambda p: cmath.exp(1j * float(p[0])),
}


class _Bits:
    """Bit arithmetic on key arrays, uint64 or Python-int object arrays."""

    def __init__(self, nq):
        self.wide = nq > 63
        self.dtype = object if self.wide else np.uint64
        self.one = 1 if self.wide else np.uint64(1)

    def sh(self, q):
        return q if self.wide else np.uint64(q)

    def mask(self, q):
        return (1 << q) if self.wide else np.uint64(1 << q)

    def bit(self, keys, q):
        return ((keys >> self.sh(q)) & self.one).astype(bool)

    def keys(self, values):
        return np.array(values, dtype=self.dtype)


class Branch:
    """One measurement branch: a normalised sparse state plus classical bits."""

    def __init__(self, keys, amps, clbits, prob, sim):
        self.keys, self.amps, self.clbits, self.prob = keys, amps, clbits, prob
        self._sim = sim

    def __len__(self):
        return len(self.keys)

    def values(self, reg):
        """The value of `reg` in each key (an int array, one entry per key)."""
        B, idx = self._sim.B, self._sim.idx
        out = np.zeros(len(self.keys), dtype=object)
        for i, q in enumerate(reg):
            out = out + (B.bit(self.keys, idx[q]).astype(object) * (1 << i))
        return out

    def value(self, reg):
        """The value of `reg`, asserting the branch is a single basis state."""
        self.assert_basis()
        return int(self.values(reg)[0])

    def phase(self):
        """The amplitude of the single surviving basis state (unit modulus)."""
        self.assert_basis()
        return complex(self.amps[0])

    def assert_basis(self):
        if len(self.keys) != 1:
            raise SparseSimError(f"expected a basis state, got {len(self.keys)} keys")

    def amplitudes(self, *regs):
        """{(value of reg0, value of reg1, ...): amplitude}."""
        cols = [self.values(r) for r in regs]
        out = {}
        for j in range(len(self.keys)):
            k = tuple(int(c[j]) for c in cols)
            out[k] = out.get(k, 0) + complex(self.amps[j])
        return out

    def assert_clean(self, qubits):
        """Every key has these qubits at |0>: the register is unentangled 0."""
        B, idx = self._sim.B, self._sim.idx
        for q in qubits:
            if B.bit(self.keys, idx[q]).any():
                raise SparseSimError("register is not |0> in every branch key")

    def cbit(self, clbit):
        return (self.clbits >> self._sim.cidx[clbit]) & 1

    def creg(self, reg):
        return sum(self.cbit(c) << i for i, c in enumerate(reg))


class SparseResult:
    def __init__(self, branches, sim):
        self.branches, self._sim = branches, sim

    def distribution(self):
        """{classical value: probability}, as `_dynsim.outcome_distribution`."""
        d = {}
        for b in self.branches:
            d[b.clbits] = d.get(b.clbits, 0.0) + b.prob
        return d

    def only(self):
        if len(self.branches) != 1:
            raise SparseSimError(f"{len(self.branches)} branches, expected one")
        return self.branches[0]


class _Sim:
    """The engine.  One instance per top-level circuit run."""

    def __init__(self, qc, measure, seed, cutoff):
        self.qc = qc
        self.nq = qc.num_qubits
        self.B = _Bits(self.nq)
        self.idx = {q: i for i, q in enumerate(qc.qubits)}
        self.cidx = {c: i for i, c in enumerate(qc.clbits)}
        self.cutoff = cutoff
        self.rng = random.Random(seed)
        self.mode = measure
        self.n_meas = 0
        if callable(measure) or isinstance(measure, (list, tuple)):
            seq = measure
            self.pick = (seq if callable(seq)
                         else (lambda i, _c, s=list(seq): s[i]))
            self.mode = "given"

    # -- gates on one branch --------------------------------------------------
    def _fire(self, keys, ctrls, cstate):
        B = self.B
        f = np.ones(len(keys), dtype=bool)
        for i, c in enumerate(ctrls):
            b = B.bit(keys, c)
            f &= b if (cstate >> i) & 1 else ~b
        return f

    def _phase_where(self, br, sel, ph):
        if sel.any():
            br.amps = br.amps.copy()
            br.amps[sel] *= ph

    def _matrix(self, br, qs, M):
        """Apply a dense 2^k x 2^k matrix on qubits `qs`, merging duplicates."""
        B, k = self.B, len(qs)
        local = np.zeros(len(br.keys), dtype=np.int64)
        base = br.keys.copy()
        for i, q in enumerate(qs):
            b = B.bit(br.keys, q)
            local |= b.astype(np.int64) << i
            base = np.where(b, base ^ B.mask(q), base) if not B.wide else \
                np.array([kk ^ (1 << q) if bb else kk for kk, bb in zip(base, b)],
                         dtype=object)
        out_keys, out_amps = [], []
        for j in range(1 << k):
            col = M[j, local] * br.amps
            nz = np.abs(col) > self.cutoff
            if not nz.any():
                continue
            kk = base[nz]
            for i, q in enumerate(qs):
                if (j >> i) & 1:
                    kk = kk ^ B.mask(q)
            out_keys.append(kk)
            out_amps.append(col[nz])
        if not out_keys:
            br.keys, br.amps = self.B.keys([]), np.zeros(0, dtype=complex)
            return
        keys = np.concatenate(out_keys)
        amps = np.concatenate(out_amps)
        uk, inv = np.unique(keys, return_inverse=True)
        acc = np.zeros(len(uk), dtype=complex)
        np.add.at(acc, inv.reshape(-1), amps)
        keep = np.abs(acc) > self.cutoff
        br.keys, br.amps = uk[keep], acc[keep]
        if not self.B.wide:
            br.keys = br.keys.astype(np.uint64)

    def _apply(self, br, op, w):
        """Apply a unitary op to branch `br`; w = qubit indices."""
        B, name = self.B, op.name
        if name in _SKIP:
            return
        if name == "global_phase":
            br.amps = br.amps * cmath.exp(1j * float(op.params[0]))
            return
        if name == "x":
            br.keys = br.keys ^ B.mask(w[0])
            return
        if name == "swap":
            d = B.bit(br.keys, w[0]) ^ B.bit(br.keys, w[1])
            br.keys = np.where(d, br.keys ^ (B.mask(w[0]) | B.mask(w[1])), br.keys) \
                if not B.wide else self._objwhere(d, br.keys, B.mask(w[0]) | B.mask(w[1]))
            return
        if name == "ecand":
            if B.bit(br.keys, w[2]).any():
                raise SparseSimError("AND target was not |0>")
            f = B.bit(br.keys, w[0]) & B.bit(br.keys, w[1])
            br.keys = self._xor_where(br.keys, f, B.mask(w[2]))
            return
        if name == "ecand_dg":
            t = B.bit(br.keys, w[2])
            if (t != (B.bit(br.keys, w[0]) & B.bit(br.keys, w[1]))).any():
                raise SparseSimError("AND-dagger target did not hold a AND b")
            br.keys = self._xor_where(br.keys, t, B.mask(w[2]))
            return
        if name in _PHASE1:
            self._phase_where(br, B.bit(br.keys, w[0]), _PHASE1[name](op.params))
            return
        if name == "rz":
            th = float(op.params[0])
            b = B.bit(br.keys, w[0])
            br.amps = br.amps * np.where(b, cmath.exp(0.5j * th), cmath.exp(-0.5j * th))
            return

        base, nctrl, cstate = None, 0, None
        if isinstance(op, ControlledGate):
            base, nctrl, cstate = op.base_gate.name, op.num_ctrl_qubits, op.ctrl_state
        elif name in ("cx", "ccx", "mcx", "c3x", "c4x", "mcx_gray"):
            base, nctrl = "x", len(w) - 1
        elif name in ("cz", "ccz"):
            base, nctrl = "z", len(w) - 1
        elif name in ("cswap", "fredkin"):
            base, nctrl = "swap", 1
        elif name in ("cp", "cu1", "mcp", "mcphase"):
            base, nctrl = "p", len(w) - 1
        if base is not None:
            if cstate is None:
                cstate = (1 << nctrl) - 1
            cs, ts = w[:nctrl], w[nctrl:]
            f = self._fire(br.keys, cs, cstate)
            if base == "x":
                br.keys = self._xor_where(br.keys, f, B.mask(ts[0]))
                return
            if base == "swap":
                d = f & (B.bit(br.keys, ts[0]) ^ B.bit(br.keys, ts[1]))
                br.keys = self._xor_where(br.keys, d, B.mask(ts[0]) | B.mask(ts[1]))
                return
            if base in ("z", "p", "u1"):
                ph = -1.0 if base == "z" else cmath.exp(1j * float(op.params[0]))
                self._phase_where(br, f & B.bit(br.keys, ts[0]), ph)
                return
            # other controlled bases fall through to the definition

        d = getattr(op, "definition", None)
        if d is not None and name not in ("unitary", "h", "sx", "sxdg", "y",
                                          "rx", "ry", "u", "u2", "u3", "r"):
            if d.global_phase:
                br.amps = br.amps * cmath.exp(1j * float(d.global_phase))
            dq = {q: w[i] for i, q in enumerate(d.qubits)}
            for ci in d.data:
                self._apply(br, ci.operation, [dq[q] for q in ci.qubits])
            return
        if op.num_qubits > 10:
            raise SparseSimError(f"gate {name!r} on {op.num_qubits} qubits has no definition")
        self._matrix(br, w, np.asarray(op.to_matrix(), dtype=complex))

    def _xor_where(self, keys, sel, mask):
        if not sel.any():
            return keys
        if self.B.wide:
            return self._objwhere(sel, keys, mask)
        return np.where(sel, keys ^ mask, keys)

    @staticmethod
    def _objwhere(sel, keys, mask):
        return np.array([k ^ mask if s else k for k, s in zip(keys, sel)], dtype=object)

    # -- measurement ------------------------------------------------------------
    def _measure(self, branches, q, c, record=True, consume=True):
        """Z-measure qubit q into clbit c.  `consume=False` (a reset) never
        takes a supplied outcome: it follows the state, sampling if it must."""
        mode = self.mode if (consume or self.mode != "given") else "sample"
        out = []
        for br in branches:
            one = self.B.bit(br.keys, q)
            w = np.abs(br.amps) ** 2
            tot = float(w.sum())
            p1 = float(w[one].sum()) / tot
            opts = [(0, 1.0 - p1, ~one), (1, p1, one)]
            if mode == "enumerate":
                chosen = [o for o in opts if o[1] * br.prob > self.cutoff ** 2]
            elif mode == "sample":
                chosen = [opts[1] if self.rng.random() < p1 else opts[0]]
            else:
                b = int(self.pick(self.n_meas, c)) & 1
                if opts[b][1] < self.cutoff:
                    raise SparseSimError(f"supplied outcome {b} has probability 0")
                chosen = [opts[b]]
            for b, pb, sel in chosen:
                keys, amps = br.keys[sel], br.amps[sel] / math.sqrt(pb * tot)
                cl = br.clbits
                if record and c is not None:
                    cl = (cl & ~(1 << c)) | (b << c)
                nb = Branch(keys, amps, cl, br.prob * (pb if mode == "enumerate" else 1.0), self)
                nb.last = b
                out.append(nb)
        if consume:
            self.n_meas += 1
        return out

    # -- the walk -----------------------------------------------------------------
    def run(self, circ, qmap, cmap, branches, checks=None):
        cidx_local = {c: i for i, c in enumerate(circ.clbits)}
        qidx_local = {q: i for i, q in enumerate(circ.qubits)}
        by_pos = {}
        for pos, qs, val in (checks or ()):
            by_pos.setdefault(pos, []).append((qs, val))

        def check(pos):
            for qs, val in by_pos.get(pos, ()):
                for br in branches:
                    v = br.values(qs)
                    if len(v) and not all(int(x) == val for x in v):
                        raise SparseSimError(
                            f"ancilla check failed at instruction {pos}: expected {val}")

        for pos, ci in enumerate(circ.data):
            if checks:
                check(pos)
            op = ci.operation
            w = [qmap[qidx_local[q]] for q in ci.qubits]
            cs = [cmap[cidx_local[c]] for c in ci.clbits]
            name = op.name
            if name in _SKIP or name.startswith("save_"):
                continue
            if name == "measure":
                branches = self._measure(branches, w[0], cs[0])
            elif name == "reset":
                branches = self._measure(branches, w[0], None, record=False,
                                         consume=False)
                for br in branches:
                    if br.last:
                        br.keys = br.keys ^ self.B.mask(w[0])
            elif name == "if_else":
                cond = op.condition
                target, val = cond if isinstance(cond, tuple) else (None, None)
                if isinstance(target, Clbit):
                    bits = [cmap[cidx_local[target]]]
                elif isinstance(target, ClassicalRegister):
                    bits = [cmap[cidx_local[b]] for b in target]
                else:
                    raise NotImplementedError("only bit/register conditions")
                hit = lambda cl: sum(((cl >> b) & 1) << i for i, b in enumerate(bits)) == val
                yes = [b for b in branches if hit(b.clbits)]
                no = [b for b in branches if not hit(b.clbits)]
                tb = op.blocks[0]
                fb = op.blocks[1] if len(op.blocks) > 1 else None
                if yes:
                    yes = self.run(tb, w, cs, yes)
                if no and fb is not None:
                    no = self.run(fb, w, cs, no)
                branches = yes + no
            elif getattr(op, "blocks", None):
                raise NotImplementedError(f"control flow {name!r}")
            elif name in ("initialize", "state_preparation"):
                for br in branches:
                    self._prepare(br, w, op)
            else:
                for br in branches:
                    self._apply(br, op, w)
        if checks:
            check(len(circ.data))
        return branches

    def _prepare(self, br, w, op):
        for q in w:
            if self.B.bit(br.keys, q).any():
                raise SparseSimError("state preparation on a qubit that is not |0>")
        vec = np.asarray(op.params, dtype=complex)
        if vec.ndim != 1 or len(vec) != 1 << len(w):
            raise NotImplementedError("state preparation needs an amplitude list")
        M = np.zeros((len(vec), len(vec)), dtype=complex)
        M[:, 0] = vec / np.linalg.norm(vec)
        self._matrix(br, w, M)


def basis_key(qc, init=None):
    """The integer key of a basis state given as {qubit-or-index: bit} or
    {register(list of qubits): int}."""
    idx = {q: i for i, q in enumerate(qc.qubits)}
    key = 0
    for k, v in (init or {}).items():
        if isinstance(k, (list, tuple, QuantumRegister)):
            for i, q in enumerate(k):
                if (int(v) >> i) & 1:
                    key |= 1 << idx[q]
        elif int(v) & 1:
            key |= 1 << (idx[k] if k in idx else k)
    return key


def simulate_sparse(qc, init=None, state=None, checks=(), measure="enumerate",
                    seed=0, cutoff=1e-12):
    """Run `qc` and return a SparseResult.

    init    : a basis state, {qubit-or-index: bit} as for `ec_sim.simulate`,
              or {register: int}
    state   : alternatively {key: amplitude}, a superposition (normalised here)
    checks  : `Machine.checks` -- ancillas asserted |0> where they were freed
    measure : "enumerate" (every branch, Born-weighted), "sample" (one branch),
              or the outcomes themselves: a list of bits consumed in order, or
              a callable (measurement number, clbit index) -> bit
    """
    sim = _Sim(qc, measure, seed, cutoff)
    if state is None:
        state = {basis_key(qc, init): 1.0}
    keys = sim.B.keys(list(state.keys()))
    amps = np.array(list(state.values()), dtype=complex)
    amps = amps / np.linalg.norm(amps)
    br = Branch(keys, amps, 0, 1.0, sim)
    branches = sim.run(qc, list(range(sim.nq)), list(range(qc.num_clbits)), [br],
                       checks)
    return SparseResult(branches, sim)


def uniform_state(qc, fixed=None, **over):
    """Uniform superposition over the given register values.

    uniform_state(qc, {acc: 5}, a=(addr_reg, range(4))) puts `addr_reg` in
    (|0>+|1>+|2>+|3>)/2 with `acc` = 5.  Returns {key: amplitude}.
    """
    base = basis_key(qc, fixed)
    idx = {q: i for i, q in enumerate(qc.qubits)}
    keys = [base]
    for reg, vals in over.values():
        nk = []
        for k in keys:
            for v in vals:
                kk = k
                for i, q in enumerate(reg):
                    if (v >> i) & 1:
                        kk |= 1 << idx[q]
                nk.append(kk)
        keys = nk
    a = 1 / math.sqrt(len(keys))
    return {k: a for k in keys}


def assert_coherent(qc, inputs, outcomes, checks=(), expect=None, tol=1e-9):
    """Measurement-based uncomputation is phase-correct on these inputs.

    For each outcome sequence in `outcomes`, run every basis input in
    `inputs` (dicts as for `simulate_sparse(init=...)`) with those
    measurement results.  Each must end in a single basis state, and all
    inputs must end with the *same* phase: a relative phase between inputs is
    exactly what an unrepaired X-basis measurement leaves behind.

    `expect(input, branch)` may additionally check the output values.
    Returns {outcome index: common phase}.
    """
    phases = {}
    for oi, seq in enumerate(outcomes):
        ref = None
        for inp in inputs:
            br = simulate_sparse(qc, init=inp, checks=checks, measure=seq).only()
            br.assert_basis()
            if expect is not None:
                expect(inp, br)
            ph = br.phase()
            if ref is None:
                ref = ph
            elif abs(ph - ref) > tol:
                raise SparseSimError(
                    f"outcome sequence {oi}: phase {ph:.6f} differs from {ref:.6f} "
                    f"-- the uncomputation left a relative phase")
        phases[oi] = ref
    return phases


class Session:
    """Run a circuit piece by piece, measuring in between -- for hybrid
    circuits whose later gates depend on outcomes classically (the fix-up table
    of a measurement-based unlookup, say).

        s = Session(template_qc, init={...})
        s.run(segment)            # segment: a circuit on template_qc's qubits
        m = s.mx(register)        # X-basis measure + reset; returns the int

    Outcomes are sampled (seeded) unless `outcomes` supplies them.
    """

    def __init__(self, qc, init=None, state=None, seed=0, outcomes=None, cutoff=1e-12):
        self.qc = qc
        self.sim = _Sim(qc, outcomes if outcomes is not None else "sample", seed, cutoff)
        if state is None:
            state = {basis_key(qc, init): 1.0}
        keys = self.sim.B.keys(list(state.keys()))
        amps = np.array(list(state.values()), dtype=complex)
        self.branch = Branch(keys, amps / np.linalg.norm(amps), 0, 1.0, self.sim)

    def run(self, segment, checks=()):
        qmap = [self.sim.idx[q] for q in segment.qubits] if all(
            q in self.sim.idx for q in segment.qubits) else list(range(segment.num_qubits))
        cmap = list(range(segment.num_clbits))
        (self.branch,) = self.sim.run(segment, qmap, cmap, [self.branch], checks)
        return self

    def mx(self, qubits):
        """Measure `qubits` in the X basis, reset them to |0>, return the int."""
        m = 0
        for i, q in enumerate(qubits):
            j = self.sim.idx[q]
            self.sim._apply(self.branch, _H, [j])
            (self.branch,) = self.sim._measure([self.branch], j, None, record=False)
            if self.branch.last:
                self.branch.keys = self.branch.keys ^ self.sim.B.mask(j)
                m |= 1 << i
        return m

    def mz(self, qubits):
        m = 0
        for i, q in enumerate(qubits):
            j = self.sim.idx[q]
            (self.branch,) = self.sim._measure([self.branch], j, None, record=False)
            m |= self.branch.last << i
        return m


from qiskit.circuit.library import HGate as _HGate  # noqa: E402
_H = _HGate()
