"""Exact peephole passes over a built circuit (ECDSA.Fail Sec 5.3.7, exact forms).

ECDSA.Fail's largest source of commits was dead-code elimination.  Its exact
forms are the ones that need no sampling:

  constant propagation   an ancilla starts |0>, and a constant is loaded with
                         X gates, so on *every* branch its value is known until
                         a gate with an unknown control writes it.  A Toffoli
                         or AND with a control known to be 0 does nothing; with
                         a control known to be 1 it is a CNOT (no T gates, no
                         magic state).  ECDSA.Fail's "CCX -> CX where a control
                         is implied" is the same rule.
  cancellation           a gate followed directly (on all its qubits) by its
                         inverse is the identity.

Both are exact by construction -- they only use values that hold in every
branch -- and the tests re-run the exhaustive suites on the optimised
circuits.  The fire-census stripping ECDSA.Fail also used (dropping gates that
never fired on a billion samples) is deliberately absent: it is not exact.

Ancilla checks are kept, re-indexed, and a cancellation never spans a check on
one of its qubits, so every "freed ancilla is |0>" assertion still holds.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CXGate, XGate

from ec_gates import AndDgGate, AndGate

_INVERSE_PAIRS = {("x", "x"), ("cx", "cx"), ("ccx", "ccx"), ("swap", "swap"),
                  ("cswap", "cswap"), ("ecand", "ecand_dg"), ("ecand_dg", "ecand")}


def _names(ci):
    return ci.operation.name


def constprop(m, inputs):
    """Rewrite m's circuit using values known on every branch.

    `inputs`: the registers whose values are NOT known (everything else starts
    at |0>).  Returns a new Machine."""
    unknown = {q for reg in inputs for q in reg}
    known = {q: 0 for q in m.qc.qubits if q not in unknown}
    out = []                                     # (operation, qubits) kept
    old_to_new = []

    def forget(qs):
        for q in qs:
            known.pop(q, None)

    for ci in m.qc.data:
        old_to_new.append(len(out))
        op, qs, name = ci.operation, list(ci.qubits), ci.operation.name
        k = [known.get(q) for q in qs]
        if name == "x":
            if k[0] is not None:
                known[qs[0]] ^= 1
            out.append((op, qs))
        elif name == "cx":
            c, t = qs
            if k[0] == 0:
                continue
            if k[0] == 1:
                out.append((XGate(), [t]))
                if k[1] is not None:
                    known[t] ^= 1
                continue
            forget([t])
            out.append((op, qs))
        elif name in ("ccx", "ecand", "ecand_dg"):
            a, b, t = qs
            if k[0] == 0 or k[1] == 0:
                if name != "ccx":
                    known[t] = 0                 # an AND's target is 0 here
                continue                         # the target is unchanged
            if k[0] == 1 and k[1] == 1:
                out.append((XGate(), [t]))
                if k[2] is not None:
                    known[t] ^= 1
                continue
            if k[0] == 1 or k[1] == 1:
                c = b if k[0] == 1 else a
                out.append((CXGate(), [c, t]))
                forget([t])
                continue
            forget([t])
            out.append((op, qs))
        elif name == "swap":
            a, b = qs
            ka, kb = known.pop(a, None), known.pop(b, None)
            if ka is not None:
                known[b] = ka
            if kb is not None:
                known[a] = kb
            out.append((op, qs))
        elif name == "cswap":
            c, a, b = qs
            if k[0] == 0 or (k[1] is not None and k[1] == k[2]):
                continue
            if k[0] == 1:
                ka, kb = known.pop(a, None), known.pop(b, None)
                if ka is not None:
                    known[b] = ka
                if kb is not None:
                    known[a] = kb
                from qiskit.circuit.library import SwapGate
                out.append((SwapGate(), [a, b]))
                continue
            forget([a, b])
            out.append((op, qs))
        elif name in ("cz", "z", "s", "sdg", "t", "tdg", "p", "barrier"):
            out.append((op, qs))                 # diagonal: values unchanged
        else:
            forget(qs)                           # anything else: be conservative
            out.append((op, qs))
    old_to_new.append(len(out))
    return _rebuild(m, out, old_to_new)


def cancel(m):
    """Remove adjacent gate/inverse pairs (adjacent on all their qubits, with
    no ancilla check on those qubits in between).  Returns a new Machine."""
    data = [(ci.operation, list(ci.qubits)) for ci in m.qc.data]
    checks_at = {}
    for pos, qs, _ in m.checks:
        checks_at.setdefault(pos, set()).update(qs)
    alive = [True] * len(data)
    stacks = {q: [] for q in m.qc.qubits}        # per qubit: indices of live ops
    last_check = {q: -1 for q in m.qc.qubits}
    for i, (op, qs) in enumerate(data):
        for q in checks_at.get(i, ()):
            last_check[q] = i
        prev = {stacks[q][-1] if stacks[q] else None for q in qs}
        if len(prev) == 1:
            j = prev.pop()
            if j is not None and alive[j]:
                opj, qsj = data[j]
                if (qsj == qs and (opj.name, op.name) in _INVERSE_PAIRS
                        and all(last_check[q] <= j for q in qs)):
                    alive[j] = alive[i] = False
                    for q in qs:
                        stacks[q].pop()
                    continue
        for q in qs:
            stacks[q].append(i)
    out, old_to_new = [], []
    for i, (op, qs) in enumerate(data):
        old_to_new.append(len(out))
        if alive[i]:
            out.append((op, qs))
    old_to_new.append(len(out))
    return _rebuild(m, out, old_to_new)


def peephole(m, inputs, rounds=3):
    """constprop then cancel, repeated until nothing changes."""
    for _ in range(rounds):
        before = len(m.qc.data)
        m = cancel(constprop(m, inputs))
        if len(m.qc.data) == before:
            break
    return m


def _rebuild(m, ops, old_to_new):
    from ec_sim import Machine
    m2 = Machine(m.mode, m.qc.name)
    qc = QuantumCircuit(*m.qc.qregs, *m.qc.cregs, name=m.qc.name)
    for op, qs in ops:
        qc.append(op, qs)
    m2.qc = qc
    m2.ctx.qc = qc
    m2.checks = [(old_to_new[pos], qs, v) for pos, qs, v in m.checks]
    m2._pool, m2._nanc = list(m._pool), m._nanc
    m2._live, m2.peak_live = m._live, m.peak_live
    return m2
