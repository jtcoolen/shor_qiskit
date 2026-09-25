"""Exact outcome distribution of a dynamic circuit, by enumerating branches.

Aer runs a circuit with mid-circuit measurement shot by shot, so it can only
*sample* the distribution.  This computes it: every measurement splits each
branch in two, weighted by its Born probability, and `if_test` blocks run on
the branches whose classical bits satisfy them.  That is the literal semantics
of measurement and feed-forward, written independently of Aer and of the
deferred-measurement argument the semiclassical QFT rests on -- which is what
makes it a fair referee between the two.

Exponential in the number of measurements, so for small checks.  Branches
below `cutoff` are dropped, which keeps sharply peaked circuits cheap.
"""
import numpy as np
from qiskit.circuit import Clbit
from qiskit.quantum_info import Operator, Statevector


def outcome_distribution(qc, initial=None, cutoff=1e-14):
    """{classical value: probability}, bit i of the value = qc.clbits[i]."""
    sv = initial if initial is not None else Statevector.from_int(0, 2**qc.num_qubits)
    branches = [(np.asarray(sv.data, dtype=complex), 0, 1.0)]
    branches = _run(qc, list(range(qc.num_qubits)), list(range(qc.num_clbits)),
                    branches, qc.num_qubits, cutoff)
    dist = {}
    for _, bits, p in branches:
        dist[bits] = dist.get(bits, 0.0) + p
    return dist


def _run(circ, qmap, cmap, branches, nq, cutoff):
    idx = np.arange(2**nq)
    for inst in circ.data:
        op = inst.operation
        qs = [qmap[circ.find_bit(q).index] for q in inst.qubits]
        cs = [cmap[circ.find_bit(c).index] for c in inst.clbits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            q, c = qs[0], cs[0]
            one = ((idx >> q) & 1).astype(bool)
            new = []
            for data, bits, p in branches:
                for b, sel in ((0, ~one), (1, one)):
                    pb = float(np.sum(np.abs(data[sel]) ** 2))
                    if p * pb > cutoff:
                        d = np.where(sel, data, 0) / np.sqrt(pb)
                        new.append((d, (bits & ~(1 << c)) | (b << c), p * pb))
            branches = new
        elif op.name == "if_else":
            bit, val = op.condition
            if not isinstance(bit, Clbit):
                raise NotImplementedError("only single-bit conditions")
            ci = cmap[circ.find_bit(bit).index]
            yes = [br for br in branches if ((br[1] >> ci) & 1) == val]
            no = [br for br in branches if ((br[1] >> ci) & 1) != val]
            true_body, false_body = op.blocks[0], (op.blocks[1] if len(op.blocks) > 1 else None)
            yes = _run(true_body, qs, cs, yes, nq, cutoff) if yes else []
            if false_body is not None and no:
                no = _run(false_body, qs, cs, no, nq, cutoff)
            branches = yes + no
        elif op.name in ("reset", "delay") or getattr(op, "blocks", None):
            raise NotImplementedError(op.name)
        else:
            U = Operator(op)                         # built once, applied to all
            branches = [(Statevector(d).evolve(U, qargs=qs).data, bits, p)
                        for d, bits, p in branches]
    return branches
