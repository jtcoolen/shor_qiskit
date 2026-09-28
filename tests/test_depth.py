"""depth.py: Toffoli depth, reaction depth and expected Toffoli count.

Checked against closed forms on circuits whose critical path is known by
construction, and against `resources.count` for the Toffoli total, so the two
counters can never disagree about what a Toffoli is.
"""
import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

import depth as D
import resources as R
from ec_adders import cdkm_add, gidney_add
from ec_sim import Machine
from qrom import lookup_ui
from semiclassical import semiclassical_iqft


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def main():
    for n in (2, 3, 5, 8, 13):
        m = Machine("and")
        x, y, a = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n - 1, "a")
        gidney_add(m.ctx, x, y, a)
        pr = D.profile(m)
        assert pr["toffoli_depth"] == n - 1, (n, pr)
        assert pr["reaction_depth"] == n - 1, (n, pr)
        assert D.reaction_depth(m, and_dg_reacts=True) == 2 * (n - 1), n
    ok("Gidney adder: Toffoli depth n-1 (the carry chain; AND-dagger is free)")

    for n in (2, 4, 7, 12):
        m = Machine("and")
        x, y, c = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "c")
        cdkm_add(m.ctx, x, y, c[0])
        pr = D.profile(m)
        assert pr["toffoli_depth"] == 2 * n, (n, pr)
        assert pr["toffoli"] == 2 * n, (n, pr)
    ok("CDKM adder: Toffoli depth 2n (MAJ chain up, UMA chain down)")

    for w in (1, 2, 3, 4, 5):
        L = 1 << w
        addr, out = QuantumRegister(w, "a"), QuantumRegister(4, "o")
        one, anc = QuantumRegister(1, "one"), QuantumRegister(w, "anc")
        qc = QuantumCircuit(one, addr, out, anc)
        qc.x(one[0])
        lookup_ui(qc, one[0], addr, out, list(range(3, 3 + L)), anc)
        pr = D.profile(qc)
        assert pr["toffoli"] == R.count(qc).toffoli == 2 * (L - 1), (w, pr)
        assert pr["toffoli_depth"] == 2 * (L - 1), (w, pr)
    ok("unary-iteration lookup: 2(L-1) Toffolis, all on one chain")

    for t in (1, 2, 4, 7):
        ctrl, out = QuantumRegister(1, "c"), ClassicalRegister(t, "o")
        tgt = QuantumRegister(1, "t")
        qc = QuantumCircuit(ctrl, tgt, out)
        semiclassical_iqft(qc, ctrl[0], out,
                           [(lambda k: (lambda q, c: q.cp(math.pi / 2 ** k, c, tgt[0])))(k)
                            for k in range(t)])
        pr = D.profile(qc)
        assert pr["toffoli"] == 0 and pr["reaction_depth"] == t, (t, pr)
    ok("semiclassical inverse QFT: reaction depth t (each outcome steers the next)")

    # expected count: a Toffoli inside a classically controlled block counts
    # p_fire times; the worst case (resources.count) counts it fully
    q, c = QuantumRegister(3, "q"), ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)
    qc.ccx(0, 1, 2)
    qc.measure(0, c[0])
    with qc.if_test((c[0], 1)):
        qc.ccx(0, 1, 2)
        qc.ccx(0, 1, 2)
    qc.measure(1, c[1])
    with qc.if_test((c[1], 1)):
        with qc.if_test((c[0], 1)):
            qc.ccx(0, 1, 2)
    pr = D.profile(qc)
    assert pr["toffoli"] == 4 == R.count(qc).toffoli, pr
    assert pr["expected_toffoli"] == 1 + 2 * 0.5 + 0.25, pr
    assert D.expected_toffoli(qc, p_fire=1.0) == 4
    ok("expected Toffoli: blocks weighted by p_fire, nested blocks multiply")

    # composite gates are walked into, and agree with resources on the total
    from shor_essentials import qft
    from windowed import windowed_c_ua  # noqa: F401  (imports cleanly)
    from rc_adder import rc_c_add_mod
    n, N = 4, 13
    qc = QuantumCircuit(QuantumRegister(2, "c"), QuantumRegister(n + 1, "y"),
                        QuantumRegister(2, "sf"), QuantumRegister(n + 2, "anc"))
    rc_c_add_mod(qc, [qc.qubits[0], qc.qubits[1]], 5, qc.qubits[2:2 + n],
                 qc.qubits[2 + n], qc.qubits[3 + n], N, qc.qubits[4 + n:])
    qc.append(qft(3), qc.qubits[2:5])
    assert D.profile(qc)["toffoli"] == R.count(qc).toffoli
    ok("Toffoli total agrees with resources.count through composite gates")


def test_depth():
    main()


if __name__ == "__main__":
    main()
    print("\ntest_depth: all passed")
