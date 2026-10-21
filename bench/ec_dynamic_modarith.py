"""Benchmark: Gidney 2507.23079 dynamic modular constant addition vs unitary.

Writes bench/ec_dynamic_modarith.json.  For each toy prime: Toffolis, clean
and dirty qubit needs, vent measurements and exact all-branch verification of
the dynamic `cmodadd_const_dyn`, next to `ec_modarith.cmodadd_const`.  Then the
toy affine ECDLP oracle (CLASSIQ p=7, m_bits=2) with the four constant
additions per point addition made dynamic: oracle basis/branch proof and an
Aer CPU end-to-end run (set DYN_BENCH_SHOTS, default 1024; 0 skips).

Run: PYTHONPATH=shor_qiskit:tests venv/bin/python bench/ec_dynamic_modarith.py
"""
import itertools
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shor_qiskit"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import ec_classical as C  # noqa: E402
import ec_cost  # noqa: E402
import ec_dynamic_modarith as D  # noqa: E402
import ec_modarith as MA  # noqa: E402
import ec_shor as S  # noqa: E402
import shor_stats as ST  # noqa: E402
from ec_sim import Machine, simulate  # noqa: E402


def _init(pairs):
    out = {}
    for reg, v in pairs:
        for i, q in enumerate(reg):
            out[q] = (v >> i) & 1
    return out


def _val(bits, qc, reg):
    idx = {q: i for i, q in enumerate(qc.qubits)}
    return sum(bits[idx[q]] << i for i, q in enumerate(reg))


def modadd_row(p, k):
    n = p.bit_length()
    mu = Machine("and", "u")
    yu, qu = mu.alloc(n, "y"), mu.alloc(1, "q")
    MA.cmodadd_const(mu, qu[0], yu, k, p)
    mt = Machine("toffoli", "t")
    yt, qt = mt.alloc(n, "y"), mt.alloc(1, "q")
    MA.cmodadd_const(mt, qt[0], yt, k, p)
    md = Machine("toffoli", "d")
    yd, gd, qd = md.alloc(n, "y"), md.alloc(n - 1, "g"), md.alloc(1, "q")
    D.cmodadd_const_dyn(md, qd[0], yd, k, p, gd)
    sig, inputs = set(), 0
    for yv, gv, qv in itertools.product(range(p), range(1 << (n - 1)), (0, 1)):
        bits, s0, s = D.branch_sim(md.qc, _init([(yd, yv), (gd, gv), (qd, qv)]),
                                   md.checks)
        bu = simulate(mu.qc, _init([(yu, yv), (qu, qv)]), mu.checks)
        assert _val(bits, md.qc, yd) == (yv + qv * k) % p == _val(bu, mu.qc, yu)
        assert _val(bits, md.qc, gd) == gv
        sig.add((s0, tuple(s)))
        inputs += 1
    assert len(sig) == 1
    vents = D.measure_count(md.qc)
    return {
        "p": p, "n": n, "k": k,
        "unitary": {"toffoli_paper": ec_cost.count(mu)["toffoli_paper"],
                    "t_and_mode": ec_cost.count(mu)["t"],
                    "toffoli_toffoli_mode": ec_cost.count(mt)["toffoli"],
                    "ancilla_qubits": mu.qc.num_qubits - n - 1,
                    "peak_live_ancilla": mu.peak_live},
        "dynamic": {"toffoli": D.toffoli_count(md.qc),
                    "clean_ancilla": md.qc.num_qubits - n - 1 - (n - 1),
                    "dirty_borrowed": n - 1,
                    "peak_live_clean": md.peak_live,
                    "vent_measurements": vents,
                    "t_if_7_per_toffoli": 7 * D.toffoli_count(md.qc),
                    "branches_per_input": 2 ** vents,
                    "verified_inputs": inputs,
                    "phase_signature_constant": True},
    }


def adder_counts():
    from qiskit import QuantumCircuit, QuantumRegister
    rows = []
    for n in (4, 8, 16, 32):
        t, g, c, k = (QuantumRegister(n), QuantumRegister(n - 2),
                      QuantumRegister(2), QuantumRegister(1))
        row = {"n": n, "formula_classical_d0_1": 3 * n - 7}
        for label, lits in (("classical_odd", D.select_literals(n, (1 << n) - 1)),
                            ("controlled_odd", None)):
            qc = QuantumCircuit(t, g, c, k)
            if lits is None:
                lits = D.select_literals(n, (1 << n) - 1, 0, k[0])
            D.append_literal_adder(qc, list(t), list(g), list(c), lits)
            row[label] = {"toffoli": D.toffoli_count(qc),
                          "vents": D.measure_count(qc)}
        rows.append(row)
    return rows


def oracle():
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    out = {"curve": "CLASSIQ p=7 a=5 b=4", "order": order, "m_bits": 2}
    for inversion in ("kaliski", "divsteps"):
        qc, info = D.ecdlp_circuit_dyn(curve, P, Q, order, m_bits=2,
                                       inversion=inversion)
        qa, ia = S.ecdlp_circuit(curve, P, Q, order, oracle="arith", m_bits=2,
                                 inversion=inversion)
        m = info["machine"]
        kr, lr, px, py = info["regs"]
        Ps, Qs = S._rungs(curve, P, Q, 2)
        sig = set()
        persistent = set(kr) | set(lr) | set(px) | set(py)
        for u in range(4):
            for v in range(4):
                bits, s0, s = D.branch_sim(m.qc, _init([(kr, u), (lr, v)]), m.checks)
                R = info["offset"]
                for i in range(2):
                    if (u >> i) & 1:
                        R = curve.add(R, Ps[i])
                    if (v >> i) & 1:
                        R = curve.add(R, Qs[i])
                assert (_val(bits, m.qc, px), _val(bits, m.qc, py)) == (R.x, R.y)
                assert all(b == 0 for i, b in enumerate(bits)
                           if m.qc.qubits[i] not in persistent)
                sig.add((s0, tuple(s)))
        assert len(sig) == 1
        out[inversion] = {
            "qubits_dynamic": qc.num_qubits, "qubits_unitary": qa.num_qubits,
            "toffoli_dynamic": D.toffoli_count(m.qc),
            "toffoli_unitary": ec_cost.count(ia["machine"])["toffoli_paper"],
            "vent_measurements": D.measure_count(m.qc),
            "basis_inputs_verified": 16, "all_branch_phase_constant": True,
            "scratch_clean_all_branches": True,
        }
    shots = int(os.environ.get("DYN_BENCH_SHOTS", "1024"))
    if shots:
        t = time.time()
        counts, cands, info = D.run_ecdlp_dyn(curve, P, Q, order, shots=shots,
                                              seed=5, m_bits=2, inversion="divsteps")
        k = next(k for k in range(order) if curve.mul(k, P) == Q)
        exact = ST.ecdlp_probs(order, k, 2)
        emp = ST.empirical(S.pair_counts(counts), (4, 4))
        tvd, null = ST.tvd(emp, exact), ST.tvd_null(exact, shots)
        out["aer_e2e"] = {
            "inversion": "divsteps", "qubits": info["qubits"], "device": info["aer"],
            "shots": shots, "seconds": round(time.time() - t, 1),
            "k_true": k, "top_candidate": cands[0][0] if cands else None,
            "tvd": round(tvd, 4), "tvd_null_9999": round(null, 4),
            "pass": bool(cands and cands[0][0] == k and tvd <= null),
            "note": "kaliski oracle is 67 qubits, above Aer's 63-qubit MPS "
                    "coupling-map limit; it is basis/branch-verified only",
        }
    return out


def main():
    rows = [modadd_row(p, k) for p in (3, 5, 7, 11, 13) for k in (1, p - 1)]
    res = {
        "source": "Gidney, arXiv:2507.23079v1 (2025), Figs. 2-4, 6, Eq. 8",
        "status": "component + selectable toy hybrid oracle; NOT a Pareto/E2E "
                  "resource claim: only 4 of 11 point-add steps are constant "
                  "additions, and the dynamic circuits are not invertible by "
                  "Machine.emit_inverse",
        "finding": "the dynamic modadd trades gates for space: 3 clean + n-1 "
                   "borrowed dirty qubits vs 3n+1 clean ancillas, but ~9n-9 "
                   "dirty-target Toffolis vs the AND-based unitary's ~5n "
                   "toffoli_paper (its ANDs are uncomputed by measurement, "
                   "so it is the cheaper-in-T baseline at these sizes)",
        "workspace": "modadd: 3 clean (2 vent carries + 1 flag) + n-1 dirty",
        "python": platform.python_version(),
        "literal_adder": adder_counts(),
        "modadd": rows,
        "oracle": oracle(),
    }
    path = os.path.join(HERE, "ec_dynamic_modarith.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["oracle"], indent=1))


if __name__ == "__main__":
    main()
