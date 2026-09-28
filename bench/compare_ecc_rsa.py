"""ECDLP-256 against RSA, whole algorithms, logical and physical.

Everything "built" here is read off circuits this package builds at size
(`ec_toffoli_256.json`, `ec_space_256.json`, `ec_hier_256.json`, `rsa_g25.json`);
the physical columns apply one model to all of them (`physical.estimate`,
Gidney 2025's surface-code assumptions: p = 1e-3, 1 us cycles, 10 us reaction,
six CCZ factories).  Published points are quoted with their sources.

    ./venv/bin/python bench/compare_ecc_rsa.py      # writes bench/compare_ecc_rsa.json
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "bench"))

import physical as PH

J = lambda name: json.loads((ROOT / "bench" / name).read_text())

# windowed ECDLP: 28 additions + one first-window lookup of 2^16 ([1128], IonQ)
ADDITIONS, LOOKUP = 28, 3 * 2 ** 16


def space_full():
    """The whole ECDLP-256 circuit on the qubit-lean cells, built."""
    import ec_classical as C
    import ec_gcd as G
    import ec_shor as S
    import ec_space as SP
    import ec_window as W
    import hier as H
    from ec_hier_256 import CURVE, GEN, CMP, MSBS, N, ORDER, WBITS
    P = CURVE.p
    pml = G.PMSpace(P, msbs=MSBS, lean=True)
    cfg = W.PointAddCfg(
        lookup="mbu", merge_xy=True, offsets=True, free_xy1=True, add=pml,
        square=lambda m, c, s, a, p: SP.csub_square_pm_space(m, c, s, a, p, msbs=MSBS,
                                                             sqr_space=200, lean=True),
        mul=G.Dialog(arith=pml, fused_cmp=True, cmp_msbs=CMP, c_pad=2.3, share=True,
                     compress="fig1", walk_space=240))
    rng = random.Random(3)
    Q = CURVE.mul(rng.randrange(1, ORDER), GEN)
    S0 = CURVE.mul(rng.randrange(1, ORDER), GEN)
    with H.tracing():
        m, info = S.ecdlp_windowed(CURVE, GEN, Q, ORDER, WBITS, m_bits=N, offset=S0, drop=3,
                                   masks=True, oracle="arith", cfg=cfg, first_lookup=True,
                                   seed=7)
        c = H.count(m)
        c["toffoli_depth"] = H.exact_depth(m)
    semi = c["qubits"] - info["bits_k"] - info["bits_l"] + WBITS
    return {"toffoli": c["toffoli_paper"], "expected": round(c["toffoli_expected"]),
            "toffoli_depth": c["toffoli_depth"], "qubits": semi}


def main():
    tg, rsa = J("ec_toffoli_256.json"), J("rsa_g25.json")
    rows = []

    def add(label, kind, family, qubits, toffoli, depth=None, shots=1.0, cold=0,
            source="built here"):
        prof = PH.LogicalProfile(label, cold=cold, hot=qubits - cold, toffoli=toffoli / shots,
                                 reaction_depth=(depth or 0) / shots, shots=shots)
        est = PH.estimate(prof)
        rows.append({"label": label, "kind": kind, "family": family, "logical_qubits": qubits,
                     "toffoli": toffoli, "toffoli_depth": depth, "shots": shots,
                     "physical_qubits": est["physical_total"], "days": est["days"],
                     "hours_per_shot": est["hours_per_shot"],
                     "runtime_model": est["runtime_model"], "source": source})

    f = tg["full_algorithm_one_circuit"]              # IonQ's count and qubits, one circuit
    add("ECDLP-256, IonQ's cells, signed windows", "built", "ECC", f["qubits_semiclassical"], f["expected"],
        f.get("toffoli_depth"))
    sp = space_full()
    add("ECDLP-256, fewest qubits", "built", "ECC", sp["qubits"], sp["expected"],
        sp["toffoli_depth"])
    for n in ("1024", "2048", "3072"):
        if n not in rsa:
            continue
        r = rsa[n]
        add(f"RSA-{n}, Gidney 2025 residue arithmetic", "built", "RSA", r["qubits"],
            r["per_factoring_expected"], r.get("depth_per_shot", 0) * r["shots"],
            shots=r["shots"], cold=r["params"]["m"])
    # published, whole algorithm, logical level
    add("RSA-2048, Gidney 2025", "published", "RSA", 1399, 6.5e9, shots=9.2, cold=1280,
        source="arXiv:2505.15917 Table 5")
    pub = [
        ("ECDLP-256, IonQ", "ECC", 1457, 39.0e6, "arXiv:2609.05625"),
        ("ECDLP-256, [1128] space-optimised", "ECC", 1208, ADDITIONS * (2 ** 21.19 + LOOKUP),
         "ePrint 2026/1128 Table 2"),
        ("ECDLP-256, [1128] gate-optimised", "ECC", 1462, ADDITIONS * (2 ** 20.83 + LOOKUP),
         "ePrint 2026/1128 Table 2"),
        ("RSA-2048, Gidney-Ekera 2019", "RSA", 6189, 2.7e9, "Quantum 5, 433 (2021)"),
    ]
    for label, fam, q, t, src in pub:
        add(label, "published", fam, q, t, source=src)
    g25 = PH.estimate(PH.g25_rsa2048())
    out = {"rows": rows, "g25_reproduced": {"physical": g25["physical_total"], "days": g25["days"]},
           "security": {"ECDLP-256": 128, "RSA-2048": 112, "RSA-3072": 128, "RSA-1024": 80}}
    for r in rows:
        print(f"  {r['label']:<45} {r['logical_qubits']:>5} logical {r['toffoli']:>10.3e} Toffolis "
              f"{(r['toffoli_depth'] or 0):>10.3e} depth -> {r['physical_qubits']:>9,.0f} physical, "
              f"{r['days']:7.2f} days ({r['runtime_model']})")
    (ROOT / "bench" / "compare_ecc_rsa.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
