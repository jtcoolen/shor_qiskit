"""Estimates for optimisations not built here, from numbers that are.

Every input is a measured count from the bench JSON files; every formula is
stated next to its result.  These are estimates, not builds:

  cold_record     keep the GCD record in yoked cold storage (it is touched two
                  bits per round) -- physical qubits, same model as
                  compare_ecc_rsa.py
  factories       more CCZ factories: runtime against physical qubits, down to
                  the reaction-limited floor (depth x 10 us)
  windows         the window size w: additions against lookup cost, unsigned
                  and signed, with the 3x load on SELECT-SWAP (2 words) as built
  qt              qubits x Toffolis (ECDSA.Fail's score) of the built points

    ./venv/bin/python bench/estimates.py      # writes bench/estimates.json
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))

import physical as PH

J = lambda name: json.loads((ROOT / "bench" / name).read_text())
N, W = 256, 16


def cold_record():
    import ec_gcd as G
    tg, sp = J("ec_toffoli_256.json"), J("ec_space_256.json")
    rec_ci = G.record_qubits(G.CondInv(c_pad=2.3, compress="fig1"), N)   # packed, both
    rec_fig1 = sp["_record_qubits"]
    rows = []
    full = tg["full_algorithm_select_swap"]
    few = {r["label"]: r for r in J("compare_ecc_rsa.json")["rows"]}[
        "ECDLP-256, dialog on the space cells"]
    for label, q, t, depth, rec in (
            ("IonQ's cells, signed windows", full["qubits_semiclassical"], full["expected"],
             full["toffoli_depth"], rec_ci),
            ("dialog on the space cells", few["logical_qubits"], few["toffoli"],
             few["toffoli_depth"], rec_fig1)):
        base = PH.estimate(PH.LogicalProfile(label, cold=0, hot=q, toffoli=t, reaction_depth=depth))
        cold = PH.estimate(PH.LogicalProfile(label, cold=rec, hot=q - rec, toffoli=t,
                                             reaction_depth=depth))
        rows.append({"config": label, "logical": q, "record": rec,
                     "physical_all_hot": base["physical_total"],
                     "physical_record_cold": cold["physical_total"]})
    return rows


def factories():
    full = J("ec_toffoli_256.json")["full_algorithm_select_swap"]
    out = []
    for k in (6, 12, 24, 48):
        code = PH.SurfaceCode(factories=k, compute_patches=(7, 18 + 4 * (k - 6) // 6 * 3))
        e = PH.estimate(PH.LogicalProfile("ECDLP-256", cold=0, hot=full["qubits_semiclassical"],
                                          toffoli=full["expected"],
                                          reaction_depth=full["toffoli_depth"]), code)
        out.append({"factories": k, "physical": e["physical_total"],
                    "minutes": e["hours_per_shot"] * 60, "limited_by": e["runtime_model"]})
    return out


def windows():
    tg = J("ec_toffoli_256.json")["rows"]
    one = "+ Fig. 1, x's qubits, lean careful cell, CNOT ends, shared walk, phase fold"
    arith = tg[one]["expected"] - 3 * ((1 << W) - 2)               # without the lookups
    out = []
    for w in range(12, 21):
        adds = 2 * math.ceil(N / w) - 1 - 3                          # first lookup, 3 dropped
        for signed in (False, True):
            a = w - 1 if signed else w
            lookups = 2 * ((1 << a) - 2) + ((1 << (a - 1)) - 2 + N)   # 3x load: SELECT-SWAP
            total = adds * (arith + lookups) + (1 << w)
            out.append({"w": w, "signed": signed, "additions": adds, "toffoli": total})
    return out


def qt():
    rows = []
    for name, c in J("ec_toffoli_256.json")["rows"].items():
        rows.append({"config": name, "qubits": c["qubits"], "toffoli": c["expected"],
                     "qt": c["qubits"] * c["expected"]})
    for name, c in J("ec_space_256.json").items():
        if not name.startswith("_"):
            rows.append({"config": name, "qubits": c["qubits"], "toffoli": c["toffoli"],
                         "qt": c["qubits"] * c["toffoli"]})
    rows.append({"config": "ECDSA.Fail best Q x T (published, classical addend)",
                 "qubits": 1151, "toffoli": 1299453, "qt": 1151 * 1299453})
    rows.append({"config": "ECDSA.Fail windowed variant (published)",
                 "qubits": 1162, "toffoli": 1684161, "qt": 1162 * 1684161})
    rows.append({"config": "IonQ (published)", "qubits": 1457, "toffoli": 1392608,
                 "qt": 1457 * 1392608})
    return sorted(rows, key=lambda r: r["qt"])


def main():
    out = {"cold_record": cold_record(), "factories": factories(),
           "windows": windows(), "qt": qt()}
    (ROOT / "bench" / "estimates.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"== {k}")
        for r in v[:12]:
            print("  ", r)


if __name__ == "__main__":
    main()
