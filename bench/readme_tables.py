"""Print the README's n = 256 / RSA tables from the bench JSON files, so the
hand-written README cannot drift from what was built.

    ./venv/bin/python bench/readme_tables.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
J = lambda name: json.loads((ROOT / "bench" / name).read_text())


def fmt(x):
    return f"{int(round(x)):,}"


def main():
    h, q = J("ec_hier_256.json"), J("qualtran_compare.json")
    print("| one windowed point addition, secp256k1, w = 16 | Toffolis | Toffoli depth | qubits |")
    print("|---|---:|---:|---:|")
    lit = q["qualtran_litinski_ecadd"]
    print(f"| Qualtran's own `ECAdd` (Litinski 2023), for reference | {fmt(lit['toffoli'])} | | "
          f"{fmt(lit['qubits'])} |")
    for k, c in h["addition"].items():
        print(f"| {k} | {fmt(c['toffoli_paper'])} | {fmt(c['toffoli_depth'])} | {fmt(c['qubits'])} |")
    f = h["full_algorithm"]
    print(f"\nwhole ECDLP-256 ({f['config']}): {fmt(f['toffoli_paper'])} Toffolis, depth "
          f"{fmt(f['toffoli_depth'])}, {fmt(f['qubits_semiclassical'])} qubits; Qualtran "
          f"{fmt(f['qualtran']['clifford'])} Cliffords, {fmt(f['qualtran']['measurement'])} measurements\n")
    t = J("ec_toffoli_256.json")
    print("| one windowed point addition, secp256k1 | Toffolis | executed | depth | qubits |")
    print("|---|---:|---:|---:|---:|")
    for k, c in t["rows"].items():
        print(f"| {k} | {fmt(c['toffoli'])} | {fmt(c['expected'])} | {fmt(c['toffoli_depth'])} | "
              f"{fmt(c['qubits'])} |")
    fa = t["full_algorithm"]
    print(f"\nwhole algorithm on IonQ's cells: {fmt(fa['toffoli'])} ({fmt(fa['expected'])} executed), "
          f"depth {fmt(fa['toffoli_depth'])}, {fmt(fa['qubits_semiclassical'])} qubits\n")
    print("| one windowed point addition, secp256k1 | qubits | Toffolis | depth |")
    print("|---|---:|---:|---:|")
    for k, c in J("ec_space_256.json").items():
        if not k.startswith("_"):
            print(f"| {k} | {fmt(c['qubits'])} | {fmt(c['toffoli'])} | {fmt(c['toffoli_depth'])} |")
    print("\n| one windowed point addition, carry-lookahead | qubits | Toffolis | depth |")
    print("|---|---:|---:|---:|")
    for k, c in J("ec_depth_256.json").items():
        if isinstance(c, dict):
            print(f"| {k} | {fmt(c['qubits'])} | {fmt(c['toffoli'])} | {fmt(c['toffoli_depth'])} |")
    r = J("rsa_g25.json")
    print("\n| n | primes | one iteration | per shot (executed) | per factoring | Gidney 2025 | qubits (paper) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in sorted(r, key=int):
        x = r[n]
        print(f"| {n} | {fmt(x['primes'])} | {fmt(x['iteration']['toffoli_expected'])} | "
              f"{x['per_shot_expected']:.3g} | {x['per_factoring_expected']:.3g} | "
              f"{x['published']['toffoli_per_factoring']:.2g} | {x['qubits']} ({x['published']['qubits']}) |")
    c = J("compare_ecc_rsa.json")
    print("\n| whole algorithm | logical qubits | Toffolis | physical qubits | days |")
    print("|---|---:|---:|---:|---:|")
    for x in c["rows"]:
        tag = "" if x["kind"] == "built" else " *(published)*"
        print(f"| {x['label']}{tag} | {fmt(x['logical_qubits'])} | {x['toffoli']:.3g} | "
              f"{fmt(x['physical_qubits'])} | {x['days']:.2f} |")


if __name__ == "__main__":
    main()
