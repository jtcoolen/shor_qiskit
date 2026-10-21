"""Generate Part VIII of shor-complete.tex from bench/rsa_g25.json and
bench/compare_ecc_rsa.json.  `ec_check_tex.py` diffs it against the document."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from ec_make_part7 import num  # noqa: E402

RSA = json.loads((ROOT / "bench" / "rsa_g25.json").read_text())
CMP = json.loads((ROOT / "bench" / "compare_ecc_rsa.json").read_text())
TOY = json.loads((ROOT / "bench" / "toy_distributions.json").read_text())


def sci(x):
    e = len(str(int(x))) - 1
    return f"{x / 10 ** e:.2f}\\times10^{{{e}}}"


def rows_rsa():
    out = []
    for n in sorted(RSA, key=int):
        r = RSA[n]
        out.append(f"${n}$ & ${num(r['primes'])}$ & ${num(r['iteration']['toffoli_expected'])}$ & "
                   f"${sci(r['per_shot_expected'])}$ & ${sci(r['per_factoring_expected'])}$ & "
                   f"${sci(r['published']['toffoli_per_factoring'])}$ & "
                   f"${num(r['qubits'])}$ (${num(r['published']['qubits'])}$)\\\\")
    return "\n".join(out)


def rows_gap():
    out = []
    for r in CMP["rows"]:
        tag = "" if r["kind"] == "built" else " \\emph{(published)}"
        out.append(f"{r['label']}{tag} & ${num(r['logical_qubits'])}$ & ${sci(r['toffoli'])}$ & "
                   f"${num(round(r['physical_qubits']))}$ & ${r['days']:.2f}$\\\\")
    return "\n".join(out)


def facts():
    f = {}
    r2 = RSA["2048"]
    f["shots2048"] = f"{r2['shots']:.1f}"
    f["over2048"] = f"{100 * (r2['per_factoring_expected'] / r2['published']['toffoli_per_factoring'] - 1):.0f}"
    f["tally_q2048"] = num(r2["tally_qubits"])
    g = CMP["g25_reproduced"]
    f["g25_phys"] = num(round(g["physical"]))
    f["g25_days"] = f"{g['days']:.2f}"
    rows = {r["label"]: r for r in CMP["rows"]}
    ecc, ecc_min = rows["ECDLP-256, IonQ's cells, signed windows"], rows["ECDLP-256, fewest qubits"]
    rsa = rows["RSA-2048, Gidney 2025 residue arithmetic"]
    f["ecc_q"], f["ecc_q_min"], f["rsa_q"] = (num(ecc["logical_qubits"]),
                                              num(ecc_min["logical_qubits"]),
                                              num(rsa["logical_qubits"]))
    f["ecc_n_lo"] = f"{ecc_min['logical_qubits'] / 256:.1f}"
    f["ecc_n_hi"] = f"{ecc['logical_qubits'] / 256:.1f}"
    f["ecc_t"], f["rsa_t"] = f"${sci(ecc['toffoli'])}$"[1:-1], f"${sci(rsa['toffoli'])}$"[1:-1]
    f["ratio"] = f"{rsa['toffoli'] / ecc['toffoli']:.0f}"
    r3 = rows.get("RSA-3072, Gidney 2025 residue arithmetic")
    f["rsa3072_t"] = sci(r3["toffoli"]) if r3 else "---"
    f["ratio3072"] = f"{r3['toffoli'] / ecc['toffoli']:.0f}" if r3 else "---"
    f["ecc_days"] = f"{ecc['days']:.2f}"
    f["rsa_days"] = f"{rsa['days']:.1f}"
    f["days_ratio"] = f"{rsa['days'] / ecc['days']:.0f}"
    f["ecc_minutes"] = f"{ecc['days'] * 1440:.0f}"
    t = TOY["rsa"]
    m7 = next(x for x in t["masks"] if x["mask_bits"] == 7)
    f["toy_primes"], f["toy_lmod"], f["toy_f"] = t["primes"], t["L_mod_N"], t["f"]
    f["toy_dev"] = f"{100 * t['worst_deviation']:.2f}"
    f["toy_bound"] = f"{100 * t['deviation_bound']:.1f}"
    f["toy_tvd"], f["toy_peak"] = f"{m7['tvd']:.3f}", f"{m7['peak_mass']:.3f}"
    f["toy_epeak"] = f"{m7['exact_peak_mass']:.3f}"
    f["toy_window"], f["toy_period"] = num(m7["window"]), num(m7["period"])
    return f


def main():
    tpl = (ROOT / "bench" / "part8_template.tex").read_text()
    tpl = tpl.replace("%%RSA_ROWS%%", rows_rsa()).replace("%%GAP_ROWS%%", rows_gap())
    for k, v in facts().items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    assert "@@" not in tpl and "%%" not in tpl.replace("%%%", ""), "unsubstituted placeholder"
    sys.stdout.write(tpl)


if __name__ == "__main__":
    main()
