"""Figures for Part VII: Pareto frontiers and the ECC / RSA comparison.

Reads only the bench JSON files (every point built by this package, or quoted
from a paper and marked so) and writes PDF + PNG into doc/figures/.

    ./venv/bin/python bench/make_plots.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "doc" / "figures"
J = lambda name: json.loads((ROOT / "bench" / name).read_text())

LOOKUPS = 3 * 2 ** 16
# per windowed point addition, secp256k1, w = 16: (qubits incl. window, Toffolis incl. 3 lookups)
PUBLISHED_PADD = [
    ("[1128] space", 1192 + 16, 2 ** 21.19 + LOOKUPS),
    ("[1128] gate", 1446 + 16, 2 ** 20.83 + LOOKUPS),
    ("Babbush space", 1175 + 16, 2 ** 21.36 + LOOKUPS),
    ("Babbush gate", 1425 + 16, 2 ** 21.00 + LOOKUPS),
    ("IonQ", 1457, 1.196e6 + LOOKUPS),
    ("ECDSA.Fail (windowed)", 1162, 1_684_161),
]


def pareto(points):
    """Points not dominated in (x, y) -- both minimised -- sorted by x."""
    pts = sorted(points)
    front, best = [], float("inf")
    for x, y, *rest in pts:
        if y < best:
            front.append((x, y, *rest))
            best = y
    return front


def built_padd():
    """(qubits, toffolis, depth, label, series) for every built point addition."""
    out = []
    h = J("ec_hier_256.json")["addition"]
    for k, c in h.items():
        out.append((c["qubits"], c["toffoli_paper"], c.get("toffoli_depth"), k, "configurations"))
    for k, c in J("ec_space_256.json").items():
        if k.startswith("_"):
            continue
        out.append((c["qubits"], c["toffoli"], c.get("toffoli_depth"), k, "qubit frontier"))
    for k, c in J("ec_toffoli_256.json")["rows"].items():
        out.append((c["qubits"], c["expected"], c.get("toffoli_depth"), k, "IonQ's cells"))
    for k, c in J("ec_depth_256.json").items():
        if isinstance(c, dict):
            out.append((c["qubits"], c["expected"], c["toffoli_depth"], k, "carry-lookahead"))
    return out


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, which="both", alpha=0.25)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


COLORS = {"configurations": "tab:blue", "qubit frontier": "tab:green",
          "IonQ's cells": "tab:red", "carry-lookahead": "tab:purple", "published": "black"}


def fig_padd_toffoli():
    pts = built_padd()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for series in ("configurations", "qubit frontier", "IonQ's cells", "carry-lookahead"):
        xs = [p[0] for p in pts if p[4] == series]
        ys = [p[1] for p in pts if p[4] == series]
        ax.scatter(xs, ys, s=18, color=COLORS[series], label=f"built: {series}", zorder=3)
    for lab, q, t in PUBLISHED_PADD:
        ax.scatter([q], [t], marker="x", color="black", s=30, zorder=4)
        ax.annotate(lab, (q, t), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    front = pareto([(p[0], p[1]) for p in pts])
    ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="tab:gray",
            lw=1, label="Pareto front (built)")
    ax.set_yscale("log")
    ax.set_ylim(1.0e6, 1.5 * max(p[1] for p in pts))
    for p in pts:
        if "Luo" in p[3]:
            ax.annotate("Luo's EEA", (p[0], p[1]), fontsize=6.5, xytext=(5, -3),
                        textcoords="offset points")
    ax.legend(fontsize=7, loc="upper right")
    style(ax, "One windowed point addition, secp256k1, w = 16",
          "logical qubits", "Toffolis (executed)")
    save(fig, "ecdlp_padd_qubits_toffoli")


def fig_padd_depth():
    pts = [p for p in built_padd() if p[2]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for series in ("configurations", "qubit frontier", "IonQ's cells", "carry-lookahead"):
        xs = [p[0] for p in pts if p[4] == series]
        ys = [p[2] for p in pts if p[4] == series]
        ax.scatter(xs, ys, s=18, color=COLORS[series], label=f"built: {series}", zorder=3)
    front = pareto([(p[0], p[2]) for p in pts])
    ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="tab:gray",
            lw=1, label="Pareto front")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="lower left")
    ax.axhline(3 * 2 ** 16, color="tab:gray", ls=":", lw=1)
    ax.annotate("3 lookups of 2^16 (floor)", (ax.get_xlim()[0], 3 * 2 ** 16), fontsize=6.5,
                xytext=(4, 3), textcoords="offset points")
    style(ax, "One windowed point addition: qubits against Toffoli depth",
          "logical qubits", "Toffoli depth (exact)")
    save(fig, "ecdlp_padd_qubits_depth")


SHORT = {
    "ECDLP-256, IonQ's cells, signed windows": "ours, IonQ's cells, signed",
    "ECDLP-256, fewest qubits": "ours, fewest qubits",
    "RSA-1024, Gidney 2025 residue arithmetic": "RSA-1024",
    "RSA-2048, Gidney 2025 residue arithmetic": "RSA-2048",
    "RSA-3072, Gidney 2025 residue arithmetic": "RSA-3072",
    "RSA-2048, Gidney 2025": "RSA-2048 [G25]",
    "ECDLP-256, IonQ": "IonQ",
    "ECDLP-256, [1128] space-optimised": "[1128] space",
    "ECDLP-256, [1128] gate-optimised": "[1128] gate",
    "RSA-2048, Gidney-Ekera 2019": "RSA-2048 [GE19]",
}


def fig_ecc_rsa():
    rows = J("compare_ecc_rsa.json")["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in rows:
        mk = "o" if r["kind"] == "built" else "x"
        col = "tab:red" if r["family"] == "ECC" else "tab:blue"
        lab = SHORT.get(r["label"], r["label"])
        dy = -9 if "[" in lab or lab == "IonQ" else 3
        axs[0].scatter([r["logical_qubits"]], [r["toffoli"]], marker=mk, color=col, s=30)
        axs[0].annotate(lab, (r["logical_qubits"], r["toffoli"]), fontsize=6.5,
                        xytext=(4, dy), textcoords="offset points")
        axs[1].scatter([r["physical_qubits"]], [r["days"] * 24], marker=mk, color=col, s=30)
        axs[1].annotate(lab, (r["physical_qubits"], r["days"] * 24), fontsize=6.5,
                        xytext=(4, dy), textcoords="offset points")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axs[0].scatter([], [], color="tab:red", label="ECDLP-256 (secp256k1)")
    axs[0].scatter([], [], color="tab:blue", label="RSA")
    axs[0].legend(fontsize=7, loc="center right")
    style(axs[0], "Whole algorithm, logical (o built here, x published)", "logical qubits",
          "Toffolis per solved instance")
    style(axs[1], "Same, in one surface-code model (p = 1e-3, 1 us cycle)", "physical qubits",
          "hours, expected")
    save(fig, "ecc_vs_rsa")


def fig_security():
    rsa = J("rsa_g25.json")
    tg = J("ec_toffoli_256.json")["full_algorithm_one_circuit"]
    sec = {1024: 80, 2048: 112, 3072: 128}              # NIST SP 800-57 Part 1, Table 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axs[0]
    xs = [sec[int(n)] for n in rsa if int(n) in sec]
    ys = [rsa[n]["per_factoring_expected"] for n in rsa if int(n) in sec]
    ax.plot(sorted(xs), [y for _, y in sorted(zip(xs, ys))], "o-", color="tab:blue",
            label="RSA, Gidney 2025 (built)")
    for n in rsa:
        if int(n) in sec:
            ax.annotate(f"RSA-{n}", (sec[int(n)], rsa[n]["per_factoring_expected"]),
                        fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.scatter([128], [tg["expected"]], color="tab:red", zorder=3,
               label="ECDLP-256, secp256k1 (built)")
    ax.annotate("ECC-256", (128, tg["expected"]), fontsize=7, xytext=(4, 4),
                textcoords="offset points")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="center left")
    style(ax, "Against classical security", "classical security, bits (SP 800-57)",
          "Toffolis per solved instance")
    ax = axs[1]
    ns = sorted(int(n) for n in rsa)
    ax.plot(ns, [rsa[str(n)]["per_factoring_expected"] for n in ns], "o-", color="tab:blue",
            label="RSA-n (built)")
    ax.scatter([256], [tg["expected"]], color="tab:red", zorder=3, label="ECDLP-256 (built)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="center right")
    style(ax, "Against key size", "modulus / field size n (bits)", "Toffolis per solved instance")
    save(fig, "security_vs_toffoli")


def main():
    fig_padd_toffoli()
    fig_padd_depth()
    fig_ecc_rsa()
    fig_security()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
