"""Reproduce the toy optimisation figures from saved end-to-end benchmarks.

Run: venv/bin/python bench/plot_optimisation_frontiers.py
The tasks are deliberately plotted separately; component-only schedules,
unsampled affine rows, and CFS's different factoring task are not Aer-certified
points on the windowed ECDLP frontier.
"""

import json
from pathlib import Path

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def _report(name):
    return json.loads((HERE / name).read_text())


def _ec_panel(ax, report, title, *, log=False):
    rows = report["rows"]
    frontier = set(report["frontier"])
    for row in rows:
        sampled = row.get("tvd") is not None
        on_frontier = row["label"] in frontier
        ax.scatter(
            row["qubits"], row["toffoli_paper"], s=95 if on_frontier else 43,
            marker="o" if sampled else "s",
            facecolors=("#147d65" if on_frontier else "#9ca7b1")
            if sampled else "none",
            edgecolors="#16523f" if on_frontier else "#5c6670",
            linewidths=1.4 if on_frontier else 0.8, zorder=3,
        )
    if log:
        ax.set_yscale("log")
    ax.set(xlabel="logical qubits", ylabel="Toffolis (paper convention)",
           title=title)
    ax.grid(alpha=0.22, zorder=0)
    return frontier


def ec_figure():
    window = _report("ec_pareto.json")
    affine = _report("ec_gu_affine.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9),
                             layout="constrained")
    _ec_panel(axes[0], window, "p=7 windowed ECDLP (4,000 Aer shots)")
    # Identical Jump-2 and codec resource triples overlay one another.
    labels = {
        "jump2+reference": ("1", 10),
        "pingpong+reference": ("2", 10),
        "pingpong+symmetric": ("3", 14),
        "jump2+symmetric": ("4", -18),
    }
    for row in window["rows"]:
        if row["label"] in labels:
            label, offset = labels[row["label"]]
            axes[0].annotate(label,
                             (row["qubits"], row["toffoli_paper"]),
                             xytext=(0, offset), textcoords="offset points",
                             ha="center", fontsize=8, fontweight="bold")
    axes[0].text(
        .03, .96, "1  Jump-2 (+ codec), generic\n"
                  "2  ping-pong, generic\n"
                  "3  ping-pong, symmetric\n"
                  "4  Jump-2 (+ codec), symmetric",
        transform=axes[0].transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": .9},
    )
    _ec_panel(axes[1], affine,
              "p=5 affine ECDLP (1,024 shots where sampled)", log=True)
    offsets = {
        "kaliski": (5, 12), "divsteps": (-32, -19),
        "divsteps-decoded": (5, 11),
        "divsteps-arithmetic": (-78, 11),
        "packed-euclid": (-78, -19),
    }
    for row in affine["rows"]:
        if row["label"] in offsets:
            axes[1].annotate(row["label"],
                             (row["qubits"], row["toffoli_paper"]),
                             xytext=offsets[row["label"]],
                             textcoords="offset points", fontsize=7)
    fig.suptitle("Equal-task resource frontiers (width, Toffoli, depth)",
                 fontsize=13)
    fig.text(0.5, -0.025,
             "Green = three-metric nondominated; grey = dominated. "
             "Hollow square = basis-checked only (not Aer sampled). "
             "Vertical axis is logarithmic on the affine panel.",
             ha="center", fontsize=8)
    return fig


def rns_figure():
    report = _report("rns_gidney_toy.json")
    # The two masks have identical resource counts; one representative suffices.
    rows = [row for row in report["rows"]
            if "mask_bits=1" not in row["label"]]
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for row, label in zip(rows, ("exact modular", "high-bit CRT toy",
                                 "Gidney A.1 toy (mask 0)")):
        optimal = label == "exact modular"
        ax.scatter(row["qubits"], row["toffoli"], s=100,
                   c="#147d65" if optimal else "#9ca7b1", zorder=3)
        ax.annotate(label, (row["qubits"], row["toffoli"]),
                    xytext=(7, -11 if label == "high-bit CRT toy" else 7),
                    textcoords="offset points", fontsize=9)
    ax.set(xlabel="logical qubits", ylabel="Toffolis (unsynthesised rotations excluded)",
           title="N=15, base 2: same-task order-finding resources")
    ax.set_xlim(11, 27)
    ax.grid(alpha=0.22, zorder=0)
    return fig


def dynamic_figure():
    """Show component workspace separately from full-oracle costs."""
    report = _report("ec_dynamic_modarith.json")
    component = next(row for row in report["modadd"]
                     if row["p"] == 7 and row["k"] == 1)
    oracle = report["oracle"]["divsteps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8),
                             layout="constrained")
    labels = ("unitary", "dynamic")
    pos = (0, 1)
    axes[0].bar(pos, (component["unitary"]["ancilla_qubits"],
                       component["dynamic"]["clean_ancilla"]),
                color=("#6992b8", "#147d65"), label="clean workspace")
    axes[0].bar(1, component["dynamic"]["dirty_borrowed"],
                bottom=component["dynamic"]["clean_ancilla"],
                facecolor="none", edgecolor="#147d65", hatch="///",
                label="borrowed dirty workspace")
    axes[0].set(xticks=pos, xticklabels=labels, ylabel="ancilla wires",
                title="p=7 controlled add +1 (component)")
    ax_cost = axes[0].twinx()
    costs = (component["unitary"]["toffoli_paper"],
             component["dynamic"]["toffoli"])
    ax_cost.plot(pos, costs, "ko--", markersize=6)
    ax_cost.set_ylabel("Toffolis", color="black")
    axes[1].bar(pos, (oracle["toffoli_unitary"],
                       oracle["toffoli_dynamic"]),
                color=("#6992b8", "#147d65"))
    axes[1].set(xticks=pos, xticklabels=labels, ylabel="Toffolis",
                title="p=7 divsteps affine ECDLP (same oracle)")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].set_ylim(0, max(oracle["toffoli_unitary"],
                            oracle["toffoli_dynamic"]) * 1.18)
    axes[1].text(.5, .97,
                 f"{oracle['qubits_unitary']} qubits each; "
                 f"dynamic: {oracle['vent_measurements']} X measurements",
                 ha="center", va="top", transform=axes[1].transAxes, fontsize=8)
    fig.suptitle("Dirty-ancilla modular composition: component space, "
                 "no oracle-level width saving")
    fig.text(.25, -.04,
             "Solid = clean workspace; hatched = borrowed dirty wires; "
             "dashed = component Toffolis (right axis).",
             ha="center", fontsize=8)
    return fig


def edwards_kim_rows(mode):
    """Equal-task ECDLP resources; keep gate modes on separate frontiers."""
    from ec_pareto import frontier

    rows = []
    for result in _report("ec_edwards_kim.json")["oracles"]:
        label = (f"Horner {result['style']}" if result["adder"] == "horner"
                 else f"QROM {result['style']} {result['layout']} "
                      f"w{result['w']} "
                      f"{'load' if result['first_lookup'] else 'add'}")
        rows.append({
            "label": label, **{
                key: result[mode][key]
                for key in ("qubits", "toffoli_paper", "depth")},
            "sampled": any(run["mode"] == mode for run in result["aer_runs"]),
        })
    return rows, set(frontier(rows))


def edwards_kim_figure():
    """Separate toy ECDLP Pareto panels; unsampled rows are never certified."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, mode in zip(axes, ("toffoli", "and")):
        rows, frontier = edwards_kim_rows(mode)
        for row in rows:
            best = row["label"] in frontier
            ax.scatter(
                row["qubits"], row["toffoli_paper"],
                s=115 if best else 40,
                marker="o" if row["sampled"] else "s",
                facecolors=("#147d65" if best else "#9ca7b1")
                if row["sampled"] else "none",
                edgecolors="#16523f" if best else "#5c6670",
                linewidths=1.4 if best else .8, zorder=3,
            )
            if best:
                ax.annotate(row["label"],
                            (row["qubits"], row["toffoli_paper"]),
                            xytext=(5, 8),
                            textcoords="offset points", fontsize=7)
        ax.set(xlabel="logical qubits",
               ylabel="Toffoli/AND (paper convention)",
               title=f"p=5, 3-bit ECDLP: {mode} gates")
        ax.set_yscale("log")
        ax.set_xlim(48, 145)
        ax.grid(alpha=.22, zorder=0)
    fig.suptitle("Kim Sec. 4.4 three-constant QROM and Horner baseline: "
                 "same-task resource frontiers")
    fig.text(.5, -.04,
             "Green = nondominated in (qubits, Toffoli, depth); "
             "filled circle = Aer sampled; hollow square = basis only. "
             "Paper-layout rows above 63 qubits are not Aer-certified.",
             ha="center", fontsize=8)
    return fig


if __name__ == "__main__":
    for stem, figure in (("ec_optimisation_frontiers", ec_figure()),
                         ("rns_order_frontier", rns_figure()),
                         ("ec_dynamic_composition", dynamic_figure()),
                         ("ec_edwards_kim_frontier", edwards_kim_figure())):
        output = HERE / f"{stem}.pdf"
        figure.savefig(output, bbox_inches="tight")
        figure.savefig(HERE / f"{stem}.png", dpi=160, bbox_inches="tight")
        print(output)
        plt.close(figure)
