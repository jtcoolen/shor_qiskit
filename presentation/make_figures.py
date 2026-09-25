"""Figures and numbers for shor_circuits.tex, computed from the code in shor_qiskit/.

    ./venv/bin/python presentation/make_figures.py      # ~2 min; writes figures/ and numbers.json

Same simulator seeds as notebooks/shor_walkthrough.ipynb, so every number agrees with it.
"""
import json
import math
import pathlib
import sys
import warnings

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "shor_qiskit"), str(ROOT / "tests")]
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ControlledGate, Gate
from qiskit_aer import AerSimulator

import ec_classical as C
import ec_shor as S
import resources as R
import shor_stats as ST
from shor_essentials import add_const, c_add_const, c_add_mod, c_ua, order_circuit, qft

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
SIM = AerSimulator(seed_simulator=2026)
NUM = {}

# --- the deck's palette (sampled from shor_pres_v4_1.pdf) ----------------------
BLUE, VERM, PURPLE, INDIGO = "#1a5fa0", "#b3401a", "#63389b", "#52439a"
TEXT, SUBTLE, GREY, RULE = "#1a1c20", "#3a4550", "#757d84", "#c7c4be"
plt.rcParams.update({
    "font.family": "Palatino", "font.size": 9,
    "mathtext.fontset": "custom", "mathtext.rm": "Palatino",
    "mathtext.it": "Palatino:italic", "mathtext.bf": "Palatino:bold",
    "axes.edgecolor": SUBTLE, "axes.labelcolor": TEXT, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": SUBTLE, "ytick.color": SUBTLE, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CIRCUIT_STYLE = {
    "name": "bw", "fontsize": 16, "subfontsize": 11,
    "textcolor": TEXT, "gatetextcolor": "#ffffff", "gatefacecolor": VERM, "linecolor": SUBTLE,
    "creglinecolor": "#9aa0a6", "backgroundcolor": "#ffffff",
    "displaycolor": {
        "h": [BLUE, "#ffffff"], "x": [SUBTLE, "#ffffff"], "measure": [GREY, "#ffffff"],
        "p": [PURPLE, "#ffffff"], "cp": [PURPLE, "#ffffff"],
        "qft": [PURPLE, "#ffffff"], "qft_dg": [PURPLE, "#ffffff"],
        "cbox": [VERM, "#ffffff"], "U": [VERM, "#ffffff"], "box": [VERM, "#ffffff"],
        "MAJ": [VERM, "#ffffff"], "UMA": [VERM, "#ffffff"],
    },
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("  wrote", name)


def draw(qc, name, scale=0.7, fold=-1):
    save(qc.draw("mpl", style=CIRCUIT_STYLE, fold=fold, scale=scale), name)


def sample(qc, shots):
    return SIM.run(transpile(qc, SIM, optimization_level=1), shots=shots).result().get_counts()


def exact_probs(qc, qubits):
    q = qc.remove_final_measurements(inplace=False)
    q.save_probabilities(list(qubits))
    return np.asarray(SIM.run(transpile(q, SIM, optimization_level=0)).result().data(0)["probabilities"])


def box(label, nq, build):
    sub = QuantumCircuit(nq)
    build(sub, list(sub.qubits))
    return sub.to_gate(label=label)


def cbox(label, n_ctrl, n_tgt, build):
    sub = QuantumCircuit(n_ctrl + n_tgt)
    build(sub, list(sub.qubits))
    return ControlledGate("cbox", n_ctrl + n_tgt, [], label=label, num_ctrl_qubits=n_ctrl,
                          definition=sub, base_gate=Gate("U", n_tgt, [], label=label))


# =============================================================================
# Circuits
# =============================================================================
def circuits():
    print("circuits")
    q = QuantumCircuit(3)
    q.append(qft(3), range(3))
    draw(q.decompose(), "qft3", 0.8)

    q = QuantumCircuit(QuantumRegister(3, "y"))
    add_const(q, 3, list(q.qubits))
    draw(q, "draper3", 0.8)

    from rc_adder import _maj, _uma
    cr = QuantumRegister(1, "c")
    yr = [QuantumRegister(1, f"y{i}") for i in range(3)]
    xr = [QuantumRegister(1, f"x{i}") for i in range(3)]
    q = QuantumCircuit(cr, yr[0], xr[0], yr[1], xr[1], yr[2], xr[2])
    maj = lambda: box("MAJ", 3, lambda qq, b: _maj(qq, b[0], b[1], b[2]))
    uma = lambda: box("UMA", 3, lambda qq, b: _uma(qq, b[0], b[1], b[2]))
    q.append(maj(), [cr[0], yr[0][0], xr[0][0]])          # carry chain up ...
    q.append(maj(), [xr[0][0], yr[1][0], xr[1][0]])
    q.append(maj(), [xr[1][0], yr[2][0], xr[2][0]])
    q.append(uma(), [xr[1][0], yr[2][0], xr[2][0]])       # ... and back down, writing the sum
    q.append(uma(), [xr[0][0], yr[1][0], xr[1][0]])
    q.append(uma(), [cr[0], yr[0][0], xr[0][0]])
    draw(q, "rc_adder", 0.75)

    from qrom import and_compute, and_uncompute
    a3, cb = QuantumRegister(3, "q"), ClassicalRegister(1, "m")
    q = QuantumCircuit(a3, cb)
    and_compute(q, a3[0], a3[1], a3[2])
    q.barrier()
    and_uncompute(q, a3[0], a3[1], a3[2], cb[0])
    draw(q, "and_gadget", 0.75)

    # the semiclassical loop on its own: rungs are phase gates P(2 pi y 2^k / 2^t), y = 5, t = 3
    from semiclassical import semiclassical_iqft
    t, yv = 3, 5
    ctr, out = QuantumRegister(1, "ctr"), ClassicalRegister(t, "m")
    q = QuantumCircuit(ctr, out)
    semiclassical_iqft(q, ctr[0], out,
                       [lambda qc, c, k=k: qc.p(2 * np.pi * yv * 2**k / 2**t, c) for k in range(t)])
    draw(q, "semiclassical_phase", 0.8, fold=13)
    NUM["semiclassical_phase_counts"] = sample(q, 256)


# =============================================================================
# QFT-dagger in action: a phase ramp becomes an integer
# =============================================================================
def ramp_circuit(t, phi):
    """What the rungs leave on the counting register: qubit q turned by 2^q * phi."""
    qc = QuantumCircuit(t, t)
    qc.h(range(t))
    for q in range(t):
        qc.p(2 * np.pi * phi * 2**q, q)
    return qc


def qft_readout():
    print("qft readout")
    from fractions import Fraction
    t, shots = 3, 1000
    res = {}
    for name, phi in (("qft_readout_exact", Fraction(1, 4)), ("qft_readout_inexact", Fraction(1, 6))):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 1.6), gridspec_kw={"width_ratios": [1.5, 1]})
        xs = np.arange(2**t)
        th = 2 * np.pi * xs * float(phi)
        for x in xs:
            a1.add_patch(plt.Circle((x, 0), 0.44, fill=False, color=RULE, lw=0.9))
            a1.text(x, -0.62, str(x), ha="center", va="top", fontsize=8, color=SUBTLE)
        a1.quiver(xs, np.zeros_like(xs), 0.38 * np.cos(th), 0.38 * np.sin(th), angles="xy",
                  scale_units="xy", scale=1, color=PURPLE, width=0.011, headwidth=3.6,
                  headlength=3.6, headaxislength=3.2)
        a1.set_xlim(-0.6, 2**t - 0.4)
        a1.set_ylim(-0.95, 0.55)
        a1.set_aspect("equal")
        a1.axis("off")
        a1.set_title(rf"the ramp: amplitude of $|x\rangle$ turns ${360 * phi.numerator // phi.denominator}^\circ$ per step ($s/r={phi.numerator}/{phi.denominator}$)",
                     fontsize=8.5)
        qc = ramp_circuit(t, float(phi))
        qc.append(qft(t).inverse(), range(t))
        qc.measure(range(t), range(t))
        emp = ST.empirical(ST.order_counts(sample(qc, shots)), 2**t)
        exact = np.array([abs(sum(np.exp(2j * np.pi * x * (float(phi) - y / 2**t)) for x in range(2**t)) / 2**t) ** 2
                          for y in range(2**t)])
        a2.bar(range(2**t), emp, width=0.7, color=BLUE, alpha=0.8, label=f"Aer, {shots} shots")
        a2.plot(range(2**t), exact, "o", ms=3.5, color=PURPLE, label="exact")
        a2.axvline(2**t * float(phi), color=GREY, ls=":", lw=0.9)
        a2.set_xticks(range(2**t))
        a2.set_ylim(0, 1.08)
        a2.tick_params(labelsize=8)
        a2.set_xlabel("measured y", fontsize=8.5)
        a2.set_title(rf"after $\mathrm{{QFT}}^\dagger$ ($2^3 s/r={2**t * float(phi):.3g}$)", fontsize=8.5)
        a2.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        res[f"{phi.numerator}/{phi.denominator}"] = {"exact": exact.tolist(), "measured": emp.tolist()}
        save(fig, name)
    NUM["qft_readout"] = res


# =============================================================================
# Uncomputation: the same order-finding circuit, clean or with garbage left behind
# =============================================================================
def order_with_garbage(A, N, t, garbage):
    n = math.ceil(math.log2(N))
    ctr, tgt, anc, sf = QuantumRegister(t, "ctr"), QuantumRegister(n, "tgt"), QuantumRegister(n, "anc"), QuantumRegister(2, "sf")
    g = QuantumRegister(max(t, n), "g")
    out = ClassicalRegister(t, "out")
    qc = QuantumCircuit(ctr, tgt, anc, sf, g, out)
    qc.h(ctr)
    qc.x(tgt[0])
    for i in range(t):
        c_ua(qc, ctr[i], pow(A, 2**i, N), list(tgt), list(anc), sf[0], sf[1], N)
    if garbage == "f(x)":
        for i in range(n):
            qc.cx(tgt[i], g[i])               # a copy of f(x) left behind
    if garbage == "x":
        for i in range(t):
            qc.cx(ctr[i], g[i])               # a copy of x left behind
    qc.append(qft(t).inverse(), ctr)
    qc.measure(ctr, out)
    return qc


def uncomputation():
    print("uncomputation")
    A, N, t, shots = 7, 15, 4, 2048
    fig, axes = plt.subplots(1, 3, figsize=(5.9, 1.55), sharey=True)
    res = {}
    for ax, (garbage, title) in zip(axes, ((None, "clean: no garbage"),
                                           ("f(x)", "garbage = copy of f(x)"),
                                           ("x", "garbage = copy of x"))):
        qc = order_with_garbage(A, N, t, garbage)
        exact = exact_probs(qc, range(t))
        emp = ST.empirical(ST.order_counts(sample(qc, shots)), 2**t)
        ax.bar(range(2**t), emp, width=0.75, color=BLUE, alpha=0.75, label=f"Aer, {shots} shots")
        ax.plot(range(2**t), exact, "o", ms=2.6, color=PURPLE, label="exact")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 4, 8, 12, 15])
        ax.set_xlabel("measured y", fontsize=8)
        res[garbage or "clean"] = {"max_p": float(exact.max()), "qubits": qc.num_qubits,
                                   "tvd_vs_clean": None}
        res[garbage or "clean"]["exact"] = exact.tolist()
    axes[0].set_ylabel("probability", fontsize=8)
    axes[-1].legend(fontsize=7, loc="upper right")
    clean = np.array(res["clean"]["exact"])
    for k in res:
        res[k]["tvd_vs_clean"] = ST.tvd(np.array(res[k]["exact"]), clean)
        del res[k]["exact"]
    NUM["garbage"] = res
    save(fig, "garbage_demo")


# =============================================================================
# Gate synthesis: Solovay-Kitaev against Ross-Selinger
# =============================================================================
def synthesis():
    print("synthesis")
    from qiskit.circuit.library import RZGate
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import SolovayKitaevDecomposition, gridsynth_rz
    theta = np.pi / 16
    target = Operator(RZGate(theta)).data

    def dist(qc):
        U = Operator(qc).data
        ph = np.vdot(U.flatten(), target.flatten())
        return float(np.linalg.norm(U * np.exp(-1j * np.angle(ph)) - target, 2))

    def tcount(qc):
        ops = qc.count_ops()
        return ops.get("t", 0) + ops.get("tdg", 0)

    gs = [(dist(q), tcount(q)) for q in (gridsynth_rz(theta, epsilon=e) for e in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10))]
    skd = SolovayKitaevDecomposition()
    sk = [(dist(q), tcount(q)) for q in (skd.run(target, recursion_degree=d) for d in (1, 2, 3))]
    NUM["synthesis"] = {"gridsynth": gs, "sk": sk}
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    e = np.logspace(-10.6, -1.5, 50)
    ax.plot(np.log10(1 / e), 3 * np.log2(1 / e), ":", color=GREY, label=r"$3\log_2(1/\varepsilon)$")
    ax.plot([np.log10(1 / d) for d, _ in gs], [T for _, T in gs], "o-", color=BLUE, ms=3.5, label="Ross–Selinger")
    ax.plot([np.log10(1 / d) for d, _ in sk], [T for _, T in sk], "s-", color=VERM, ms=3.5, label="Solovay–Kitaev")
    ax.set_xlabel(r"precision  $\log_{10}(1/\varepsilon)$")
    ax.set_ylabel("T gates")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "synthesis")


# =============================================================================
# Factoring on Aer, and the post-processing
# =============================================================================
def factoring_runs():
    print("factoring runs")
    N, A, t = 15, 7, 8
    qc = order_circuit(A, N)
    counts = sample(qc, 4096)
    theory = ST.order_finding_probs(A, N, t)
    measured = ST.order_counts(counts)
    emp = ST.empirical(measured, 2**t)
    hit = ST.order_success_mask(A, N, t)
    NUM["n15"] = {"qubits": qc.num_qubits, "counts": {str(k): v for k, v in sorted(measured.items())},
                  "tvd": ST.tvd(emp, theory), "bound": ST.tvd_null(theory, 4096),
                  "aer_exact_eq_closed_form": bool(np.allclose(exact_probs(qc, range(t)), theory)),
                  "p_success": float(theory[hit].sum()), "p_success_measured": float(emp[hit].sum())}
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ys = sorted(measured)
    ax.bar(ys, [emp[y] for y in ys], width=7, color=BLUE, alpha=0.75, label="Aer, 4096 shots")
    ax.plot(ys, [theory[y] for y in ys], "o", color=PURPLE, ms=4, label="exact")
    ax.set_xlim(-8, 263)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_ylim(0, 0.32)
    ax.set_xlabel("measured y  (t = 8 counting qubits)")
    ax.set_ylabel("probability")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    save(fig, "n15_hist")

    from onectrl import order_circuit_1c
    N, A, t, shots = 21, 2, 10, 1024
    q1 = order_circuit_1c(A, N)
    c1 = sample(q1, shots)
    th = ST.order_finding_probs(A, N, t)
    e1 = ST.empirical(ST.order_counts(c1), 2**t)
    hit = ST.order_success_mask(A, N, t)
    from shor_essentials import order_from_counts
    NUM["n21"] = {"qubits": q1.num_qubits, "full_qubits": 4 * 5 + 2, "tvd": ST.tvd(e1, th),
                  "bound": ST.tvd_null(th, shots), "p_success": float(th[hit].sum()),
                  "p_success_measured": float(e1[hit].sum()), "r": order_from_counts(c1, A, N, t)}
    for name, counts_ in (("n21_exact", None), ("n21_measured", c1)):
        fig, axes = plt.subplots(1, 6, figsize=(5.9, 1.45), sharey=True)
        for s, ax in enumerate(axes):
            cen = s * 2**t / 6
            yy = np.arange(math.floor(cen) - 5, math.floor(cen) + 7)
            if counts_ is not None:
                ax.bar(yy, e1[yy % 2**t], width=0.8, color=BLUE, alpha=0.75)
            ax.plot(yy, th[yy % 2**t], "o-", ms=2.2, lw=0.8, color=PURPLE)
            ax.axvline(cen, color=GREY, ls=":", lw=0.7)
            ax.set_title(f"$s={s}$: $y\\approx{cen:.1f}$", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
        axes[0].set_ylabel("probability", fontsize=7.5)
        save(fig, name)


# =============================================================================
# What it costs
# =============================================================================
def costs():
    print("costs")
    rows = []
    for n, N in ((4, 15), (5, 21), (6, 33), (7, 65), (8, 129)):
        r = R.count(order_circuit(2, N))
        m, t = n + 1, 2 * n
        rows.append({"n": n, "N": N, "qubits": r.qubits, "toffoli": r.toffoli, "cphase": r.cphase,
                     "rotations": r.rotations, "T": r.T, "T_rot": r.T_with_rotations(1e-3),
                     "toffoli_formula": 8 * n**3 + 22 * n**2,
                     "cphase_formula": 4 * n * n * (5 * m * (m - 1) + 2 * m) + t * (t - 1) // 2})
    NUM["factoring_counts"] = rows
    ns = np.array([r_["n"] for r_ in rows], float)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for key, lab, col in (("toffoli", "Toffoli", VERM), ("cphase", "controlled phases", PURPLE),
                          ("rotations", "small rotations", BLUE)):
        ys = np.array([r_[key] for r_ in rows], float)
        slope = np.polyfit(np.log(ns), np.log(ys), 1)[0]
        ax.loglog(ns, ys, "o-", color=col, ms=3.5, label=f"{lab}  (slope {slope:.1f})")
    ax.set_xticks(ns, [str(int(v)) for v in ns])
    ax.minorticks_off()
    ax.set_xlabel("n = bits of N")
    ax.set_ylabel("count (measured)")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "factoring_scaling")

    from rc_adder import rc_c_ua
    from windowed import windowed_c_ua
    per = {"fourier": [], "rc": [], "win4": []}
    NS = (4, 6, 8, 10, 12)
    MODS = {4: 15, 6: 33, 8: 129, 10: 513, 12: 2049}
    for n in NS:
        N = MODS[n]
        c, x, yq, sf = QuantumRegister(1), QuantumRegister(n), QuantumRegister(n), QuantumRegister(2)
        q = QuantumCircuit(c, x, yq, sf)
        c_ua(q, c[0], 2, list(x), list(yq), sf[0], sf[1], N)
        per["fourier"].append(R.count(q).T_with_rotations(1e-3))
        a = QuantumRegister(n + 2)
        q = QuantumCircuit(c, x, yq, sf, a)
        rc_c_ua(q, c[0], 2, list(x), list(yq), sf[0], sf[1], N, list(a))
        per["rc"].append(R.count(q).T_with_rotations(1e-3))
        tmp, un = QuantumRegister(n), QuantumRegister(4)
        q = QuantumCircuit(c, x, yq, sf, tmp, un)
        windowed_c_ua(q, c[0], 2, list(x), list(yq), sf[0], sf[1], N, list(tmp), list(un), 4)
        per["win4"].append(R.count(q).T_with_rotations(1e-3))
    NUM["per_rung_T"] = {"n": list(NS), **per}
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.semilogy(NS, per["fourier"], "o-", color=PURPLE, ms=3.5, label="Fourier adders (Beauregard)")
    ax.semilogy(NS, per["win4"], "^-", color=BLUE, ms=3.5, label="windowed, w = 4")
    ax.semilogy(NS, per["rc"], "s-", color=VERM, ms=3.5, label="ripple-carry (Toffoli only)")
    for n, a_, b_ in zip(NS, per["fourier"], per["rc"]):
        ax.annotate(f"÷{a_ / b_:.0f}", (n, math.sqrt(a_ * b_)), fontsize=7, color=GREY, ha="center")
    ax.set_xlabel("n = bits of N")
    ax.set_ylabel("T per rung (rotations synthesised)")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "rung_T")


def variants():
    print("variants")
    from nested import nested_order_circuit
    from onectrl import order_circuit_1c
    from rc_adder import rc_c_ua
    from semiclassical import semiclassical_iqft
    from windowed import windowed_order_circuit

    def rc_order(A, N, one_control=False):
        n = math.ceil(math.log2(N))
        t = 2 * n
        ctr = QuantumRegister(1 if one_control else t, "ctr")
        tgt, anc, sf, rca = QuantumRegister(n, "tgt"), QuantumRegister(n, "anc"), QuantumRegister(2, "sf"), QuantumRegister(n + 2, "rc")
        out = ClassicalRegister(t, "out")
        qc = QuantumCircuit(ctr, tgt, anc, sf, rca, out)
        qc.x(tgt[0])
        rung = lambda k: (lambda q, c: rc_c_ua(q, c, pow(A, 2**k, N), list(tgt), list(anc), sf[0], sf[1], N, list(rca)))
        if one_control:
            semiclassical_iqft(qc, ctr[0], out, [rung(k) for k in range(t)])
        else:
            qc.h(ctr)
            for k in range(t):
                rung(k)(qc, ctr[k])
            qc.append(qft(t).inverse(), ctr)
            qc.measure(ctr, out)
        return qc

    vs = [("Beauregard (textbook)", order_circuit(7, 15)),
          ("windowed, w = 2", windowed_order_circuit(7, 15, w=2)),
          ("nested windows, 2 × 2", nested_order_circuit(7, 15, 2, 2)),
          ("ripple-carry", rc_order(7, 15)),
          ("one counting qubit", order_circuit_1c(7, 15)),
          ("one counting qubit + ripple-carry", rc_order(7, 15, one_control=True))]
    rows = []
    for lab, q in vs:
        r = R.count(q)
        rows.append({"label": lab, "qubits": r.qubits, "toffoli": r.toffoli, "T_toffoli": r.T,
                     "rotations": r.rotations, "T_total": r.T_with_rotations(1e-3)})
    NUM["variants_n15"] = rows
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.9, 1.9), gridspec_kw={"width_ratios": [1, 1.6]})
    labs = [r_["label"] for r_ in rows]
    a1.barh(labs, [r_["qubits"] for r_ in rows], color=BLUE, alpha=0.8)
    a1.invert_yaxis()
    a1.set_xlabel("qubits")
    for i, r_ in enumerate(rows):
        a1.text(r_["qubits"] + 0.4, i, str(r_["qubits"]), va="center", fontsize=7)
    a2.barh(labs, [r_["T_toffoli"] for r_ in rows], color=VERM, alpha=0.85, label="T from Toffolis")
    a2.barh(labs, [r_["T_total"] - r_["T_toffoli"] for r_ in rows], left=[r_["T_toffoli"] for r_ in rows],
            color=PURPLE, alpha=0.45, label="T from synthesised rotations")
    a2.invert_yaxis()
    a2.set_yticklabels([])
    a2.set_xlabel(r"T gates  (total synthesis error $\varepsilon = 10^{-3}$)")
    for i, r_ in enumerate(rows):
        a2.text(r_["T_total"] * 1.01 + 8000, i, f"{r_['T_total'] / 1000:,.0f}k", va="center", fontsize=7)
    a2.set_xlim(0, 1.18e6)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    a2.legend(fontsize=7, loc="lower right")
    save(fig, "variants")


# =============================================================================
# ECDLP
# =============================================================================
def ecdlp():
    print("ecdlp")
    curve, G, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    r = curve.point_order(G)
    k = next(k for k in range(r) if curve.mul(k, G) == Q)
    pts = [P for P in curve.points() if not P.inf]
    fig, ax = plt.subplots(figsize=(2.4, 2.25))
    ax.scatter([P.x for P in pts], [P.y for P in pts], s=26, color=RULE, edgecolor=SUBTLE, zorder=2, lw=0.6)
    for i in range(1, r):
        P = curve.mul(i, G)
        col = PURPLE if P == Q else VERM
        ax.scatter([P.x], [P.y], s=38, color=col, zorder=3)
        ax.annotate(f"[{i}]G" if P != Q else f"Q=[{i}]G", (P.x, P.y), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color=col)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xlim(-0.5, 6.9)
    ax.set_ylim(-0.5, 7.3)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$y^2=x^3+5x+4$ over $\mathbb{F}_7$", fontsize=8.5)
    save(fig, "curve_p7")

    counts, cands, info = S.run_ecdlp(curve, G, Q, r, shots=4096, seed=7)
    mb = info["m_bits"]
    q_ = 1 << mb
    pairs = S.pair_counts(counts)
    exact = ST.ecdlp_probs(r, k, mb)
    emp = ST.empirical(pairs, (q_, q_))
    hit = ST.ecdlp_success_mask(r, k, mb)
    total = sum(w for _, w in cands)
    NUM["ecdlp_p7"] = {"qubits": info["qubits"], "k": k, "r": r, "tvd": ST.tvd(emp, exact),
                       "bound": ST.tvd_null(exact, 4096), "p_success": float(exact[hit].sum()),
                       "p_success_measured": float(emp[hit].sum()),
                       "votes": [(kk, w / total) for kk, w in cands]}
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.0))
    for ax, (lab, M) in zip(axes, (("exact  P(j₁, j₂)", exact), ("Aer, 4096 shots", emp))):
        ax.imshow(M.T, origin="lower", cmap="Purples", vmin=0, vmax=0.2)
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlabel("j₁  (register u)", fontsize=8)
        ax.set_xticks(range(q_))
        ax.set_yticks(range(q_))
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("j₂  (register v)", fontsize=8)
    save(fig, "ecdlp_heat")

    fig, ax = plt.subplots(figsize=(2.3, 1.6))
    ax.bar([str(kk) for kk, _ in cands], [w / total for _, w in cands],
           color=[PURPLE if kk == k else RULE for kk, _ in cands])
    ax.set_xlabel("candidate k")
    ax.set_ylabel("share of votes")
    save(fig, "ecdlp_votes")

    curve11, G11 = C.TOY11, C.TOY11_G
    r11 = curve11.point_order(G11)
    k11 = 9
    c11, cands11, info11 = S.run_ecdlp(curve11, G11, curve11.mul(k11, G11), r11, shots=4096, seed=11, one_control=True)
    NUM["ecdlp_p11_1c"] = {"qubits": info11["qubits"], "full_qubits": S.ecdlp_circuit(curve11, G11, G11, r11, oracle="table")[1]["qubits"],
                           "k": k11, "recovered": next(kk for kk, _ in cands11 if C.verify_dlog(curve11, G11, curve11.mul(k11, G11), kk))}

    # one point addition, step by step
    import ec_kaliski as K
    import ec_modarith as MA
    import ec_mult as MU
    import ec_pointadd as PA
    from ec_sim import Machine
    p, n = 7, 3
    x2, y2 = G.x, G.y
    mch = Machine("and")
    qb = mch.alloc(1, "q")[0]
    x1, y1 = mch.alloc(n, "x"), mch.alloc(n, "y")
    lam = mch.anc(n, "lam")
    steps = [(r"1   $x \leftarrow x_1 - x_2$", lambda: MA.modsub_const(mch, x1, x2, p)),
             (r"2   $y \leftarrow y_1 - y_2$", lambda: MA.cmodsub_const(mch, qb, y1, y2, p)),
             (r"3   $\lambda \leftarrow y/x$  (division)", lambda: K.mod_div(mch, qb, x1, y1, lam, p)),
             (r"4   $y \leftarrow y \oplus x\lambda = 0$", lambda: MU.modmul_xor(mch, x1, lam, y1, p)),
             (r"5   $x \leftarrow x + 3x_2$", lambda: MA.modadd_const(mch, x1, 3 * x2 % p, p)),
             (r"6   $x \leftarrow x - \lambda^2$", lambda: MU.modsqr_sub(mch, lam, x1, p)),
             (r"7   $y \leftarrow y + x\lambda$", lambda: MU.modmul_add(mch, x1, lam, y1, p)),
             (r"8   $\lambda \leftarrow 0$  (division)", lambda: K.mod_div(mch, qb, x1, y1, lam, p)),
             (r"9   $x \leftarrow -x_3$", lambda: PA._csub_2x2_or_x2(mch, qb, x1, x2, p)),
             (r"10  $y \leftarrow y_3$", lambda: MA.cmodsub_const(mch, qb, y1, y2, p)),
             (r"11  $x \leftarrow x_3$", lambda: MA.cmodneg(mch, qb, x1, p))]
    vals = []
    for lab, fn in steps:
        mark = mch.begin()
        fn()
        sub = QuantumCircuit(mch.qc.qubits)
        for ci in mch.since(mark):
            sub.append(ci)
        rc_ = R.count(sub)
        vals.append((lab, rc_.toffoli + rc_.and_))
    mch.free(lam)
    tot = sum(v for _, v in vals)
    NUM["padd_steps"] = {"steps": vals, "total": tot, "division_share": (vals[2][1] + vals[7][1]) / tot,
                         "qubits": mch.qc.num_qubits}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    cols = [VERM if "division" in lab else RULE for lab, _ in vals]
    ax.barh([lab for lab, _ in vals], [v for _, v in vals], color=cols)
    ax.invert_yaxis()
    ax.set_xlabel("Toffoli-equivalents (p = 7)")
    ax.tick_params(axis="y", labelsize=7.5)
    save(fig, "padd_steps")

    rows = []
    for p in (7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093):
        n = p.bit_length()
        mm = Machine("and")
        PA.point_add_ctrl(mm, mm.alloc(1, "q")[0], mm.alloc(n, "x"), mm.alloc(n, "y"), 2, 1, p)
        rc_ = R.count(mm)
        rows.append({"p": p, "n": n, "qubits": rc_.qubits, "toffoli": rc_.toffoli + rc_.and_, "T": rc_.T})
    NUM["padd_scaling"] = rows
    nn = np.array([r_["n"] for r_ in rows], float)
    tof = np.array([r_["toffoli"] for r_ in rows], float)
    qub = np.array([r_["qubits"] for r_ in rows], float)
    slope = np.polyfit(np.log(nn[3:]), np.log(tof[3:]), 1)[0]
    qfit = np.polyfit(nn, qub, 1)
    NUM["padd_fit"] = {"toffoli_slope": slope, "qubits_linear": qfit.tolist()}
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.loglog(nn, tof, "o-", color=VERM, ms=3.5, label=f"Toffoli per addition (slope {slope:.1f})")
    ax.loglog(nn, qub, "s-", color=BLUE, ms=3.5, label=f"qubits (≈ {qfit[0]:.0f}n + {qfit[1]:.0f})")
    ax.set_xticks(nn, [str(int(v)) for v in nn])
    ax.minorticks_off()
    ax.set_xlabel("n = bits of p")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "ecdlp_scaling")

    abl = json.load(open(ROOT / "bench" / "ec_ablation.json"))
    pick = [("adder_n16", "CDKM [CDKM04]", "Gidney [Gid18]", "adder: Gidney vs CDKM"),
            ("cadder_n16", "reconstruct (CDKM)", "copy-then-add [106] Fig 4b", "controlled adder: copy-then-add"),
            ("mul_n16_w4", "schoolbook double-and-add", "Montgomery+QCSA [106] w=4", "multiplier: Montgomery + carry-save"),
            ("inv_n12", "reference Kaliski [HJN+20]", "unconditional+postponed [106]", "inversion: Kaliski, 106 §3.3"),
            ("inplace_n6_p61", "Kaliski division (out-of-place)", "EEA dialog + Bezout replay [1128]", "in-place multiply: Euclidean dialog"),
            ("padd_p=11", "affine in-place [106] Alg 3", "Jacobian out-of-place [106] Alg 4", "point addition: Jacobian"),
            ("moddbl_q1021", "exact (Alg 5)", "pseudo-Mersenne (Alg 7)", "doubling: pseudo-Mersenne (approx.)"),
            ("tempand_n12", "all Toffoli (7 T each)", "temporary AND (4 T / 0 T)", "T of an inversion: temporary AND")]
    bars = []
    for key, b, o, lab in pick:
        e = abl[key]
        metric = "t" if key.startswith("tempand") else "toffoli_paper"
        bars.append((lab, 1 - e["variants"][o][metric] / e["variants"][b][metric]))
    NUM["ablation"] = bars
    fig, ax = plt.subplots(figsize=(3.7, 2.3))
    ax.barh([b[0] for b in bars], [100 * b[1] for b in bars], color=INDIGO, alpha=0.8)
    ax.invert_yaxis()
    for i, (_, v) in enumerate(bars):
        ax.text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, 98)
    ax.set_xlabel("saved (Toffoli; T for the last row)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5)
    save(fig, "ablation")


# =============================================================================
# RSA against elliptic curves
# =============================================================================
def rsa_vs_ecc():
    print("rsa vs ecc")
    from rc_adder import rc_c_ua
    padd = {r_["n"]: r_ for r_ in NUM["padd_scaling"]}          # measured controlled point additions
    ns = sorted(padd)
    mult = {}
    for n in ns:                                                 # measured controlled multiplications
        N = (1 << n) - 1
        c, x, yq, sf, a = QuantumRegister(1), QuantumRegister(n), QuantumRegister(n), QuantumRegister(2), QuantumRegister(n + 2)
        q = QuantumCircuit(c, x, yq, sf, a)
        rc_c_ua(q, c[0], 2, list(x), list(yq), sf[0], sf[1], N, list(a))
        mult[n] = R.count(q).toffoli
    ratio = {n: padd[n]["toffoli"] / mult[n] for n in ns}

    # extrapolation of this repo's circuits, fitted on the measured n = 3..12
    nn = np.array(ns, float)
    A = np.vstack([nn**2 * np.log2(nn), nn**2, nn]).T
    coef, *_ = np.linalg.lstsq(A, np.array([padd[n]["toffoli"] for n in ns], float), rcond=None)
    qfit = np.polyfit(nn, [padd[n]["qubits"] for n in ns], 1)
    t_add = lambda n: float(coef @ [n * n * math.log2(n), n * n, n])
    repo_ecc = lambda n: (qfit[0] * n + qfit[1], 2 * n * t_add(n))            # one counting qubit
    repo_rsa = lambda n: (3 * n + 5, 2 * n * (36 * n * n + 43 * n))          # ripple-carry, one counting qubit
    # literature, minimal width: Roetteler-Naehrig-Svore-Lauter (curves), Haner-Roetteler-Svore (RSA)
    lit_ecc = lambda n: (9 * n + 2 * math.ceil(math.log2(n)) + 10, (448 * math.log2(n) + 4090) * n**3)
    lit_rsa = lambda n: (2 * n + 2, (64 * (math.log2(n) - 2) + 29.46) * n**3)
    ge_rsa = lambda n: 0.3 * n**3 + 0.0005 * n**3 * math.log2(n)             # Gidney-Ekera formula
    levels = [(80, 1024, 160), (112, 2048, 224), (128, 3072, 256), (192, 7680, 384), (256, 15360, 521)]
    NUM["rsa_ecc"] = {
        "per_op": {"n": ns, "point_add": [padd[n]["toffoli"] for n in ns], "mult": [mult[n] for n in ns],
                   "ratio": [ratio[n] for n in ns]},
        "repo_128": {"ecc": repo_ecc(256), "rsa": repo_rsa(3072), "ratio": repo_rsa(3072)[1] / repo_ecc(256)[1]},
        "repo_192": {"ecc": repo_ecc(384), "rsa": repo_rsa(7680), "ratio": repo_rsa(7680)[1] / repo_ecc(384)[1]},
        "lit_128": {"ecc": (2330, 1.26e11), "rsa": (6146, 1.86e13), "ratio": 1.86e13 / 1.26e11},
        "ge_3072": ge_rsa(3072), "litinski_256": 5e7, "ge_2048_reported": 2.7e9,
        "levels": [{"sigma": sg, "rsa_n": rn, "ecc_n": en, "rsa_toff": lit_rsa(rn)[1], "ecc_toff": lit_ecc(en)[1],
                    "ratio": lit_rsa(rn)[1] / lit_ecc(en)[1]} for sg, rn, en in levels],
    }

    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    ax.semilogy(ns, [padd[n]["toffoli"] for n in ns], "o-", color=PURPLE, ms=3.5, label="controlled point addition")
    ax.semilogy(ns, [mult[n] for n in ns], "s-", color=VERM, ms=3.5, label="controlled modular multiplication")
    for n in (4, 8, 12):
        ax.annotate(f"×{ratio[n]:.1f}", (n, math.sqrt(padd[n]["toffoli"] * mult[n])), ha="center", fontsize=8, color=SUBTLE)
    ax.set_xlabel("n = bits of the modulus / field")
    ax.set_ylabel("Toffolis per operation")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "group_op_cost")

    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    groups = [("minimal width\n(literature)", 1.26e11, 1.86e13),
              ("this repo's circuits\n(extrapolated)", repo_ecc(256)[1], repo_rsa(3072)[1]),
              ("optimised\n(literature)", 5e7, ge_rsa(3072))]
    xs = np.arange(len(groups))
    ax.bar(xs - 0.18, [g[1] for g in groups], width=0.34, color=PURPLE, alpha=0.85, label="P-256 (ECDSA)")
    ax.bar(xs + 0.18, [g[2] for g in groups], width=0.34, color=VERM, alpha=0.85, label="RSA-3072")
    for x_, g in zip(xs, groups):
        ax.text(x_, g[2] * 2.2, f"÷{g[2] / g[1]:.0f}", ha="center", fontsize=9.5, color=TEXT)
    ax.set_yscale("log")
    ax.set_ylim(1e7, 1e15)
    ax.set_xticks(xs, [g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("Toffolis, whole attack")
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    save(fig, "rsa_vs_ecc")

    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    sg = [L["sigma"] for L in NUM["rsa_ecc"]["levels"]]
    ax.semilogy(sg, [L["rsa_toff"] for L in NUM["rsa_ecc"]["levels"]], "s-", color=VERM, ms=3.5, label="RSA")
    ax.semilogy(sg, [L["ecc_toff"] for L in NUM["rsa_ecc"]["levels"]], "o-", color=PURPLE, ms=3.5, label="elliptic curve")
    for L in NUM["rsa_ecc"]["levels"]:
        ax.annotate(f"÷{L['ratio']:.0f}", (L["sigma"], math.sqrt(L["rsa_toff"] * L["ecc_toff"])), ha="center",
                    fontsize=7.5, color=SUBTLE)
    ax.set_xticks(sg)
    ax.set_xlabel("classical security level (bits)")
    ax.set_ylabel("Toffolis (minimal width)")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "security_scaling")

    # RSA-2048 with Beauregard's circuit, exact and approximate QFT (formulas verified for n = 4..8)
    n, eps = 2048, 1e-3
    m, t = n + 1, 2 * n
    tof = 8 * n**3 + 22 * n**2
    cp_exact = 4 * n * n * (5 * m * (m - 1) + 2 * m) + t * (t - 1) // 2
    n_qft = 40 * n * n + 1
    d = math.ceil(math.log2(n_qft * m / eps))
    cp_q = lambda w: d * w - d * (d + 1) // 2 if w > d else w * (w - 1) // 2
    cp_approx = 4 * n * n * (10 * cp_q(m) + 2 * m) + cp_q(t)
    T = lambda rot: 7 * tof + rot * (3 * math.log2(rot / eps) + 4)
    NUM["rsa2048"] = {"qubits": 4 * n + 2, "toffoli": tof, "rot_exact": 3 * cp_exact, "T_exact": T(3 * cp_exact),
                      "d": d, "rot_approx": 3 * cp_approx, "T_approx": T(3 * cp_approx)}


if __name__ == "__main__":
    circuits()
    qft_readout()
    uncomputation()
    synthesis()
    factoring_runs()
    costs()
    variants()
    ecdlp()
    rsa_vs_ecc()
    (HERE / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print("numbers.json written")
