"""Toy complete twisted-Edwards ECDLP on Aer CPU: resources, histogram, graph.

Standard HWCD 2008 / RFC 8032 a = -1 extended-coordinate law on the 8-point
curve -x^2 + y^2 = 1 + 3x^2y^2 over F_5 -- *not* Kim 2026/106 Sec 4.4 and not
Ed25519 scale.  Two oracle styles are measured on the same task:

  inplace  every controlled addition is affine in -> affine out, uncomputing
           the old point (no retained input; Aer-samplable, 54 qubits)
  bennett  extended-coordinate additions keep each projective input live
           until one final normalisation + Bennett unwind (117 qubits: above
           the 63-qubit Aer limit, so basis-verified only, marked unsampled)

Baselines: the existing Weierstrass arithmetic oracles (ec_shor affine and
windowed) are run on the isomorphic curve y^2 = x^3 + 4x + 4 and refuse it
-- every offset hits an exceptional addition -- so no arithmetic cost
comparison is possible; the Weierstrass *lookup* oracle is sampled only as
a distribution reference.  No Pareto claim is made.
"""

import itertools
import json
import pathlib

import ec_classical as C
import ec_cost as CO
import ec_edwards as E
import ec_shor as S
import shor_stats as ST
from aer_device import describe
from ec_sim import getv, setv, simulate

OUT = pathlib.Path(__file__).with_suffix(".json")
PNG = pathlib.Path(__file__).with_suffix(".png")
K, SHOTS, SEED, AER_LIMIT = 5, 4000, 3, 63


def basis_ok(curve, P, mb, m, info):
    kr, lr, ox, oy = info["regs"]
    keep = set(list(kr) + list(lr) + list(ox) + list(oy))
    for u, v in itertools.product(range(1 << mb), repeat=2):
        init = {}
        setv(init, kr, u)
        setv(init, lr, v)
        b = simulate(m.qc, init, m.checks)
        if (getv(b, m.qc, ox), getv(b, m.qc, oy)) != curve.mul(u + K * v, P):
            return False
        if any(b[i] for i, q in enumerate(m.qc.qubits) if q not in keep):
            return False
    return True


def component_costs(c):
    n = c.p.bit_length()
    R = E.generator(c)
    rows = {}
    m = E.Machine("toffoli")
    ctrl = m.alloc(1, "c")
    P1 = tuple(m.alloc(n, s) for s in "XYZT")
    P3 = tuple(m.alloc(n, s + "3") for s in "XYZT")
    E.ext_add_ctrl(m, ctrl[0], P1, R, c, P3)
    rows["controlled extended add (out of place, input retained)"] = (m, 4 * n)
    m = E.Machine("toffoli")
    ctrl = m.alloc(1, "c")
    x, y = m.anc(n, "x"), m.anc(n, "y")
    E.affine_add_ctrl(m, ctrl[0], x, y, R, c)
    rows["controlled affine add (in place, input uncomputed)"] = (m, 0)
    out = {}
    for name, (m, retained) in rows.items():
        cnt = CO.count(m)
        out[name] = {"qubits": m.qc.num_qubits, "toffoli": cnt["toffoli"],
                     "gates": cnt["gates"], "retained_qubits": retained}
    return out


def measure():
    c = E.EDW5
    P = E.generator(c)
    order = c.point_order(P)
    Q = c.mul(K, P)
    mb = 3
    exact = ST.ecdlp_probs(order, K, mb)
    null = ST.tvd_null(exact, SHOTS)
    rows = []
    hist = None
    for style in ("inplace", "bennett"):
        for mode in ("toffoli", "and"):
            m, info = E.ecdlp_oracle(c, P, Q, mb, mode, style=style)
            cnt = CO.count(m)
            row = {"style": style, "mode": mode, "oracle_qubits": m.qc.num_qubits,
                   "peak_live": m.peak_live, "toffoli": cnt["toffoli"],
                   "toffoli_paper": cnt.get("toffoli_paper"),
                   "gates": cnt["gates"],
                   "retained_projective_qubits_peak": info["retained_qubits_peak"],
                   "basis_verified_all_64": basis_ok(c, P, mb, m, info)}
            if m.qc.num_qubits <= AER_LIMIT:
                pairs, scr, cands, rinfo = E.run_ecdlp(
                    c, P, Q, order, shots=SHOTS, seed=SEED, mode=mode, style=style)
                emp = ST.empirical(pairs, (1 << mb, 1 << mb))
                tvd = ST.tvd(emp, exact)
                if cands[0][0] != K or set(scr) != {0} or tvd > null:
                    raise AssertionError((style, mode, cands[:2], scr, tvd))
                row.update(aer=rinfo["aer"], circuit_qubits=rinfo["qubits"],
                           shots=SHOTS, recovered_k=cands[0][0],
                           scratch_qubits_measured=rinfo["scratch_qubits"],
                           scratch_all_zero_shots=scr.get(0, 0),
                           tvd=round(tvd, 4), tvd_noise_bound=round(null, 4))
                if hist is None:
                    hist = {f"{j1},{j2}": [round(float(emp[j1, j2]), 5),
                                           round(float(exact[j1, j2]), 5)]
                            for j1, j2 in itertools.product(range(8), repeat=2)}
            else:
                row["aer"] = (f"unsampled: {m.qc.num_qubits} oracle qubits > "
                              f"{AER_LIMIT}-qubit Aer limit (basis-verified only)")
            rows.append(row)

    aw, bw, f = c.weierstrass()
    W = C.Curve(c.p, aw, bw)
    Pw, Qw = C.Point(*f(P)), C.Point(*f(Q))
    refusals = {}
    for label, kw in (("affine", dict(arithmetic="affine")),
                      ("windowed w=1..3", None)):
        msgs = set()
        for mbits in (1, 2, 3):
            for kws in ([kw] if kw else [dict(arithmetic="windowed",
                                              window_size=w) for w in (1, 2, 3)]):
                try:
                    S.ecdlp_circuit(W, Pw, Qw, order, oracle="arith",
                                    m_bits=mbits, **kws)
                    msgs.add("ACCEPTED")
                except ValueError as e:
                    msgs.add(str(e).split(" for the ")[0])
        refusals[label] = sorted(msgs)
    counts, cands, tinfo = S.run_ecdlp(W, Pw, Qw, order, shots=SHOTS, seed=SEED)
    temp = ST.empirical(S.pair_counts(counts), (8, 8))
    baseline = {
        "curve": f"y^2 = x^3 + {aw}x + {bw} over F_5 (birationally isomorphic)",
        "arith_oracles_m_bits_1_3": refusals,
        "table_lookup_reference": {
            "note": "classical-table lookup oracle, not arithmetic: "
                    "distribution reference only, not a cost baseline",
            "qubits": tinfo["qubits"], "recovered_k": cands[0][0],
            "tvd": round(ST.tvd(temp, exact), 4)},
    }
    return {
        "scope": "toy only; standard complete extended twisted-Edwards "
                 "(HWCD 2008 add-2008-hwcd-3 / RFC 8032 5.1.4) a=-1; NOT Kim "
                 "ePrint 2026/106 Sec 4.4 (unverified, PDF inaccessible); NOT "
                 "Ed25519 scale; no Pareto claim",
        "curve": {"p": c.p, "a": -1, "d": c.d, "points": len(c.points()),
                  "P": P, "Q": Q, "k": K, "order": order, "m_bits": mb},
        "components": component_costs(c),
        "oracles": rows,
        "baseline": baseline,
        "histogram_measured_vs_exact": hist,
        "tvd_noise_bound": round(null, 4),
    }


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    h = data["histogram_measured_vs_exact"]
    keys = [k for k in h if h[k][0] or h[k][1]]
    xs = range(len(keys))
    a1.bar([x - 0.2 for x in xs], [h[k][0] for k in keys], 0.4,
           label=f"Aer CPU, {SHOTS} shots")
    a1.bar([x + 0.2 for x in xs], [h[k][1] for k in keys], 0.4,
           label="exact Shor distribution")
    a1.set_xticks(list(xs), keys, rotation=90, fontsize=7)
    a1.set_xlabel("(j1, j2)")
    a1.set_ylabel("probability")
    r0 = data["oracles"][0]
    a1.set_title(f"Edwards ECDLP p=5, k={K}: recovered {r0['recovered_k']}, "
                 f"TVD {r0['tvd']} <= {r0['tvd_noise_bound']}", fontsize=9)
    a1.legend(fontsize=8)
    rows = data["oracles"]
    labels = [f"{r['style']}\n{r['mode']}" for r in rows]
    metrics = (("oracle_qubits", "qubits"), ("toffoli_paper", "Toffoli+AND / 1000"),
               ("retained_projective_qubits_peak", "retained qubits"))
    w = 0.27
    for i, (key, name) in enumerate(metrics):
        vals = [r[key] / (1000 if key == "toffoli_paper" else 1) for r in rows]
        a2.bar([x + (i - 1) * w for x in range(len(rows))], vals, w, label=name)
    a2.axhline(AER_LIMIT, ls="--", c="grey", lw=0.8)
    a2.text(len(rows) - 0.6, AER_LIMIT + 1, "Aer 63-qubit limit", fontsize=7,
            ha="right")
    a2.set_xticks(range(len(rows)), labels, fontsize=8)
    a2.set_title("oracle resources (same task; trade-off, no Pareto claim)",
                 fontsize=9)
    a2.legend(fontsize=8)
    fig.suptitle("TOY ONLY: standard HWCD/RFC 8032 extended twisted-Edwards "
                 "(a=-1, d=3, F_5); not Kim 2026/106 Sec 4.4, not Ed25519 "
                 "scale. Weierstrass arithmetic oracles refuse this instance.",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG, dpi=150)


def main():
    data = measure()
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    plot(data)
    for r in data["oracles"]:
        print(r["style"], r["mode"], r["oracle_qubits"], r["toffoli"],
              r["retained_projective_qubits_peak"], r.get("tvd"), r["aer"])
    print(json.dumps(data["baseline"], indent=1))


if __name__ == "__main__":
    main()
