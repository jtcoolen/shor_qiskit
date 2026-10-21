"""Kim ePrint 2026/106 Sec 4.4 QROM mixed addition vs the Horner baseline:
toy ECDLP on Aer CPU, resources, JSON (no plot).

Same task as bench/ec_edwards_toy.py -- the 8-point complete curve
-x^2 + y^2 = 1 + 3x^2y^2 over F_5, P = (1, 2), Q = [5]P, 3-bit u and v --
with two oracle families selected by `ec_edwards_kim.build_oracle`:

  horner    `ec_edwards` baseline: one control bit per addition, the addend
            constants folded into (controlled) constant additions
  kim-qrom  `ec_edwards_kim`: w-bit windows; per window the three constants
            (y-x, y+x, 2d x y) are loaded coherently by one QROM walk into
            three n-bit registers, then the 7-product Sec 4.4 law, then the
            QROM unload

Every oracle variant is basis-verified on all 64 (u, v) (scratch clean).
Variants of <= 63 qubits in the AER_RUNS list are sampled end to end on Aer
CPU (MPS) -- k recovered, all scratch measured |0>, TVD within shot noise --
and the rest are marked unsampled with the reason.

Resource convention: `toffoli_paper` = Toffoli + Gidney AND + CSWAP gates
(ec_cost.count) of the whole oracle; "toffoli" mode builds every AND as a
ccx and its uncompute as a ccx, "and" mode uses measurement-based AND-dagger
(no Toffoli).  Qubits are the circuit's allocated qubits incl. the 2 m_bits
scalar register and the output.  The paper's Table 4 counts all gates of a
ProjectQ synthesis at n = 255 with QCSA adders: it is reproduced separately
under "paper_table4_projection" and is NOT comparable to, nor reproduced by,
these toy measurements.
"""

import itertools
import json
import pathlib
import sys
import time

# this script shares its name with shor_qiskit/ec_edwards_kim.py: put the
# package first so the import below is the module, not this file
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
sys.path.insert(0, str(HERE.parent / "shor_qiskit"))

import ec_cost as CO  # noqa: E402
import ec_edwards as E  # noqa: E402
import ec_edwards_kim as KM  # noqa: E402
import shor_stats as ST  # noqa: E402
from ec_sim import getv, setv, simulate  # noqa: E402

OUT = pathlib.Path(__file__).with_suffix(".json")
K, SHOTS, SEED, AER_LIMIT, MB = 5, 4000, 3, 63, 3
# (adder, style, layout, w, first_lookup, mode) sampled on Aer
AER_RUNS = [
    ("horner", "inplace", None, None, None, "toffoli"),
    ("horner", "inplace", None, None, None, "and"),
    ("kim-qrom", "inplace", "compact", 1, True, "toffoli"),
    ("kim-qrom", "inplace", "compact", 2, True, "toffoli"),
    ("kim-qrom", "inplace", "compact", 3, True, "toffoli"),
    ("kim-qrom", "inplace", "compact", 3, True, "and"),
    ("kim-qrom", "bennett", "compact", 3, True, "toffoli"),
    ("kim-qrom", "bennett", "compact", 3, True, "and"),
]

PAPER_TABLE4 = {
    "source": "Kim et al., ePrint 2026/106, Table 4 (Ed25519, n = 255, one "
              "point addition, ProjectQ resource counts)",
    "note": "the paper's projections, quoted -- NOT measured here and not "
            "comparable to the toy counts (different field size, adders, "
            "gate set and depth-optimised layout)",
    "sec4_1_inplace_via_weierstrass": {
        "total_gates_log2": 28.07, "qubits": 7038, "t_depth": 144717,
        "full_depth": 2182096, "td_x_m_log2": 29.92, "fd_x_m_log2": 33.84},
    "sec4_4_out_of_place_edwards": {
        "total_gates_log2": 25.06, "qubits": 64947, "t_depth": 3577,
        "full_depth": 42377, "td_x_m_log2": 27.79, "fd_x_m_log2": 31.36},
}


def basis_ok(curve, P, m, info):
    kr, lr, ox, oy = info["regs"]
    keep = set(list(kr) + list(lr) + list(ox) + list(oy))
    for u, v in itertools.product(range(1 << MB), repeat=2):
        init = {}
        setv(init, kr, u)
        setv(init, lr, v)
        b = simulate(m.qc, init, m.checks)
        if (getv(b, m.qc, ox), getv(b, m.qc, oy)) != curve.mul(u + K * v, P):
            return False
        if any(b[i] for i, q in enumerate(m.qc.qubits) if q not in keep):
            return False
    return True


def variants():
    for style in ("inplace", "bennett"):
        yield ("horner", style, None, None, None)
    for style, layout, w, fl in itertools.product(
            KM.STYLES, KM.LAYOUTS, (1, 2, 3), (True, False)):
        yield ("kim-qrom", style, layout, w, fl)


def aer_row(c, P, Q, order, key, exact, null):
    adder, style, layout, w, fl, mode = key
    t0 = time.time()
    if adder == "horner":
        pairs, scr, cands, info = E.run_ecdlp(
            c, P, Q, order, shots=SHOTS, seed=SEED, mode=mode, style=style)
    else:
        pairs, scr, cands, info = KM.run_ecdlp(
            c, P, Q, order, w, shots=SHOTS, seed=SEED, mode=mode, style=style,
            layout=layout, first_lookup=fl)
    emp = ST.empirical(pairs, (1 << MB, 1 << MB))
    tvd = ST.tvd(emp, exact)
    if cands[0][0] != K or set(scr) != {0} or tvd > null:
        raise AssertionError((key, cands[:2], scr, tvd))
    return {"mode": mode, "aer": info["aer"], "circuit_qubits": info["qubits"],
            "shots": SHOTS, "seed": SEED, "recovered_k": cands[0][0],
            "scratch_qubits_measured": info["scratch_qubits"],
            "scratch_all_zero_shots": scr.get(0, 0), "tvd": round(tvd, 4),
            "tvd_noise_bound": round(null, 4),
            "wall_seconds": round(time.time() - t0, 1)}


def measure():
    c = E.EDW5
    P = E.generator(c)
    order = c.point_order(P)
    Q = c.mul(K, P)
    exact = ST.ecdlp_probs(order, K, MB)
    null = ST.tvd_null(exact, SHOTS)
    rows = []
    for adder, style, layout, w, fl in variants():
        row = {"adder": adder, "style": style, "layout": layout, "w": w,
               "first_lookup": fl}
        for mode in ("toffoli", "and"):
            m, info = KM.build_oracle(adder, c, P, Q, MB, mode, style, w=w,
                                      layout=layout or "compact",
                                      first_lookup=bool(fl))
            cnt = CO.count(m)
            row[mode] = {"qubits": m.qc.num_qubits, "peak_live": m.peak_live,
                         "toffoli_paper": cnt["toffoli_paper"],
                         "gates": cnt["gates"], "depth": cnt["depth"]}
            if mode == "toffoli":
                row["basis_verified_all_64"] = basis_ok(c, P, m, info)
                row["retained_projective_qubits_peak"] = info[
                    "retained_qubits_peak"]
                if adder == "kim-qrom":
                    row["mixed_additions"] = info["mixed_additions"]
                    row["windows"] = info["windows"]
                    row["qrom_constant_registers"] = 3
        row["aer_runs"] = []
        for key in AER_RUNS:
            if key[:5] == (adder, style, layout, w, fl):
                row["aer_runs"].append(aer_row(c, P, Q, order, key, exact, null))
                print("  aer", key, row["aer_runs"][-1]["wall_seconds"], "s",
                      file=sys.stderr)
        if not row["aer_runs"]:
            q = row["toffoli"]["qubits"]
            row["aer"] = (f"unsampled: {q} qubits > {AER_LIMIT}-qubit Aer limit "
                          "(basis-verified only)" if q > AER_LIMIT else
                          "unsampled: CPU time budget (basis-verified only)")
        rows.append(row)
        print(adder, style, layout, w, fl, row["toffoli"]["qubits"],
              row["toffoli"]["toffoli_paper"], file=sys.stderr)
    return {
        "scope": "toy only (p = 5, n = 3 bits): Kim ePrint 2026/106 Sec 4.4 "
                 "out-of-place mixed extended-Edwards addition with the three "
                 "addend constants loaded by a coherent windowed QROM, inside "
                 "a full Shor ECDLP circuit; compared with the ec_edwards "
                 "controlled-Horner baseline. NOT Ed25519 scale, NOT the "
                 "paper's QCSA/ProjectQ synthesis; same-task toy resource "
                 "frontier only, not a cryptographic-scale Pareto claim",
        "resource_convention": "toffoli_paper = Toffoli + AND + CSWAP of the "
                               "whole oracle (ec_cost.count); 'toffoli' mode: "
                               "ANDs and their uncompute both as ccx; 'and' "
                               "mode: measurement-based AND-dagger; qubits = "
                               "allocated circuit qubits incl. u, v, output",
        "like_for_like": "kim-qrom w=1, first_lookup=False performs the same "
                         "6 additions as horner (accumulator from identity); "
                         "first_lookup=True replaces the first u window by a "
                         "QROM load straight into the accumulator",
        "curve": {"p": c.p, "a": -1, "d": c.d, "points": len(c.points()),
                  "P": P, "Q": Q, "k": K, "order": order, "m_bits": MB},
        "aer_limit_qubits": AER_LIMIT,
        "oracles": rows,
        "paper_table4_projection": PAPER_TABLE4,
    }


def main():
    data = measure()
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
