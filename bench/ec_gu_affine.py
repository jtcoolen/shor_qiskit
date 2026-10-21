"""Compare exact affine ECDLP oracles on the same p=5 toy task.

Aer CPU cannot accept circuits wider than 63 qubits here. Rows beyond its
reported limit are basis-verified, but explicitly marked unsampled.
"""

import json
import pathlib

import ec_classical as C
import ec_cost as CO
import ec_shor as S
import shor_stats as ST
from aer_device import describe, device, run, simulator
from ec_pareto import frontier
from ec_sim import run as run_basis


OUT = pathlib.Path(__file__).with_suffix(".json")
BACKENDS = ("kaliski", "kaliski-opt", "divsteps")


def measure(shots=1024, seed=5):
    dev = device()
    curve = C.Curve(5, 0, 1, "p5")
    base = next(P for P in curve.points()
                if not P.inf and curve.point_order(P) == 3)
    double = curve.mul(2, base)
    order, mb, k = 3, 2, 2
    exact = ST.ecdlp_probs(order, k, mb)
    limit = ST.tvd_null(exact, shots)
    aer_max = simulator(64).configuration().n_qubits
    rows = []
    for backend in BACKENDS:
        qc, info = S.ecdlp_circuit(
            curve, base, double, order, oracle="arith",
            m_bits=mb, inversion=backend)
        kr, lr, x, y = info["regs"]
        for u in range(1 << mb):
            for v in range(1 << mb):
                expected = curve.add(info["offset"],
                                     curve.mul((u + k * v) % order, base))
                result = run_basis(info["machine"], {kr: u, lr: v})
                if (result(x), result(y)) != (expected.x, expected.y):
                    raise AssertionError((backend, u, v))
        cost = CO.count(qc)
        row = {
            "label": backend,
            **{metric: cost[metric]
               for metric in ("qubits", "toffoli_paper", "depth", "t")},
            "basis_cases": (1 << mb) ** 2,
        }
        if qc.num_qubits > aer_max:
            row["aer"] = f"not sampled: {qc.num_qubits} > {aer_max} Aer {dev} qubits"
            row["tvd"] = None
        else:
            counts = run(qc, shots=shots, seed=seed)
            candidates = C.ecdlp_postprocess(S.pair_counts(counts), order, mb)
            q = 1 << mb
            tvd = ST.tvd(ST.empirical(S.pair_counts(counts), (q, q)), exact)
            if not candidates or candidates[0][0] != k or tvd > limit:
                raise AssertionError((backend, candidates[:3], tvd, limit))
            row["aer"], row["tvd"] = describe(qc.num_qubits), round(tvd, 6)
        rows.append(row)
        print(f"{backend}: {row['qubits']} qubits, "
              f"{row['toffoli_paper']} Toffoli, "
              f"depth {row['depth']}, {row['aer']}", flush=True)
    return {
        "task": "affine arithmetic ECDLP p=5, order=3, k=2, m_bits=2",
        "shots": shots, "seed": seed, "device": dev,
        "aer_limit": aer_max,
        "tvd_limit": round(limit, 6), "rows": rows,
        "metrics": ["qubits", "toffoli_paper", "depth"],
        "frontier": frontier(rows),
    }


if __name__ == "__main__":
    result = measure()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Frontier: {result['frontier']}; wrote {OUT}", flush=True)
