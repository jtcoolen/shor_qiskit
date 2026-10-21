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
BACKENDS = ("kaliski", "kaliski-opt", "divsteps", "divsteps-decoded")


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

    p7, base7, double7 = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order7, mb7 = p7.point_order(base7), 2
    k7 = next(i for i in range(order7) if p7.mul(i, base7) == double7)
    shots7 = 4096
    validations = []
    for backend, bits in (("divsteps", mb7), ("divsteps-decoded", 3)):
        qc7, _ = S.ecdlp_circuit(
            p7, base7, double7, order7, oracle="arith",
            m_bits=bits, inversion=backend)
        if qc7.num_qubits > aer_max:
            raise RuntimeError(
                f"p=7 {backend} requires {qc7.num_qubits} qubits, "
                f"but Aer {dev} offers only {aer_max}")
        exact = ST.ecdlp_probs(order7, k7, bits)
        bound = ST.tvd_null(exact, shots7)
        counts7 = run(qc7, shots=shots7, seed=seed)
        candidates7 = C.ecdlp_postprocess(
            S.pair_counts(counts7), order7, bits)
        empirical7 = ST.empirical(
            S.pair_counts(counts7), (1 << bits, 1 << bits))
        tvd7 = ST.tvd(empirical7, exact)
        if not candidates7 or candidates7[0][0] != k7 or tvd7 > bound:
            raise AssertionError((backend, candidates7[:3], tvd7, bound))
        validations.append({
            "task": f"affine arithmetic ECDLP p=7, order=5, k=4, m_bits={bits}",
            "backend": backend, "qubits": qc7.num_qubits, "shots": shots7,
            "aer": describe(qc7.num_qubits),
            "tvd": round(tvd7, 6), "tvd_limit": round(bound, 6),
        })
        print(f"p=7 {backend}: {qc7.num_qubits} qubits on "
              f"{describe(qc7.num_qubits)}, TVD {tvd7:.3f} <= {bound:.3f}",
              flush=True)
    return {
        "task": "affine arithmetic ECDLP p=5, order=3, k=2, m_bits=2",
        "shots": shots, "seed": seed, "device": dev,
        "aer_limit": aer_max,
        "tvd_limit": round(limit, 6), "rows": rows,
        "metrics": ["qubits", "toffoli_paper", "depth"],
        "frontier": frontier(rows),
        "additional_validation": validations,
    }


if __name__ == "__main__":
    result = measure()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Frontier: {result['frontier']}; wrote {OUT}", flush=True)
