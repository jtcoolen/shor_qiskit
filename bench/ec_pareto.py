"""Measure the windowed toy ECDLP backends on one comparable end-to-end task.

Run with PYTHONPATH=shor_qiskit:tests venv/bin/python bench/ec_pareto.py.
SHOR_AER_DEVICE=GPU requires a CUDA-enabled Aer installation; auto uses CPU
when no GPU is available. Results are written only after every run passes.
"""

import json
import pathlib

import ec_classical as C
import ec_cost as CO
import ec_shor as S
import shor_stats as ST
from aer_device import available_devices, describe, device, run


ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "ec_pareto.json"
GCDS = ("dialog", "pingpong", "jump2", "jump2-codec")
SQUARES = ("reference", "symmetric", "karatsuba")


def frontier(rows):
    """Non-dominated points in (qubits, Toffoli, circuit depth)."""
    metrics = ("qubits", "toffoli_paper", "depth")
    return [
        row["label"] for row in rows
        if not any(
            other is not row
            and all(other[k] <= row[k] for k in metrics)
            and any(other[k] < row[k] for k in metrics)
            for other in rows
        )
    ]


def measure(shots=4000, seed=5):
    dev = device()  # Reject an unavailable explicitly requested GPU up front.
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order, m_bits, window_size = curve.point_order(P), 3, 2
    k = next(i for i in range(order) if curve.mul(i, P) == Q)
    exact = ST.ecdlp_probs(order, k, m_bits)
    limit = ST.tvd_null(exact, shots)
    rows = []
    for gcd in GCDS:
        for square in SQUARES:
            label = f"{gcd}+{square}"
            qc, info = S.ecdlp_circuit(
                curve, P, Q, order, oracle="arith", arithmetic="windowed",
                window_size=window_size, m_bits=m_bits, gcd=gcd, square=square)
            cost = CO.count(qc)
            counts = run(qc, shots=shots, seed=seed)
            candidates = C.ecdlp_postprocess(S.pair_counts(counts), order, m_bits)
            q = 1 << m_bits
            tvd = ST.tvd(ST.empirical(S.pair_counts(counts), (q, q)), exact)
            if not candidates or candidates[0][0] != k or tvd > limit:
                raise AssertionError(
                    f"{label}: expected k={k}, got {candidates[:3]}, "
                    f"TVD {tvd:.4f} (limit {limit:.4f})")
            row = {
                "label": label, "gcd": gcd, "square": square,
                "qubits": cost["qubits"],
                "toffoli_paper": cost["toffoli_paper"],
                "depth": cost["depth"], "t": cost["t"],
                "tvd": round(tvd, 6),
            }
            rows.append(row)
            print(f"{label}: {row['qubits']} qubits, "
                  f"{row['toffoli_paper']} Toffoli, depth {row['depth']}, "
                  f"TVD {tvd:.3f} <= {limit:.3f} on {describe(qc.num_qubits)}",
                  flush=True)
    return {
        "task": f"CLASSIQ p={curve.p}, order={order}, k={k}, "
                f"window_size={window_size}, m_bits={m_bits}",
        "shots": shots, "seed": seed,
        "device": dev, "available_devices": list(available_devices()),
        "method": describe(rows[0]["qubits"]), "tvd_limit": round(limit, 6),
        "metrics": ["qubits", "toffoli_paper", "depth"],
        "rows": rows, "frontier": frontier(rows),
    }


if __name__ == "__main__":
    result = measure()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Frontier: {', '.join(result['frontier'])}; wrote {OUT}", flush=True)
