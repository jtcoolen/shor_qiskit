"""Benchmark of the CFS ePrint 2024/222 low-bit RNS toy -> bench/rns_cfs_toy.json.

Run: PYTHONPATH=shor_qiskit python bench/rns_cfs_toy.py
"""

import json
import os
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from rns_cfs import (CfsConfig, CfsFactoringToy, cfs_factoring_circuit,
                     cfs_multiproduct, cfs_order_circuit, eh_postprocess,
                     ideal_distribution, ideal_order_distribution,
                     order_postprocess, split_counts)

SHOTS, SEED = 2000, 7
MPS = AerSimulator(method="matrix_product_state", device="CPU")


def describe(conf):
    qc, regs, tally = cfs_multiproduct(conf)
    tq = transpile(qc, MPS, optimization_level=0)
    return dict(
        N=conf.modulus, multipliers=list(conf.multipliers),
        primes=list(conf.primes), r=conf.out_bits, nu=conf.nu,
        u=conf.u, u_prime=conf.u_prime, t_N=conf.t_N,
        registers={k: len(v) for k, v in regs.items()},
        qubits=qc.num_qubits, gate_counts=dict(qc.count_ops()),
        aer_transpiled_gates=len(tq.data), tally=vars(tally),
        failures={str(k): v for k, v in conf.failures().items()},
        internal_failures={str(k): v
                           for k, v in conf.failures(internal=True).items()})


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def eh_run(toy):
    qc, _, _ = cfs_factoring_circuit(toy)
    t = time.time()
    counts = MPS.run(transpile(qc, MPS, optimization_level=0), shots=SHOTS,
                     seed_simulator=SEED).result().get_counts()
    secs = time.time() - t
    out = split_counts(counts)
    ideal = ideal_distribution(toy)
    emp = np.zeros_like(ideal)
    for (j, k), c in out.items():
        emp[j, k] += c / SHOTS
    pq = eh_postprocess(toy, out)
    return dict(beta=toy.beta, qubits=qc.num_qubits, seconds=round(secs, 2),
                counts={f"{j},{k}": c for (j, k), c in sorted(out.items())},
                ideal={f"{j},{k}": round(float(ideal[j, k]), 6)
                       for j, k in np.argwhere(ideal > 1e-9)},
                tv_distance=round(tv(ideal, emp), 4),
                p_zero=round(float(ideal[0, 0]), 6),
                theorem3_bound=0.5 + 0.01 + 0.25 / 2,
                result=list(pq) if pq else None)


def order_run(conf, beta, base):
    qc, _, _ = cfs_order_circuit(conf, beta)
    t = time.time()
    counts = MPS.run(transpile(qc, MPS, optimization_level=0), shots=SHOTS,
                     seed_simulator=SEED).result().get_counts()
    secs = time.time() - t
    emp = np.zeros(1 << conf.m)
    for k, c in counts.items():
        emp[int(k, 2)] += c / SHOTS
    ideal = ideal_order_distribution(conf, beta)
    true_r = next(r for r in range(1, conf.modulus)
                  if pow(base, r, conf.modulus) == 1)
    per_shot = sum(c for k, c in counts.items()
                   if order_postprocess({k: 1}, base, conf.modulus, conf.m)
                   == true_r) / SHOTS
    return dict(beta=beta, qubits=qc.num_qubits, seconds=round(secs, 2),
                counts=dict(sorted(counts.items())),
                ideal={format(y, f"0{conf.m}b"): round(float(p), 6)
                       for y, p in enumerate(ideal) if p > 1e-9},
                tv_distance=round(tv(ideal, emp), 4),
                order=order_postprocess(counts, base, conf.modulus, conf.m),
                true_order=true_r, per_shot_success=per_shot)


def main():
    good = CfsFactoringToy(15, 2, (7, 11, 13, 19), 2, 1)
    bad = CfsFactoringToy(15, 2, (3, 5, 7, 11), 2, 1)
    n21 = CfsConfig(21, [pow(2, 1 << i, 21) for i in range(5)],
                    (5, 7, 11, 13, 29), 2, 2)
    report = dict(
        paper="Chevignard-Fouque-Schrottenloher, ePrint 2024/222",
        scope="toy Algorithm 1 (coherent, clean) + toy Algorithm 2 for "
              "N=15; Shor order finding (deviation) for N=21",
        shots=SHOTS, seed=SEED, simulator="Aer MPS CPU",
        algorithm1=dict(n15_good=describe(good.config()),
                        n15_failing=describe(bad.config()),
                        n21_order=describe(n21)),
        algorithm2_n15=[eh_run(CfsFactoringToy(15, 2, (7, 11, 13, 19), 2, b))
                        for b in (1, 2, 3)],
        order_n21=order_run(n21, 1, 2),
    )
    path = os.path.join(os.path.dirname(__file__), "rns_cfs_toy.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    for run in report["algorithm2_n15"]:
        print("EH beta", run["beta"], "tv", run["tv_distance"], "->",
              run["result"])
    o = report["order_n21"]
    print("N=21 order", o["order"], "tv", o["tv_distance"], "per-shot",
          o["per_shot_success"])


if __name__ == "__main__":
    main()
