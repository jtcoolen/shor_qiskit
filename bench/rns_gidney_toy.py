"""Gidney arXiv:2505.15917 App. A.1 toy against the N=15, A=2, t=3 baselines.

Same task as bench/rns_toy.py.  The Gidney rows run the App. A.1 dataflow
transcribed in shor_qiskit/rns_gidney.py (Shor-style g**e input, toy widths,
loop3 startup lookup only, window4 = 1).  Resource counts treat every
classically conditioned block as firing; Toffoli/rotations exclude
fault-tolerant synthesis overhead.  The CFS ePrint 2024/222 least-significant-
bit algorithm is not implemented: its full text was not retrievable.
"""

import json
from pathlib import Path

from qiskit import transpile
from qiskit_aer import AerSimulator

from resources import count
from rns_gidney import (ApproxModexpConfig, approx_period_circuit,
                        ideal_joint_distribution, split_counts)
from rns_order import rns_order_circuit
from rns_reference import ResidueOracle
from shor_essentials import order_circuit, order_from_counts

OUT = Path(__file__).with_suffix(".json")


def _row(label, circuit, counts, backend, **extra):
    r = count(circuit)
    return {"label": label, "qubits": r.qubits, "toffoli": r.toffoli,
            "rotations": r.rotations, "t": r.T, "measure": r.measure,
            "conditional": r.conditional,
            "counts": dict(sorted(counts.items())),
            "order": order_from_counts(counts, 2, 15, 3),
            "backend": backend, **extra}


def measure(shots=512, seed=5):
    if shots < 1:
        raise ValueError("shots must be positive")
    rows = []
    mps = AerSimulator(method="matrix_product_state", device="CPU")
    high = ResidueOracle(15, 2, 3, (7, 11, 13, 19), 3)
    for label, circuit in (("exact Fourier modular", order_circuit(2, 15, 3)),
                           ("truncated RNS high bits (rns_order)",
                            rns_order_circuit(high))):
        counts = mps.run(transpile(circuit, mps, optimization_level=0),
                         shots=shots, seed_simulator=seed).result().get_counts()
        rows.append(_row(label, circuit, counts,
                         "Aer CPU/matrix_product_state"))
    sv = AerSimulator(method="statevector", device="CPU")
    for mask_bits in (0, 1):
        conf = ApproxModexpConfig(15, 2, 3, (17, 23), 2, window1=2,
                                  mask_bits=mask_bits)
        circuit, _, tally = approx_period_circuit(conf)
        raw = sv.run(transpile(circuit, sv, optimization_level=0),
                     shots=shots, seed_simulator=seed).result().get_counts()
        freq, joint = {}, {}
        for fields, c in split_counts(raw, circuit):
            bits = format(fields["frequency"], "03b")
            freq[bits] = freq.get(bits, 0) + c
            key = (fields["approx"], fields["frequency"])
            joint[key] = joint.get(key, 0) + c
        ideal = ideal_joint_distribution(conf)
        tv = 0.5 * sum(abs(joint.get(k, 0) / shots - ideal.get(k, 0))
                       for k in set(joint) | set(ideal))
        row = _row(f"Gidney A.1 approx_modexp toy, mask_bits={mask_bits}",
                   circuit, freq, "Aer CPU/statevector (dynamic circuit)",
                   primes=list(conf.primes), ell=conf.ell,
                   kept_bits=conf.kept_bits, window1=conf.window1,
                   truncated_modulus=conf.truncated_modulus,
                   approximate=[conf.approximate(e) for e in range(8)],
                   max_deviation=max(conf.deviation(e) for e in range(8)),
                   paper_tally=vars(tally),
                   tv_vs_ideal_eq26=round(tv, 4))
        if row["order"] != 4 or any(int(b, 2) % 2 for b in freq):
            raise AssertionError(row)
        rows.append(row)
    return {
        "task": "N=15, A=2, exponent_bits=3, order=4", "shots": shots,
        "rows": rows,
        "paper_tally_rule": "Gidney Sec. 3.1/Table 3: subtractions and "
                            "comparisons count as additions; GHZ lookups free",
        "deviations": [
            "Shor-style g**e input (as Sec. 2 describes), not Ekera-Hastad",
            "Zenodo ExecutionConfig unavailable: lookup1 (Eq. 52 dlog "
            "differences of window products) and lookup3c (pow(g_i, l, p_i)) "
            "rebuilt from equations/docstrings",
            "only loop3 startup lookup (2*window3 >= ell); lookup3a/b body "
            "not reconstructed",
            "window4 = 1, so lookup4 is the Eq. 21 constant C[i, k]",
            "ancilla-free Fourier arithmetic and address-controlled phases "
            "instead of materialised unary-iteration lookups (no vents)",
            "X-measured qubits reset by conditional X",
            "toy widths far outside the paper's parameter grid",
        ],
        "not_implemented": "CFS ePrint 2024/222 low-bit algorithm: ePrint "
                           "PDF HTTP 403, HAL hal-05436363 behind an "
                           "anti-bot challenge; no details verified",
        "note": "At N=15 the low-space circuits are wider and costlier than "
                "exact Fourier modular; the paper's advantage is asymptotic.",
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(measure(), indent=2) + "\n")
    print(f"Wrote {OUT}")
