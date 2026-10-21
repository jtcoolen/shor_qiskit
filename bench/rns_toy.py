"""Compare two full N=15, A=2, t=3 toy order-finding circuits.

The RNS oracle approximates high bits, not the low-bit CFS algorithm. Equal
task and frequency register size; the two output representations differ.
"""

import json
from pathlib import Path

from qiskit import transpile
from qiskit_aer import AerSimulator

from resources import count
from rns_order import rns_order_circuit
from rns_reference import ResidueOracle, modular_deviation
from shor_essentials import order_circuit, order_from_counts


OUT = Path(__file__).with_suffix(".json")


def measure(shots=512, seed=5):
    if shots < 1:
        raise ValueError("shots must be positive")
    config = ResidueOracle(15, 2, 3, (7, 11, 13, 19), 3)
    sim = AerSimulator(method="matrix_product_state", device="CPU")
    circuits = (
        ("exact Fourier modular", order_circuit(2, 15, 3)),
        ("truncated RNS high bits", rns_order_circuit(config)),
    )
    rows = []
    for name, circuit in circuits:
        counts = sim.run(transpile(circuit, sim, optimization_level=0),
                         shots=shots, seed_simulator=seed).result().get_counts()
        if order_from_counts(counts, 2, 15, 3) != 4:
            raise AssertionError((name, counts))
        if any(int(bits, 2) % 2 for bits in counts):
            raise AssertionError((name, "unexpected nonperiodic frequency", counts))
        r = count(circuit)
        rows.append({
            "label": name, "qubits": r.qubits, "toffoli": r.toffoli,
            "rotations": r.rotations, "t": r.T,
            "counts": dict(sorted(counts.items())),
            "order": 4, "backend": "Aer CPU/matrix_product_state",
        })
    return {
        "task": "N=15, A=2, exponent_bits=3, order=4", "shots": shots,
        "rows": rows, "rns_primes": list(config.primes),
        "crt_product": config.crt_modulus, "kept_bits": config.kept_bits,
        "wrap_deviation": modular_deviation(config.crt_modulus, 0, 15),
        "max_output_deviation": max(config.deviation(i) for i in range(8)),
        "exact_high_bit_matches": sum(
            config.approximate(i) == config.exact(i) >> 1 for i in range(8)),
        "input_cases": 8,
        "frontier": ["exact Fourier modular"],
        "note": "RNS high-bit toy circuit is dominated at this width; "
                "Toffoli/rotations exclude fault-tolerant synthesis overhead.",
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(measure(), indent=2) + "\n")
    print(f"Wrote {OUT}")
