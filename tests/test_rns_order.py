"""Toy coherent RNS high-bit CRT oracle and full period-finding circuit."""

import json
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from rns_order import rns_oracle, rns_order_circuit
from rns_reference import ResidueOracle
from shor_essentials import order_from_counts


def main():
    config = ResidueOracle(15, 2, 3, (7, 11, 13, 19), 3)
    oracle, regs = rns_oracle(config)
    exponent, _, _, _, output = regs
    sim = AerSimulator(method="matrix_product_state", device="CPU")
    for value in range(1 << config.exponent_bits):
        qc = QuantumCircuit(*oracle.qregs)
        for bit in range(config.exponent_bits):
            if value >> bit & 1:
                qc.x(exponent[bit])
        qc.compose(oracle, inplace=True)
        qc.measure_all()
        counts = sim.run(transpile(qc, sim, optimization_level=0),
                         shots=8, seed_simulator=5).result().get_counts()
        output_index = qc.find_bit(output[0]).index
        expected = value | (config.approximate(value) << output_index)
        assert {int(key.replace(" ", ""), 2) for key in counts} == {expected}, (
            value, counts, expected)
    print("RNS oracle: all eight toy exponents and scratch registers exact")

    coherence = QuantumCircuit(*oracle.qregs)
    coherence.h(exponent)
    coherence.compose(oracle, inplace=True)
    coherence.compose(oracle.inverse(), inplace=True)
    coherence.h(exponent)
    coherence.measure_all()
    interference = sim.run(
        transpile(coherence, sim, optimization_level=0),
        shots=32, seed_simulator=5).result().get_counts()
    assert {int(key.replace(" ", ""), 2) for key in interference} == {0}
    print("RNS oracle: exact superposition phase after inverse and clean scratch")

    circuit = rns_order_circuit(config)
    counts = sim.run(transpile(circuit, sim, optimization_level=0),
                     shots=512, seed_simulator=5).result().get_counts()
    assert order_from_counts(counts, 2, 15, config.exponent_bits) == 4
    assert sum(count for bits, count in counts.items()
               if int(bits, 2) % 2 == 0) == 512
    print(f"RNS E2E: {circuit.num_qubits} qubits, "
          "512 Aer CPU shots recover order 4")
    report = json.loads((Path(__file__).resolve().parent.parent
                         / "bench" / "rns_toy.json").read_text())
    rows = report["rows"]
    assert report["task"] == "N=15, A=2, exponent_bits=3, order=4"
    assert report["frontier"] == ["exact Fourier modular"]
    assert rows[0]["qubits"] < rows[1]["qubits"] == circuit.num_qubits
    assert rows[0]["toffoli"] < rows[1]["toffoli"]
    assert rows[0]["rotations"] < rows[1]["rotations"]
    assert report["max_output_deviation"] == 4 / 15
    assert report["exact_high_bit_matches"] == 4

    second = ResidueOracle(15, 7, 3, (3, 5, 11, 23), 3)
    second_circuit = rns_order_circuit(second)
    second_counts = sim.run(
        transpile(second_circuit, sim, optimization_level=0),
        shots=256, seed_simulator=7).result().get_counts()
    assert order_from_counts(second_counts, 7, 15, 3) == 4
    assert all(int(bits, 2) % 2 == 0 for bits in second_counts)
    print("RNS E2E: independent A=7 CRT set recovers order 4 on Aer CPU")


if __name__ == "__main__":
    main()
