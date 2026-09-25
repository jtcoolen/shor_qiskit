from qiskit import transpile
from qiskit_aer import AerSimulator
from shor_essentials import order_circuit, order_from_counts

sim = AerSimulator(seed_simulator=2026)
qc = order_circuit(7, 15)                        # a = 7, N = 15: 18 qubits
counts = sim.run(transpile(qc, sim), shots=4096).result().get_counts()
r = order_from_counts(counts, 7, 15, t=8)
print(qc.num_qubits, r)
