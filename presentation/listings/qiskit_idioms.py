from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator

def add_one(qc, y):                  # a builder appends gates
    for i in reversed(range(1, len(y))):
        qc.mcx(y[:i], y[i])          # carry: multi-controlled X
    qc.x(y[0])

y, out = QuantumRegister(3, "y"), ClassicalRegister(3, "out")
qc = QuantumCircuit(y, out)
qc.x([y[0], y[1]])                   # y = 3, qubit 0 = lowest bit
add_one(qc, y)                       # y = 4
inc = QuantumCircuit(3)
add_one(inc, inc.qubits)
qc.append(inc.to_gate().inverse(), y)  # run backwards: y = 3
qc.measure(y, out)
sim = AerSimulator()
print(sim.run(transpile(qc, sim), shots=100).result().get_counts())
