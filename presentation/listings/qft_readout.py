import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from shor_essentials import qft

t, phi = 3, 1 / 4                   # s/r = 1/4, t = 3
qc = QuantumCircuit(t, t)
qc.h(range(t))
for q in range(t):                  # the ramp the rungs leave:
    qc.p(2 * np.pi * phi * 2**q, q) # qubit q turns 2^q s/r
qc.append(qft(t).inverse(), range(t))  # ramp -> integer
qc.measure(range(t), range(t))
sim = AerSimulator()
print(sim.run(transpile(qc, sim), shots=1000).result().get_counts())
