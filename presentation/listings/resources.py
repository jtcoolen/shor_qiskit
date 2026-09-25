import resources as R
from shor_essentials import order_circuit

r = R.count(order_circuit(7, 15))
print(r.qubits, r.toffoli, r.cphase, r.rotations)
print(r.T, r.T_with_rotations(1e-3))
