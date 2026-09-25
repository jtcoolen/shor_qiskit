# the toy curve y^2 = x^3 + 5x + 4 mod 7, with G of order 5
from ec_classical import CLASSIQ as curve, CLASSIQ_G as G
from ec_shor import solve

Q = curve.mul(4, G)                         # public key: Q = [4]G
# build the circuit, run it on Aer, post-process the counts
k, counts, info = solve(curve, G, Q, order=5)
print(k, info["qubits"])
