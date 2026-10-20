"""Oblivious carry runways: piecewise addition agrees with plain addition except
with probability <= (#runways * #additions) / 2^m  (Gidney 1905.08488, Thm 4.2)."""
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
import runway as rw

SIM = AerSimulator(method="statevector")


def error_rate(L, positions, m, g, ks, shots=3000):
    qc = QuantumCircuit(L + m * len(positions))
    reg = list(range(L))
    rws = []
    off = L
    for p in positions:
        rws.append((p, list(range(off, off + m)))); off += m
    for i in range(L):
        if (g >> i) & 1:
            qc.x(i)
    rw.encode_runways(qc, reg, rws)
    for k in ks:
        rw.piecewise_add_const(qc, k, reg, rws)
    qc.measure_all()
    res = SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots).result().get_counts()
    want = (g + sum(ks)) % (1 << L)
    bad = 0
    for b, cnt in res.items():
        v = int(b.replace(" ", ""), 2)
        main = v & ((1 << L) - 1)
        rv = [(v >> (L + i * m)) & ((1 << m) - 1) for i in range(len(positions))]
        if rw.decode(main, rv, positions, L) != want:
            bad += cnt
    return bad / shots


# Fig. 2/3: one runway; the error shrinks as 2^-m with the number of additions
L, p, ks = 6, 3, [5, 7, 3, 6]
print(f"L={L}, runway at {p}, {len(ks)} additions: error vs bound  len(ks)*2^-m")
for m in (1, 2, 3, 4):
    e = max(error_rate(L, [p], m, g, ks) for g in (0, 9, 33))
    bound = len(ks) / 2 ** m
    print(f"  m={m}: worst error {e:.4f}   bound {bound:.4f}")
    assert e <= bound + 0.03, (m, e, bound)

# a single addition deviates by at most 2^-m (Thm 4.2), even for a 1-qubit runway
for m in (1, 2):
    e = max(error_rate(L, [p], m, g, [k]) for g in (0, 9, 33) for k in (3, 5, 7))
    print(f"  one addition, m={m}: worst error {e:.4f}   bound {2 ** -m:.4f}")
    assert e <= 2 ** -m + 0.03, (m, e)

# two runways, three pieces
e = error_rate(8, [3, 6], 2, 5, [13, 7, 21, 9])
print(f"L=8, runways at 3,6 (m=2), 4 additions: error {e:.4f}   bound {2 * 4 / 4:.2f}")
e = error_rate(7, [3], 3, 5, [13, 7])
assert e <= 2 / 8 + 0.03
print("oblivious carry runways: OK")
