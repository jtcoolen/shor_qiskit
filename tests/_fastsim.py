"""Fast exact outcome for permutation circuits: one Aer shot."""
from qiskit import transpile
from qiskit_aer import AerSimulator
_SIM = AerSimulator(method="statevector")

def outcome(qc):
    m = qc.copy(); m.measure_all()
    r = _SIM.run(transpile(m, _SIM, optimization_level=0), shots=1).result()
    counts = r.get_counts()
    assert len(counts) == 1, f"not deterministic: {counts}"
    return int(next(iter(counts)).replace(" ", ""), 2)

def setv(qc, v, qs):
    for i, q in enumerate(qs):
        if (v >> i) & 1:
            qc.x(q)
