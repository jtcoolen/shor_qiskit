"""Full coset order-finding circuit: unit fidelity of one coset multiplication,
then end-to-end order finding.  Covers the plain, runway and windowed variants."""
import math
import time
import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

import runway as rw
from coset import coset_state, coset_width
from coset_order import (c_ua_coset, coset_order_circuit, encode_coset)
from shor_essentials import order_from_counts

SIM = AerSimulator(method="statevector")


def layout(N, cpad, rspacing, rlen, window):
    n = math.ceil(math.log2(N)); m = coset_width(N, cpad)
    qc_regs = dict(c=QuantumRegister(1), x=QuantumRegister(m), y=QuantumRegister(m),
                   enc=QuantumRegister(2))
    pos = rw.runway_positions(m, rspacing, rlen) if rspacing else []
    rws = [(p, QuantumRegister(rlen)) for p in pos]
    extra = []
    tmp = anc = None
    if window:
        tmp, anc = QuantumRegister(n), QuantumRegister(window)
        extra = [tmp, anc]
    return qc_regs, rws, tmp, anc, extra


def _sv(qc):
    c = qc.copy()
    c.save_statevector()
    return SIM.run(transpile(c, SIM, optimization_level=0)).result().get_statevector().data


def unit_fidelity(N, cpad, A, x0, rspacing=None, rlen=None, window=None):
    r, rws, tmp, anc, extra = layout(N, cpad, rspacing, rlen, window)
    regs = [r["c"], r["x"], r["y"], r["enc"]] + [q for _, q in rws] + extra
    runs = [(p, list(q)) for p, q in rws]
    ideal = QuantumCircuit(*regs)
    ideal.x(r["c"][0])
    encode_coset(ideal, r["x"], (A * x0) % N, N, cpad, r["enc"][0], r["enc"][1])
    encode_coset(ideal, r["y"], 0, N, cpad, r["enc"][0], r["enc"][1])
    qc = QuantumCircuit(*regs)
    qc.x(r["c"][0])
    encode_coset(qc, r["x"], x0, N, cpad, r["enc"][0], r["enc"][1])
    encode_coset(qc, r["y"], 0, N, cpad, r["enc"][0], r["enc"][1])
    c_ua_coset(qc, r["c"][0], A, list(r["x"]), list(r["y"]), N, runs, window,
               list(tmp) if tmp else None, list(anc) if anc else None)
    return abs(np.vdot(_sv(ideal), _sv(qc))) ** 2


def run_case(title, N, cpad, As, xs, **kw):
    t0 = time.time()
    worst = min(unit_fidelity(N, cpad, A, x0, **kw) for A in As for x0 in xs)
    print(f"  {title:34s} cpad={cpad}: worst fidelity {worst:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return worst


# the encoder is exact: |k>_coset built by circuit equals the ideal state
for N, cpad, k in [(7, 3, 5), (15, 3, 11)]:
    r = QuantumRegister(coset_width(N, cpad)); e = QuantumRegister(2)
    qc = QuantumCircuit(r, e)
    encode_coset(qc, r, k, N, cpad, e[0], e[1])
    v = np.asarray(Statevector(qc).data).reshape(4, -1)
    assert abs(np.vdot(coset_state(k, N, cpad), v[0])) > 1 - 1e-9 and abs(v[1:]).max() < 1e-9
print("encoder: circuit == ideal coset state")

print("one coset multiplication  |x>|0> -> |A x>|0>,  fidelity with the ideal product state")
print("  (every addition deviates by <= 2^-cpad + runways*2^-len, subadditively: F >= 1 - total)")
N = 7
cases = [  # title, cpad, runway (spacing, len), window
    ("plain", 3, None, None), ("plain", 4, None, None), ("plain", 5, None, None),
    ("windowed w=2", 4, None, 2),
    ("runways 2x len 2", 4, (3, 2), None),
    ("runway 1x len 4", 4, (3, 4), None),
    ("runway 1x len 4 + windowed w=2", 4, (3, 4), 2),
]
fid = {}
for title, cpad, rwy, window in cases:
    m = coset_width(N, cpad)
    adds = 2 * (m if window is None else -(-m // window))
    nr = len(rw.runway_positions(m, *rwy)) if rwy else 0
    eps = adds * (2.0 ** -cpad + (nr * 2.0 ** -rwy[1] if rwy else 0))
    kw = dict(rspacing=rwy[0], rlen=rwy[1]) if rwy else {}
    t0 = time.time()
    f = min(unit_fidelity(N, cpad, 3, x0, window=window, **kw) for x0 in (2, 6))
    fid[title, cpad] = f
    print(f"  {title:32s} cpad={cpad} m={m}: worst fidelity {f:.4f}  bound {max(0, 1 - eps):.4f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    assert f >= 1 - eps - 1e-6, (title, cpad, f, eps)
# more padding / longer runways -> closer to the ideal state
assert fid["plain", 3] < fid["plain", 4] < fid["plain", 5]
assert fid["runways 2x len 2", 4] < fid["runway 1x len 4", 4] <= fid["plain", 4] + 1e-9
assert fid["runway 1x len 4", 4] > 0.5 * fid["plain", 4]

print("end-to-end order finding (N=15, A=7, r=4, t=3: exact phases, y in 0,2,4,6)")
for title, kw in [("coset", {}), ("coset + runway", dict(runway_spacing=4, runway_len=3))]:
    t0 = time.time()
    qc, info = coset_order_circuit(7, 15, 4, t=3, **kw)
    shots = 60
    counts = SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots).result().get_counts()
    good = sum(c for b, c in counts.items() if int(b, 2) % 2 == 0)
    r = order_from_counts(counts, 7, 15, 3)
    print(f"  {title:16s}: {qc.num_qubits} qubits, {good}/{shots} shots on even outcomes, "
          f"order = {r}  ({time.time()-t0:.0f}s)", flush=True)
    min_even = 0.45 if kw else 0.7
    assert good >= min_even * shots and r == 4
print("coset order-finding circuit: OK")
