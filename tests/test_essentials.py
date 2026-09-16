"""Verify every level of shor_essentials.py on a noise-free simulator."""
import math
import random

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from shor_essentials import (add_const, c_add_mod, c_ua, find_factor,
                             order_circuit, order_from_counts)

SIM = AerSimulator()
PM = generate_preset_pass_manager(backend=SIM, optimization_level=1)
SAMPLER = Sampler(seed=7)


def run_counts(qc, shots=256):
    return (
        SAMPLER.run([PM.run(qc)], shots=shots).result()[0]
        .data.out.get_counts()
    )


def set_value(qc, val, qubits):
    for i, q in enumerate(qubits):
        if (val >> i) & 1:
            qc.x(q)


# --- Level 1: add_const exhaustively, n=4, including negative X -------------
for n in (3, 4):
    for y0 in range(2**n):
        for X in (0, 1, 3, -2, 2**n - 1, 5 - 2**n):
            reg = QuantumRegister(n)
            out = ClassicalRegister(n, "out")
            qc = QuantumCircuit(reg, out)
            set_value(qc, y0, reg)
            add_const(qc, X, list(reg))
            qc.measure(reg, out)
            counts = run_counts(qc, shots=64)
            assert len(counts) == 1, (y0, X, counts)
            got = int(next(iter(counts)), 2)
            want = (y0 + X) % 2**n
            assert got == want, (n, y0, X, got, want)
print("Level 1  add_const: OK (exhaustive, n=3,4, incl. negative constants)")

# --- Level 2: c_add_mod exhaustively for N=15 and N=9, both control values --
for N in (9, 15):
    n = math.ceil(math.log2(N))
    for ctrl_on in (True, False):
        for y0 in range(N):
            for X in range(N):
                c = QuantumRegister(1)
                y = QuantumRegister(n)
                sf = QuantumRegister(2)
                out = ClassicalRegister(n + 2, "out")  # y + sign + flag
                qc = QuantumCircuit(c, y, sf, out)
                if ctrl_on:
                    qc.x(c[0])
                set_value(qc, y0, y)
                c_add_mod(qc, [c[0]], X, list(y), sf[0], sf[1], N)
                qc.measure(list(y) + list(sf), out)
                counts = run_counts(qc, shots=16)
                assert len(counts) == 1, (N, y0, X, counts)
                v = int(next(iter(counts)), 2)
                got_y, got_sf = v % 2**n, v >> n
                want = (y0 + X) % N if ctrl_on else y0
                assert got_y == want and got_sf == 0, \
                    (N, ctrl_on, y0, X, got_y, got_sf, want)
    print(f"Level 2  c_add_mod: OK (exhaustive, N={N}, control on and off)")

# --- Level 4: c_ua for N=15, all coprime A, all x < N, both controls --------
N, n = 15, 4
for A in (2, 4, 7, 8, 11, 13):
    for ctrl_on in (True, False):
        for x0 in range(N):
            c = QuantumRegister(1)
            x = QuantumRegister(n)
            y = QuantumRegister(n)
            sf = QuantumRegister(2)
            out = ClassicalRegister(2 * n + 2, "out")
            qc = QuantumCircuit(c, x, y, sf, out)
            if ctrl_on:
                qc.x(c[0])
            set_value(qc, x0, x)
            c_ua(qc, c[0], A, list(x), list(y), sf[0], sf[1], N)
            qc.measure(list(x) + list(y) + list(sf), out)
            counts = run_counts(qc, shots=16)
            assert len(counts) == 1, (A, x0, counts)
            v = int(next(iter(counts)), 2)
            got_x, rest = v % 2**n, v >> n
            want = (A * x0) % N if ctrl_on else x0
            assert got_x == want and rest == 0, \
                (A, ctrl_on, x0, got_x, rest, want)
    print(f"Level 4  c_ua: OK (N=15, A={A}, scratch clean, control on/off)")

# --- Level 5: end-to-end distribution, N=15, A=7, t=8 -----------------------
qc = order_circuit(7, 15)
counts = run_counts(qc, shots=4096)
peaks = {int(b, 2) for b, c in counts.items() if c > 100}
assert peaks == {0, 64, 128, 192}, sorted(
    (int(b, 2), c) for b, c in counts.items())
total_on_peaks = sum(c for b, c in counts.items() if int(b, 2) in peaks)
print(f"Level 5  order_circuit(7,15): peaks at {sorted(peaks)} "
      f"({total_on_peaks}/4096 shots on peaks)")
r = order_from_counts(counts, 7, 15, 8)
assert r == 4, r
print(f"Post-processing: order r = {r}")

# A=2 as well
counts2 = run_counts(order_circuit(2, 15), shots=1024)
r2 = order_from_counts(counts2, 2, 15, 8)
assert r2 == 4, r2
print(f"Post-processing: order of 2 mod 15 = {r2}")

# --- find_factor end to end -------------------------------------------------
random.seed(3)
f = find_factor(15, lambda qc: run_counts(qc, shots=512))
assert f in (3, 5), f
print(f"find_factor(15) = {f}")
print("ALL TESTS PASSED")
