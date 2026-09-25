"""The resource counter, checked against closed forms and against Qiskit.

The closed forms are derived by reading the circuits, and they are what the
notebook quotes to explain the counts: if a formula and a count disagree here,
one of them is wrong.  With n = ceil(log2 N) and m = n + 1 (the adder width,
value plus sign bit):

  QFT on n qubits          n H, n(n-1)/2 controlled phases, floor(n/2) swaps
  Beauregard c_add_mod     2m + 3 Toffolis, 5m(m-1) + 2m controlled phases
  CDKM rc_c_add_mod        18m + 3 Toffolis, no rotations at all
  c_ua (Fourier)           4n^2 + 11n Toffolis        (2n modular adds + n CSWAP)
  rc_c_ua                  36n^2 + 43n Toffolis
  order_circuit            4n + 2 qubits, 8n^3 + 22n^2 Toffolis
  order_circuit_1c         2n + 3 qubits, same arithmetic, t measurements
"""
import math
import random

from _ec_util import ok, section

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister

import ec_classical as C
import ec_cost as CO
import ec_kaliski as K
import ec_shor as S
import resources as R
from ec_sim import Machine
from onectrl import order_circuit_1c
from rc_adder import rc_c_add_mod, rc_c_ua
from shor_essentials import c_add_mod, c_ua, order_circuit, qft


def one(gate_fn, n=4):
    qc = QuantumCircuit(n)
    gate_fn(qc)
    return R.count(qc)


def test_prices():
    section("1. each primitive is priced by its standard decomposition")
    pi = math.pi
    cases = [
        ("CCX",                 lambda q: q.ccx(0, 1, 2),          dict(toffoli=1, T=7)),
        ("CSWAP",               lambda q: q.cswap(0, 1, 2),        dict(toffoli=1, T=7, cx=2)),
        ("MCX, 4 controls",     lambda q: q.mcx([0, 1, 2, 3], 4),  dict(toffoli=5, T=35, T_and=15, ancillas=2)),
        ("MCP, 2 controls",     lambda q: q.mcp(0.3, [0, 1], 2),   dict(toffoli=2, rotations=3, T_and=4)),
        ("T",                   lambda q: q.t(0),                  dict(T=1)),
        ("P(pi/4) = T",         lambda q: q.p(pi / 4, 0),          dict(T=1, rotations=0)),
        ("P(pi/2) = S",         lambda q: q.p(pi / 2, 0),          dict(T=0, rotations=0)),
        ("P(0.3)",              lambda q: q.p(0.3, 0),             dict(T=0, rotations=1)),
        ("CP(pi) = CZ",         lambda q: q.cp(pi, 0, 1),          dict(T=0, rotations=0, cx=1)),
        ("CP(pi/2) = CS",       lambda q: q.cp(pi / 2, 0, 1),      dict(T=3, rotations=0, cx=2)),
        ("CP(pi/4)",            lambda q: q.cp(pi / 4, 0, 1),      dict(T=0, rotations=3, cx=2)),
    ]
    for label, fn, want in cases:
        got = one(fn, 5).as_dict()
        for k, v in want.items():
            assert got[k] == v, (label, k, got[k], v)
        ok(f"{label:18s} " + ", ".join(f"{k} {v}" for k, v in want.items()))


def test_agrees_with_qiskit():
    section("2. T pricing agrees with Qiskit's own Clifford+T decomposition")
    rnd = random.Random(1)
    gates = {"ccx": lambda qc, q: qc.ccx(*q), "cswap": lambda qc, q: qc.cswap(*q),
             "cx": lambda qc, q: qc.cx(q[0], q[1]), "x": lambda qc, q: qc.x(q[0]),
             "h": lambda qc, q: qc.h(q[0]), "t": lambda qc, q: qc.t(q[0]),
             "tdg": lambda qc, q: qc.tdg(q[0]), "s": lambda qc, q: qc.s(q[0])}
    for _ in range(20):
        qc = QuantumCircuit(5)
        for _ in range(30):
            gates[rnd.choice(list(gates))](qc, rnd.sample(range(5), 3))
        tq = transpile(qc, basis_gates=["h", "t", "tdg", "s", "sdg", "cx", "x", "z"],
                       optimization_level=0)
        ops = tq.count_ops()
        assert ops.get("t", 0) + ops.get("tdg", 0) == R.count(qc).T
    ok("20 random Toffoli/Clifford+T circuits: identical T-counts")

    for theta in (math.pi / 4, 0.3, -math.pi / 8):
        qc = QuantumCircuit(2)
        qc.cp(theta, 0, 1)
        tq = transpile(qc, basis_gates=["p", "cx"], optimization_level=0)
        angles = sorted(abs(float(i.operation.params[0])) for i in tq.data
                        if i.operation.name == "p")
        assert angles == [abs(theta) / 2] * 3 and tq.count_ops()["cx"] == 2
    ok("CP(theta) -> three P(+-theta/2) and two CNOTs, as Qiskit decomposes it")

    for p in (31, 251):
        n = p.bit_length()
        m = Machine("and")
        K.mod_inv(m, m.alloc(n, "x"), m.alloc(n, "o"), p)
        a, b = CO.count(m), R.count(m)
        assert (b.toffoli + b.and_, b.T) == (a["toffoli_paper"], a["t"])
    ok("ECDLP circuits: identical to ec_cost's Toffoli and T counts (inversion mod 31, 251)")


def test_closed_forms():
    section("3. the counts are the closed forms")
    for n in range(1, 9):
        qc = QuantumCircuit(n)
        qc.append(qft(n), range(n))
        r = R.count(qc)
        assert (r.h, r.cphase, r.cx) == (n, n * (n - 1) // 2, n * (n - 1) + 3 * (n // 2))
        assert (r.T, r.rotations) == (3 * (n - 1), 3 * (n - 1) * (n - 2) // 2)
    ok("QFT_n: n H + n(n-1)/2 CP(pi/2^d); the d=1 ones are CS (3 T), "
       "the rest are 3 small rotations each")

    for N in (15, 21, 33, 63):
        n = math.ceil(math.log2(N))
        m = n + 1
        c, y, sf, a = (QuantumRegister(2), QuantumRegister(n), QuantumRegister(2),
                       QuantumRegister(n + 2))
        qc = QuantumCircuit(c, y, sf)
        c_add_mod(qc, list(c), 7 % N, list(y), sf[0], sf[1], N)
        r = R.count(qc)
        assert (r.toffoli, r.cphase) == (2 * m + 3, 5 * m * (m - 1) + 2 * m), N
        qc = QuantumCircuit(c, y, sf, a)
        rc_c_add_mod(qc, list(c), 7 % N, list(y), sf[0], sf[1], N, list(a))
        r = R.count(qc)
        assert (r.toffoli, r.cphase, r.rotations) == (18 * m + 3, 0, 0), N
    ok("modular adder: Fourier 2m+3 Toffoli + 5m(m-1)+2m CP; ripple-carry 18m+3 "
       "Toffoli, zero rotations (N = 15, 21, 33, 63)")

    for N, A in ((15, 7), (21, 2), (33, 5)):
        n = math.ceil(math.log2(N))
        t = 2 * n
        c, x, y, sf, a = (QuantumRegister(1), QuantumRegister(n), QuantumRegister(n),
                          QuantumRegister(2), QuantumRegister(n + 2))
        qc = QuantumCircuit(c, x, y, sf)
        c_ua(qc, c[0], A, list(x), list(y), sf[0], sf[1], N)
        rung = R.count(qc)
        assert rung.toffoli == 4 * n * n + 11 * n
        qc = QuantumCircuit(c, x, y, sf, a)
        rc_c_ua(qc, c[0], A, list(x), list(y), sf[0], sf[1], N, list(a))
        assert R.count(qc).toffoli == 36 * n * n + 43 * n
        full, semi = R.count(order_circuit(A, N)), R.count(order_circuit_1c(A, N))
        assert (full.qubits, full.toffoli) == (4 * n + 2, 8 * n**3 + 22 * n**2)
        assert full.cphase == t * rung.cphase + t * (t - 1) // 2
        assert (semi.qubits, semi.toffoli) == (2 * n + 3, full.toffoli)
        assert semi.cphase == full.cphase - t * (t - 1) // 2
        assert (semi.measure, semi.conditional) == (t, t * (t - 1) // 2 + t)
        ok(f"N={N}: rung 4n^2+11n = {rung.toffoli} Toffoli; circuit {full.qubits} "
           f"qubits, 8n^3+22n^2 = {full.toffoli}; one-control {semi.qubits} qubits "
           f"and {t*(t-1)//2} CP traded for classically controlled phases")

    m, _ = S.ecdlp_circuit(C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q, 5, oracle="arith")
    r = R.count(m)
    assert r.rotations == 0 and r.cphase == 0
    ok(f"ECDLP arithmetic oracle, p=7: {r.toffoli + r.and_} Toffoli-eq, "
       f"{r.T} T, zero rotations -- pure Toffoli/Clifford")


if __name__ == "__main__":
    test_prices()
    test_agrees_with_qiskit()
    test_closed_forms()
    print("\ntest_resources: all passed")
