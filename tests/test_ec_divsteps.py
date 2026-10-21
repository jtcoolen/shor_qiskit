"""Toy reversible divsteps: controlled inverse, phase, and clean ancillas."""

from _ec_util import ok, section
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import ec_classical as C
import ec_pointadd as PA
import ec_shor as S
import shor_stats as ST
from ec_divsteps import mod_inv
from ec_sim import Machine, run


def main():
    section("controlled divsteps inverse, exhaustive toy units")
    for p in (3, 5, 7):
        n = p.bit_length()
        m = Machine("toffoli")
        c = m.alloc(1, "c")
        x, out = m.alloc(n, "x"), m.alloc(n, "out")
        mod_inv(m, x, out, p, c[0])
        for ctl in (0, 1):
            for value in range(p):
                if ctl and not value:
                    continue
                result = run(m, {c: ctl, x: value})
                expected = pow(value, -1, p) if ctl else 0
                assert result(out) == expected, (p, ctl, value, result(out))
                assert result(x) == value and result(c) == ctl
        ok(f"p={p}: exact inverse and ancilla cleanup on controlled basis states")

    section("toy-width boundary primes and signed-register limits")
    for p in (11, 13, 17, 31, 61, 127):
        n = p.bit_length()
        m = Machine("toffoli")
        c, x, out = (m.alloc(size, label) for size, label in (
            (1, "c"), (n, "x"), (n, "out")))
        mod_inv(m, x, out, p, c[0])
        for value in sorted({1, 2, p // 2, p - 2, p - 1}):
            result = run(m, {c: 1, x: value})
            assert result(out) == pow(value, -1, p)
        ok(f"p={p}: boundary inputs, exact inverse and clean ancillas")

    section("Aer coherence and phase on an enabled/disabled superposition")
    m = Machine("and")
    c, x, out = (m.alloc(size, label) for size, label in (
        (1, "c"), (2, "x"), (2, "out")))
    mod_inv(m, x, out, 3, c[0])
    qc = QuantumCircuit(*m.qc.qregs)
    qc.x(x[0])
    qc.h(c[0])
    qc.compose(m.qc, inplace=True)
    qc.cx(out[0], c[0])
    qc.h(out[0])
    qc.measure_all()
    sim = AerSimulator(method="matrix_product_state", device="CPU")
    counts = sim.run(
        transpile(qc, sim, optimization_level=0), shots=64).result().get_counts()
    expected = 1 << qc.find_bit(x[0]).index
    assert {int(bits.replace(" ", ""), 2) for bits in counts} == {expected}, counts
    ok("controlled inverse has correct interference and no leaked ancilla")

    section("affine controlled division and point addition")
    for p in (5, 7):
        n = p.bit_length()
        m = Machine("toffoli")
        c = m.alloc(1, "c")
        x, y, out = (m.alloc(n, label) for label in ("x", "y", "out"))
        PA._moddiv(m, c[0], x, y, out, p, "reference", None, "divsteps")
        for ctl in (0, 1):
            for value in range(1, p):
                for numerator in range(p):
                    result = run(m, {c: ctl, x: value, y: numerator})
                    expected = numerator * pow(value, -1, p) % p if ctl else 0
                    assert result(out) == expected, (p, ctl, value, numerator)
        ok(f"p={p}: division exact, including disabled control")

    curve = C.CLASSIQ
    pts = [P for P in curve.points() if not P.inf]
    addend = pts[0]
    m = Machine("toffoli")
    c = m.alloc(1, "c")
    x, y = m.alloc(3, "x"), m.alloc(3, "y")
    PA.point_add_ctrl(
        m, c[0], x, y, addend.x, addend.y, 7, inversion="divsteps")
    checked = 0
    for point in pts:
        if C.point_add_exceptional(curve, point, addend):
            continue
        for ctl in (0, 1):
            result = run(m, {c: ctl, x: point.x, y: point.y})
            expected = curve.add(point, addend) if ctl else point
            assert (result(x), result(y)) == (expected.x, expected.y)
            checked += 1
    ok(f"p=7: {checked} controlled point additions, ancillas clean")

    section("full arithmetic ECDLP oracle with divsteps")
    P, Q = C.CLASSIQ_G, C.CLASSIQ_Q
    qc, info = S.ecdlp_circuit(
        curve, P, Q, curve.point_order(P), oracle="arith", m_bits=1,
        inversion="divsteps")
    scalars = info["regs"][:2]
    x, y = info["regs"][2:]
    for u in range(2):
        for v in range(2):
            expected = curve.add(info["offset"], curve.add(
                curve.mul(u, P), curve.mul(v, Q)))
            result = run(info["machine"], {scalars[0]: u, scalars[1]: v})
            assert (result(x), result(y)) == (expected.x, expected.y)
    assert qc.count_ops()["measure"] == 2
    ok("full QFT/readout circuit has all four correct arithmetic basis outputs")

    section("Aer end to end: toy ECDLP with arithmetic divsteps")
    small = C.Curve(5, 0, 1, "p5")
    base = next(P for P in small.points()
                if not P.inf and small.point_order(P) == 3)
    double = small.mul(2, base)
    counts, candidates, info = S.run_ecdlp_arith(
        small, base, double, 3, inversion="divsteps",
        m_bits=2, shots=1024, seed=5)
    q = 1 << info["m_bits"]
    exact = ST.ecdlp_probs(3, 2, info["m_bits"])
    measured = ST.empirical(S.pair_counts(counts), (q, q))
    tvd, bound = ST.tvd(measured, exact), ST.tvd_null(exact, 1024)
    assert candidates[0][0] == 2 and tvd <= bound
    ok(f"{info['qubits']}-qubit circuit on {info['aer']}: "
       f"k=2, TVD {tvd:.3f} <= {bound:.3f}")


if __name__ == "__main__":
    main()
    print("\ntest_ec_divsteps: all passed")
