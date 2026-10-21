"""End-to-end affine checks for signed Appendix A arithmetic decoding."""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

import ec_classical as C
import ec_pointadd as PA
import ec_shor as S
import shor_stats as ST
from _ec_util import ok, section
from ec_divsteps import mod_inv
from ec_sim import Machine, run


def main():
    section("signed arithmetic inversion with coherent control and cleanup")
    for p in (3, 5, 7, 11, 13, 17, 31, 61, 127):
        m = Machine("toffoli")
        c = m.alloc(1, "control")
        x, out = m.alloc(p.bit_length(), "x"), m.alloc(p.bit_length(), "out")
        mod_inv(m, x, out, p, c[0], history="arithmetic")
        values = range(p) if p <= 31 else {0, 1, 2, p // 2, p - 1}
        for value in values:
            for control in (0, 1):
                if control and value == 0:
                    continue
                result = run(m, {x: value, c: control})
                expected = pow(value, -1, p) if control else 0
                assert (result(out), result(x), result(c)) == (
                    expected, value, control)
        ok(f"p={p}: signed inverse on controlled branches, all scratch clean")

    section("final round keeps its decisions across the copy-out")
    m = Machine("and")
    c = m.alloc(1, "control")
    x, out = m.alloc(3, "x"), m.alloc(3, "out")
    mod_inv(m, x, out, 5, c[0], history="arithmetic")
    prep = StatePreparation([0.5 if 0 < v < 5 else 0 for v in range(8)])
    qc = QuantumCircuit(*m.qc.qregs)
    qc.h(c[0])
    qc.append(prep, x)
    qc.compose(m.qc, inplace=True)
    for value in range(1, 5):
        flips = [x[j] for j in range(3) if not (value >> j) & 1]
        for wire in flips:
            qc.x(wire)
        for i in range(3):
            if (pow(value, -1, 5) >> i) & 1:
                qc.mcx([c[0], *x], out[i])
        for wire in flips:
            qc.x(wire)
    qc.append(prep.inverse(), x)
    qc.h(c[0])
    qc.measure_all()
    sim = AerSimulator(method="matrix_product_state", device="CPU")
    counts = sim.run(transpile(qc, sim, optimization_level=0),
                     shots=64, seed_simulator=5).result().get_counts()
    assert {int(key.replace(" ", ""), 2) for key in counts} == {0}, counts
    ok("p=5 Aer CPU: controlled unit superposition returns to |0>")

    section("controlled division, including the disabled branch")
    for p in (5, 7):
        m = Machine("toffoli")
        control = m.alloc(1, "control")
        x, numerator, out = (m.alloc(p.bit_length(), label)
                             for label in ("divisor", "numerator", "out"))
        PA._moddiv(m, control[0], x, numerator, out, p,
                   "reference", None, "divsteps-arithmetic")
        for value in range(1, p):
            for number in range(p):
                for enabled in (0, 1):
                    result = run(m, {control: enabled, x: value,
                                     numerator: number})
                    expected = number * pow(value, -1, p) % p if enabled else 0
                    assert result(out) == expected, (p, value, number, enabled)
        ok(f"p={p}: controlled quotient and erased work registers")

    section("nonexceptional in-place affine point addition")
    curve = C.CLASSIQ
    points = [point for point in curve.points() if not point.inf]
    addend = points[0]
    m = Machine("toffoli")
    control = m.alloc(1, "control")
    x, y = m.alloc(3, "x"), m.alloc(3, "y")
    PA.point_add_ctrl(m, control[0], x, y, addend.x, addend.y, 7,
                      inversion="divsteps-arithmetic")
    checked = 0
    for point in points:
        if C.point_add_exceptional(curve, point, addend):
            continue
        for enabled in (0, 1):
            result = run(m, {control: enabled, x: point.x, y: point.y})
            expected = curve.add(point, addend) if enabled else point
            assert (result(x), result(y)) == (expected.x, expected.y)
            checked += 1
    ok(f"p=7: {checked} in-place additions with clean ancillas")

    section("full QFT/readout arithmetic ECDLP on Aer CPU")
    curve = C.Curve(5, 0, 1, "p5")
    base = next(point for point in curve.points()
                if not point.inf and curve.point_order(point) == 3)
    double = curve.mul(2, base)
    circuit, info = S.ecdlp_circuit(
        curve, base, double, 3, oracle="arith", m_bits=2,
        inversion="divsteps-arithmetic")
    kr, lr, x, y = info["regs"]
    for u in range(4):
        for v in range(4):
            result = run(info["machine"], {kr: u, lr: v})
            expected = curve.add(
                info["offset"], curve.mul((u + 2 * v) % 3, base))
            assert (result(x), result(y)) == (expected.x, expected.y)
    assert circuit.count_ops()["measure"] == 4
    counts, candidates, details = S.run_ecdlp_arith(
        curve, base, double, 3, inversion="divsteps-arithmetic",
        m_bits=2, shots=1024, seed=5)
    exact = ST.ecdlp_probs(3, 2, 2)
    tvd = ST.tvd(ST.empirical(S.pair_counts(counts), (4, 4)), exact)
    bound = ST.tvd_null(exact, 1024)
    assert candidates[0][0] == 2 and tvd <= bound
    ok(f"{details['qubits']}-qubit Aer {details['aer']} oracle recovered "
       f"k=2, TVD {tvd:.3f} <= {bound:.3f}")


if __name__ == "__main__":
    main()
    print("\ntest_ec_divsteps_arithmetic: all passed")
