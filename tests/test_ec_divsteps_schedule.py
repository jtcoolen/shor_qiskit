"""Opt-in bounded divsteps schedules signal every unfinished toy branch."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from _ec_util import ok, section
from ec_divsteps import mod_inv
from ec_divsteps_reference import walk
from ec_sim import Machine, run


def main():
    section("bounded schedule failures are flagged, never false inverses")
    for p in (5, 7, 11):
        for bound in range(1, max(walk(w, p)[1] for w in range(1, p)) + 1):
            m = Machine("toffoli")
            control, x, out, failure = (
                m.alloc(1, "control"), m.alloc(p.bit_length(), "input"),
                m.alloc(p.bit_length(), "inverse"), m.alloc(1, "failure"))
            mod_inv(m, x, out, p, control[0], history="arithmetic",
                    round_limit=bound, failure=failure[0])
            misses = 0
            for enabled in (0, 1):
                for value in range(1 << p.bit_length()):
                    result = run(m, {control: enabled, x: value})
                    failed = int(bool(enabled) and (
                        value == 0 or value >= p
                        or walk(value, p)[1] > bound))
                    expected = pow(value, -1, p) if enabled and not failed else 0
                    assert (result(out), result(failure),
                            result(x), result(control)) == (
                                expected, failed, value, enabled), (
                                    p, bound, enabled, value)
                    misses += failed
            ok(f"p={p}, steps={bound}: {misses}/{1 << p.bit_length()} "
               "representable enabled inputs failed, "
               "all scratch clean")

    section("Aer phase on a success/failure superposition")
    m = Machine("and")
    control, x, out, failure = (
        m.alloc(1, "control"), m.alloc(3, "input"),
        m.alloc(3, "inverse"), m.alloc(1, "failure"))
    mod_inv(m, x, out, 5, control[0], history="arithmetic",
            round_limit=4, failure=failure[0])
    qc = QuantumCircuit(*m.qc.qregs)
    qc.x(control[0])
    qc.x(x[1])
    qc.h(x[0])
    qc.compose(m.qc, inplace=True)
    qc.cx(x[0], out[1])
    qc.x(failure[0])
    qc.cx(x[0], failure[0])
    qc.h(x[0])
    qc.x(x[1])
    qc.x(control[0])
    qc.measure_all()
    sim = AerSimulator(method="matrix_product_state", device="CPU")
    counts = sim.run(transpile(qc, sim, optimization_level=0),
                     shots=32, seed_simulator=5).result().get_counts()
    assert {int(key.replace(" ", ""), 2) for key in counts} == {0}, counts
    ok("bounded success/failure paths interfere with no leaked work")

    section("invalid truncated schedules fail before circuit construction")
    m = Machine()
    x, out, status = m.alloc(3, "x"), m.alloc(3, "out"), m.alloc(1, "status")
    for opts in ({"history": "arithmetic", "round_limit": 0,
                  "failure": status[0]},
                 {"history": "arithmetic", "round_limit": 2},
                 {"history": "decoded", "round_limit": 2,
                  "failure": status[0]}):
        before = len(m.qc.data)
        try:
            mod_inv(m, x, out, 5, **opts)
        except ValueError:
            assert len(m.qc.data) == before
        else:
            raise AssertionError(opts)


if __name__ == "__main__":
    main()
    print("\ntest_ec_divsteps_schedule: all passed")
