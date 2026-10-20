import random

import ec_classical as C
import ec_eea as E
from ec_sim import Machine, SimError, run


def _rates(q, inputs):
    n = q.bit_length()
    results = [
        C.eea_dialog_register_sharing(q, x, n)
        for x in inputs
    ]
    return results


def test_register_sharing_exhaustive_success():
    for q in (251, 4093):
        results = _rates(q, range(1, q))
        assert all(not result.overflow for result in results)
        assert all(result.converged for result in results)
        assert len(results) == q - 1


def test_register_sharing_large_prime_overflow_rate_is_explicit():
    q = 65521
    results = _rates(q, range(1, q))
    overflows = sum(result.overflow for result in results)
    gcd_failures = sum(not result.converged for result in results)
    assert overflows == 2
    assert gcd_failures == 0
    assert abs((q - 1 - overflows) / (q - 1) - 0.99996947) < 1e-8


def test_register_sharing_seeded_random_inputs():
    q = 4093
    rng = random.Random(1128)
    inputs = [rng.randrange(1, q) for _ in range(500)]
    results = _rates(q, inputs)
    assert all(not result.overflow and result.converged for result in results)


def test_register_sharing_circuit_reduces_width_and_matches_dialog():
    q = 251
    n = q.bit_length()
    iters = C.eea_iterations(n)
    baseline = Machine("and")
    x_baseline = baseline.alloc(n, "x")
    E.dialog(baseline, x_baseline, q, iters)

    shared = Machine("and")
    x_shared = shared.alloc(n, "x")
    recs, u, v = E.dialog(
        shared, x_shared, q, iters, register_sharing=True)
    assert shared.qc.num_qubits < baseline.qc.num_qubits

    rng = random.Random(20261128)
    for value in [1, 2, 3, 7, 42, 128, 250] + [
            rng.randrange(1, q) for _ in range(20)]:
        state = run(shared, {x_shared: value})
        expected, final_u, final_v = C.eea_dialog(q, value, iters)
        assert state(u) == final_u == 1
        assert state(v) == final_v == 0
        assert [(state(a), state(b)) for a, b in recs] == expected


def test_register_sharing_with_compressed_records_matches_dialog():
    q = 251
    n = q.bit_length()
    iters = C.eea_iterations(n)
    m = Machine("and")
    x = m.alloc(n, "x")
    recs, u, v = E.dialog(
        m, x, q, iters, compress=True, register_sharing=True)

    for value in (1, 2, 3, 7, 42, 128, 250):
        state = run(m, {x: value})
        expected, final_u, final_v = C.eea_dialog(q, value, iters)
        decoded = []
        full = len(recs) - len(recs) % 3
        for start in range(0, full, 3):
            group = [bit for pair in recs[start:start + 3]
                     for reg in pair for bit in reg]
            packed = sum(state([bit]) << i for i, bit in enumerate(group[:5]))
            decoded.extend(C.decompress_triple(packed))
        decoded.extend((state([a[0]]), state([b[0]])) for a, b in recs[full:])
        assert state(u) == final_u == 1
        assert state(v) == final_v == 0
        assert decoded == expected


def test_register_sharing_inplace_multiplication():
    q = 31
    n = q.bit_length()
    baseline = Machine("and")
    xb, yb = baseline.alloc(n, "x"), baseline.alloc(n, "y")
    E.inplace_mul(baseline, xb, yb, q)

    shared = Machine("and")
    x, y = shared.alloc(n, "x"), shared.alloc(n, "y")
    E.inplace_mul(shared, x, y, q, register_sharing=True)
    assert shared.qc.num_qubits < baseline.qc.num_qubits

    for xv in range(1, q):
        for yv in (0, 1, q - 1):
            state = run(shared, {x: xv, y: yv})
            assert state(x) == xv
            assert state(y) == xv * yv % q

    inverse = Machine("and")
    xi, yi = inverse.alloc(n, "x"), inverse.alloc(n, "y")
    E.inplace_div(inverse, xi, yi, q, register_sharing=True)
    for xv in (1, 2, 7, q - 1):
        for yv in (0, 1, q - 1):
            state = run(inverse, {xi: xv, yi: yv})
            assert state(xi) == xv
            assert state(yi) == yv * pow(xv, -1, q) % q


def test_register_sharing_overflow_raises_instead_of_truncating():
    m = Machine("and")
    x = m.alloc(8, "x")
    E.dialog(m, x, 251, iters=4, register_sharing=True, c_pad=0)
    try:
        run(m, {x: 8})
    except SimError:
        pass
    else:
        raise AssertionError("overflowing layout was silently accepted")


def test_register_sharing_overflow_and_gcd_failure_are_separate():
    result = C.eea_dialog_register_sharing(
        15, 5, n=4, c_pad=100, iters=C.eea_iterations(4))
    assert not result.overflow
    assert not result.converged


def main():
    test_register_sharing_exhaustive_success()
    test_register_sharing_large_prime_overflow_rate_is_explicit()
    test_register_sharing_seeded_random_inputs()
    test_register_sharing_circuit_reduces_width_and_matches_dialog()
    test_register_sharing_with_compressed_records_matches_dialog()
    test_register_sharing_inplace_multiplication()
    test_register_sharing_overflow_raises_instead_of_truncating()
    test_register_sharing_overflow_and_gcd_failure_are_separate()
    print("EEA register-sharing tests passed")


if __name__ == "__main__":
    main()
