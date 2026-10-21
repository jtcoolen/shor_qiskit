"""Exhaustive invariants and branch recovery for Gu et al. Eq. (6)/(A11)."""

from math import gcd

from _ec_util import ok, section
from ec_divsteps_reference import State, decode, quotient, reverse_step, step, walk
from ec_euclid_reference import (binary_divmod, inverse as euclid_inverse,
                                 pack_state, packed_trace, unpack_state)


def main():
    section("exact divstep recurrence, decision decoder, and inverse")
    for p in (3, 5, 7, 11, 13, 17, 31, 61, 127):
        for w in range(1, p):
            if gcd(w, p) != 1:
                continue
            state = State(0, p, w, 0, 1)
            limit = 4 * p.bit_length() ** 2
            path = [state]
            for t in range(limit):
                if state.g == 0:
                    break
                updated, decision = step(state)
                assert updated.f * updated.gv - updated.g * updated.fv == p
                assert updated.f % 2 and updated.gv % 2 and not updated.fv % 2
                assert updated.fv * w % p == (1 << (t + 1)) * updated.f % p
                assert updated.gv * w % p == (1 << (t + 1)) * updated.g % p
                assert decode(updated) == decision, (p, w, state, updated)
                assert reverse_step(updated) == state, (p, w, state, updated)
                path.append(updated)
                state = updated
            assert state.g == 0 and abs(state.f) == 1
            end, active = walk(w, p)
            assert (end, active) == (state, len(path) - 1)
            assert state.f * state.fv * pow(2, -active, p) % p == pow(w, -1, p)
            for a in range(p):
                assert quotient(w, a, p) == a * pow(w, -1, p) % p
        ok(f"p={p}: every unit, numerator, decision and reverse step exact")
    for w, p, limit in ((0, 7, None), (7, 7, None), (3, 9, None), (1, 7, 0)):
        try:
            walk(w, p, limit)
        except ValueError:
            pass
        else:
            raise AssertionError((w, p, limit))

    section("exact quotient-Euclidean reference and binary long division")
    for numerator in range(128):
        for denominator in range(1, 128):
            assert binary_divmod(numerator, denominator) == divmod(
                numerator, denominator)
    for p in (3, 5, 7, 9, 11, 13, 17, 31, 61, 127):
        for w in range(1, p):
            if gcd(w, p) == 1:
                assert euclid_inverse(w, p) == pow(w, -1, p)
                assert euclid_inverse(w, p) == quotient(w, 1, p)
            else:
                try:
                    euclid_inverse(w, p)
                except ValueError:
                    pass
                else:
                    raise AssertionError((w, p))
    ok("binary scans and signed Euclidean coefficients exact for toy inputs")

    section("two-word packed boundary contract")
    for t in range(8):
        for q in range(4):
            for rem in range(1, 8):
                for prev_t in range(4):
                    for next_rem in range(1, 4):
                        values = (t, q, rem, prev_t, next_rem)
                        assert unpack_state(pack_state(*values, 12)) == values
    for p in (3, 5, 7, 11, 13, 17, 31, 61, 127):
        for w in range(1, p):
            for snapshot in packed_trace(w, p):
                t, q, rem, prev_t, next_rem = unpack_state(snapshot)
                assert snapshot.width == p.bit_length() + 3
                assert t >= 0 and q >= 0 and rem >= 0
                assert prev_t >= 0 and next_rem >= 0
                assert snapshot.work1 < 1 << snapshot.width
                assert snapshot.work2 < 1 << snapshot.width
                assert snapshot.t_bits + 1 + snapshot.q_bits + snapshot.r_bits <= snapshot.width
                assert snapshot.t_prev_bits + snapshot.r_next_bits <= snapshot.width
    ok("all toy units fit two opposed-end packed words at each boundary")


if __name__ == "__main__":
    main()
    print("\ntest_ec_divsteps_reference: all passed")
