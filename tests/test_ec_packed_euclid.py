"""Check the long-division quantum primitive before building a packed EEA."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from _ec_util import ok, section
from ec_packed_euclid import binary_divmod, rotate_word_if, shift_work2, swap_at
from ec_sim import Machine, run


def main():
    section("coherent Work2 shifts with moving position counter")
    for direction, steps in (("left", 1), ("right", 1),
                             ("left", 2), ("right", 2)):
        m = Machine("toffoli")
        word, shift, control = (
            m.alloc(3, "word"), m.alloc(3, "shift"), m.alloc(1, "control"))
        shift_work2(m, word, shift, control[0], direction, steps)
        for value in range(8):
            for old_shift in range(2, 5):
                for enabled in (0, 1):
                    result = run(m, {word: value, shift: old_shift,
                                     control: enabled})
                    count = steps if enabled else 0
                    if direction == "left":
                        expected_word = ((value << count) |
                                         (value >> (3 - count))) & 7
                        expected_shift = old_shift + count
                    else:
                        expected_word = ((value >> count) |
                                         (value << (3 - count))) & 7
                        expected_shift = old_shift - count
                    assert (result(word), result(shift), result(control)) == (
                        expected_word, expected_shift, enabled)
        ok(f"Work2 {direction} x{steps}: counters and words clean")

    m = Machine("and")
    word, shift, control = (
        m.alloc(3, "word"), m.alloc(3, "shift"), m.alloc(1, "control"))
    shift_work2(m, word, shift, control[0], "left")
    qc = QuantumCircuit(*m.qc.qregs)
    qc.x(word[0])
    qc.x(shift[1])
    qc.h(control[0])
    qc.compose(m.qc, inplace=True)
    got = np.asarray(Statevector.from_instruction(qc))
    want = np.zeros_like(got)
    for value, counter, enabled in ((1, 2, 0), (2, 3, 1)):
        basis = sum(((number >> bit) & 1) << qc.find_bit(reg[bit]).index
                    for reg, number in ((word, value), (shift, counter),
                                        (control, enabled))
                    for bit in range(len(reg)))
        want[basis] = 2 ** -0.5
    assert abs(np.vdot(got, want)) > 1 - 1e-9
    assert np.max(np.abs(got[want == 0])) < 1e-9
    ok("Work2 counter and rotation preserve coherent relative phases")
    for bad_word, bad_steps in (([], 1), (word, 1.0), (word, True)):
        try:
            shift_work2(m, bad_word, shift, control[0], "left", bad_steps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shift should be rejected")
    paper = Machine("and")
    word8, small_shift, paper_control = (
        paper.alloc(8, "paper_width_work2"),
        paper.alloc(3, "paper_shift"), paper.alloc(1, "paper_control"))
    shift_work2(paper, word8, small_shift, paper_control[0], "left")
    assert run(paper, {word8: 1, small_shift: 4,
                       paper_control: 1})(small_shift) == 5
    ok("paper-sized eight-wire word accepts a three-bit shift counter")

    section("conditional packed-workspace circular shifts")
    for width in (2, 3, 4):
        for direction in ("left", "right"):
            m = Machine("toffoli")
            word, control = m.alloc(width, "word"), m.alloc(1, "control")
            rotate_word_if(m, word, control[0], direction)
            for value in range(1 << width):
                for enabled in (0, 1):
                    result = run(m, {word: value, control: enabled})
                    rotated = (((value << 1) | (value >> (width - 1)))
                               & ((1 << width) - 1) if direction == "left"
                               else (value >> 1) | ((value & 1) << (width - 1)))
                    assert (result(word), result(control)) == (
                        rotated if enabled else value, enabled)
            ok(f"width={width} {direction}: every word/control exact")
    m = Machine("and")
    word, control = m.alloc(3, "word"), m.alloc(1, "control")
    rotate_word_if(m, word, control[0], "left")
    qc = QuantumCircuit(*m.qc.qregs)
    qc.x(word[0])
    qc.h(control[0])
    qc.compose(m.qc, inplace=True)
    got = np.asarray(Statevector.from_instruction(qc))
    want = np.zeros_like(got)
    for value, enabled in ((1, 0), (2, 1)):
        basis = (value << qc.find_bit(word[0]).index) | (
            enabled << qc.find_bit(control[0]).index)
        want[basis] = 2 ** -0.5
    assert abs(np.vdot(got, want)) > 1 - 1e-9
    ok("coherent conditional rotation preserves phase and word layout")
    try:
        rotate_word_if(m, [word[0], word[1], word[0]], control[0], "right")
    except ValueError:
        pass
    else:
        raise AssertionError("repeated word bits must be rejected")

    section("coherent indexed swap for packed-word boundaries")
    for width in (2, 3):
        m = Machine("toffoli")
        word = m.alloc(width, "word")
        index = m.alloc((width - 1).bit_length(), "index")
        flag = m.alloc(1, "flag")
        swap_at(m, word, index, flag[0])
        for value in range(1 << width):
            for address in range(1 << len(index)):
                for initial_flag in (0, 1):
                    result = run(m, {word: value, index: address, flag: initial_flag})
                    expected_word, expected_flag = value, initial_flag
                    if address < width:
                        expected_flag = (value >> address) & 1
                        expected_word ^= (initial_flag ^ expected_flag) << address
                    assert (result(word), result(index), result(flag)) == (
                        expected_word, address, expected_flag)
        ok(f"width={width}: all words, flags and addresses, clean scratch")

    m = Machine("and")
    word = m.alloc(3, "word")
    index = m.alloc(2, "index")
    flag = m.alloc(1, "flag")
    swap_at(m, word, index, flag[0])
    qc = QuantumCircuit(*m.qc.qregs)
    qc.x(word[1])
    qc.x(flag[0])
    for bit in index:
        qc.h(bit)
    qc.compose(m.qc, inplace=True)
    got = np.asarray(Statevector.from_instruction(qc))
    want = np.zeros_like(got)
    for address in range(4):
        value, new_flag = (3, 0) if address == 0 else (
            (2, 1) if address in (1, 3) else (6, 0))
        assignments = {word: value, index: address, flag: new_flag}
        basis = sum(((number >> bit) & 1) << qc.find_bit(reg[bit]).index
                    for reg, number in assignments.items()
                    for bit in range(len(reg)))
        want[basis] = 0.5
    assert abs(np.vdot(got, want)) > 1 - 1e-9
    assert np.max(np.abs(got[want == 0])) < 1e-9
    ok("indexed swap preserves relative phases and clears selection scratch")

    section("coherent compare/subtract binary long division")
    for n in (2, 3, 4):
        m = Machine("toffoli")
        a, b, q, r = (m.alloc(n, label)
                      for label in ("numerator", "divisor", "quotient", "remainder"))
        binary_divmod(m, a, b, q, r)
        for aa in range(1 << n):
            for bb in range(1, 1 << n):
                result = run(m, {a: aa, b: bb})
                assert (result(a), result(b)) == (aa, bb)
                assert (result(q), result(r)) == divmod(aa, bb), (n, aa, bb)
        ok(f"n={n}: all nonzero divisors and numerators, clean scratch")

    section("superposition phase and ancilla cleanup")
    m = Machine("and")
    a, b, q, r = (m.alloc(2, label)
                  for label in ("numerator", "divisor", "quotient", "remainder"))
    binary_divmod(m, a, b, q, r)
    qc = QuantumCircuit(*m.qc.qregs)
    qc.x(b[0])
    qc.x(b[1])
    for bit in a:
        qc.h(bit)
    qc.compose(m.qc, inplace=True)
    got = np.asarray(Statevector.from_instruction(qc))
    want = np.zeros_like(got)
    for aa in range(4):
        word = {a: aa, b: 3, q: aa // 3, r: aa % 3}
        index = sum(((value >> bit) & 1) << qc.find_bit(reg[bit]).index
                    for reg, value in word.items() for bit in range(2))
        want[index] = 0.5
    assert abs(np.vdot(got, want)) > 1 - 1e-9
    assert np.max(np.abs(got[want == 0])) < 1e-9
    ok("n=2: exact amplitudes/phases, no dirty or entangled scratch")


if __name__ == "__main__":
    main()
    print("\ntest_ec_packed_euclid: all passed")
