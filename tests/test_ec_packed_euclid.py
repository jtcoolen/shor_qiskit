"""Check the long-division quantum primitive before building a packed EEA."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from _ec_util import ok, section
from ec_packed_euclid import binary_divmod, swap_at
from ec_sim import Machine, run


def main():
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
