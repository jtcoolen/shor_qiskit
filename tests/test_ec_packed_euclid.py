"""Packed Euclidean primitives and the Luo Algorithms 1-3 inversion walk."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from _ec_util import ok, section
import ec_classical as C
import ec_euclid_reference as R
import ec_packed_euclid as PE
import ec_pointadd as PA
import ec_shor as S
from _ec_util import FULL, primes
from ec_packed_euclid import (binary_divmod, rotate_word_if, shift_work2,
                              swap_at, xor_at)
from ec_sim import Machine, run, simulate

# Luo et al. arXiv:2607.13816 Table 4 (p = 37, x = 13): per step
# (t, q, r, t', r', l_t, l_q, l_r', l_s, Phase1, Phase2, Iter, Sign).
TABLE4 = [
    (1, 0, 37, 0, 13, 1, 0, 4, 0, 0, 0, 0, 0),
    (1, 0, 37, 0, 13, 1, 0, 4, 1, 0, 0, 0, 0),
    (1, 0, 37, 0, 13, 1, 0, 4, 2, 0, 1, 0, 0),
    (1, 2, 11, 0, 13, 1, 1, 4, 1, 0, 1, 0, 0),
    (1, 2, 11, 0, 13, 1, 2, 4, 0, 1, 0, 0, 0),
    (1, 2, 11, 0, 13, 1, 1, 4, 1, 1, 0, 0, 0),
    (1, 0, 11, 2, 13, 1, 0, 4, 2, 1, 1, 0, 1),
    (1, 0, 11, 2, 13, 1, 0, 4, 1, 1, 1, 0, 0),
    (2, 0, 13, 1, 11, 2, 0, 4, 0, 0, 0, 1, 0),
    (2, 0, 13, 1, 11, 2, 0, 4, 1, 0, 1, 1, 0),
    (2, 1, 2, 1, 11, 2, 1, 4, 0, 1, 0, 1, 0),
    (2, 0, 2, 3, 11, 2, 0, 4, 1, 1, 1, 1, 1),
    (3, 0, 11, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0),
    (3, 0, 11, 2, 2, 2, 0, 2, 1, 0, 0, 0, 0),
    (3, 0, 11, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0),
    (3, 0, 11, 2, 2, 2, 0, 2, 3, 0, 1, 0, 0),
    (3, 4, 3, 2, 2, 2, 1, 2, 2, 0, 1, 0, 0),
    (3, 4, 3, 2, 2, 2, 2, 2, 1, 0, 1, 0, 0),
    (3, 5, 1, 2, 2, 2, 3, 2, 0, 1, 0, 0, 0),
    (3, 4, 1, 5, 2, 2, 2, 2, 1, 1, 0, 0, 0),
    (3, 4, 1, 5, 2, 2, 1, 2, 2, 1, 0, 0, 0),
    (3, 0, 1, 17, 2, 2, 0, 2, 3, 1, 1, 0, 1),
    (3, 0, 1, 17, 2, 2, 0, 2, 2, 1, 1, 0, 0),
    (3, 0, 1, 17, 2, 2, 0, 2, 1, 1, 1, 0, 0),
    (17, 0, 2, 3, 1, 5, 0, 1, 0, 0, 0, 1, 0),
    (17, 0, 2, 3, 1, 5, 0, 1, 1, 0, 0, 1, 0),
    (17, 0, 2, 3, 1, 5, 0, 1, 2, 0, 1, 1, 0),
    (17, 2, 0, 3, 1, 5, 1, 1, 1, 0, 1, 1, 0),
    (17, 2, 0, 3, 1, 5, 2, 1, 0, 1, 0, 1, 0),
    (17, 2, 0, 3, 1, 5, 1, 1, 1, 1, 0, 1, 0),
    (17, 0, 0, 37, 1, 5, 0, 1, 2, 1, 1, 1, 1),
    (17, 0, 0, 37, 1, 5, 0, 1, 1, 1, 1, 1, 0),
    (37, 0, 1, 17, 0, 6, 0, 0, 0, 0, 0, 0, 0),
    (37, 0, 1, 17, 0, 6, 0, 0, 1, 0, 0, 0, 0),
    (37, 0, 1, 17, 0, 6, 0, 0, 2, 0, 0, 0, 0),
    (37, 0, 1, 17, 0, 6, 0, 0, 3, 0, 0, 0, 0),
    (37, 0, 1, 17, 0, 6, 0, 0, 4, 0, 0, 0, 0),
]


def integer_step(d):
    """Luo's integer model (eea_model/one_iter_opt.py), independent of layout."""
    t, q, r, tp, rp, lt, lq, lrp, ls, p1, p2, it, sg = d
    if not p1:
        ls += 1 - 2 * p2
        r -= rp << ls
        sg ^= r < 0
        sg ^= p2
        if not p2 or not sg:
            r += rp << ls
    if p1 ^ p2:
        if sg and not (q >> ls) & 1:
            sg, q = 0, q + (1 << ls)
        elif not sg and (q >> ls) & 1:
            sg, q = 1, q - (1 << ls)
        lq += -1 if p1 else 1
    if p1:
        if p2 or not sg:
            tp -= t << ls
        sg ^= 1
        sg ^= tp < 0
        tp += t << ls
        ls += 1 - 2 * p2
    if not lq and lrp:
        p2 ^= sg ^ p1
        sg ^= p2
    if not ls:
        p1, p2 = p1 ^ 1, p2 ^ 1
    if not lq and not ls:
        t, tp, r, rp = tp, t, rp, r
        lt += t.bit_length() - tp.bit_length()
        lrp += rp.bit_length() - r.bit_length()
        it ^= 1
    return (t, q, r, tp, rp, lt, lq, lrp, ls, p1, p2, it, sg)


def model_row(s):
    return tuple(R.luo_decode(s)) + (s.lt, s.lq, s.lrp, s.ls, s.phase1,
                                     s.phase2, s.iter, s.sign)


def load(regs, s):
    init = {}
    for k in range(regs.width):
        init[regs.w1[k]], init[regs.w2[k]] = s.work1[k], s.work2[k]
    for reg, value in ((regs.lt, s.lt), (regs.lq, s.lq), (regs.lrp, s.lrp),
                       (regs.ls, s.ls)):
        for k, q in enumerate(reg):
            init[q] = (value >> k) & 1
    for q, value in ((regs.p1, s.phase1), (regs.p2, s.phase2),
                     (regs.iter, s.iter), (regs.sign, s.sign)):
        init[q] = value
    return init


def main():
    section("coherent indexed packed-word read")
    for width in (2, 3, 4):
        m = Machine("toffoli")
        word = m.alloc(width, "word")
        index = m.alloc((width - 1).bit_length(), "index")
        output = m.alloc(1, "output")
        xor_at(m, word, index, output[0])
        for value in range(1 << width):
            for address in range(1 << len(index)):
                for initial in (0, 1):
                    result = run(m, {word: value, index: address,
                                     output: initial})
                    expected = initial ^ ((value >> address) & 1
                                          if address < width else 0)
                    assert (result(word), result(index),
                            result(output)) == (value, address, expected)
        ok(f"width={width}: indexed read exact with out-of-range address")
    m = Machine("and")
    read_word, read_index = m.alloc(3, "word"), m.alloc(2, "index")
    read_target = m.alloc(1, "target")
    xor_at(m, read_word, read_index, read_target[0])
    read_qc = QuantumCircuit(*m.qc.qregs)
    read_qc.x(read_word[1])
    for wire in read_index:
        read_qc.h(wire)
    read_qc.compose(m.qc, inplace=True)
    amplitudes = np.asarray(Statevector.from_instruction(read_qc))
    expected = np.zeros_like(amplitudes)
    for address in range(4):
        assignments = ((read_word, 2), (read_index, address),
                       (read_target, int(address == 1)))
        basis = sum(((number >> bit) & 1) << read_qc.find_bit(reg[bit]).index
                    for reg, number in assignments for bit in range(len(reg)))
        expected[basis] = 0.5
    assert abs(np.vdot(amplitudes, expected)) > 1 - 1e-9
    assert np.max(np.abs(amplitudes[expected == 0])) < 1e-9
    ok("coherent indexed read preserves phase and clears selection scratch")

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

    luo_walk()


def luo_walk():
    section("Luo Algorithm 3 position model reproduces Table 4")
    s = R.luo_init(13, 37)
    for step, row in enumerate(TABLE4):
        assert model_row(s) == row, (step, model_row(s), row)
        if step < len(TABLE4) - 1:
            R.luo_step(s)
    assert R.luo_output(s, 37) == pow(13, -1, 37)
    ok("p=37, x=13: all 37 rows, then 13^-1 = 20")

    section("position model equals Luo's integer model; exact inverses")
    for p in primes(127 if not FULL else 257)[1:]:
        steps = R.luo_step_bound(p.bit_length())
        for x in range(p):
            s = R.luo_init(x, p)
            d = model_row(s)
            for _ in range(steps):
                R.luo_step(s)
                if d[7]:
                    d = integer_step(d)
                    assert model_row(s) == d, (p, x, model_row(s), d)
            assert d[7] == 0 or x == 0, (p, x)
            assert R.luo_output(s, p) == (pow(x, -1, p) if x else 0)
    ok("odd primes 5..127: every step to termination, 4 ceil(1.5795 n) "
       "steps suffice")

    section("gate-level Luo step equals the model on every reachable state")
    for p in (5, 7, 11, 13):
        m = Machine("toffoli")
        regs = PE.LuoRegs(m, p)
        PE.luo_step(m, regs)
        index = {q: k for k, q in enumerate(m.qc.qubits)}
        persistent = {q for reg in regs.regs() for q in reg}
        states = {}
        for x in range(p):
            for s in R.luo_run(x, p)[:-1]:
                states[s.key()] = s
        for s in states.values():
            bits = simulate(m.qc, load(regs, s), m.checks)
            want = R.luo_step(R.LuoState(**{**s.__dict__,
                                            "work1": list(s.work1),
                                            "work2": list(s.work2)}))
            assert all(bits[index[q]] == v for q, v in load(regs, want).items())
            assert not any(bits[k] for q, k in index.items()
                           if q not in persistent)
        ok(f"p={p}: {len(states)} states, {m.width} qubits, scratch clean")

    section("controlled packed-Euclid inversion with full uncompute")
    for p in (5, 7, 11, 13):
        m = Machine("toffoli")
        c = m.alloc(1, "control")
        x, out = m.alloc(p.bit_length(), "x"), m.alloc(p.bit_length(), "out")
        assert PE.mod_inv(m, x, out, p, c[0]) == []
        for value in range(p):
            for control in (0, 1):
                for preset in (0, 1):
                    result = run(m, {x: value, c: control, out: preset})
                    want = pow(value, -1, p) if control and value else 0
                    assert (result(out), result(x), result(c)) == (
                        preset ^ want, value, control)
        ok(f"p={p}: {m.width} qubits, all inputs, ancillas returned to 0")

    section("controlled division and point addition with packed-euclid")
    for p in (5, 7):
        m = Machine("toffoli")
        control = m.alloc(1, "control")
        x, numerator, out = (m.alloc(p.bit_length(), label)
                             for label in ("divisor", "numerator", "out"))
        PA._moddiv(m, control[0], x, numerator, out, p,
                   "reference", None, "packed-euclid")
        for value in range(1, p):
            for number in range(p):
                for enabled in (0, 1):
                    result = run(m, {control: enabled, x: value,
                                     numerator: number})
                    want = number * pow(value, -1, p) % p if enabled else 0
                    assert result(out) == want, (p, value, number, enabled)
        ok(f"p={p}: controlled quotient, erased work registers")
    curve = C.CLASSIQ
    points = [point for point in curve.points() if not point.inf]
    addend = points[0]
    m = Machine("toffoli")
    control = m.alloc(1, "control")
    x, y = m.alloc(3, "x"), m.alloc(3, "y")
    PA.point_add_ctrl(m, control[0], x, y, addend.x, addend.y, 7,
                      inversion="packed-euclid")
    checked = 0
    for point in points:
        if C.point_add_exceptional(curve, point, addend):
            continue
        for enabled in (0, 1):
            result = run(m, {control: enabled, x: point.x, y: point.y})
            want = curve.add(point, addend) if enabled else point
            assert (result(x), result(y)) == (want.x, want.y)
            checked += 1
    ok(f"p=7: {checked} in-place additions with clean ancillas")

    section("packed-euclid ECDLP oracle on every basis input")
    curve = C.Curve(5, 0, 1, "p5")
    base = next(point for point in curve.points()
                if not point.inf and curve.point_order(point) == 3)
    _, info = S.ecdlp_circuit(curve, base, curve.mul(2, base), 3,
                              oracle="arith", m_bits=2,
                              inversion="packed-euclid")
    kr, lr, x, y = info["regs"]
    for u in range(4):
        for v in range(4):
            result = run(info["machine"], {kr: u, lr: v})
            want = curve.add(info["offset"], curve.mul((u + 2 * v) % 3, base))
            assert (result(x), result(y)) == (want.x, want.y)
    ok(f"p=5: 16 inputs, {info['machine'].width} qubits; Aer sampling is in "
       "bench/ec_gu_affine.py")


if __name__ == "__main__":
    main()
    print("\ntest_ec_packed_euclid: all passed")
