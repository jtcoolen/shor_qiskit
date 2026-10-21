"""Reversible primitives for a future packed Euclidean walk.

Luo et al. arXiv:2607.13816 Sec. 3.1 and Gu et al. SI II.2 scan quotient
bits by comparing and subtracting shifted divisors, and address work bits
using coherent length counters. This module implements those primitives,
but not their packed workspaces or point adder.
Do not advertise it as an ECDLP backend until those circuits exist.
"""

import ec_adders as A
from ec_sim import Reg


def swap_at(m, word, index, flag):
    """Swap flag with the word bit addressed by a coherent binary index.

    Out-of-range indices leave both unchanged. The index and temporary
    selection bit are restored, even when the addressed word bit is occupied.
    """
    if not word or not index or 1 << len(index) < len(word):
        raise ValueError("address must cover a nonempty word")
    if len(set(word + index + [flag])) != len(word) + len(index) + 1:
        raise ValueError("word, index and flag must be disjoint")
    ctx = m.ctx
    selected = m.anc(1, "selected")
    for position, bit in enumerate(word):
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                ctx.x(address_bit)
        ctx.mcx(index, selected[0])
        ctx.cswap(selected[0], flag, bit)
        ctx.mcx(index, selected[0])
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                ctx.x(address_bit)
    m.free(selected)


def xor_at(m, word, index, target):
    """XOR an addressed packed-word bit into a separate target qubit.

    Unlike ``swap_at`` this does not change the work word, so it can
    temporarily select operands for a location-controlled comparison.
    The caller must reverse the selection before modifying its source.
    """
    if (not word or not index or 1 << len(index) < len(word)
            or len(set(word + index + [target]))
            != len(word) + len(index) + 1):
        raise ValueError("packed read needs distinct wires and a covering index")
    for position, bit in enumerate(word):
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                m.ctx.x(address_bit)
        m.ctx.mcx((*index, bit), target)
        for shift, address_bit in enumerate(index):
            if not (position >> shift) & 1:
                m.ctx.x(address_bit)


def rotate_word_if(m, word, control, direction):
    """Conditionally rotate a packed work word by one physical wire.

    Luo et al. Alg. 3 uses this reversible circular shift while the live
    coefficient and remainder boundaries move. A rotation alone does not
    update those boundaries or implement the full packed walk.
    """
    if (direction not in ("left", "right") or not word
            or control in word or len(set(word)) != len(word)):
        raise ValueError("rotation needs unique word wires and a disjoint control")
    if direction == "left":
        positions = range(len(word) - 2, -1, -1)
    else:
        positions = range(len(word) - 1)
    for i in positions:
        m.ctx.cswap(control, word[i], word[i + 1])


def shift_work2(m, word, shift, control, direction, steps=1):
    """Rotate Work2 and update its coherent shift counter together.

    Algorithms 2-3 use one-position pre/post shifts and two-position
    corrections. The caller must ensure the counter does not overflow its
    promised live range; modulo-register wrapping is not a valid boundary.
    """
    if (not word or not shift or type(steps) is not int
            or not 1 <= steps <= 2
            or len(set(word + shift + [control])) != len(word) + len(shift) + 1
            or direction not in ("left", "right")):
        raise ValueError("Work2 shift needs a nonempty word, counter, separate "
                         "control and a one- or two-position direction")
    cp = m.anc(len(shift), "constant")
    carry = m.anc(len(shift), "carry")
    for _ in range(steps):
        rotate_word_if(m, word, control, direction)
    A.cadd_const(m.ctx, control, shift,
                 steps if direction == "left" else -steps, cp, carry)
    m.free(cp, carry)


def binary_divmod(m, numerator, denominator, quotient, remainder):
    """XOR quotient and remainder of numerator / denominator into clean outputs.

    The input and denominator are n-bit unsigned integers; denominator must
    be nonzero. The scan holds a partial remainder in n+1 bits, obtains each
    quotient bit by comparing it with the divisor, conditionally subtracts,
    copies the output and reverses the scan to clear its work registers.
    """
    n = len(numerator)
    if not n or any(len(reg) != n for reg in (denominator, quotient, remainder)):
        raise ValueError("binary division needs four equal nonempty registers")
    if set(numerator) & set(denominator + quotient + remainder) or (
            set(denominator) & set(quotient + remainder) or
            set(quotient) & set(remainder)):
        raise ValueError("division registers must be disjoint")
    ctx = m.ctx
    rem = m.anc(n + 1, "partial")
    quot = m.anc(n, "quotient")
    hi = m.anc(1, "divisor_hi")
    divisor = Reg(list(denominator) + list(hi), "divisor")
    mark = m.begin()
    for bit in range(n - 1, -1, -1):
        for i in range(n, 0, -1):
            ctx.swap(rem[i], rem[i - 1])
        ctx.cx(numerator[bit], rem[0])
        comparator = m.anc(n + 1, "compare")
        A.lt_uint(ctx, rem, divisor, quot[bit], comparator)
        m.free(comparator)
        ctx.x(quot[bit])
        copy, carry = m.anc(n + 1, "copy"), m.anc(n + 1, "carry")
        A.csub(ctx, quot[bit], divisor, rem, copy, carry)
        m.free(copy, carry)
    body = m.since(mark)
    for a, b in zip(quot, quotient):
        ctx.cx(a, b)
    for a, b in zip(rem, remainder):
        ctx.cx(a, b)
    m.undo(body)
    m.free(rem, quot, hi)
