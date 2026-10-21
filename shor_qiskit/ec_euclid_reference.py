"""Exact classical oracle for the quotient Euclidean walk in Gu et al. SI II.2.

Binary long division here specifies the arithmetic; this module does not
implement the paper's packed qubit layout, merged scans, or point-adder circuit.
"""

from math import gcd
from dataclasses import dataclass


@dataclass(frozen=True)
class PackedSnapshot:
    """Two opposed-end words and the live boundaries of an EEA state."""

    work1: int
    work2: int
    t_bits: int
    q_bits: int
    r_bits: int
    t_prev_bits: int
    r_next_bits: int
    width: int


def pack_state(t, quotient, r_prev, t_prev, r, width):
    """Store (t, quotient, r_prev) and (t_prev, r) from opposite ends.

    A spare bit after t separates the growing coefficient from the quotient.
    This models the live boundaries of Luo et al.'s Work1/Work2; it is a
    classical layout specification, not a quantum permutation or gate count.
    """
    values = (t, quotient, r_prev, t_prev, r)
    if width < 1 or any(value < 0 for value in values):
        raise ValueError("packed EEA values must be nonnegative")
    bt, bq, br, btp, brn = (value.bit_length() for value in values)
    if bt + 1 + bq + br > width or btp + brn > width:
        raise ValueError("packed EEA values overlap at this width")
    first = t | (quotient << (bt + 1)) | (r_prev << (width - br))
    second = t_prev | (r << (width - brn))
    return PackedSnapshot(first, second, bt, bq, br, btp, brn, width)


def unpack_state(snapshot):
    """Read back the five integers using the recorded coherent boundaries."""
    s = snapshot
    low = lambda bits: (1 << bits) - 1
    return (s.work1 & low(s.t_bits),
            (s.work1 >> (s.t_bits + 1)) & low(s.q_bits),
            s.work1 >> (s.width - s.r_bits),
            s.work2 & low(s.t_prev_bits),
            s.work2 >> (s.width - s.r_next_bits))


def binary_divmod(numerator, denominator):
    """Binary long division by compare/subtract at each scanned bit."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("expected a nonnegative numerator and positive divisor")
    remainder = quotient = 0
    for bit in range(numerator.bit_length() - 1, -1, -1):
        remainder = 2 * remainder + ((numerator >> bit) & 1)
        if remainder >= denominator:
            remainder -= denominator
            quotient |= 1 << bit
    return quotient, remainder


def inverse(w, p):
    """Compute the signed-coefficient inverse from Eq. (60), without truncation."""
    if p < 3 or p % 2 == 0 or not 0 < w < p or gcd(w, p) != 1:
        raise ValueError("expected 0 < w < odd p with gcd(w, p) = 1")
    prev_r, r = p, w
    prev_t, t = 0, 1
    parity = 0
    while r != 1:
        quotient, next_r = binary_divmod(prev_r, r)
        if next_r == 0:
            raise AssertionError("unit input reached a zero remainder")
        prev_r, r = r, next_r
        prev_t, t = t, prev_t + quotient * t
        parity ^= 1
    return (-t if parity else t) % p


def packed_trace(w, p):
    """Return exact toy packed-boundary snapshots before/after quotient scans."""
    if p < 3 or p % 2 == 0 or not 0 < w < p or gcd(w, p) != 1:
        raise ValueError("expected a unit w modulo odd p")
    width = p.bit_length() + 3
    prev_r, r, prev_t, t = p, w, 0, 1
    trace = [pack_state(t, 0, prev_r, prev_t, r, width)]
    while r != 1:
        q, next_r = binary_divmod(prev_r, r)
        if next_r == 0:
            raise AssertionError("unit input reached zero remainder")
        trace.append(pack_state(t, q, next_r, prev_t, r, width))
        prev_r, r, prev_t, t = r, next_r, t, prev_t + q * t
        if prev_r * t + r * prev_t != p:
            raise AssertionError("packed Euclidean determinant changed")
        trace.append(pack_state(t, 0, prev_r, prev_t, r, width))
    return trace
