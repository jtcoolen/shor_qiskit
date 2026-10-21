"""Exact classical oracle for the quotient Euclidean walk in Gu et al. SI II.2.

Binary long division here specifies the arithmetic; this module does not
implement the paper's packed qubit layout, merged scans, or point-adder circuit.
"""

from math import gcd


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
