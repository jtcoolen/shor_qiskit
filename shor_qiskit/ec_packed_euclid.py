"""Exact binary long division primitive for a future packed Euclidean walk.

Luo et al. arXiv:2607.13816 Sec. 3.1 and Gu et al. SI II.2 scan quotient
bits by comparing and subtracting shifted divisors. This module implements
that arithmetic reversibly, but not their packed workspaces or point adder.
Do not advertise it as an ECDLP backend until those circuits exist.
"""

import ec_adders as A
from ec_sim import Reg


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
