"""Oblivious carry runways (Gidney, arXiv:1905.08488, Sec. 4).

A runway of length m at bit position p is m extra qubits that catch the carry
out of the bits below p, so the register can be added to as independent
pieces instead of through one long carry chain.  To keep the carry from
leaking information about the additions performed, the runway is
*oblivious*: it starts in |+>^m and is then subtracted out of the register at
position p (Fig. 2),

    |g>|+^m>  ->  sum_r |g - r*2^p mod 2^L>|r>,

so every runway value r is an equally good encoding of g.  An addition then
becomes piecewise (Fig. 3): each piece (the register bits between two
runways, plus the runway sitting on top of it) is added to on its own.  The
register decodes as  (main + sum_i r_i 2^(p_i)) mod 2^L.  One piecewise
addition deviates from the ideal addition with probability <= 2^-m per
runway (Thm 4.2); it is wrong only if a runway is at 2^m - 1 and overflows.

A runway is removed by running the encoder backwards (add it back, H): that
returns it to |0>, exactly when no overflow happened.

A `runways` argument below is a list of (p, qubits) sorted by position p.
"""

import numpy as np

from shor_essentials import add_const, c_add_const
from windowed import add_quantum


def runway_positions(width, spacing, length):
    """Evenly spaced runway positions strictly inside a register of `width`
    bits; the top of every runway must still fit in the register."""
    return [p for p in range(spacing, width, spacing) if p + length <= width]


def encode_runways(qc, reg, runways):
    """Attach the runways (all runway qubits start |0>)."""
    for p, qs in runways:
        qc.h(list(qs))
        add_quantum(qc, list(qs), list(reg[p:]), -1)


def remove_runways(qc, reg, runways):
    """Exact inverse of encode_runways."""
    for p, qs in reversed(runways):
        add_quantum(qc, list(qs), list(reg[p:]), +1)
        qc.h(list(qs))


def _pieces(reg, runways):
    """[(start, end, qubits)]: main bits [start, end) plus the runway above."""
    bounds = [0] + [p for p, _ in runways] + [len(reg)]
    out = []
    for i in range(len(bounds) - 1):
        qs = list(reg[bounds[i]:bounds[i + 1]])
        if i < len(runways):
            qs += list(runways[i][1])
        out.append((bounds[i], bounds[i + 1], qs))
    return out


def piecewise_add_const(qc, k, reg, runways, ctrls=()):
    """reg += k (approximately, piece by piece), optionally controlled."""
    k %= 1 << len(reg)
    for lo, hi, qs in _pieces(reg, runways):
        kk = (k >> lo) & ((1 << (hi - lo)) - 1)
        if not kk:
            continue
        if ctrls:
            c_add_const(qc, list(ctrls), kk, qs)
        else:
            add_const(qc, kk, qs)


def piecewise_add_quantum(qc, a, reg, runways):
    """reg += a, where a is a quantum register no wider than reg."""
    for lo, hi, qs in _pieces(reg, runways):
        sl = list(a[lo:hi])
        if sl:
            add_quantum(qc, sl, qs, +1)


def decode(main, runway_vals, runways_pos, width):
    """Integer encoded by a measured register and its measured runways."""
    return (main + sum(r << p for r, p in zip(runway_vals, runways_pos))) % (1 << width)
