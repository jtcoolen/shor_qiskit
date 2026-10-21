"""Exact integer model of Gu et al.'s divsteps inversion (Eq. 6, Appendix A).

This is a reference for building and testing a future reversible gate circuit,
not a quantum backend: Python branches and integers are not coherent gates.
"""

from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class State:
    z: int
    f: int
    g: int
    fv: int
    gv: int


def step(state):
    """Apply one active divstep; return the next state and (swap, odd) bits."""
    z, f, g, fv, gv = state.z, state.f, state.g, state.fv, state.gv
    if g == 0:
        raise ValueError("a terminated walk must be frozen")
    if g & 1:
        if z >= 0:
            return State(-z, g, (g - f) // 2, 2 * gv, gv - fv), (1, 1)
        return State(z + 1, f, (g + f) // 2, 2 * fv, gv + fv), (0, 1)
    return State(z + 1, f, g // 2, 2 * fv, gv), (0, 0)


def decode(state):
    """Recover an active step's decision from *only* its updated state.

    The ambiguous case implements Appendix A, Eq. (A11). No history bits
    or classical input w are consulted.
    """
    if state.fv % 4 == 2:
        return 1, 1
    if state.z >= 1:
        return 0, 0
    if state.g == 0:
        return 0, 1
    if state.fv == 0:
        raise ValueError("negative-counter nonswap has zero coefficient")
    j = (abs(state.fv) & -abs(state.fv)).bit_length() - 1
    if j < 2:
        raise ValueError("ambiguous divstep needs excursion depth >= 2")
    modulus = 1 << j
    q = state.fv >> j
    b = ((state.gv * pow(q, -1, modulus)) % modulus) >> (j - 1)
    return 0, b


def reverse_step(state):
    """Undo an active step using its decoded decision, with no stored record."""
    swap, odd = decode(state)
    if swap:
        old_gv = state.fv // 2
        return State(-state.z, state.f - 2 * state.g, state.f,
                     old_gv - state.gv, old_gv)
    old_fv = state.fv // 2
    return State(state.z - 1, state.f, 2 * state.g - odd * state.f,
                 old_fv, state.gv - odd * old_fv)


def walk(w, p, max_steps=None):
    """Return (terminal state, active steps), failing explicitly on bad input."""
    if p < 3 or p % 2 == 0 or not 0 < w < p or gcd(w, p) != 1:
        raise ValueError("expected 0 < w < odd p with gcd(w, p) = 1")
    if max_steps is None:
        max_steps = 4 * p.bit_length() ** 2
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    state = State(0, p, w, 0, 1)
    for t in range(max_steps):
        if state.g == 0:
            return state, t
        state, _ = step(state)
    if state.g:
        raise ValueError(f"divsteps did not terminate within {max_steps} steps")
    return state, max_steps


def quotient(w, a, p, max_steps=None):
    """Classical oracle for a/w mod p, with reverse-walk scaling verification."""
    state, t = walk(w, p, max_steps)
    r = (a * state.f * state.fv) % p
    for _ in range(t):
        state = reverse_step(state)
        r = (r * pow(2, -1, p)) % p
    if state != State(0, p, w, 0, 1):
        raise AssertionError("reverse divsteps did not restore the input")
    return r
