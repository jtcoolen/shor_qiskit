"""Windowed in-place multiplication (Gidney, arXiv:1905.07682, Sec. 3.2).

    |x>  ->  |k * x mod 2^L>      for odd k, on an L-qubit two's-complement register.

Qubit i of the product depends only on qubits <= i of x, so the register can be
rewritten from the top window down: for the window w at bits [i, i+W) add the
*carry* of w*k, namely (w*k) >> W, into every bit above the window (one table
lookup + one plain addition), then multiply the window by k mod 2^W in place
by the same trick with W = 1 (a controlled constant addition per bit).
Windows are consumed top to bottom, so each window is still the original one
when it is read, and the later additions into higher bits only add carries.

Toffoli count O((L/W)(L + 2^W + W^2)), i.e. O(L^2 / lg L) at W = lg L.

This is a plain (non-modular) multiplication.  Modular multiplication is built
elsewhere from two out-of-place product additions; this construction is what
Gidney's paper uses when the modulus is a power of two.
"""

from qrom import lookup_ui
from rc_adder import rc_add, rc_add_const


def _fixup(qc, ctrl, win, k, scratch):
    """win *= k mod 2^len(win), controlled on ctrl.  Bit-by-bit, top down:
    a set bit i adds (k >> 1) into the bits above it."""
    L = len(win)
    k %= 1 << L
    for i in reversed(range(L - 1)):
        X = (k >> 1) % (1 << (L - i - 1))
        if X:
            rc_add_const(qc, X, win[i + 1:], scratch[:L - i], ctrls=[ctrl, win[i]])


def times_equal(qc, ctrl, target, k, window, tmp, anc, carry):
    """target *= k (mod 2^len(target)) when ctrl is 1; identity otherwise.

    k      odd classical constant (even factors are not reversible)
    tmp    len(target) clean qubits for the looked-up value   (returned clean)
    anc    `window` clean qubits for the unary-iteration QROM (returned clean)
    carry  one clean qubit for the adder                      (returned clean)
    """
    L = len(target)
    assert k % 2 == 1, "even factors are not reversible"
    k %= 1 << L
    if k == 1:
        return
    for i in reversed(range(0, L, window)):
        win = list(target[i:i + window])
        W = len(win)
        hi = list(target[i + W:])
        if hi:
            table = [((j * k) >> W) % (1 << len(hi)) for j in range(1 << W)]
            t = list(tmp[:len(hi)])
            lookup_ui(qc, ctrl, win, t, table, list(anc)[:W])
            rc_add(qc, t, hi, carry)
            lookup_ui(qc, ctrl, win, t, table, list(anc)[:W])      # uncompute
        _fixup(qc, ctrl, win, k, [carry] + list(tmp))
