"""Gidney's 2-clean / (n-2)-dirty constant adder, as a standalone dynamic circuit.

Source: C. Gidney, "A Classical-Quantum Adder with Constant Workspace and
Linear Gates", arXiv:2507.23079, Sec. 2 (streaming adder by venting carries,
Fig. "streaming-adder"; phasing by carries via Haner et al.'s carry-xor,
Fig. "carry-xor"; assembled in Fig. "add-2-clean-n-dirty").

OPT-IN / NOT UNITARY.  The circuit measures qubits in the X basis and applies
classically-controlled gates.  It therefore has no inverse and must NOT be
routed through `ec_sim.Machine`/`Ctx` or its `emit_inverse` path.  Nothing in
the package imports this module.

Semantics (offset `d` is a classical constant, any integer; carry-in is folded
into it):

    |x>_target |h>_dirty |0>_clean  ->  |x + d mod 2^n> |h> |0>

for every x and *every* dirty content h, including superpositions and states
entangled with outside qubits.  Exact, with no residual phase (up to the
usual unobservable global phase of a measurement branch).

Construction (c_k = carry into bit k of x + d, c_0 = 0, p_k = y_k ^ d_k):
 1. Stream k = 0..n-1 with two clean qubits alternating as the carry holder:
    c_{k+1} = p_k c_k ^ d_k y_k (one Toffoli), target_k <- x_k^d_k^c_k, then
    - for k >= 2 CNOT c_k into dirty g_k (the "merged" first carry-xor),
    - vent c_k: X-basis measurement, outcome m_k, qubit reset.
    Venting leaves the phase (-1)^{m_k c_k}.  Since carry(~x', d) == carry(x, d)
    (paper's Z-redundancy identity), c_k is a Z-function of ~target'.
 2. Z^{m_k} on g_k, carry-xor g_k ^= c_k(~target') (k>=2), Z^{m_k} on g_k.
    Z U Z U == Z on the carries, so with the merged first xor this cancels the
    vented phases and restores g.  c_1 = d_0 * x_0 needs no dirty qubit:
    phase m_1 * d_0 * (1 ^ target'_0) is applied as X Z X (= -Z) on target_0.
 Toffoli count: (n-2) streaming + (2n-5) carry-xor  = 3n - 7 (n >= 3).

The 4n / 3-clean "borrow back and forth" variant of the paper is not implemented.
"""
from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def _vent(qc, q, bit):
    """Z-redundant deletion: X-basis measure, then reset to |0>."""
    qc.h(q)
    qc.measure(q, bit)
    qc.reset(q)


def _carry_xor_inverted(qc, t, g, d, n):
    """g_j ^= carry_j(~t, d, 0) for j = 2..n-1, leaving t and other dirty content as-is.

    `g[j-2]` is g_j.  Two Toffoli ladders (Haner et al.): a descending pass cancels
    the dirty content, an ascending pass writes the carries (c_1 = d_0 y_0 is read
    straight from y_0, no qubit needed).  y = ~t; p_j = y_j ^ d_j.
    """
    if n < 3:
        return
    bit = lambda j: (d >> j) & 1
    for q in t:
        qc.x(q)                                  # t now holds y

    def ccx_p(j, c, tgt):                        # tgt ^= p_j & c
        if bit(j):
            qc.x(t[j])
        qc.ccx(t[j], c, tgt)
        if bit(j):
            qc.x(t[j])

    for j in range(n - 1, 2, -1):
        ccx_p(j - 1, g[j - 3], g[j - 2])
    for j in range(2, n):
        if j == 2:
            if bit(0):
                ccx_p(1, t[0], g[0])
        else:
            ccx_p(j - 1, g[j - 3], g[j - 2])
        if bit(j - 1):
            qc.cx(t[j - 1], g[j - 2])
    for q in t:
        qc.x(q)


def append_dirty_const_adder(qc, target, dirty, clean, offset, carry_in=0):
    """Append target += offset + carry_in (mod 2^n) to `qc`.

    target: n qubits (LSB first).  dirty: >= n-2 qubits in arbitrary state,
    returned unchanged.  clean: 2 qubits in |0>, returned in |0>.
    Adds a 1-bit ClassicalRegister per vented carry; returns the list of them.
    """
    n = len(target)
    if n < 1:
        raise ValueError("target register must not be empty")
    if carry_in not in (0, 1):
        raise ValueError("carry_in must be 0 or 1")
    if len(dirty) < max(n - 2, 0):
        raise ValueError(f"need n-2={max(n - 2, 0)} dirty qubits, got {len(dirty)}")
    if n >= 2 and len(clean) < 2:
        raise ValueError("need 2 clean qubits")
    allq = list(target) + list(dirty[:max(n - 2, 0)]) + list(clean[:2] if n >= 2 else [])
    if len(set(allq)) != len(allq):
        raise ValueError("target, dirty and clean qubits must be disjoint")
    d = (offset + carry_in) % (1 << n)
    g = list(dirty[:max(n - 2, 0)])
    bit = lambda j: (d >> j) & 1
    if n == 0:
        return []
    if n == 1:
        if bit(0):
            qc.x(target[0])
        return []

    m = {k: ClassicalRegister(1, f"vent{k}") for k in range(1, n)}
    for r in m.values():
        qc.add_register(r)
    t = target
    # k = 0: c_0 = 0, so c_1 = d_0 x_0 and target_0 ^= d_0.
    if bit(0):
        qc.cx(t[0], clean[0])
        qc.x(t[0])
    for k in range(1, n):
        c, nxt = clean[(k + 1) % 2], clean[k % 2]
        if bit(k):
            qc.x(t[k])                           # t_k = p_k = x_k ^ d_k
        if k < n - 1:
            qc.ccx(t[k], c, nxt)
            if bit(k):
                qc.cx(t[k], nxt)                 # d_k y_k = d_k p_k ^ d_k
                qc.x(nxt)
        qc.cx(c, t[k])
        if k >= 2:
            qc.cx(c, g[k - 2])
        _vent(qc, c, m[k][0])

    for k in range(2, n):
        with qc.if_test((m[k], 1)):
            qc.z(g[k - 2])
    _carry_xor_inverted(qc, t, g, d, n)
    for k in range(2, n):
        with qc.if_test((m[k], 1)):
            qc.z(g[k - 2])
    if bit(0):
        with qc.if_test((m[1], 1)):
            qc.x(t[0])
            qc.z(t[0])
            qc.x(t[0])
    return list(m.values())


def dirty_const_adder_circuit(n, offset, carry_in=0):
    """Standalone circuit with registers target(n), dirty(n-2), clean(2) and vent bits."""
    if n < 1:
        raise ValueError("n must be positive")
    tgt = QuantumRegister(n, "target")
    dirty = QuantumRegister(max(n - 2, 0), "dirty")
    clean = QuantumRegister(2 if n >= 2 else 0, "clean")
    qc = QuantumCircuit(*[r for r in (tgt, dirty, clean) if len(r)], name=f"gidney_add_{offset}")
    append_dirty_const_adder(qc, list(tgt), list(dirty), list(clean), offset, carry_in)
    return qc


def toffoli_count(qc):
    return qc.count_ops().get("ccx", 0)
