"""Gidney's dirty-ancilla dynamic adder, composed into modular constant addition.

Source: C. Gidney, "A Classical-Quantum Adder with Constant Workspace and
Linear Gates", arXiv:2507.23079v1 (2025).  Paper facts used here:

* Sec 2.2 / Fig. 2: stream the ripple carry through two clean qubits and
  "vent" each spent carry (X-basis measurement).  Eq. (8):
  carry(~x', d, c0) == carry(x, d, c0), so every vented carry is a Z-function
  of the finished register and outcome 1 only leaves a phase task.
* Sec 2.3 / Figs. 3-4: those phases are paid by Z^m, Haener et al.'s
  carry-xor into n-2 dirty qubits, Z^m; the first carry-xor is merged into the
  streaming pass.  2 clean + n-2 dirty, 3n +- O(1) Toffolis.
* Sec 2.4 (end) / Fig. 6: d_k is only used to invert controls and to control
  bit flips, so a *controlled* addition substitutes d_k -> control and costs no
  extra Toffoli.  We use the obvious generalisation: every d_k may be any
  parity "literal" c xor q1 xor q2 ..., which also lets the reduction flag
  select between two offsets.

This module writes every majority in the paper's form
maj(a, b, c) = ((a ^ c)(b ^ c)) ^ c, so a quantum d_k only enters as a CNOT
onto a control.  The single exception is the lowest carry c_1 = x_0 d_0
(the carry-in is 0): for a quantum d_0 it costs one Toffoli in the streaming
pass and one in the carry-xor (the paper's figures always spend these on its
carry-in qubit).  Toffolis: 3n-7 for a classical offset with d_0 = 1,
3n-5 for a quantum d_0.

OPT-IN / NOT UNITARY.  Venting measures and feeds forward.  Nothing here may
run under `ec_sim.Machine.emit_inverse` (it cannot reverse a measurement; Qiskit
raises on `Measure.inverse()`), and nothing in the package's default unitary
paths imports this module.  Inverses are built *explicitly*: the inverse of
"+k mod p" is "+(p-k) mod p", which is again a forward dynamic circuit.

Modular composition (`cmodadd_const_dyn`), y in [0, p), p < 2^n, k in [0, p):

  1. (n+1)-bit dynamic add on (y, hi) of  ctrl ? k-p : -p     hi = [y + ctrl k < p]
  2. n-bit dynamic add on y of literal hi * p                  y = y + ctrl k mod p
  3. dynamic carry-out (no target change): hi ^= ctrl [y >= k],
     then hi ^= not ctrl, which returns hi to |0>.
Workspace: 3 clean (2 vent carries + flag hi) and n-1 dirty qubits, which may
be in any state, entangled with anything, and are returned untouched.

Exact phase check.  `branch_sim` pushes one basis state through a dynamic
circuit and returns, besides the output bits, the phase as an affine function
of *all* vent outcomes at once: every vent is an H on a basis qubit (50/50), the
feed-forward bodies are checked to be diagonal, so the trajectory is outcome
independent and the branch phase is (-1)^(s0 + sum_i m_i s_i).  The circuit
is exact, on every one of the 2^V branches, iff the output map is right and
every s_i is the same for all inputs.
"""
from __future__ import annotations

import math

from qiskit.circuit import ClassicalRegister, Clbit, ControlledGate
from qiskit.circuit.library import SwapGate, XGate

# =============================================================================
# literals: d_k = const XOR parity(qubits)
# =============================================================================


def select_literals(n, when1, when0=0, ctrl=None):
    """Per-bit literals of the offset `ctrl ? when1 : when0` (mod 2^n)."""
    out = []
    for k in range(n):
        a, b = (when1 >> k) & 1, (when0 >> k) & 1
        if ctrl is None or a == b:
            out.append((a if ctrl is None else b, ()))
        else:
            out.append((b, (ctrl,)))
    return out


def _is_const(L):
    return not L[1]


def _flip(qc, L, q):
    c, qs = L
    if c:
        qc.x(q)
    for r in qs:
        qc.cx(r, q)


def _materialize(qc, L):
    c, qs = L
    q0 = qs[0]
    for r in qs[1:]:
        qc.cx(r, q0)
    if c:
        qc.x(q0)
    return q0


def _dematerialize(qc, L):
    c, qs = L
    q0 = qs[0]
    if c:
        qc.x(q0)
    for r in qs[1:]:
        qc.cx(r, q0)


def _ccx_lit(qc, L, a, t):
    """t ^= d & a (Toffoli only when d is quantum)."""
    if _is_const(L):
        if L[0]:
            qc.cx(a, t)
        return
    q0 = _materialize(qc, L)
    qc.ccx(q0, a, t)
    _dematerialize(qc, L)


def _cz_lit(qc, L, a):
    """Phase (-1)^(d & a)."""
    if _is_const(L):
        if L[0]:
            qc.z(a)
        return
    q0 = _materialize(qc, L)
    qc.cz(q0, a)
    _dematerialize(qc, L)


def _maj(qc, y, L, s, t):
    """t ^= maj(y, d, s) = ((y ^ s)(d ^ s)) ^ s; y and s restored."""
    qc.cx(s, y)
    _flip(qc, L, s)
    qc.ccx(y, s, t)
    _flip(qc, L, s)
    qc.cx(s, y)
    qc.cx(s, t)


def _vent(qc, q, bit):
    """Z-redundant deletion: X-basis measurement, then reset to |0>."""
    qc.h(q)
    qc.measure(q, bit)
    qc.reset(q)


def _vent_bits(qc, size):
    """A classical scratch register of at least `size` bits, shared by calls.

    Every adder consumes its outcomes before the next one starts, so the bits
    can be reused; this keeps the classical width O(n).
    """
    for r in qc.cregs:
        if r.name.startswith("dynvent") and r.size >= size:
            return r
    r = ClassicalRegister(max(size, 1), f"dynvent{size}")
    qc.add_register(r)
    return r


def _check_disjoint(*groups):
    flat = [q for g in groups for q in g]
    if len(set(flat)) != len(flat):
        raise ValueError("target, dirty, clean, output and literal qubits must be disjoint")


def _lit_qubits(lits):
    return sorted({q for _, qs in lits for q in qs}, key=id)


# =============================================================================
# the dirty-ancilla adder (Fig. 4) with literal offsets
# =============================================================================


def _carry_xor(qc, y, lits, g, top, A, bits, c1_mode):
    """g_j ^= carry_j(y, d) for j = 2..top, y unchanged (Haener ladder, maj form).

    `g[j-2]` is g_j.  The lowest carry c_1 = y_0 d_0 is read from y_0 when d_0
    is the constant 1, is zero when d_0 is 0, and is otherwise computed into the
    clean qubit A and vented again (fix: CZ(d_0, y_0)), so this costs one extra
    Toffoli only for a quantum d_0.
    """
    if top < 2:
        return
    for j in range(top, 2, -1):                              # descending pass
        _flip(qc, lits[j - 1], y[j - 1])
        qc.ccx(y[j - 1], g[j - 3], g[j - 2])
        _flip(qc, lits[j - 1], y[j - 1])
    if c1_mode == "quantum":
        _ccx_lit(qc, lits[0], y[0], A)
    for j in range(2, top + 1):                              # ascending pass
        if j == 2:
            if c1_mode == "zero":
                _ccx_lit(qc, lits[1], y[1], g[0])
                continue
            s = y[0] if c1_mode == "one" else A
        else:
            s = g[j - 3]
        _maj(qc, y[j - 1], lits[j - 1], s, g[j - 2])
    if c1_mode == "quantum":
        _vent(qc, A, bits)
        with qc.if_test((bits, 1)):
            _cz_lit(qc, lits[0], y[0])


def _c1_mode(L0):
    if _is_const(L0):
        return "one" if L0[0] else "zero"
    return "quantum"


def append_literal_adder(qc, target, dirty, clean, lits):
    """target <- target + D (mod 2^n), D = sum_k d_k 2^k with literal d_k.

    dirty: >= n-2 qubits, any state, restored.  clean: 2 qubits |0>, restored.
    Literal qubits are only used as controls and are restored.
    """
    n = len(target)
    if n < 1:
        raise ValueError("target register must not be empty")
    if len(lits) != n:
        raise ValueError("need one literal per target bit")
    nd = max(n - 2, 0)
    if len(dirty) < nd:
        raise ValueError(f"need n-2={nd} dirty qubits, got {len(dirty)}")
    if n >= 2 and len(clean) < 2:
        raise ValueError("need 2 clean qubits")
    t, g = list(target), list(dirty[:nd])
    cl = list(clean[:2]) if n >= 2 else []
    _check_disjoint(t, g, cl, _lit_qubits(lits))
    if n == 1:
        _flip(qc, lits[0], t[0])
        return
    mode = _c1_mode(lits[0])
    m = _vent_bits(qc, n)
    # k = 0: c_0 = 0, c_1 = x_0 d_0, t_0 ^= d_0
    if mode != "zero":
        _ccx_lit(qc, lits[0], t[0], cl[0])
    _flip(qc, lits[0], t[0])
    for k in range(1, n):
        c, nxt = cl[(k + 1) % 2], cl[k % 2]
        known_zero = k == 1 and mode == "zero"
        if k < n - 1:
            if known_zero:
                _ccx_lit(qc, lits[k], t[k], nxt)
            else:
                qc.cx(c, t[k])                   # t_k = x_k ^ c_k
                _flip(qc, lits[k], c)
                qc.ccx(t[k], c, nxt)
                _flip(qc, lits[k], c)
                qc.cx(c, nxt)                    # nxt = maj(x_k, d_k, c_k)
        elif not known_zero:
            qc.cx(c, t[k])
        _flip(qc, lits[k], t[k])                 # t_k = x_k ^ d_k ^ c_k
        if k >= 2:
            qc.cx(c, g[k - 2])                   # merged first carry-xor
        if not known_zero:
            _vent(qc, c, m[k - 1])
    # phase tasks: carries of (~t', d) == carries of (x, d)  [Eq. 8]
    for q in t:
        qc.x(q)
    for k in range(2, n):
        with qc.if_test((m[k - 1], 1)):
            qc.z(g[k - 2])
    _carry_xor(qc, t, lits, g, n - 1, cl[0], m[n - 1], mode)
    for k in range(2, n):
        with qc.if_test((m[k - 1], 1)):
            qc.z(g[k - 2])
    if mode != "zero":
        with qc.if_test((m[0], 1)):
            _cz_lit(qc, lits[0], t[0])           # c_1 = y_0 d_0
    for q in t:
        qc.x(q)


def append_literal_carry_out(qc, y, dirty, clean, lits, out):
    """out ^= [y + D >= 2^n] (the carry out), y unchanged.

    Same streaming/venting/carry-xor pattern without writing the sum, so the
    vented carries are Z-functions of the untouched y.  dirty: >= n-1 qubits.
    Used as the comparator that clears the modular reduction flag.
    """
    n = len(y)
    if n < 1:
        raise ValueError("register must not be empty")
    if len(lits) != n:
        raise ValueError("need one literal per bit")
    nd = n - 1
    if len(dirty) < nd:
        raise ValueError(f"need n-1={nd} dirty qubits, got {len(dirty)}")
    if n >= 2 and len(clean) < 2:
        raise ValueError("need 2 clean qubits")
    t, g = list(y), list(dirty[:nd])
    cl = list(clean[:2]) if n >= 2 else []
    _check_disjoint(t, g, cl, [out], _lit_qubits(lits))
    if n == 1:
        _ccx_lit(qc, lits[0], t[0], out)
        return
    mode = _c1_mode(lits[0])
    m = _vent_bits(qc, n + 1)
    if mode != "zero":
        _ccx_lit(qc, lits[0], t[0], cl[0])
    for k in range(1, n):
        c, nxt = cl[(k + 1) % 2], cl[k % 2]
        if k == 1 and mode == "zero":
            _ccx_lit(qc, lits[1], t[1], nxt)
            continue
        _maj(qc, t[k], lits[k], c, nxt)
        if k >= 2:
            qc.cx(c, g[k - 2])
        _vent(qc, c, m[k - 1])
    cn = cl[(n - 1) % 2]
    qc.cx(cn, out)
    qc.cx(cn, g[n - 2])
    _vent(qc, cn, m[n - 1])
    for k in range(2, n + 1):
        with qc.if_test((m[k - 1], 1)):
            qc.z(g[k - 2])
    _carry_xor(qc, t, lits, g, n, cl[0], m[n], mode)
    for k in range(2, n + 1):
        with qc.if_test((m[k - 1], 1)):
            qc.z(g[k - 2])
    if mode != "zero":
        with qc.if_test((m[0], 1)):
            _cz_lit(qc, lits[0], t[0])


# =============================================================================
# modular constant addition on a Machine
# =============================================================================


def _anc_excluding(m, size, name, avoid):
    return m.anc(size, name, exclude=[list(avoid)])


def cmodadd_const_dyn(m, ctrl, y, k, p, dirty):
    """y <- (y + ctrl*k) mod p, k classical, y in [0, p) -- dynamic, exact.

    ctrl: control qubit or None.  dirty: >= n-1 qubits disjoint from y and
    ctrl, any state, restored.  Draws 3 clean qubits from the Machine pool
    (never from `dirty`) and returns them clean, with `Machine.free` checks.
    """
    n = len(y)
    if p >= 1 << n:
        raise ValueError("p must fit in the register")
    k %= p
    if k == 0:
        return
    if n < 2:
        raise ValueError("need at least 2 bits")
    dirty = list(dirty)
    if len(dirty) < n - 1:
        raise ValueError(f"need n-1={n - 1} dirty qubits, got {len(dirty)}")
    dirty = dirty[:n - 1]
    others = list(y) + dirty + ([ctrl] if ctrl is not None else [])
    _check_disjoint(others)
    hi = _anc_excluding(m, 1, "dhi", others)
    cl = _anc_excluding(m, 2, "dcl", others + list(hi))
    qc, N = m.qc, 1 << (n + 1)
    ye = list(y) + [hi[0]]
    append_literal_adder(qc, ye, dirty, cl,
                         select_literals(n + 1, (k - p) % N, (-p) % N, ctrl))
    append_literal_adder(qc, list(y), dirty, cl, select_literals(n, p, 0, hi[0]))
    append_literal_carry_out(qc, list(y), dirty, cl,
                             select_literals(n, (1 << n) - k, 0, ctrl), hi[0])
    if ctrl is not None:
        qc.x(hi[0])
        qc.cx(ctrl, hi[0])
    m.free(cl, hi)


def modadd_const_dyn(m, y, k, p, dirty):
    cmodadd_const_dyn(m, None, y, k, p, dirty)


def cmodsub_const_dyn(m, ctrl, y, k, p, dirty):
    """y <- (y - ctrl*k) mod p: explicitly the forward circuit for +(p-k)."""
    cmodadd_const_dyn(m, ctrl, y, (-k) % p, p, dirty)


def modsub_const_dyn(m, y, k, p, dirty):
    cmodadd_const_dyn(m, None, y, (-k) % p, p, dirty)


def dynamic_ops_since(qc, start=0):
    """Names of non-unitary instructions appended since `start`."""
    return sorted({ci.operation.name for ci in qc.data[start:]
                   if ci.operation.name in ("measure", "reset", "if_else")})


# =============================================================================
# a hybrid affine point addition and toy ECDLP circuit
# =============================================================================


def point_add_ctrl_dyn(m, q, x1, y1, x2, y2, p, multiplier="reference",
                       inversion="kaliski", word_size=None):
    """`ec_pointadd.point_add_ctrl` with steps 1, 2, 5, 10 dynamic.

    The four classical-constant modular additions use the dirty-ancilla
    composition above, borrowing the *other* coordinate register as dirty
    workspace.  Steps 3, 4, 6-9, 11 are the unchanged unitary builders (their
    internal `emit_inverse` calls only ever reverse unitary bodies).  There is
    no inverse of this routine: Shor's oracle only applies forward additions.
    """
    import ec_modarith as MA
    import ec_pointadd as PA
    n = len(x1)
    lam = m.anc(n, "lam")
    modsub_const_dyn(m, x1, x2, p, dirty=y1)                          # 1
    cmodsub_const_dyn(m, q, y1, y2, p, dirty=x1)                      # 2
    PA._moddiv(m, q, x1, y1, lam, p, multiplier, word_size, inversion)  # 3
    PA._modmul_xor(m, x1, lam, y1, p, multiplier, word_size)          # 4
    modadd_const_dyn(m, x1, 3 * x2 % p, p, dirty=y1)                  # 5
    PA._modsqr_sub(m, lam, x1, p, multiplier, word_size)              # 6
    PA._modmul_add(m, x1, lam, y1, p, multiplier, word_size)          # 7
    PA._moddiv(m, q, x1, y1, lam, p, multiplier, word_size, inversion)  # 8
    PA._csub_2x2_or_x2(m, q, x1, x2, p)                               # 9
    cmodsub_const_dyn(m, q, y1, y2, p, dirty=x1)                      # 10
    MA.cmodneg(m, q, x1, p)                                           # 11
    m.free(lam)


def ecdlp_circuit_dyn(curve, P, Q, order, m_bits=None, offset=None,
                      mode="and", multiplier="reference", inversion="kaliski",
                      word_size=None):
    """Affine arithmetic ECDLP circuit with dynamic constant additions.

    Same layout as `ec_shor.ecdlp_circuit(oracle="arith", arithmetic="affine")`
    but every point addition is `point_add_ctrl_dyn`.  Returns (qc, info);
    `info["machine"]` is the oracle-only Machine (for `branch_sim`).
    """
    from qiskit.circuit import QuantumCircuit

    import ec_shor as S
    from ec_sim import Machine
    from shor_essentials import qft

    p = curve.p
    n = p.bit_length()
    mb = max(1, math.ceil(math.log2(order))) if m_bits is None else m_bits
    if offset is None:
        Sp = S._pick_offset(curve, P, Q, order, mb, strict=True)
    else:
        Sp = offset
        if not S._offset_is_safe(curve, P, Q, mb, Sp, "affine", None):
            raise ValueError(f"offset {Sp!r} reaches an exceptional input")
    m = Machine(mode, "ecdlp_dyn")
    kr, lr = m.alloc(mb, "k"), m.alloc(mb, "l")
    px, py = m.alloc(n, "px"), m.alloc(n, "py")
    for i in range(n):
        if (Sp.x >> i) & 1:
            m.ctx.x(px[i])
        if (Sp.y >> i) & 1:
            m.ctx.x(py[i])
    Ps, Qs = S._rungs(curve, P, Q, mb)
    for regs, pts in ((kr, Ps), (lr, Qs)):
        for i, R in enumerate(pts):
            point_add_ctrl_dyn(m, regs[i], px, py, R.x, R.y, p, multiplier,
                               inversion, word_size)
    ck, cl = ClassicalRegister(mb, "ok"), ClassicalRegister(mb, "ol")
    qc = QuantumCircuit(*m.qc.qregs, ck, cl, *m.qc.cregs, name="ecdlp_dyn")
    qc.h(kr)
    qc.h(lr)
    qc.compose(m.qc, qubits=list(qc.qubits), clbits=[qc.clbits[2 * mb + i]
               for i in range(m.qc.num_clbits)], inplace=True)
    qc.append(qft(mb).inverse(), list(kr))
    qc.append(qft(mb).inverse(), list(lr))
    qc.measure(kr, ck)
    qc.measure(lr, cl)
    return qc, {"n": n, "m_bits": mb, "offset": Sp, "order": order,
                "regs": (kr, lr, px, py), "machine": m,
                "qubits": qc.num_qubits, "multiplier": multiplier,
                "inversion": inversion}


def ecdlp_counts(raw):
    """Drop the vent register(s): keep the "ol ok" part of each Aer key."""
    out = {}
    for key, c in raw.items():
        k = " ".join(key.split()[-2:])
        out[k] = out.get(k, 0) + c
    return out


def run_ecdlp_dyn(curve, P, Q, order, shots=1024, seed=7, **kw):
    """Sample `ecdlp_circuit_dyn` on Aer; counts are marginal over (ok, ol)."""
    import aer_device as AD
    import ec_classical as C
    import ec_shor as S
    qc, info = ecdlp_circuit_dyn(curve, P, Q, order, **kw)
    mb = info["m_bits"]
    raw = AD.run(qc, shots=shots, seed=seed)
    counts = ecdlp_counts(raw)
    info["aer"] = AD.describe(qc.num_qubits)
    cands = C.ecdlp_postprocess(S.pair_counts(counts), order, mb)
    return counts, cands, info


# =============================================================================
# exact all-branch basis simulation with affine phase tracking
# =============================================================================


class BranchError(AssertionError):
    pass


def branch_sim(qc, init=None, checks=()):
    """One basis state through a vented dynamic circuit, all branches at once.

    Returns (bits, s0, s) where the branch with vent outcomes m (in order of
    the measurements) maps the input to |bits> with amplitude
    2^(-V/2) (-1)^(s0 + sum_i m_i s[i]).  Raises unless every measurement is
    an H-then-measure vent of a basis qubit, every reset follows a vent (or
    hits a qubit that is already |0>), and every feed-forward body is
    diagonal and conditioned on one clbit with value 1.
    `checks` are `Machine.checks` (clean-ancilla assertions by position).
    """
    idx = {q: i for i, q in enumerate(qc.qubits)}
    cidx = {c: i for i, c in enumerate(qc.clbits)}
    bits = [0] * qc.num_qubits
    for k, v in (init or {}).items():
        bits[idx[k] if k in idx else k] = int(v) & 1
    state = {"s0": 0, "s": [], "pend": {}, "clev": {}}
    by_pos = {}
    for pos, qs, val in checks:
        by_pos.setdefault(pos, []).append((qs, val))

    def check(pos):
        for qs, val in by_pos.get(pos, ()):
            got = sum(bits[idx[q]] << i for i, q in enumerate(qs))
            if got != val:
                raise BranchError(f"ancilla check failed at {pos}: {got} != {val}")

    def apply(op, w, cw, phase_sink):
        name = op.name
        pend = state["pend"]
        if name == "barrier":
            return
        if any(i in pend for i in w) and name != "measure":
            raise BranchError(f"{name} on a qubit in X superposition")
        if name == "h":
            pend[w[0]] = bits[w[0]]
            return
        if name == "measure":
            if w[0] not in pend:
                raise BranchError("measurement that is not a vent")
            if phase_sink is not None:
                raise BranchError("measurement inside a feed-forward body")
            b = pend.pop(w[0])
            state["clev"][cw[0]] = len(state["s"])
            state["s"].append(b)                 # (-1)^(m b)
            bits[w[0]] = None                    # random until reset
            return
        if name == "reset":
            # only a vented (just measured) or already-clean qubit may be
            # reset; resetting live data would hide decoherence
            if bits[w[0]] not in (None, 0):
                raise BranchError("reset of a qubit that was not vented")
            bits[w[0]] = 0
            return
        if any(bits[i] is None for i in w):
            raise BranchError(f"{name} on a measured, un-reset qubit")
        if name == "if_else":
            cond = op.condition
            if (not isinstance(cond[0], Clbit) or cond[1] != 1
                    or (len(op.blocks) > 1 and op.blocks[1] is not None)):
                raise BranchError("only `if clbit == 1` bodies are supported")
            ev = state["clev"][cidx[cond[0]]]
            body = op.blocks[0]
            before = list(bits)
            acc = [0]
            sub = {q: w[i] for i, q in enumerate(body.qubits)}
            csub = {c: cw[i] for i, c in enumerate(body.clbits)}
            for ci in body.data:
                apply(ci.operation, [sub[q] for q in ci.qubits],
                      [csub[c] for c in ci.clbits], acc)
            if bits != before:
                raise BranchError("feed-forward body is not diagonal")
            state["s"][ev] ^= acc[0]
            return
        if name in ("z", "cz", "ccz"):
            v = 1
            for i in w:
                v &= bits[i]
            if phase_sink is None:
                state["s0"] ^= v
            else:
                phase_sink[0] ^= v
            return
        if name == "id":
            return
        if name == "x":
            bits[w[0]] ^= 1
            return
        if name == "swap":
            bits[w[0]], bits[w[1]] = bits[w[1]], bits[w[0]]
            return
        if name == "ecand":
            if bits[w[2]] != 0:
                raise BranchError("AND target not clean")
            bits[w[2]] = bits[w[0]] & bits[w[1]]
            return
        if name == "ecand_dg":
            if bits[w[2]] != bits[w[0]] & bits[w[1]]:
                raise BranchError("AND-dagger target mismatch")
            bits[w[2]] = 0
            return
        base, nctrl, cstate = None, 0, None
        if isinstance(op, ControlledGate):
            base, nctrl, cstate = op.base_gate, op.num_ctrl_qubits, op.ctrl_state
        elif name in ("cx", "ccx", "mcx", "c3x", "c4x"):
            base, nctrl, cstate = XGate(), len(w) - 1, (1 << (len(w) - 1)) - 1
        elif name in ("cswap", "fredkin"):
            base, nctrl, cstate = SwapGate(), 1, 1
        if base is not None and base.name in ("x", "swap"):
            if all(bits[c] == ((cstate >> i) & 1) for i, c in enumerate(w[:nctrl])):
                t = w[nctrl:]
                if base.name == "x":
                    bits[t[0]] ^= 1
                else:
                    bits[t[0]], bits[t[1]] = bits[t[1]], bits[t[0]]
            return
        d = getattr(op, "definition", None)
        if d is None:
            raise BranchError(f"unsupported gate {name!r}")
        sub = {q: w[i] for i, q in enumerate(d.qubits)}
        for ci in d.data:
            apply(ci.operation, [sub[q] for q in ci.qubits], [], phase_sink)

    for pos, ci in enumerate(qc.data):
        check(pos)
        apply(ci.operation, [idx[q] for q in ci.qubits],
              [cidx[c] for c in ci.clbits], None)
    check(len(qc.data))
    if state["pend"] or any(b is None for b in bits):
        raise BranchError("circuit ends with an unvented or un-reset qubit")
    return bits, state["s0"], state["s"]


def toffoli_count(qc):
    """Toffoli-class gates, recursing into feed-forward bodies."""
    n = 0
    for ci in qc.data:
        name = ci.operation.name
        if name in ("ccx", "ecand", "cswap", "ccz", "mcx"):
            n += 1
        for b in getattr(ci.operation, "blocks", ()) or ():
            if b is not None:
                n += toffoli_count(b)
    return n


def measure_count(qc):
    n = 0
    for ci in qc.data:
        n += ci.operation.name == "measure"
        for b in getattr(ci.operation, "blocks", ()) or ():
            if b is not None:
                n += measure_count(b)
    return n
