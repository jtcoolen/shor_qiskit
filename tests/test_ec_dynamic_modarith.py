"""Gidney 2507.23079 dirty-ancilla dynamic adder composed into modular arithmetic.

Exactness is proven three ways:
  * `branch_sim`: every basis input, every dirty state, both controls; the
    phase of all 2^V vent branches is an affine function of the outcomes whose
    coefficients must not depend on the input  ->  exact on every branch.
  * a numpy statevector that follows *every* branch of the dynamic circuit on
    a random entangled superposition (reset + feed-forward included).
  * Aer per-shot statevectors on the same kind of state.
Run: PYTHONPATH=shor_qiskit:tests venv/bin/python tests/test_ec_dynamic_modarith.py
"""
import itertools
import os
import random
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shor_qiskit"))
import ec_cost  # noqa: E402
import ec_dynamic_modarith as D  # noqa: E402
import ec_modarith as MA  # noqa: E402
from ec_sim import Machine, simulate  # noqa: E402

PRIMES = (3, 5, 7, 11, 13)


def _val(bits, qc, reg):
    idx = {q: i for i, q in enumerate(qc.qubits)}
    return sum(bits[idx[q]] << i for i, q in enumerate(reg))


def _init(regs_vals):
    init = {}
    for reg, v in regs_vals:
        for i, q in enumerate(reg):
            init[q] = (v >> i) & 1
    return init


class _Sig:
    """All inputs must share one (s0, s) phase signature."""

    def __init__(self):
        self.sig = None

    def __call__(self, s0, s, where):
        cur = (s0, tuple(s))
        if self.sig is None:
            self.sig = cur
        assert cur == self.sig, f"input-dependent branch phase at {where}: {cur} vs {self.sig}"


def _adder_case(n, w1, w0, controlled, comparator=False):
    t = QuantumRegister(n, "t")
    g = QuantumRegister(max(n - (1 if comparator else 2), 0), "g")
    c = QuantumRegister(2, "c")
    k = QuantumRegister(1, "k")
    o = QuantumRegister(1, "o")
    qc = QuantumCircuit(t, g, c, k, o)
    lits = D.select_literals(n, w1, w0, k[0] if controlled else None)
    if comparator:
        D.append_literal_carry_out(qc, list(t), list(g), list(c), lits, o[0])
    else:
        D.append_literal_adder(qc, list(t), list(g), list(c), lits)
    sig = _Sig()
    for x, gv, kv, ov in itertools.product(range(1 << n), range(1 << len(g)),
                                           (0, 1) if controlled else (1,), (0, 1)):
        bits, s0, s = D.branch_sim(qc, _init([(t, x), (g, gv), (k, kv), (o, ov)]))
        dd = w1 if kv else w0
        want_t = x if comparator else (x + dd) % (1 << n)
        want_o = ov ^ (x + dd >= 1 << n) if comparator else ov
        assert _val(bits, qc, t) == want_t, (n, w1, w0, x, kv)
        assert _val(bits, qc, o) == want_o, (n, w1, w0, x, kv, "carry")
        assert _val(bits, qc, g) == gv and _val(bits, qc, c) == 0
        assert _val(bits, qc, k) == kv
        sig(s0, s, (n, w1, w0, x, gv, kv))
    return qc


def test_literal_adder_exhaustive():
    rnd = random.Random(1)
    for n in range(1, 6):
        for d in range(1 << n):
            _adder_case(n, d, 0, False)
        for _ in range(12):
            _adder_case(n, rnd.randrange(1 << n), rnd.randrange(1 << n), True)
    # paper count: 3n-7 Toffolis for a classical offset with d_0 = 1
    for n in range(3, 10):
        assert D.toffoli_count(_adder_case(n, (1 << n) - 1, 0, False) if n < 6 else
                               _build_adder(n, (1 << n) - 1)) == 3 * n - 7


def _build_adder(n, d):
    t, g, c = QuantumRegister(n), QuantumRegister(n - 2), QuantumRegister(2)
    qc = QuantumCircuit(t, g, c)
    D.append_literal_adder(qc, list(t), list(g), list(c), D.select_literals(n, d))
    return qc


def test_literal_carry_out_exhaustive():
    rnd = random.Random(2)
    for n in range(1, 6):
        for d in range(1 << n):
            _adder_case(n, d, 0, False, comparator=True)
        for _ in range(12):
            _adder_case(n, rnd.randrange(1 << n), rnd.randrange(1 << n), True,
                        comparator=True)


def _modadd_machine(p, k, controlled, sub=False):
    n = p.bit_length()
    m = Machine("toffoli", "dyn")
    y = m.alloc(n, "y")
    g = m.alloc(n - 1, "g")
    q = m.alloc(1, "q")
    fn = D.cmodsub_const_dyn if sub else D.cmodadd_const_dyn
    fn(m, q[0] if controlled else None, y, k, p, g)
    return m, y, g, q


def _check_modadd(p, k, controlled, sub=False):
    n = p.bit_length()
    m, y, g, q = _modadd_machine(p, k, controlled, sub)
    sig = _Sig()
    for yv, gv, qv in itertools.product(range(p), range(1 << (n - 1)),
                                        (0, 1) if controlled else (1,)):
        bits, s0, s = D.branch_sim(m.qc, _init([(y, yv), (g, gv), (q, qv)]), m.checks)
        sgn = -1 if sub else 1
        assert _val(bits, m.qc, y) == (yv + sgn * qv * k) % p, (p, k, yv, qv)
        assert _val(bits, m.qc, g) == gv and _val(bits, m.qc, q) == qv
        assert sum(bits) == sum(bits[i] for i in range(len(bits))
                                if m.qc.qubits[i] in set(y) | set(g) | set(q))
        sig(s0, s, (p, k, yv, gv, qv))
    return m


def test_modadd_all_primes_controls_dirty_states():
    for p in PRIMES:
        for k in range(1, p):
            for controlled in (False, True):
                for sub in (False, True):
                    _check_modadd(p, k, controlled, sub)


def test_compare_against_unitary_cmodadd_const():
    """Same toy task, same answers; dynamic needs 3 clean qubits, not 3n+1.

    Toffolis are compared with the unitary builder in plain-Toffoli mode; its
    AND mode (measurement-uncomputed ANDs) is cheaper -- see the benchmark.
    """
    for p in PRIMES:
        n = p.bit_length()
        for k in (1, p // 2 + 1, p - 1):
            mu = Machine("toffoli", "u")
            yu, qu = mu.alloc(n, "y"), mu.alloc(1, "q")
            MA.cmodadd_const(mu, qu[0], yu, k, p)
            md, yd, gd, qd = _modadd_machine(p, k, True)
            for yv, qv in itertools.product(range(p), (0, 1)):
                bu = simulate(mu.qc, _init([(yu, yv), (qu, qv)]), mu.checks)
                bd, _, _ = D.branch_sim(md.qc, _init([(yd, yv), (qd, qv)]), md.checks)
                assert _val(bu, mu.qc, yu) == _val(bd, md.qc, yd)
            tu = ec_cost.count(mu)["toffoli_paper"]
            td = D.toffoli_count(md.qc)
            clean_u = mu.qc.num_qubits - n - 1
            clean_d = md.qc.num_qubits - n - 1 - (n - 1)
            assert clean_d == 3 < clean_u, (p, k, clean_d, clean_u)
            assert td < tu, (p, k, td, tu)


def test_emit_inverse_refuses_dynamic_circuit():
    m = Machine("toffoli", "x")
    y, g = m.alloc(3, "y"), m.alloc(2, "g")
    try:
        m.emit_inverse(D.modadd_const_dyn, m, y, 3, 7, g)
    except Exception:
        return
    raise AssertionError("emit_inverse must not silently reverse measurements")


def test_branch_sim_rejects_reset_of_live_data():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.reset(1)
    try:
        D.branch_sim(qc, {0: 1})
    except D.BranchError:
        return
    raise AssertionError("reset of input-dependent garbage must be rejected")


def test_rejects_overlapping_dirty_workspace():
    m = Machine("toffoli", "x")
    y = m.alloc(3, "y")
    try:
        D.modadd_const_dyn(m, y, 3, 7, list(y)[:2])
    except ValueError:
        return
    raise AssertionError("dirty workspace overlapping the target must be rejected")


# -----------------------------------------------------------------------------
# full-branch statevector (numpy) on entangled superpositions
# -----------------------------------------------------------------------------


def _apply_gate(psi, mat, qs, nq):
    psi = psi.reshape([2] * nq)
    ax = [nq - 1 - q for q in qs][::-1]
    k = len(qs)
    psi = np.moveaxis(psi, ax, list(range(k)))
    sh = psi.shape
    psi = (mat @ psi.reshape(2 ** k, -1)).reshape(sh)
    return np.moveaxis(psi, list(range(k)), ax).reshape(-1)


def _branches(qc, psi):
    """[(prob, normalized state)] over every measurement branch."""
    from qiskit.quantum_info import Operator
    nq = qc.num_qubits
    qidx = {q: i for i, q in enumerate(qc.qubits)}
    cidx = {c: i for i, c in enumerate(qc.clbits)}
    out = [(1.0, psi, [0] * qc.num_clbits)]

    def run(data, br, qmap, cmap):
        for ci in data:
            op = ci.operation
            qs = [qmap[q] for q in ci.qubits]
            cs = [cmap[c] for c in ci.clbits]
            nb = []
            for pr, st, cl in br:
                if op.name == "barrier":
                    nb.append((pr, st, cl))
                elif op.name in ("measure", "reset"):
                    v = st.reshape([2] * nq)
                    for b in (0, 1):
                        proj = np.zeros_like(v)
                        sl = [slice(None)] * nq
                        sl[nq - 1 - qs[0]] = b
                        proj[tuple(sl)] = v[tuple(sl)]
                        pb = np.vdot(proj, proj).real
                        if pb < 1e-12:
                            continue
                        new = proj.reshape(-1) / np.sqrt(pb)
                        ncl = list(cl)
                        if op.name == "measure":
                            ncl[cs[0]] = b
                            nb.append((pr * pb, new, ncl))
                        else:
                            if b:
                                new = _apply_gate(new, np.array([[0, 1], [1, 0]]), qs, nq)
                            nb.append((pr * pb, new, ncl))
                    if op.name == "reset":
                        # reset is a channel: merge the two pure pieces only if
                        # they coincide (they must, for clean deletion)
                        a = nb[-2:] if len(nb) >= 2 and nb[-1][2] == nb[-2][2] else None
                        if a is not None and abs(abs(np.vdot(a[0][1], a[1][1])) - 1) > 1e-9:
                            raise AssertionError("reset of an entangled qubit")
                        if a is not None:
                            nb[-2:] = [(a[0][0] + a[1][0], a[0][1], a[0][2])]
                elif op.name == "if_else":
                    clb, val = op.condition
                    if cl[cidx[clb]] == val:
                        body = op.blocks[0]
                        sub = run(body.data, [(pr, st, cl)],
                                  {q: qs[i] for i, q in enumerate(body.qubits)},
                                  {c: cs[i] for i, c in enumerate(body.clbits)})
                        nb.extend(sub)
                    else:
                        nb.append((pr, st, cl))
                else:
                    mat = Operator(op).data
                    nb.append((pr, _apply_gate(st, mat, qs, nq), cl))
            br = nb
        return br

    return run(qc.data, out, qidx, cidx)


def _rand_state(nq, support, rng):
    psi = np.zeros(2 ** nq, complex)
    for idx in support:
        psi[idx] = rng.normal() + 1j * rng.normal()
    return psi / np.linalg.norm(psi)


def test_numpy_all_branches_modadd_entangled():
    """y and dirty g in superposition, entangled with a reference qubit."""
    rng = np.random.default_rng(3)
    for p, k, controlled in ((5, 3, True), (7, 4, True), (7, 6, False), (5, 2, False)):
        n = p.bit_length()
        m, y, g, q = _modadd_machine(p, k, controlled)
        qc = m.qc.copy()
        r = QuantumRegister(1, "ref")
        qc.add_register(r)
        nq = qc.num_qubits
        qi = {b: i for i, b in enumerate(qc.qubits)}

        def index(yv, gv, qv, rv):
            ii = 0
            for reg, v in ((y, yv), (g, gv), (q, qv), (r, rv)):
                for j, b in enumerate(reg):
                    ii |= ((v >> j) & 1) << qi[b]
            return ii
        sup = [index(yv, gv, qv, rv) for yv in range(p) for gv in range(1 << (n - 1))
               for qv in (0, 1) for rv in (0, 1)]
        psi = _rand_state(nq, sup, rng)
        want = np.zeros_like(psi)
        for yv in range(p):
            for gv in range(1 << (n - 1)):
                for qv in (0, 1):
                    for rv in (0, 1):
                        add = k if (qv or not controlled) else 0
                        want[index((yv + add) % p, gv, qv, rv)] = psi[index(yv, gv, qv, rv)]
        brs = _branches(qc, psi)
        assert abs(sum(b[0] for b in brs) - 1) < 1e-9
        for pr, st, _ in brs:
            assert abs(abs(np.vdot(want, st)) ** 2 - 1) < 1e-9, (p, k, pr)


def test_aer_pershot_fidelity_modadd():
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    from qiskit import transpile
    rng = np.random.default_rng(4)
    for p, k in ((5, 3), (7, 5)):
        n = p.bit_length()
        m, y, g, q = _modadd_machine(p, k, True)
        body = m.qc
        qc = body.copy_empty_like()
        r = QuantumRegister(1, "ref")
        qc.add_register(r)
        nq = qc.num_qubits
        qi = {b: i for i, b in enumerate(qc.qubits)}

        def index(yv, gv, qv, rv):
            ii = 0
            for reg, v in ((y, yv), (g, gv), (q, qv), (r, rv)):
                for j, b in enumerate(reg):
                    ii |= ((v >> j) & 1) << qi[b]
            return ii
        sup = [(yv, gv, qv, rv) for yv in range(p) for gv in range(1 << (n - 1))
               for qv in (0, 1) for rv in (0, 1)]
        psi = _rand_state(nq, [index(*s) for s in sup], rng)
        want = np.zeros_like(psi)
        for yv, gv, qv, rv in sup:
            want[index((yv + qv * k) % p, gv, qv, rv)] = psi[index(yv, gv, qv, rv)]
        qc.initialize(psi, qc.qubits)
        qc.compose(body, qubits=list(qc.qubits)[:body.num_qubits],
                   clbits=list(qc.clbits), inplace=True)
        qc.save_statevector(pershot=True)
        sim = AerSimulator(method="statevector", seed_simulator=11)
        res = sim.run(transpile(qc, sim, optimization_level=0), shots=64,
                      memory=True).result()
        svs = res.data(0)["statevector"]
        mem = res.get_memory(0)
        for sv in svs:
            f = abs(np.vdot(want, Statevector(sv).data)) ** 2
            assert f > 1 - 1e-9, (p, k, f)
        assert len(set(mem)) > 4, "too few distinct vent branches sampled"


# -----------------------------------------------------------------------------
# toy affine ECDLP with dynamic constant additions
# -----------------------------------------------------------------------------


def test_point_add_ctrl_dyn_basis_all_inputs():
    """The full dynamic oracle, every (u, v), all vent branches, scratch clean."""
    for inversion in ("kaliski", "divsteps"):
        _oracle_basis(inversion)


def _oracle_basis(inversion):
    import ec_shor as S
    curve, P, Q, order = _toy_curve()
    _, info = D.ecdlp_circuit_dyn(curve, P, Q, order, m_bits=2, inversion=inversion)
    m = info["machine"]
    kr, lr, px, py = info["regs"]
    Sp = info["offset"]
    Ps, Qs = S._rungs(curve, P, Q, 2)
    sig = _Sig()
    for u in range(4):
        for v in range(4):
            init = _init([(kr, u), (lr, v)])
            bits, s0, s = D.branch_sim(m.qc, init, m.checks)
            R = Sp
            for i in range(2):
                if (u >> i) & 1:
                    R = curve.add(R, Ps[i])
                if (v >> i) & 1:
                    R = curve.add(R, Qs[i])
            assert (_val(bits, m.qc, px), _val(bits, m.qc, py)) == (R.x, R.y), (u, v)
            assert _val(bits, m.qc, kr) == u and _val(bits, m.qc, lr) == v
            persistent = set(kr) | set(lr) | set(px) | set(py)
            assert all(b == 0 for i, b in enumerate(bits)
                       if m.qc.qubits[i] not in persistent), "scratch not clean"
            sig(s0, s, (u, v))


def _toy_curve():
    import ec_classical as C
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    return curve, P, Q, curve.point_order(P)


def test_toy_ecdlp_aer_e2e():
    """Sampled distribution matches the ideal Shor-ECDLP distribution."""
    if os.environ.get("SKIP_DYN_E2E"):
        print("  (skipped by SKIP_DYN_E2E)")
        return
    import shor_stats as ST
    import ec_shor as S
    curve, P, Q, order = _toy_curve()
    shots = int(os.environ.get("DYN_E2E_SHOTS", "256"))
    # kaliski is 67 qubits, above Aer's 63-qubit MPS limit; divsteps is 63
    counts, cands, info = D.run_ecdlp_dyn(curve, P, Q, order, shots=shots, seed=5,
                                          m_bits=2, inversion="divsteps")
    k_true = next(k for k in range(order) if curve.mul(k, P) == Q)
    assert cands and cands[0][0] == k_true, (cands[:3], k_true)
    emp = ST.empirical(S.pair_counts(counts), (4, 4))
    ideal = ST.ecdlp_probs(order, k_true, 2)
    tvd, null = ST.tvd(emp, ideal), ST.tvd_null(ideal, shots)
    print(f"  e2e: {info['qubits']} qubits, {info['aer']}, TVD {tvd:.3f} (null {null:.3f})")
    assert tvd <= null, (tvd, null)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"{name}: ok ({time.time() - t:.1f}s)")
