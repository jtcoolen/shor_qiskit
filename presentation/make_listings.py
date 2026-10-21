"""Code excerpts for shor_circuits.tex, taken from the source so they cannot drift.

    ./venv/bin/python presentation/make_listings.py

* whole functions are cut out of shor_qiskit/ by AST, docstrings removed;
* abridged excerpts are checked line by line against their source file
  (every line except `...` must occur there verbatim);
* usage snippets are executed, and the values they print are asserted.
"""
import ast
import pathlib
import subprocess
import sys
import textwrap

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "shor_qiskit"
OUT = HERE / "listings"
OUT.mkdir(exist_ok=True)


def function(path, name):
    """The source of one function, docstring removed."""
    src = path.read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = src.splitlines()[fn.lineno - 1:fn.end_lineno]
    doc = fn.body[0]
    if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
        del lines[doc.lineno - fn.lineno:doc.end_lineno - fn.lineno + 1]
    return textwrap.dedent("\n".join(lines))


def whole(out, *parts):
    (OUT / out).write_text("\n\n".join(function(SRC / f, n) for f, n in parts) + "\n")
    print("  wrote", out)


def abridged(out, source, text):
    text = textwrap.dedent(text).strip("\n") + "\n"
    have = {ln.strip() for ln in source.read_text().splitlines()}
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith("...") and s not in have:
            raise SystemExit(f"{out}: line not in {source.name}: {s!r}")
    (OUT / out).write_text(text)
    print("  wrote", out, "(checked against", source.name + ")")


def executed(out, text, expect):
    """Run a usage snippet; its printed output must equal `expect`."""
    text = textwrap.dedent(text).strip("\n") + "\n"
    env = {"PYTHONPATH": f"{SRC}:{ROOT / 'tests'}", "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
    code = "import warnings; warnings.filterwarnings('ignore')\n" + text
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    if got.returncode:
        raise SystemExit(f"{out} failed:\n{got.stderr}")
    if got.stdout.strip() != expect.strip():
        raise SystemExit(f"{out}: printed {got.stdout.strip()!r}, expected {expect.strip()!r}")
    (OUT / out).write_text(text)
    print("  wrote", out, "(executed:", got.stdout.strip().replace("\n", " | ") + ")")


whole("qft.py", ("shor_essentials.py", "qft"))
whole("add_const.py", ("shor_essentials.py", "add_const"), ("shor_essentials.py", "c_add_const"))
whole("c_add_mod.py", ("shor_essentials.py", "c_add_mod"))
whole("c_ua.py", ("shor_essentials.py", "c_mult_acc"), ("shor_essentials.py", "c_ua"))
whole("order_circuit.py", ("shor_essentials.py", "order_circuit"))
whole("order_from_counts.py", ("shor_essentials.py", "order_from_counts"))
whole("semiclassical.py", ("semiclassical.py", "semiclassical_iqft"))
whole("rc_add.py", ("rc_adder.py", "rc_add"))
whole("windowed.py", ("windowed.py", "windowed_c_mult_acc"))

abridged("find_factor.py", SRC / "shor_essentials.py", """
    while True:
        a = random.randrange(2, N)
        if (d := math.gcd(a, N)) > 1:
            return d                         # lucky gcd
        t = 2 * math.ceil(math.log2(N))
        r = order_from_counts(run(order_circuit(a, N)), a, N, t)
        if r and r % 2 == 0:
            d = math.gcd(pow(a, r // 2, N) - 1, N)
            if 1 < d < N:
                return d
""")

abridged("ecdlp_circuit.py", SRC / "ec_shor.py", """
    qc = QuantumCircuit(kr, lr, px, py, ck, cl)
    ...                                     # accumulator <- S  (X gates)
    qc.h(kr)
    qc.h(lr)
    pack = list(px) + list(py)
    Ps, Qs = _rungs(curve, P, Q, mb)
    for i, R in enumerate(Ps):
        permutation(qc, pack, point_perm(curve, R, n), ctrls=[kr[i]])
    for i, R in enumerate(Qs):
        permutation(qc, pack, point_perm(curve, R, n), ctrls=[lr[i]])
    qc.append(qft(mb).inverse(), list(kr))
    qc.append(qft(mb).inverse(), list(lr))
    qc.measure(kr, ck)
    qc.measure(lr, cl)
""")

abridged("ecdlp_post.py", SRC / "ec_classical.py", """
    votes, q = {}, 1 << bits
    for (j1, j2), c in items:
        for da in range(-search, search + 1):
            for db in range(-search, search + 1):
                a1 = (round(j1 * order / q) + da) % order
                a2 = (round(j2 * order / q) + db) % order
                if gcd(a1, order) != 1:
                    continue
                k = a2 * pow(a1, -1, order) % order
                w = c / (1 + abs(da) + abs(db)) ** 2
                votes[k] = votes.get(k, 0.0) + w
    return sorted(votes.items(), key=lambda kv: -kv[1])
""")

abridged("garbage.py", HERE / "make_figures.py", """
    for i in range(t):
        c_ua(qc, ctr[i], pow(A, 2**i, N), list(tgt), list(anc), sf[0], sf[1], N)
    if garbage == "x":
        for i in range(t):
            qc.cx(ctr[i], g[i])               # a copy of x left behind
    qc.append(qft(t).inverse(), ctr)
""")

executed("run_aer.py", """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from shor_essentials import order_circuit, order_from_counts

    sim = AerSimulator(seed_simulator=2026)
    qc = order_circuit(7, 15)                        # a = 7, N = 15: 18 qubits
    counts = sim.run(transpile(qc, sim), shots=4096).result().get_counts()
    r = order_from_counts(counts, 7, 15, t=8)
    print(qc.num_qubits, r)
""", "18 4")

executed("resources.py", """
    import resources as R
    from shor_essentials import order_circuit

    r = R.count(order_circuit(7, 15))
    print(r.qubits, r.toffoli, r.cphase, r.rotations)
    print(r.T, r.T_with_rotations(1e-3))
""", "18 864 7068 12855\n14309 978434")

executed("ecdlp_run.py", """
    # the toy curve y^2 = x^3 + 5x + 4 mod 7, with G of order 5
    from ec_classical import CLASSIQ as curve, CLASSIQ_G as G
    from ec_shor import solve

    Q = curve.mul(4, G)                         # public key: Q = [4]G
    # build the circuit, run it on Aer, post-process the counts
    k, counts, info = solve(curve, G, Q, order=5)
    print(k, info["qubits"])
""", "4 12")

executed("qiskit_idioms.py", """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    def add_one(qc, y):                  # a builder appends gates
        for i in reversed(range(1, len(y))):
            qc.mcx(y[:i], y[i])          # carry: multi-controlled X
        qc.x(y[0])

    y, out = QuantumRegister(3, "y"), ClassicalRegister(3, "out")
    qc = QuantumCircuit(y, out)
    qc.x([y[0], y[1]])                   # y = 3, qubit 0 = lowest bit
    add_one(qc, y)                       # y = 4
    inc = QuantumCircuit(3)
    add_one(inc, inc.qubits)
    qc.append(inc.to_gate().inverse(), y)  # run backwards: y = 3
    qc.measure(y, out)
    sim = AerSimulator()
    print(sim.run(transpile(qc, sim), shots=100).result().get_counts())
""", "{'011': 100}")

ECDLP_BUILDER = """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from ec_shor import permutation, point_perm
    from shor_essentials import qft

    def ecdlp_circuit(curve, P, Q, m, S):
        n = curve.p.bit_length()
        u, v = QuantumRegister(m, "u"), QuantumRegister(m, "v")
        pt = QuantumRegister(2 * n, "point")               # x, then y
        qc = QuantumCircuit(u, v, pt, ClassicalRegister(m, "j1"), ClassicalRegister(m, "j2"))
        for i in range(2 * n):                             # accumulator <- S
            if ((S.x + (S.y << n)) >> i) & 1:
                qc.x(pt[i])
        qc.h(u)
        qc.h(v)
        for i in range(m):                                 # controlled +[2^i]P and +[2^i]Q
            permutation(qc, pt, point_perm(curve, curve.mul(2**i, P), n), ctrls=[u[i]])
            permutation(qc, pt, point_perm(curve, curve.mul(2**i, Q), n), ctrls=[v[i]])
        qc.append(qft(m).inverse(), u)
        qc.append(qft(m).inverse(), v)
        qc.measure(u, qc.cregs[0])
        qc.measure(v, qc.cregs[1])
        return qc
"""

# the clean builder must produce exactly the library circuit's output distribution
executed("ecdlp_builder_check.py", ECDLP_BUILDER + """
    import numpy as np
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from ec_classical import CLASSIQ as curve, CLASSIQ_G as G
    from ec_shor import ecdlp_circuit as library

    def probs(qc):
        qc = qc.remove_final_measurements(inplace=False)
        qc.save_probabilities(list(range(6)))
        sim = AerSimulator()
        return np.asarray(sim.run(transpile(qc, sim, optimization_level=0)).result().data(0)["probabilities"])

    Q = curve.mul(4, G)
    ref, info = library(curve, G, Q, 5, oracle="table")
    mine = ecdlp_circuit(curve, G, Q, info["m_bits"], info["offset"])
    print(np.allclose(probs(mine), probs(ref)))
""", "True")
(OUT / "ecdlp_builder.py").write_text(textwrap.dedent(ECDLP_BUILDER).strip("\n").split("\n\n", 1)[1] + "\n")
(OUT / "ecdlp_builder_check.py").unlink()
print("  wrote ecdlp_builder.py (its distribution equals ec_shor's)")
executed("qft_readout.py", """
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from shor_essentials import qft

    t, phi = 3, 1 / 4                   # s/r = 1/4, t = 3
    qc = QuantumCircuit(t, t)
    qc.h(range(t))
    for q in range(t):                  # the ramp the rungs leave:
        qc.p(2 * np.pi * phi * 2**q, q) # qubit q turns 2^q s/r
    qc.append(qft(t).inverse(), range(t))  # ramp -> integer
    qc.measure(range(t), range(t))
    sim = AerSimulator()
    print(sim.run(transpile(qc, sim), shots=1000).result().get_counts())
""", "{'010': 1000}")
print("listings done")

# --- Act XI: the 2026 optimisations ------------------------------------------------
whole("ci_add.py", ("ec_adders.py", "ci_add"))
abridged("ci_round.py", SRC / "ec_gcd.py", """
    def _round(self, m, U, VT, rec, w):
        ctx = m.ctx
        a, mm = rec
        u, vt = Reg(U[:w + 1]), Reg(VT[:w + 1])
        ctx.cx(vt[1], a[0])
        ctx.cx(u[1], a[0])                               # a = v~[1] ^ u[1]
        ...
        A.ci_add(ctx, a[0], u, vt, anc)                  # v~ += (-1)^a u
        ...
        for i in range(len(vt) - 1):                     # v~ /= 2 (even)
            ctx.swap(vt[i], vt[i + 1])
        ...
        A.gt_top(ctx, u, vt, mm[0], sc, k, ctrl=a[0])    # m = a AND [u > v~]
        ...
        for i in range(len(u)):
            ctx.cswap(mm[0], u[i], vt[i])
""")
abridged("sqr_int.py", SRC / "ec_square.py", """
    for i in range(n):
        ctx.cx(z[n + i], z[n + i + 1])              # sign-extend by one bit
        h = list(x[i + 1:])
        for q in h + [pad[0]]:
            ctx.cx(x[i], q)                         # x_i ? NOT(0||h) : (0||h)
        operand = Reg(h + [pad[0], x[i]], "op")     # signed, sign = x_i
        window = Reg(z[2 * i + 1:n + i + 2], "win")
        ...
        A.sub(ctx, operand, window, anc)            # z -= signed operand
        ...
        for q in h + [pad[0]]:
            ctx.cx(x[i], q)
""")
abridged("select_swap.py", SRC / "ec_mbu.py", """
    lam = 1 << k
    lo, hi = list(addr[:k]), list(addr[k:])
    ...
    lookup(m, hi, [q for blk in blocks for q in blk], wide)
    for i in reversed(range(k)):                  # block 0 <- word a_lo
        s = 1 << i
        for j in range(s):
            for qa, qb in zip(blocks[j], blocks[j + s]):
                m.ctx.cswap(lo[i], qa, qb)
    ...
    unlookup(m, addr, junk, G, group=group)
""")
abridged("vented_add.py", SRC / "ec_cqadd.py", """
    for k in range(n):
        ...
            nxt = m.anc(1, "cq")[0]               # c_{k+1} = maj(x_k, d_k, c_k)
            ...
                ctx.cx(xs[k], cur)                # cur = x ^ c
                _flip(ctx, xs[k], _bit(d, k), ctrl)
                ctx.and_(xs[k], cur, nxt)         # (x ^ d)(x ^ c)
                _flip(ctx, xs[k], _bit(d, k), ctrl)
                ctx.cx(xs[k], nxt)                # maj = x ^ (x ^ d)(x ^ c)
                ctx.cx(xs[k], cur)                # cur = c again
        ...
            ctx.cx(cur, xs[k])
        _flip(ctx, xs[k], _bit(d, k), ctrl)
        if cur is not None and cur is not cin:    # vent c_k (k >= 1)
            ...
            m.free(Reg([cur]))
        cur = nxt
""")
abridged("vent.py", SRC / "ec_mbu.py", """
def vent(m, data, reg, f, key):
    g = VentGate([len(d) for d in data], len(reg), f, key)
    m.qc.append(g, [q for d in data for q in d] + list(reg))


def zfix(m, reg, key):
    m.qc.append(ZFixGate(len(reg), key), list(reg))
""")


# --- Acts VII, VIII and XII: every optimisation of shor-complete.tex ---------------
whole("rc_majuma.py", ("rc_adder.py", "_maj"), ("rc_adder.py", "_uma"))
whole("rc_c_add_mod.py", ("rc_adder.py", "rc_add_const"), ("rc_adder.py", "rc_c_add_mod"))
whole("lookup_ui.py", ("qrom.py", "_walk"), ("qrom.py", "lookup_ui"))
whole("phase_fixup.py", ("qrom.py", "phase_fixup"))
whole("add_quantum_mod.py", ("windowed.py", "add_quantum_mod"))
whole("coset_add.py", ("coset.py", "add"))
whole("cadd.py", ("ec_adders.py", "cadd"))
abridged("moddbl_pm.py", SRC / "ec_approx.py", """
def moddbl_pm(m, x, q, lsbs=None):
    ...
    u, f = pm
    ctx, n = m.ctx, len(x)
    ...
    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])                      # n+1 bits; xe[n] is the overflow
    cp, sc = m.anc(lsbs, "cp"), m.anc(lsbs, "sc")
    A.cadd_const(ctx, xe[n], Reg(list(xe[:lsbs])), f, cp, sc)
    m.free(cp, sc)
    ctx.cx(xe[0], xe[n])                          # clears the bit == subtracts 2^u
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
""")
whole("csa.py", ("ec_qcsa.py", "csa_compress"))
abridged("kaliski_opt.py", SRC / "ec_kaliski_opt.py", """
    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b[0], u, v, cp, sc)               # v -= u
    m.free(cp, sc)
    cp, sc = m.anc(W, "cp"), m.anc(W, "sc")
    A.cadd(ctx, b[0], r, s, cp, sc)               # s += r   (plain, not mod p)
    m.free(cp, sc)
    _rot_down(ctx, v)                             # v /= 2
    _rot_up(ctx, r)                               # r *= 2   (free: no reduction)
""")
whole("dialog_round.py", ("ec_eea.py", "dialog_round"))
whole("bezout_replay.py", ("ec_eea.py", "bezout_replay"))
abridged("jacobian.py", SRC / "ec_proj.py", """
    Z1sq = A(n, "Z1sq"); _, b1 = m.step(sqr, Z1, Z1sq)                    # 1
    U2 = A(n, "U2");     _, b2 = m.step(MU.modmul_const, m, Z1sq, x2, U2, p)   # 2
    Z1cu = A(n, "Z1cu"); _, b3 = m.step(mul, Z1sq, Z1, Z1cu)              # 3
    H = A(n, "H");       _, b4 = m.step(sub_into, U2, X1, H)              # 4
    ...
    T0 = A(n, "T0");     _, b12 = m.step(sub_into, Rsq, Hcu, T0)          # 12
    _, b13 = m.step(sub_into, T0, twoV, X3)                        # 13  OUTPUT
    T1 = A(n, "T1");     _, b14 = m.step(sub_into, V, X3, T1)             # 14
    T2 = A(n, "T2");     _, b15 = m.step(mul, R, T1, T2)                  # 15
    T3 = A(n, "T3");     _, b16 = m.step(mul, Y1, Hcu, T3)                # 16
    _, b17 = m.step(sub_into, T2, T3, Y3)                          # 17  OUTPUT
    _, b18 = m.step(mul, Z1, H, Z3)                                # 18  OUTPUT

    # unwind everything but the three outputs.  Note T0 goes back via step 12,
    # not step 13 -- step 13 wrote X3, which stays.
    for body in (b16, b15, b14, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1):
        m.undo(body)
""")
abridged("windowed_padd.py", SRC / "ec_window.py", """
    MA.is_nonzero(m, addr, c[0])                       # 1
    _lookup(m, one[0], addr, x1, xs)                   # 2
    _lookup(m, one[0], addr, y1, ys)
    MA.modsub(m, x1, x2, p)                            # 3
    MA.modsub(m, y1, y2, p)                            # 4
    _lookup(m, one[0], addr, x1, xs)                   # 5
    _lookup(m, one[0], addr, y1, ys)
    E.inplace_div(m, x2, y2, p, iters)                 # 6  y2 <- y2 / x2
    _lookup(m, one[0], addr, xx, x3s)                  # 7
    MA.modadd(m, xx, x2, p)                            # 8
    _lookup(m, one[0], addr, xx, x3s)                  # 9
    _csub_square(m, c[0], y2, x2, p)                   # 10 if c: x2 -= y2^2
    E.inplace_mul(m, x2, y2, p, iters)                 # 11 y2 <- y2 * x2
    MA.cmodneg(m, c[0], x2, p)                         # 12 if c: x2 <- -x2
    ...                                                # 13-17: add P back, unload
""")
abridged("modadd_pm_phase.py", SRC / "ec_approx.py", """
    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    sc = m.anc(n, "sc")
    A.add(ctx, x + ax, y + ay, sc)                   # ay = carry-out
    m.free(sc)
    cp, sc2 = m.anc(kappa, "cp"), m.anc(kappa, "sc2")
    A.cadd_const(ctx, ay[0], Reg(list(y[:kappa])), f, cp, sc2)   # 2^u -> f
    m.free(cp, sc2)
    ...
    def phase(qc, qs):
        ...
        qc.h(out)                                    # |->: the CNOT kicks back
        A.lt_uint(Ctx(qc, "and"), top_bits(yy, delta), top_bits(xx, delta), out,
                  anc[:delta])
        ...
    MB.mbu_flag(m, ay[0], list(x) + list(y), recompute, phase, nanc=delta + 1)
""")
abridged("signed.py", SRC / "ec_signedwin.py", """
    for addr, table in zip(addrs, tables):
        w = len(addr)
        s, low = addr[w - 1], Reg(list(addr[:w - 1]), "l")
        ctx.x(s)                                    # s <- [d < 0]
        for q in low:
            ctx.cx(s, q)                            # low <- l~, the |d| index
        if prev is None:
            neg(m, s, y2, p)                        # R <- -R when d < 0
        else:
            ps, plow = prev                         # close the last window and
            ctx.cx(ps, s)                           # open this one: one negation,
            neg(m, s, y2, p)                        # controlled by the XOR
            ctx.cx(ps, s)
            ...
        W.windowed_point_add_cfg(m, low, x2, y2, table, p, cfg)
        prev = (s, low)
""")
whole("cdkm_cadd.py", ("ec_space.py", "_cuma"), ("ec_space.py", "cdkm_cadd"))
abridged("inc_borrowed.py", SRC / "ec_space.py", """
def inc_borrowed(m, w, g):
    ctx, k = m.ctx, len(w)
    ...
    cy = m.anc(1, "icy")
    A.sub(ctx, Reg(g), Reg(list(w)), cy)
    for q in g:
        ctx.x(q)
    A.sub(ctx, Reg(g), Reg(list(w)), cy)
    for q in g:
        ctx.x(q)
    m.free(cy)
""")
whole("div3.py", ("ec_luo3.py", "div3"))
abridged("carry_xor.py", SRC / "ec_cqadd.py", """
    for k in range(K):                            # y_k in place, k < K
        _flip(ctx, xs[k], _bit(d, k), ctrl)
    for k in range(K - 1, 0, -1):                 # Q^{-1}: g_{k+1} ^= y_k g_k
        ctx.ccx(xs[k], g[k - 1], g[k])
    # P, k = 0: g_1 ^= d_0 ^ x_1 ^ y_0 (c_0 ^ d_0)
    ...
    for k in range(1, K):                         # P: g_{k+1} ^= x_k ^ x_{k+1} ^ y_k g_k
        orig(k, g[k])
        orig(k + 1, g[k])
        ctx.ccx(xs[k], g[k - 1], g[k])
    for k in range(1, K + 1):                     # w -> c: g_k ^= x_k
        orig(k, g[k - 1])
    for k in range(K):
        _flip(ctx, xs[k], _bit(d, k), ctrl)
""")
abridged("census.py", SRC / "ec_opt.py", """
    fired = set()
    for inp in samples:
        ...
        for pos, (name, w) in enumerate(prog):
            ...
            elif name in ("ccx", "ecand"):
                if bits[w[0]] & bits[w[1]]:
                    fired.add(pos)
                    bits[w[2]] ^= 1
            ...
    return fired
""")
abridged("boundary.py", SRC / "hier.py", """
    def wrapper(*args, **kwargs):
        ...
        (ea, ek), qubits = sh
        key = (label, ea, ek)
        node = _CACHE.get(key)
        if node is None:
            node = _build(fn, label, key, args, kwargs, qubits, m)
            ...
            _CACHE[key] = node
        anc = m.anc(node.n_anc, "h") if node.n_anc else []
        m.qc.append(NodeGate(node), list(qubits) + list(anc))
""")
abridged("ladder.py", SRC / "shor_essentials.py", """
    for i in range(t):                       # the ladder: t rungs
        c_ua(qc, ctr[i], pow(A, 2**i, N), tgt, anc, sf[0], sf[1], N)
""")
whole("and_compute.py", ("qrom.py", "and_compute"))
whole("and_uncompute.py", ("qrom.py", "and_uncompute"))
abridged("nested.py", SRC / "nested.py", """
    for i in range(0, len(e), we):
        ei = list(e[i:i + we])
        ne = len(ei)
        kes = [pow(k, (2**i) * v, N) for v in range(2**ne)]
        ...
        for j in range(0, n, wm):
            mi = A[j:j + wm]
            addr = ei + list(mi)
            table = [((kes[ad & ((1 << ne) - 1)] * (ad >> ne)) << j) % N
                     for ad in range(1 << len(addr))]
            lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])
            qc.append(madd, tmp[:n] + B + [sign, flag])
            lookup_ui(qc, one, addr, tmp[:n], table, anc[:len(addr)])
        ...                                       # A -= B * k_e^-1, the same way
        A, B = B, A                               # relabelling swap -- free
""")
abridged("inplace_mul_short.py", SRC / "ec_eea.py", """
    recs, u, v = dialog(m, x, q, iters)                  # 1: x -> dialog
    clear_dialog_end(m, u, v)

    s = m.anc(n, "bz")                                   # 2: replay onto (y, 0)
    bezout_replay(m, y, s, recs, q, dbl, cadd)
    for a, b in zip(y, s):                               # r is 0, s is the answer
        m.ctx.swap(a, b)
    m.free(s)
    ...                                                  # 3: dialog -> x
    for rec in reversed(recs):
        m.emit_inverse(dialog_round, m, u2, v2, rec)
    A.encode_const(m.ctx, u2, q)                         # and starts at u = q
""")
