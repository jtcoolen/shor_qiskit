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
