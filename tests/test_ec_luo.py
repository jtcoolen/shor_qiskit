"""ec_luo: Luo et al.'s register-shared extended Euclidean algorithm.

  * the classical model reproduces [Luo26] Table 4 (p = 37, x = 13) bit for
    bit, and every x at p in (7, 11, 13, 61, 127, 251) ends normalised
    (r = 1, r' = 0, q = 0, t = p) with x^{-1} in W2, within 4 ceil(c n) steps;
    [Luo26]'s extremal quotient sequences take 12k + 16 steps, inside it;
  * [Luo26] App. A.2's step-dependent active windows contain every position
    the schedule uses, at every step, for every x;
  * the circuit equals the model after EVERY microstep, for every x;
  * luo_inv, Luo.div and Luo.mul are exact and leave every ancilla clean;
  * qubits: the inversion is exactly [Luo26]'s 2n + 6 floor(log2 n) + 19, the
    division 4n + 6 floor(log2 n) + 19 -- set against ec_gcd.Dialog;
  * with Luo() as its multiplier, a windowed point addition adds correctly.

Exhaustive runs use a bit-parallel basis-state simulator (one input per bit
lane, the semantics of `ec_sim.simulate` including its ancilla checks),
cross-checked against `ec_sim.run` on sample inputs.
"""
import math
import random
import time

from _ec_util import FULL, ok, random_curve, random_generator, section

import ec_classical as C
import ec_cost as CO
import ec_gcd as G
import ec_luo as L
import ec_luo_classical as LC
import ec_window as W
from ec_sim import Machine, SimError, run
from qiskit.circuit import ControlledGate

PRIMES = (7, 11, 13, 61, 127, 251)


# =============================================================================
# Bit-parallel basis-state simulation
# =============================================================================
def _compile(qc):
    """Flatten to (name, qubit indices[, ctrl_state]) over X/CX/CCX/SWAP/
    CSWAP/MCX and the two AND gadgets, descending into definitions.  Also
    returns, per instruction of qc, where its flattened ops start."""
    idx = {q: i for i, q in enumerate(qc.qubits)}
    out = []

    def emit(op, qs):
        nm = op.name
        if nm in ("barrier", "measure", "id", "delay", "global_phase",
                  "z", "s", "sdg", "t", "tdg", "cz", "ccz", "p", "rz"):
            return
        if nm in ("x", "cx", "ccx", "swap", "cswap", "ecand", "ecand_dg"):
            out.append((nm, qs))
            return
        if nm in ("mcx", "c3x", "c4x", "mcx_gray") or (
                isinstance(op, ControlledGate) and op.base_gate.name == "x"):
            out.append(("mcx", qs, getattr(op, "ctrl_state", (1 << (len(qs) - 1)) - 1)))
            return
        d = op.definition
        sub = {q: qs[i] for i, q in enumerate(d.qubits)}
        for ci in d.data:
            emit(ci.operation, [sub[q] for q in ci.qubits])

    marks = []
    for ci in qc.data:
        marks.append(len(out))
        emit(ci.operation, [idx[q] for q in ci.qubits])
    marks.append(len(out))
    return out, marks, idx


class Lanes:
    """Simulate `m` on every {Reg: value} in `inputs` at once.  Enforces
    `m.checks` and the AND gadgets' preconditions on every lane.
    `advance(i)` runs up to instruction i of m.qc, incrementally."""

    def __init__(self, m, inputs):
        self.m = m
        self.ops, self.marks, self.idx = _compile(m.qc)
        self.ALL = (1 << len(inputs)) - 1
        self.b = [0] * m.qc.num_qubits
        for k, d in enumerate(inputs):
            for reg, v in d.items():
                for i, q in enumerate(reg):
                    if (v >> i) & 1:
                        self.b[self.idx[q]] |= 1 << k
        self.checks = {}
        for pos, qs, val in m.checks:
            assert val == 0
            self.checks.setdefault(self.marks[pos], []).extend(self.idx[q] for q in qs)
        self.pos = 0

    def advance(self, upto=None):
        b, ALL, ops = self.b, self.ALL, self.ops
        end = len(ops) if upto is None else self.marks[upto]
        for pos in range(self.pos, end):
            for q in self.checks.get(pos, ()):
                if b[q]:
                    raise SimError(f"ancilla check failed at op {pos}")
            o = ops[pos]
            nm, q = o[0], o[1]
            if nm == "x":
                b[q[0]] ^= ALL
            elif nm == "cx":
                b[q[1]] ^= b[q[0]]
            elif nm == "ccx":
                b[q[2]] ^= b[q[0]] & b[q[1]]
            elif nm == "swap":
                b[q[0]], b[q[1]] = b[q[1]], b[q[0]]
            elif nm == "cswap":
                t = (b[q[1]] ^ b[q[2]]) & b[q[0]]
                b[q[1]] ^= t
                b[q[2]] ^= t
            elif nm == "ecand":
                if b[q[2]]:
                    raise SimError("AND target was not |0>")
                b[q[2]] = b[q[0]] & b[q[1]]
            elif nm == "ecand_dg":
                if b[q[2]] != b[q[0]] & b[q[1]]:
                    raise SimError("AND-dagger target did not hold a AND b")
                b[q[2]] = 0
            else:                                       # mcx
                f = ALL
                for i, c in enumerate(q[:-1]):
                    f &= b[c] if (o[2] >> i) & 1 else ~b[c] & ALL
                b[q[-1]] ^= f
        self.pos = end
        if upto is None:
            for q in self.checks.get(len(ops), ()):
                if b[q]:
                    raise SimError("ancilla check failed at the end")
        return self

    def read(self, k, reg):
        return sum(((self.b[self.idx[q]] >> k) & 1) << i for i, q in enumerate(reg))

    def others_zero(self, keep):
        keep = set(keep)
        for q in self.m.qc.qubits:
            if q not in keep:
                assert self.b[self.idx[q]] == 0, "a workspace qubit came back dirty"


def lanes_run(m, inputs):
    return Lanes(m, inputs).advance()


# =============================================================================
# The classical model
# =============================================================================
def test_table4():
    section("[Luo26] Table 4: p = 37, x = 13, reproduced bit for bit")
    bad = LC.table4_mismatches()
    assert bad == [], bad[:3]
    s = LC.run(37, 13)
    assert LC.inverse_from(s, 37) == pow(13, -1, 37) == 20
    ok("all 37 rows: both banks (with Table 4's field separators), t, q, r, "
       "t', r', the four lengths, P1, P2, Iter, Sign")
    ok(f"x^-1 = 20 read from W2 after the {LC.step_bound(6)}-step schedule")


def test_classical():
    section("the model on every x: normalised end, x^{-1}, the step bound")
    for p in PRIMES + ((509, 1021) if FULL else ()):
        n = p.bit_length()
        worst, bound, blocks = LC.check_prime(p)
        assert blocks <= LC.pad_blocks_bound(n) < 1 << LC.meta_bits(n)
        print(f"      p={p:>4} n={n:>2}: worst x takes {worst:>3} steps, bound "
              f"4 ceil(c n) = {bound:>3} (4n = {4 * n:>2}); idle blocks <= {blocks}")
    ok("every x ends at r = 1, r' = 0, q = 0, t = p, phases 0, W2 = +-x^{-1}")

    # [Luo26] App. A.1: the quotients (2, 1, 2, 1, ..., 2, 2) are extremal
    for k in range(0, 61, 6):
        qs = [2] + [1, 2] * k + [2]
        a, b = 1, 0                                  # the continuant, from the end
        for q in reversed(qs):
            a, b = q * a + b, a
        p, x = a, b
        assert LC.quotients(p, x) == qs
        N = LC.steps_needed(p, x)
        assert N == 12 * k + 16 <= LC.step_bound(p.bit_length())
    print(f"      extremal sequence, k = {k}: n = {p.bit_length()}, N = {N}, bound "
          f"{LC.step_bound(p.bit_length())}; N / log2 p = {N / math.log2(p):.3f} "
          f"-> 4c = {4 * LC.C:.3f}")
    ok("the (2,1)^k sequences take 12k + 16 steps, inside 4 ceil(c n)")


def test_windows():
    section("[Luo26] App. A.2 active windows hold at every step, for every x")
    ps = (37,) + PRIMES + ((509, 701) if FULL else ())
    for p in ps:
        assert LC.window_violations(p) == [], p
    ok(f"p in {ps}: the r window, the quotient lane, the t boundary and both "
       "length scans")


# =============================================================================
# The circuit
# =============================================================================
def _stepwise(p):
    """Build init + every microstep; compare with the model after each one,
    for every x at once."""
    n = p.bit_length()
    m = Machine("and")
    x = m.alloc(n, "x")
    bk = L.Banks(m, n)
    L.eea_init(m, bk, p, x)
    marks = [len(m.qc.data)]
    Nmax = LC.step_bound(n)
    for T in range(1, Nmax + 1):
        L.eea_step(m, bk, T)
        marks.append(len(m.qc.data))
    xs = list(range(1, p))
    models = [LC.init(p, v) for v in xs]
    sim = Lanes(m, [{x: v} for v in xs])
    vec = lambda bits: sum(v << i for i, v in enumerate(bits))
    for T, mark in enumerate(marks):
        sim.advance(mark)
        for k, s in enumerate(models):
            if T:
                LC.step(s, T, "block")
            got = (sim.read(k, bk.W1), sim.read(k, bk.W2), sim.read(k, bk.lt),
                   sim.read(k, bk.lq), sim.read(k, bk.lr), sim.read(k, bk.ls),
                   sim.read(k, [bk.P1, bk.P2, bk.Sign, bk.Iter]))
            want = (vec(s.W1), vec(s.W2), s.lt, s.lq, s.lr, s.ls,
                    vec([s.P1, s.P2, s.Sign, s.Iter]))
            assert got == want, (p, xs[k], T)
    return Nmax


def test_stepwise():
    section("circuit = model after every microstep, for every x")
    for p in (7, 13, 61) + ((127, 251) if FULL else ()):
        Nmax = _stepwise(p)
        print(f"      p={p:>3}: all {Nmax} steps, all {p - 1} x")
    ok("both banks bit for bit, the four lengths and the four flags equal "
       "ec_luo_classical's")


def test_inverse():
    section("luo_inv: |x> -> |x^{-1}> |ls, Iter>, [Luo26] eq. (1)")
    for p in PRIMES:
        n = p.bit_length()
        m = Machine("and")
        x = m.alloc(n, "x")
        out, gamma = L.luo_inv(m, x, p)
        xs = list(range(1, p))
        sim = lanes_run(m, [{x: v} for v in xs])
        for k, v in enumerate(xs):
            assert sim.read(k, out) == pow(v, -1, p), (p, v)
            s = LC.run(p, v)
            assert sim.read(k, gamma) == s.ls | (s.Iter << LC.meta_bits(n)), (p, v)
        sim.others_zero(list(out) + list(gamma))
        c = CO.count(m)
        assert c["qubits"] == LC.luo_qubits(n), (p, c["qubits"])
        print(f"      p={p:>3}: {c['qubits']:>3} qubits = 2n + 6 floor(log2 n) + 19, "
              f"{c['toffoli_paper']:>6} Toffoli")
    ok("every x: the inverse, the garbage the model predicts, all else |0>")

    for p in (13, 61):                                  # the ancilla-C3X cells
        n = p.bit_length()
        m = Machine("and")
        x = m.alloc(n, "x")
        out, gamma = L.luo_inv(m, x, p, lean=False)
        sim = lanes_run(m, [{x: v} for v in range(1, p)])
        for k, v in enumerate(range(1, p)):
            assert sim.read(k, out) == pow(v, -1, p)
        sim.others_zero(list(out) + list(gamma))
        c0, c1 = CO.count(m), None
        m1 = Machine("and")
        L.luo_inv(m1, m1.alloc(n, "x"), p)
        c1 = CO.count(m1)
        assert c0["qubits"] == c1["qubits"] + 1 and c0["toffoli_paper"] < c1["toffoli_paper"]
    ok(f"lean=False (an ancilla for the cells' C3X): exact too, one qubit more, "
       f"{c0['toffoli_paper']} vs {c1['toffoli_paper']} Toffoli at p = 61")


def test_div_mul():
    section("Luo().div and .mul: exact, x preserved, ancillas clean")
    rnd = random.Random(0x10)
    cases = [(7, None), (11, None), (13, None), (61, None if FULL else 600),
             (127, 2000 if FULL else 300)]
    for p, sample in cases:
        n = p.bit_length()
        pairs = [(a, b) for a in range(1, p) for b in range(p)]
        if sample is not None:
            pairs = rnd.sample(pairs, sample)
        for fn in ("div", "mul"):
            m = Machine("and")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            getattr(L.Luo(), fn)(m, x, y, p)
            sim = lanes_run(m, [{x: a, y: v} for a, v in pairs])
            for k, (a, v) in enumerate(pairs):
                want = v * pow(a, -1, p) % p if fn == "div" else a * v % p
                assert (sim.read(k, x), sim.read(k, y)) == (a, want), (fn, p, a, v)
            sim.others_zero(list(x) + list(y))
        c = CO.count(m)
        assert c["qubits"] == L.div_qubits(n), (p, c["qubits"])
        print(f"      p={p:>3}: {len(pairs):>5} pairs x (div, mul); {c['qubits']} qubits, "
              f"{c['toffoli_paper']} Toffoli")
    ok(f"exhaustive at p = 7, 11, 13{', 61' if FULL else ''}; sampled at "
       f"{'' if FULL else '61 and '}127")

    p, n = 13, 4                                        # against the reference
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    L.Luo().div(m, x, y, p)
    pts = [(1, 0), (5, 7), (12, 12), (3, 1)]
    sim = lanes_run(m, [{x: a, y: v} for a, v in pts])
    for k, (a, v) in enumerate(pts):
        rd = run(m, {x: a, y: v})
        assert (rd(x), rd(y)) == (sim.read(k, x), sim.read(k, y)) \
            == (a, v * pow(a, -1, p) % p)
    ok("ec_sim.run agrees with the bit-parallel runs (p = 13, 4 inputs)")

    for fn in (L.luo_div, L.luo_mul):                   # the function spelling
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        fn(m, x, y, p)
        rd = run(m, {x: 5, y: 7})
        assert rd(y) == (7 * pow(5, -1, p) if fn is L.luo_div else 35) % p
    ok("luo_div / luo_mul are Luo().div / .mul")


def _prime_below(N):
    p = N - 1
    while any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
        p -= 2
    return p


def _widths(p):
    n = p.bit_length()
    mi = Machine("and")
    L.luo_inv(mi, mi.alloc(n, "x"), p)
    md = Machine("and")
    x, y = md.alloc(n, "x"), md.alloc(n, "y")
    L.Luo().div(md, x, y, p)
    mg = Machine("and")
    x, y = mg.alloc(n, "x"), mg.alloc(n, "y")
    G.Dialog().div(mg, x, y, p)
    return CO.count(mi), CO.count(md), CO.count(mg)


def test_qubits():
    section("qubits and Toffolis: Luo against ec_gcd.Dialog")
    rows = []
    for p in (13, 61, 251) + tuple(_prime_below(1 << n) for n in
                                   ((16, 32, 64) if FULL else (16, 32))):
        n = p.bit_length()
        t0 = time.time()
        ci, cd, cg = _widths(p)
        assert ci["qubits"] == LC.luo_qubits(n), (n, ci["qubits"])
        assert cd["qubits"] == L.div_qubits(n), (n, cd["qubits"])
        assert cd["qubits"] < cg["qubits"]
        rows.append((n, ci["qubits"], LC.luo_qubits(n), ci["toffoli_paper"],
                     cd["qubits"], cg["qubits"], cd["toffoli_paper"],
                     cg["toffoli_paper"], f"{cd['toffoli_paper'] / cg['toffoli_paper']:.0f}x"))
        if n >= 16:
            ref = LC.LUO_TABLE6.get(n)
            print(f"      n={n}: built, not simulated, in {time.time() - t0:.0f}s; "
                  f"inversion {ci['toffoli_paper'] / n ** 2:.0f} n^2 Toffoli"
                  + (f" ({ci['toffoli_paper'] / 1e6:.2f}M; [Luo26] Table 6: "
                     f"{ref[0]:.2f}M)" if ref else ""))
    print(CO.table(rows, ["n", "inv q", "2n+6lg+19", "inv Tof", "div q",
                          "Dialog q", "div Tof", "Dialog Tof", "ratio"]))
    print(f"      n = 256: inversion {LC.luo_qubits(256)} qubits (ECDSA.Fail: banks "
          f"+ principal metadata 567), division {L.div_qubits(256)} qubits")
    ok("inversion width = [Luo26]'s formula exactly; division "
       "4n + 6 floor(log2 n) + 19, about half of Dialog's, for ~20x the Toffolis")


# =============================================================================
# The point addition
# =============================================================================
def _build_pa(pts, p, cfg):
    n, w = p.bit_length(), (len(pts) - 1).bit_length()
    m = Machine("and")
    addr, x, y = m.alloc(w, "a"), m.alloc(n, "x"), m.alloc(n, "y")
    W.windowed_point_add_cfg(m, addr, x, y, pts, p, cfg)
    return m, addr, x, y


def _pa_cases(curve, pts):
    """Non-exceptional (R, i, R + P_i), as in tests/test_ec_window_cfg.py."""
    for R in curve.points():
        if R.inf:
            continue
        for i, T in enumerate(pts):
            if T.inf:
                if R.x != 0:
                    yield R, i, R
                continue
            if C.point_add_exceptional(curve, R, T):
                continue
            S = curve.add(R, T)
            if not S.inf:
                yield R, i, S


def test_pointadd():
    section("plug-in: windowed_point_add_cfg with Luo() as the multiplier")
    rnd = random.Random(61)
    curve, pts_all = random_curve(rnd, pmax=61, pmin=61)
    Gp = random_generator(rnd, curve, pts_all, min_order=8)
    p = curve.p
    plain = W.window_points(curve, Gp, 2)
    for label, extra in (("default", {}),
                         ("mbu+merge+free", dict(lookup="mbu", merge_xy=True,
                                                 free_xy1=True))):
        m, addr, x, y = _build_pa(plain, p, W.PointAddCfg(mul=L.Luo(), **extra))
        cases = list(_pa_cases(curve, plain))
        if not FULL:
            cases = rnd.sample(cases, min(80, len(cases)))
        sim = lanes_run(m, [{addr: i, x: R.x, y: R.y} for R, i, S in cases])
        for k, (R, i, S) in enumerate(cases):
            got = (sim.read(k, x), sim.read(k, y), sim.read(k, addr))
            assert got == (S.x, S.y, i), (label, R, i)
        R, i, S = cases[0]
        rd = run(m, {addr: i, x: R.x, y: R.y})
        assert (rd(x), rd(y)) == (S.x, S.y)
        c = CO.count(m)
        base = CO.count(_build_pa(plain, p, W.PointAddCfg(mul=G.Dialog(), **extra))[0])
        print(f"      {label:<15} {len(cases)} sums correct; {c['qubits']} qubits "
              f"(Dialog {base['qubits']}), {c['toffoli_paper']} Toffoli "
              f"(Dialog {base['toffoli_paper']})")
    ok(f"{curve.name}, w = 2: R + P_i exact on every sampled non-exceptional "
       "pair (and through ec_sim.run on one)")


def main():
    t0 = time.time()
    test_table4()
    test_classical()
    test_windows()
    test_stepwise()
    test_inverse()
    test_div_mul()
    test_qubits()
    test_pointadd()
    print(f"\n      ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
    print("\ntest_ec_luo: all passed")
