"""ec_gcd.Dialog: [1128]'s in-place multiplication with each refinement a switch.

  * Fig. 1: five Toffolis pack three records into five qubits, injectively;
  * Dialog() with default options IS ec_eea.inplace_mul, gate for gate;
  * the exact options (fused comparator, packing, reusing x) are exact;
  * the probabilistic ones (width schedule, sharing, top-bit comparisons)
    fail at a rate that falls to 0 as their parameter grows;
  * what each costs.
"""
import itertools
import math
import random

from _ec_util import FULL, ok, section

import ec_cost as CO
import ec_eea as E
import ec_gcd as G
from ec_sim import Machine, SimError, run
from qiskit.circuit import QuantumCircuit, QuantumRegister


def ops(m):
    idx = {q: i for i, q in enumerate(m.qc.qubits)}
    return [(ci.operation.name, tuple(idx[q] for q in ci.qubits)) for ci in m.qc.data]


def test_fig1():
    section("[1128] Fig. 1: three records in five qubits, five Toffolis")
    from ec_gates import Ctx
    from ec_sim import simulate
    qc = QuantumCircuit(QuantumRegister(6, "w"))
    G.fig1_compress(Ctx(qc, "and"), list(qc.qubits))
    assert sum(1 for ci in qc.data if ci.operation.name == "ccx") == 5
    seen = set()
    for pairs in itertools.product(((0, 0), (1, 0), (1, 1)), repeat=3):
        init = {i: pairs[i // 2][i % 2] for i in range(6)}
        bits = simulate(qc, init)
        assert bits[5] == 0
        seen.add(tuple(bits[:5]))
    assert len(seen) == 27
    ok("all 27 reachable inputs: last wire |0>, the other five distinct")


def build(q, backend, fn="mul"):
    n = q.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    getattr(backend, fn)(m, x, y, q)
    return m, x, y


def fail_rate(q, backend, pairs, fn="mul"):
    m, x, y = build(q, backend, fn)
    bad = 0
    for xv, yv in pairs:
        want = xv * yv % q if fn == "mul" else yv * pow(xv, -1, q) % q
        try:
            rd = run(m, {x: xv, y: yv})
            if rd(y) != want or rd(x) != xv:
                bad += 1
        except SimError:
            bad += 1
    return bad / len(pairs), m


def test_default_identity():
    section("Dialog() = ec_eea.inplace_mul, gate for gate")
    for q in (7, 13, 61):
        n = q.bit_length()
        a = Machine("and")
        x, y = a.alloc(n, "x"), a.alloc(n, "y")
        E.inplace_mul(a, x, y, q)
        b = build(q, G.Dialog())[0]
        assert ops(a) == ops(b), q
        assert [c[0] for c in a.checks] == [c[0] for c in b.checks], q
    ok("q = 7, 13, 61: identical instruction streams and ancilla checks")


EXACT = {
    "fused_cmp": dict(fused_cmp=True),
    "compress": dict(compress="fig1"),
    "reuse_x": dict(reuse_x=True),
    "fused+compress+reuse_x": dict(fused_cmp=True, compress="fig1", reuse_x=True),
}


def test_exact_options():
    section("exact options: exact on every input")
    q = 61
    pairs = [(xv, yv) for xv in range(1, q) for yv in range(q)]
    if not FULL:
        pairs = pairs[::3]
    base = CO.count(build(q, G.Dialog())[0])
    print(f"      q={q} baseline: {base['qubits']} qubits, "
          f"{base['toffoli_paper']} Toffoli-eq")
    for label, kw in EXACT.items():
        for fn in ("mul", "div"):
            r, m = fail_rate(q, G.Dialog(**kw), pairs, fn)
            assert r == 0.0, (label, fn, r)
        c = CO.count(m)
        print(f"      {label:<24} {c['qubits']:>4} qubits  "
              f"{c['toffoli_paper']:>6} Toffoli-eq")
    ok(f"q = {q}, multiplication and division, {len(pairs)} (x, y) pairs each")


def test_probabilistic_options():
    section("probabilistic options: failure falls to 0 as the parameter grows")
    rnd = random.Random(7)
    for q in (127, 251):
        n = q.bit_length()
        pairs = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(300 if FULL else 120)]
        rows = []
        for c_pad in (0.0, 0.5, 1.0, 2.3):
            for share in (False, True):
                r, m = fail_rate(q, G.Dialog(c_pad=c_pad, share=share,
                                             compress="fig1" if share else None,
                                             fused_cmp=True), pairs)
                c = CO.count(m)
                rows.append((c_pad, share, r, c["qubits"], c["toffoli_paper"]))
        for c_pad, share, r, qb, t in rows:
            print(f"      q={q} c_pad={c_pad:<4} share={share!s:<5} "
                  f"fail {100*r:5.1f}%  {qb:>4} qubits {t:>6} Toffoli-eq")
        big = [r for c_pad, share, r, *_ in rows if c_pad == 2.3]
        assert all(r == 0.0 for r in big), (q, big)
        for share in (False, True):
            rs = [r for c_pad, s, r, *_ in rows if s == share]
            assert rs[0] >= rs[-1]
        # top-bit comparisons only make sense against the scheduled width:
        # at full width the top bits of a shrinking u and v all tie
        prev = 1.0
        for msbs in (2, 4, n + 1):
            r, m = fail_rate(q, G.Dialog(cmp_msbs=msbs, c_pad=2.3), pairs)
            print(f"      q={q} c_pad=2.3 cmp_msbs={msbs:<3} fail {100*r:5.1f}%  "
                  f"{CO.count(m)['toffoli_paper']:>6} Toffoli-eq")
            assert r <= prev + 1e-12
            prev = r
        assert prev == 0.0
    ok("width schedule, sharing and top-bit comparison: exact at the paper's "
       "c_pad = 2.3 on these samples, and at full comparison width")


def test_pm_arith():
    section("replay arithmetic: pseudo-Mersenne with Algorithm 11")
    rnd = random.Random(11)
    for q in (61, 127):
        pairs = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(150)]
        n = q.bit_length()
        r_ex, m_ex = fail_rate(q, G.Dialog(), pairs)
        r_pm, m_pm = fail_rate(q, G.Dialog(arith=G.PM(q, msbs=n)), pairs)
        print(f"      q={q}: exact {CO.count(m_ex)['toffoli_paper']} Toffoli-eq, "
              f"PM {CO.count(m_pm)['toffoli_paper']} Toffoli-eq, "
              f"PM failure {100*r_pm:.1f}%")
        assert r_ex == 0.0
        # q = 2^7 - 1 is Mersenne (f = 1): exact.  q = 2^6 - 3 has f/q ~ 5%
        # per operation of sums the PM adder cannot reduce, compounded over
        # ~15 replay steps -- a toy-size effect (f/q ~ 2^-224 at secp256k1)
        if q == 127:
            assert r_pm == 0.0, r_pm
        else:
            assert r_pm < 0.35, r_pm
    ok("the PM replay composes (Alg 11 handles the x + y = q step); exact "
       "for the Mersenne prime 127")


def test_condinv():
    section("IonQ Sec VI: conditionally-inverted GCD")
    import ec_classical as C
    # the walk records exactly [1128]'s bits: checked classically on every x
    for q in (61, 127, 251):
        it = C.eea_iterations(q.bit_length())
        for xv in range(1, q):
            u, v = q, xv
            a = v & 1
            vt = v + (1 - a) * u
            mm = a and u > vt
            if mm:
                u, vt = vt, u
            recs = [(a, int(mm))]
            for _ in range(it - 1):
                a = ((vt >> 1) ^ (u >> 1)) & 1
                vt = (vt + (u if a == 0 else -u)) // 2
                mm = a and u > vt
                if mm:
                    u, vt = vt, u
                recs.append((a, int(mm)))
            assert (u, vt) == (1, 1)
            assert recs == C.eea_dialog(q, xv, it)[0], (q, xv)
    ok("the walk ends at (1, 1) and records [1128]'s dialog bits, every x "
       "for q = 61, 127, 251")

    q = 61
    pairs = [(xv, yv) for xv in range(1, q) for yv in range(q)]
    if not FULL:
        pairs = pairs[::3]
    base = CO.count(build(q, G.Dialog(fused_cmp=True))[0])["toffoli_paper"]
    for kind in ("standard", "ci"):
        for fn in ("mul", "div"):
            r, m = fail_rate(q, G.CondInv(replay=kind), pairs, fn)
            assert r == 0.0, (kind, fn, r)
        t = CO.count(m)["toffoli_paper"]
        print(f"      q={q} CondInv replay={kind:<8} {t:>5} Toffoli-eq "
              f"(Dialog fused: {base})")
        if kind == "standard":
            assert t < base
    ok(f"q = {q}: exact, both replays, multiplication and division")

    rnd = random.Random(5)
    q = 127
    pairs = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(150)]
    for fn in ("mul", "div"):
        r, m = fail_rate(q, G.CondInv(arith=G.PM(q, msbs=7), replay="ci"), pairs, fn)
        assert r == 0.0, (fn, r)
    t_ci = CO.count(m)["toffoli_paper"]
    t_std = CO.count(build(q, G.CondInv(arith=G.PM(q, msbs=7), replay="standard"))[0])["toffoli_paper"]
    t_dlg = CO.count(build(q, G.Dialog(arith=G.PM(q, msbs=7)))[0])["toffoli_paper"]
    print(f"      q=2^7-1, PM arithmetic: Dialog {t_dlg}, CondInv standard replay "
          f"{t_std}, CondInv ci replay {t_ci} Toffoli-eq")
    ok("q = 127 with pseudo-Mersenne arithmetic: IonQ's replay is exact")


def test_condinv_phase():
    section("IonQ's replay cells: phase adder, careful steps, forward multiplication")
    rnd = random.Random(9)
    q = 127
    n = q.bit_length()
    pairs = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(300 if FULL else 150)]
    prev = {}
    for K in (0, 1, 3, 37):
        for fn in ("div", "mul"):
            be = G.CondInv(arith=G.PMPhase(q, msbs=n), cmp_msbs=n + 1, replay="ci",
                           zero_steps=K)
            r, m = fail_rate(q, be, pairs, fn)
            if K == 37:
                assert r == 0.0, (fn, r)
            assert r <= prev.get(fn, 1.0) + 0.02, (K, fn, r)
            prev[fn] = r
            print(f"      q={q} zero_steps={K:<2} {fn}: fail {100 * r:5.1f}%  "
                  f"{CO.count(m)['toffoli_paper']} Toffoli-eq")
    ok("the structural zero sits in the first steps (v2-distributed): failure "
       "halves per careful step, exact at 37 -- division and multiplication")


def pingpong_worst(p):
    """Rounds the ping-pong walk needs in the worst case over all x."""
    def conv(x):
        r0, r1 = p, (x if x & 1 else x - p)
        i = 0
        while not (abs(r0) == 1 and abs(r1) == 1):
            s, t = (r0, r1) if i % 2 == 0 else (r1, r0)
            e = ((t >> 1) ^ (s >> 1)) & 1
            t = (t + (s if e == 0 else -s)) // 2
            if i % 2 == 0:
                r1 = t
            else:
                r0 = t
            i += 1
        return i
    return max(conv(x) for x in range(1, p))


def test_pingpong():
    section("ECDSA.Fail Sec 5.3.3: the comparison-free ping-pong GCD")
    q = 61
    n = q.bit_length()
    L = pingpong_worst(q)
    pairs = [(xv, yv) for xv in range(1, q) for yv in range(q)]
    if not FULL:
        pairs = pairs[::3]
    for fn in ("mul", "div"):
        r, m = fail_rate(q, G.PingPong(rounds=L), pairs, fn)
        assert r == 0.0, (fn, r)
    t_pp = CO.count(m)
    t_dl = CO.count(build(q, G.Dialog())[0])
    t_ci = CO.count(build(q, G.CondInv(replay="standard"))[0])
    print(f"      q={q}, {L} rounds (worst case; 2.75n = {math.ceil(2.75*n)}): "
          f"{t_pp['toffoli_paper']} Toffoli-eq / {t_pp['qubits']} qubits  vs  "
          f"Dialog {t_dl['toffoli_paper']} / {t_dl['qubits']}, CondInv "
          f"{t_ci['toffoli_paper']} / {t_ci['qubits']}")
    r, _ = fail_rate(q, G.PingPong(), pairs, "mul")
    print(f"      at the paper's L = 2.75n: {100*r:.1f}% of inputs fail to converge")
    ok(f"q = {q}: exact with the worst-case round budget, both directions")

    rnd = random.Random(9)
    q = 127
    L = pingpong_worst(q)
    pairs = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(120)]
    neg = lambda mm, c, v: __import__("ec_approx").cmodneg_approx(mm, c, v, q)
    for fn in ("mul", "div"):
        r, m = fail_rate(q, G.PingPong(arith=G.PM(q, msbs=7), rounds=L, neg=neg),
                         pairs, fn)
        assert r == 0.0, (fn, r)
    print(f"      q=2^7-1, PM cells + approximate negation: "
          f"{CO.count(m)['toffoli_paper']} Toffoli-eq")
    ok("q = 127 with pseudo-Mersenne replay cells: exact")


def jump2_worst(p):
    def need(x):
        L = 1
        while True:
            u, v = p, x
            for i in range(L):
                if i == 0:
                    if v % 2 == 0:
                        v //= 2
                else:
                    v //= 2
                if v % 2 == 0:
                    v //= 2
                b = v & 1
                s = b if i == 0 else int(b and v < u)
                if s:
                    u, v = v, u
                if b:
                    v -= u
            if (u, v) == (1, 0):
                return L
            L += 1
    return max(need(x) for x in range(1, p))


def test_jump2():
    section("ECDSA.Fail Sec 5.3.1-2: Jump-2 macro-steps and the base-5 codec")
    q = 61
    L = jump2_worst(q)
    pairs = [(xv, yv) for xv in range(1, q) for yv in range(q)]
    if not FULL:
        pairs = pairs[::3]
    for fn in ("mul", "div"):
        r, m = fail_rate(q, G.Jump2(steps=L), pairs, fn)
        assert r == 0.0, (fn, r)
    c = CO.count(m)
    print(f"      q={q}: {L} macro-steps (worst case over x), "
          f"{c['toffoli_paper']} Toffoli-eq / {c['qubits']} qubits")
    ok(f"q = {q}: exact, multiplication and division")

    for k in (2, 3):
        m = Machine("and")
        r = m.alloc(3 * k, "r")
        freed = G.base5_pack(m, list(r))
        seen = set()
        for syms in itertools.product(range(5), repeat=k):
            v = 0
            for j, d in enumerate(syms):
                b, s_, s2 = G.JUMP2_SYMBOLS[d]
                v |= (b | s_ << 1 | s2 << 2) << (3 * j)
            out = run(m, {r: v})(r)
            assert out >> (3 * k - len(freed)) == 0
            seen.add(out)
        assert len(seen) == 5 ** k
    assert G.base5_record_qubits(261) == 609
    for fn in ("mul", "div"):
        r, m = fail_rate(q, G.Jump2Packed(steps=L + 2), pairs, fn)
        assert r == 0.0, (fn, r)
    c0 = CO.count(build(q, G.Jump2(steps=L + 2))[0])
    c1 = CO.count(m)
    print(f"      codec wired in (q={q}, {L + 2} steps): {c0['qubits']} -> {c1['qubits']} "
          f"qubits, {c0['toffoli_paper']} -> {c1['toffoli_paper']} Toffoli-eq "
          f"(a generic permutation codec, ~500 per group)")
    assert c1["qubits"] < c0["qubits"]
    ok("base-5 codec: 9 -> 7 and 6 -> 5 qubits, injective on the 5^k reachable "
       "patterns; 261 steps -> 609 qubits, ECDSA.Fail's figure")


def main():
    test_fig1()
    test_default_identity()
    test_exact_options()
    test_probabilistic_options()
    test_pm_arith()
    test_condinv()
    test_condinv_phase()
    test_pingpong()
    test_jump2()


if __name__ == "__main__":
    main()
    print("\ntest_ec_gcd: all passed")
