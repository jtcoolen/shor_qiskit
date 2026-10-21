"""ec_luo3: [Luo26] Sec 5's point addition on three field registers.

  * Luo3.div / .mul are exact on every input at p = 7, 11, 13, and take
    exactly [Luo26]'s 3n + 6 floor(log2 n) + 19 qubits;
  * run literally -- y measured in the X basis, the Z^b correction applied
    from the actual outcome -- every input ends with the same phase; built
    without the Z^b it does not, so the check can see the vent;
  * the controlled addition of a classical point ([Luo26] Fig. 14) and the
    windowed addition (serial loads, signed windows) add correctly;
  * with the lean pseudo-Mersenne cells, the failures are the cells' ~f/q.
"""
import random

from _ec_util import FULL, ok, random_curve, random_generator, section

import ec_classical as C
import ec_cost as CO
import ec_gcd as G
import ec_luo3 as L3
import ec_mbu as MB
import ec_signedwin as SW
import ec_window as W
from ec_sim import Machine, SimError, run
from sparse_sim import SparseSimError


def build(p, fn, backend=None):
    n = p.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    getattr(backend or L3.Luo3(), fn)(m, x, y, p)
    return m, x, y


def test_div_mul():
    section("Luo3.div / .mul: exact, on 3n + 6 floor(log2 n) + 19 qubits")
    for p in (7, 11, 13):
        n = p.bit_length()
        for fn in ("div", "mul"):
            m, x, y = build(p, fn)
            for a in range(1, p):
                for v in range(p):
                    want = v * pow(a, -1, p) % p if fn == "div" else a * v % p
                    rd = run(m, {x: a, y: v})
                    assert (rd(x), rd(y)) == (a, want), (p, fn, a, v)
            c = CO.count(m)
            assert c["qubits"] == L3.point_add_qubits(n), (p, fn, c["qubits"])
            assert c["measure"] == n
            print(f"      p={p:>2} {fn}: {c['qubits']} qubits, {c['toffoli_paper']} Toffoli-eq, "
                  f"{c['measure']} X-measurements")
    ok("every (x != 0, y): the quotient / product, x kept, every ancilla clean")


def test_live():
    section("the vent, run literally: X-measure y, cancel (-1)^(b.y) with Z^b")
    p, n = 7, 3
    inputs = [(a, v) for a in range(1, p) for v in range(p)]
    if not FULL:
        inputs = inputs[::3]
    for fn in ("div", "mul"):
        m, x, y = build(p, fn)
        MB.live_coherent(m.qc, [{x: a, y: v} for a, v in inputs],
                         [MB.random_outcomes(s) for s in range(3)], checks=m.checks)
    orig = MB.zfix
    MB.zfix = lambda m, reg, key: None                   # the same circuit, no Z^b
    try:
        m, x, y = build(p, "div")
    finally:
        MB.zfix = orig
    try:
        MB.live_coherent(m.qc, [{x: a, y: v} for a, v in inputs],
                         [MB.all_ones_outcome()], checks=m.checks)
    except SparseSimError:
        pass
    else:
        raise AssertionError("without Z^b the phases should differ")
    ok(f"div and mul coherent on {len(inputs)} inputs x 3 outcome sequences; "
       f"without the Z^b correction they are not")


def _curve(p, seed):
    rnd = random.Random(seed)
    curve, pts = random_curve(rnd, pmax=p, pmin=p)
    return curve, random_generator(rnd, curve, pts, 4), rnd


def test_point_add():
    section("[Luo26] Fig. 14: the controlled addition of a classical point")
    for p in (11, 13):
        curve, G0, _ = _curve(p, p)
        n = p.bit_length()
        T = curve.mul(3, G0)
        m = Machine("and")
        c, x, y = m.alloc(1, "c"), m.alloc(n, "x"), m.alloc(n, "y")
        L3.point_add_ctrl_luo(m, c[0], x, y, T.x, T.y, p)
        good = 0
        for R in curve.points():
            for cv in (0, 1):
                if R.inf or R.x == 0 or (cv and (C.point_add_exceptional(curve, R, T)
                                                 or curve.add(R, T).inf)):
                    continue
                S = curve.add(R, T) if cv else R
                rd = run(m, {c: cv, x: R.x, y: R.y})
                assert (rd(x), rd(y), rd(c)) == (S.x, S.y, cv), (p, R, cv)
                good += 1
        q = CO.count(m)["qubits"]
        assert q == L3.point_add_qubits(n) + 1, (p, q)
        print(f"      p={p}: {good} (R, control) pairs exact, {q} qubits (formula + control)")
    ok("control on: the sum; control off: the accumulator unchanged")


def test_windowed():
    section("windowed: the point loaded one coordinate at a time, signed windows")
    p = 13
    curve, G0, rnd = _curve(p, 13)
    n = p.bit_length()
    tab = W.masked_window_points(curve, G0, 2, random.Random(3))[0]
    m = Machine("and")
    a, x, y = m.alloc(2, "a"), m.alloc(n, "x"), m.alloc(n, "y")
    W.windowed_point_add_cfg(m, a, x, y, tab, p, L3.windowed_cfg(p))
    good = 0
    for R in curve.points():
        for i, T in enumerate(tab):
            if R.inf or R.x == 0 or C.point_add_exceptional(curve, R, T) or curve.add(R, T).inf:
                continue
            S = curve.add(R, T)
            assert run(m, {a: i, x: R.x, y: R.y})(x) == S.x
            assert run(m, {a: i, x: R.x, y: R.y})(y) == S.y
            good += 1
    q = CO.count(m)["qubits"]
    assert q == L3.point_add_qubits(n) + 2, q
    print(f"      p={p}, w=2: {good} pairs exact, {q} qubits (formula + window)")

    from test_ec_windowed import _prime_odd_curve
    curve, P, order = _prime_odd_curve(seed=21, pmin=17, pmax=31)
    p, n = curve.p, curve.p.bit_length()
    stab, delta = SW.signed_window_points(curve, P, 2, order=order)
    m = Machine("and")
    a, x, y = m.alloc(2, "a"), m.alloc(n, "x"), m.alloc(n, "y")
    SW.windowed_point_add_signed(m, a, x, y, stab, p, L3.windowed_cfg(p), neg=L3.neg_lean)
    good = 0
    for R in curve.points():
        if R.inf or R.x == 0:
            continue
        for i in range(4):
            neg, j = SW.signed_digit(i, 2)
            if C.point_add_exceptional(curve, curve.neg(R) if neg else R, stab[j]):
                continue
            S = curve.add(curve.add(R, curve.mul(i, P)), delta)
            rd = run(m, {a: i, x: R.x, y: R.y})
            assert (rd(x), rd(y), rd(a)) == (S.x, S.y, i), (R, i)
            good += 1
    print(f"      signed, {curve.name} (order {order}): {good} (R, i) pairs exact, "
          f"{CO.count(m)['qubits']} qubits")
    ok("both windowed forms add; the width is the three registers plus the window")


def test_pm():
    section("with the lean pseudo-Mersenne cells (the n = 256 build): ~f/q failures")
    q = 127
    n = q.bit_length()
    be = L3.Luo3(arith=G.PMSpace(q, msbs=n, lean=True))
    rnd = random.Random(7)
    pairs = [(rnd.randrange(1, q), rnd.randrange(q)) for _ in range(120 if FULL else 40)]
    for fn in ("div", "mul"):
        m, x, y = build(q, fn, be)
        bad = 0
        for a, v in pairs:
            want = v * pow(a, -1, q) % q if fn == "div" else a * v % q
            try:
                rd = run(m, {x: a, y: v})
                bad += (rd(x), rd(y)) != (a, want)
            except SimError:
                bad += 1
        print(f"      q=2^7-1 {fn}: {bad}/{len(pairs)} wrong, {CO.count(m)['qubits']} qubits")
        assert bad <= 0.1 * len(pairs), (fn, bad)
    ok("failures only at the pseudo-Mersenne cells' rate (f/q ~ 1% here, 2^-224 at secp256k1)")


def main():
    test_div_mul()
    test_live()
    test_point_add()
    test_windowed()
    test_pm()


if __name__ == "__main__":
    main()
    print("\ntest_ec_luo3: all passed")
