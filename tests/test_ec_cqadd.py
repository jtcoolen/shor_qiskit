"""ec_cqadd: Gidney's classical-quantum adder with constant workspace [Gid25b].

  * carry_xor XORs every carry of x + c d + cin into a dirty register,
    every input, every dirty value, with and without control and carry in;
  * cq_add_dirty (n - 1 dirty, 2 clean) and cq_add (3 clean) add exactly,
    dirty qubits restored, on every input up to n = 7, at ~3n and ~4n
    Toffolis;
  * run literally -- every vented carry X-measured, every Z from its
    outcome -- all inputs end with the same phase;
  * GidneyArith (b += c a, b -= c a, 2x, x/2 mod p, classical p) is exact for
    every odd p tried, on 4-5 clean qubits, and phase-coherent;
  * Luo3 on it is exact for any prime, on 3n + 6 floor(log2 n) + 19 qubits.
"""
import itertools
import random

from _ec_util import FULL, ok, random_curve, random_generator, section

import ec_classical as C
import ec_cost as CO
import ec_cqadd as CQ
import ec_luo3 as L3
import ec_mbu as MB
import ec_window as W
from ec_sim import Machine, run


def _opt(m, name, use):
    r = m.alloc(1, name) if use else None
    return r, (r[0] if use else None)


def test_carry_xor():
    section("carry_xor: g ^= the carries of x + c d + cin, g dirty")
    rnd = random.Random(1)
    cases = 0
    for n in range(1, 6 if FULL else 5):
        for K in range(1, n + 1):
            for use_c, use_i in itertools.product((False, True), repeat=2):
                d = rnd.randrange(1 << n)
                m = Machine("and")
                x, g = m.alloc(n, "x"), m.alloc(K, "g")
                cr, c = _opt(m, "c", use_c)
                ir, ci = _opt(m, "ci", use_i)
                CQ.carry_xor(m.ctx, x, d, g, c, ci)
                for xv, G in itertools.product(range(1 << n), range(1 << K)):
                    for cv, iv in itertools.product((0, 1) if use_c else (1,),
                                                    (0, 1) if use_i else (0,)):
                        init = {x: xv, g: G}
                        if use_c:
                            init[cr] = cv
                        if use_i:
                            init[ir] = iv
                        rd = run(m, init)
                        want = G
                        for k in range(1, K + 1):
                            want ^= CQ.carry(xv, d * cv, iv, k) << (k - 1)
                        assert (rd(g), rd(x)) == (want, xv), (n, K, d, xv, G, cv, iv)
                        cases += 1
                tof = CO.count(m)["toffoli_paper"]
                assert tof <= 2 * K, (n, K, tof)
    ok(f"{cases} (x, g, control, carry-in) cases exact, x restored; at most 2K Toffolis")


def _adder(kind, n, d, use_c, use_i):
    m = Machine("and")
    x = m.alloc(n, "x")
    cr, c = _opt(m, "c", use_c)
    ir, ci = _opt(m, "ci", use_i)
    g = m.alloc(n - 1, "g") if kind == "dirty" else None
    if kind == "dirty":
        CQ.cq_add_dirty(m, x, d, g, c, ci)
    else:
        CQ.cq_add(m, x, d, c, ci)
    return m, x, g, cr, ir


def test_adders():
    section("cq_add_dirty (n-1 dirty, 2 clean) and cq_add (3 clean): exact")
    rnd = random.Random(2)
    for kind in ("dirty", "clean"):
        for n in range(2, 8 if FULL else 7):
            for use_c, use_i in itertools.product((False, True), repeat=2):
                d = rnd.randrange(1 << n)
                m, x, g, cr, ir = _adder(kind, n, d, use_c, use_i)
                for xv in range(1 << n):
                    G = rnd.randrange(1 << (n - 1)) if g is not None else 0
                    for cv, iv in itertools.product((0, 1) if use_c else (1,),
                                                    (0, 1) if use_i else (0,)):
                        init = {x: xv}
                        if g is not None:
                            init[g] = G
                        if use_c:
                            init[cr] = cv
                        if use_i:
                            init[ir] = iv
                        rd = run(m, init)
                        assert rd(x) == (xv + d * cv + iv) % (1 << n), (kind, n, d, xv, cv, iv)
                        if g is not None:
                            assert rd(g) == G
        for n in (8, 16, 32):
            d = (1 << n) - 977 if n > 16 else (1 << n) - 3
            m, *_ = _adder(kind, n, d, False, False)
            c = CO.count(m)
            clean = c["qubits"] - n - (n - 1 if kind == "dirty" else 0)
            print(f"      {kind:<5} n={n:>2}: {c['toffoli_paper']:>3} Toffolis "
                  f"({c['toffoli_paper'] / n:.2f} n), {clean} clean qubits, "
                  f"{c['measure']} vented carries")
            assert clean == (2 if kind == "dirty" else 3), clean
            assert c["toffoli_paper"] <= (3 if kind == "dirty" else 4) * n
    ok("every input to n = 6 (7 with SHOR_EC_FULL), with control and carry in; "
       "dirty qubits restored; 2 / 3 clean qubits, <= 3n / 4n Toffolis")


def test_live():
    section("run literally: every vented carry measured, every Z from its outcome")
    rnd = random.Random(3)
    for kind in ("dirty", "clean"):
        for n in (3, 5, 6):
            d = rnd.randrange(1, 1 << n)
            m, x, g, cr, _ = _adder(kind, n, d, True, False)
            inputs = [{x: xv, cr: rnd.randrange(2),
                       **({g: rnd.randrange(1 << (n - 1))} if g is not None else {})}
                      for xv in range(1 << n)]
            MB.live_coherent(m.qc, inputs, [MB.random_outcomes(s) for s in range(3)],
                             checks=m.checks)
    orig = MB.zfix                                         # without the phase fix
    MB.zfix = lambda m, reg, key: None
    try:
        m, x, g, cr, _ = _adder("dirty", 5, 21, True, False)
    finally:
        MB.zfix = orig
    from sparse_sim import SparseSimError
    try:
        MB.live_coherent(m.qc, [{x: xv, cr: 1, g: 0} for xv in range(32)],
                         [MB.all_ones_outcome()], checks=m.checks)
    except SparseSimError:
        pass
    else:
        raise AssertionError("without the Zs the phases should differ")
    ok("both adders phase-coherent at n = 3, 5, 6; without the Zs they are not")


def test_arith():
    section("GidneyArith: b +- c a, 2x, x/2 mod p for a classical p, exact")
    rnd = random.Random(4)
    for p in (7, 11, 13, 29) + ((61,) if FULL else ()):
        n = p.bit_length()
        ar = CQ.GidneyArith(p)
        for op in ("cadd", "csub", "dbl", "half", "dbl_dirty"):
            m = Machine("and")
            a, b = m.alloc(n, "a"), m.alloc(n, "b")
            cr = m.alloc(1, "c")
            dz = None
            if op in ("cadd", "csub"):
                getattr(ar, op)(m, cr[0], a, b)
            elif op == "dbl_dirty":
                dz = m.alloc(n + 1, "dz")
                ar.dbl(m, b, dirty=list(dz))
            else:
                getattr(ar, op)(m, b)
            for av, bv in itertools.product(range(p), repeat=2):
                for cv in ((0, 1) if op in ("cadd", "csub") else (1,)):
                    init = {a: av, b: bv, cr: cv}
                    if dz is not None:
                        init[dz] = rnd.randrange(1 << (n + 1))
                    want = {"cadd": bv + cv * av, "csub": bv - cv * av, "dbl": 2 * bv,
                            "half": bv * pow(2, -1, p), "dbl_dirty": 2 * bv}[op] % p
                    rd = run(m, init)
                    assert (rd(b), rd(a)) == (want, av), (p, op, av, bv, cv)
    p, n = 13, 4
    ar = CQ.GidneyArith(p)
    for op in ("cadd", "csub", "dbl", "half"):
        m = Machine("and")
        a, b = m.alloc(n, "a"), m.alloc(n, "b")
        cr = m.alloc(1, "c")
        if op in ("cadd", "csub"):
            getattr(ar, op)(m, cr[0], a, b)
        else:
            getattr(ar, op)(m, b)
        inputs = [{a: rnd.randrange(p), b: rnd.randrange(p), cr: rnd.randrange(2)}
                  for _ in range(12)]
        MB.live_coherent(m.qc, inputs, [MB.random_outcomes(s) for s in range(2)],
                         checks=m.checks)
    ok("every (a, b, c) at p = 7, 11, 13, 29: exact, a restored; coherent run literally")


def test_luo3():
    section("Luo3 on these cells: exact for any prime, on 3n + 6 floor(log2 n) + 19 qubits")
    for p in (7, 11, 13):
        n = p.bit_length()
        for fn in ("div", "mul"):
            m = Machine("and")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            getattr(L3.Luo3(arith=CQ.GidneyArith(p)), fn)(m, x, y, p)
            for a in range(1, p):
                for v in range(p):
                    want = v * pow(a, -1, p) % p if fn == "div" else a * v % p
                    rd = run(m, {x: a, y: v})
                    assert (rd(x), rd(y)) == (a, want), (p, fn, a, v)
            assert CO.count(m)["qubits"] == L3.point_add_qubits(n)
    m = Machine("and")
    x, y = m.alloc(3, "x"), m.alloc(3, "y")
    L3.Luo3(arith=CQ.GidneyArith(7)).div(m, x, y, 7)
    inputs = [{x: a, y: v} for a in range(1, 7) for v in range(7)][::4]
    MB.live_coherent(m.qc, inputs, [MB.random_outcomes(s) for s in range(2)], checks=m.checks)
    p = 13
    rnd = random.Random(13)
    curve, pts = random_curve(rnd, pmax=p, pmin=p)
    G0 = random_generator(rnd, curve, pts, 4)
    tab = W.masked_window_points(curve, G0, 2, random.Random(3))[0]
    m = Machine("and")
    a, x, y = m.alloc(2, "a"), m.alloc(4, "x"), m.alloc(4, "y")
    W.windowed_point_add_cfg(m, a, x, y, tab, p, L3.windowed_cfg(p, CQ.GidneyArith(p)))
    good = 0
    for R in curve.points():
        for i, T in enumerate(tab):
            if R.inf or R.x == 0 or C.point_add_exceptional(curve, R, T) or curve.add(R, T).inf:
                continue
            S = curve.add(R, T)
            rd = run(m, {a: i, x: R.x, y: R.y})
            assert (rd(x), rd(y)) == (S.x, S.y)
            good += 1
    assert CO.count(m)["qubits"] == L3.point_add_qubits(4) + 2
    ok(f"div / mul exact at p = 7, 11, 13 on the formula's qubits, coherent run literally; "
       f"the windowed addition exact on {good} pairs at the formula + window")


def main():
    test_carry_xor()
    test_adders()
    test_live()
    test_arith()
    test_luo3()


if __name__ == "__main__":
    main()
    print("\ntest_ec_cqadd: all passed")
