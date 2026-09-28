"""Carry-lookahead arithmetic ([DKRS04], [106] Sec 3.1): cla_add, cla_sub,
cla_carry_out, cla_lt, cla_geq_const, cla_modadd, cla_modsub, cla_cost."""
import math
import random

from _ec_util import FULL, ok, section

import depth as D
import ec_adders as A
import ec_cla as C
import ec_cost as CO
import ec_modarith as MA
from ec_sim import Machine, run


def cost(m):
    c = CO.count(m)
    return c["toffoli_paper"], D.toffoli_depth(m), c["qubits"]


def test_add():
    section("cla_add / cla_sub: exhaustive at n = 1..6, with and without carry")
    for n in range(1, 7):
        N = 1 << n
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        C.cla_add(m, x, y)
        mc = Machine("and")
        xc, yc, c = mc.alloc(n, "x"), mc.alloc(n, "y"), mc.alloc(1, "c")
        C.cla_add(mc, xc, yc, c[0])
        ms = Machine("and")
        xs, ys, b = ms.alloc(n, "x"), ms.alloc(n, "y"), ms.alloc(1, "b")
        C.cla_sub(ms, xs, ys, b[0])
        for xv in range(N):
            for yv in range(N):
                rd = run(m, {x: xv, y: yv})
                assert rd(y) == (xv + yv) % N and rd(x) == xv, (n, xv, yv)
                for cv in (0, 1):
                    rd = run(mc, {xc: xv, yc: yv, c: cv})
                    assert rd(yc) == (xv + yv) % N and rd(xc) == xv
                    assert rd(c) == cv ^ ((xv + yv) >> n), (n, xv, yv, cv)
                rd = run(ms, {xs: xv, ys: yv})
                assert rd(ys) == (yv - xv) % N and rd(xs) == xv
                assert rd(b) == (xv > yv), (n, xv, yv)
        cc = C.cla_cost(n)
        assert cost(m)[0] == cc["add"]["toffoli"], (n, cost(m), cc)
        assert cost(mc)[0] == cc["add_carry"]["toffoli"], (n, cost(mc), cc)
    ok("n = 1..6: y + x mod 2^n, the carry-out, y - x and its borrow; x restored, "
       "every ancilla clean where it was freed")

    for n in range(1, 6):
        N = 1 << n
        for k in range(N):
            m = Machine("and")
            y, c = m.alloc(n, "y"), m.alloc(1, "c")
            C.cla_add(m, k, y, c[0])
            for yv in range(N):
                rd = run(m, {y: yv})
                assert rd(y) == (yv + k) % N and rd(c) == (yv + k) >> n, (n, k, yv)
            assert cost(m)[0] == C.cla_cost(n)["add_carry"]["toffoli"] - n, (n, k)
        m = Machine("and")
        x, y = m.alloc(max(n - 2, 1), "x"), m.alloc(n, "y")
        C.cla_add(m, x, y)
        for xv in range(1 << len(x)):
            for yv in range(N):
                assert run(m, {x: xv, y: yv})(y) == (xv + yv) % N
    ok("classical operands at n = 1..5 (every constant; n fewer ANDs), and a "
       "narrower x zero-extended")


def test_compare():
    section("cla_carry_out, cla_lt, cla_geq_const: exhaustive at n = 1..6")
    for n in range(1, 7):
        N = 1 << n
        mo = Machine("and")
        x, y, o = mo.alloc(n, "x"), mo.alloc(n, "y"), mo.alloc(1, "o")
        C.cla_carry_out(mo, x, y, o[0])
        ml = Machine("and")
        xl, yl, ol = ml.alloc(n, "x"), ml.alloc(n, "y"), ml.alloc(1, "o")
        C.cla_lt(ml, xl, yl, ol[0])
        for xv in range(N):
            for yv in range(N):
                for ov in (0, 1):
                    rd = run(mo, {x: xv, y: yv, o: ov})
                    assert rd(o) == ov ^ ((xv + yv) >> n), (n, xv, yv)
                    assert rd(x) == xv and rd(y) == yv
                rd = run(ml, {xl: xv, yl: yv})
                assert rd(ol) == (xv < yv) and rd(xl) == xv and rd(yl) == yv
        cc = C.cla_cost(n)["carry_out"]
        assert cost(mo)[0] == cc["toffoli"] and cost(mo)[1] == cc["depth"], (n, cost(mo), cc)
        for k in range(-1, N + 2):
            m = Machine("and")
            y, o = m.alloc(n, "y"), m.alloc(1, "o")
            C.cla_geq_const(m, y, k, o[0])
            for yv in range(N):
                rd = run(m, {y: yv})
                assert rd(o) == (yv >= k) and rd(y) == yv, (n, k, yv)
    ok("n = 1..6: carry-out, [x < y] and [y >= k] for every k (ends included); "
       "inputs restored, the whole tree unwound clean")


def test_random():
    section("random samples at n = 16, 33, 64")
    rnd = random.Random(0xC1A)
    reps = 400 if FULL else 120
    for n in (16, 33, 64):
        N = 1 << n
        ms = {}
        for lab in ("add", "carry", "sub", "out", "lt", "const"):
            m = Machine("and")
            x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
            ms[lab] = (m, x, y, o)
        kc = rnd.randrange(N)
        m, x, y, o = ms["add"]; C.cla_add(m, x, y)
        m, x, y, o = ms["carry"]; C.cla_add(m, x, y, o[0])
        m, x, y, o = ms["sub"]; C.cla_sub(m, x, y, o[0])
        m, x, y, o = ms["out"]; C.cla_carry_out(m, x, y, o[0])
        m, x, y, o = ms["lt"]; C.cla_lt(m, x, y, o[0])
        m, x, y, o = ms["const"]; C.cla_add(m, kc, y, o[0])
        want = {
            "add": lambda a, b: ((a + b) % N, 0),
            "carry": lambda a, b: ((a + b) % N, (a + b) >> n),
            "sub": lambda a, b: ((b - a) % N, int(a > b)),
            "out": lambda a, b: (b, (a + b) >> n),
            "lt": lambda a, b: (b, int(a < b)),
            "const": lambda a, b: ((b + kc) % N, (b + kc) >> n),
        }
        for _ in range(reps):
            # bias towards long carry chains: all-ones runs, 0, N - 1
            xv = rnd.choice([rnd.randrange(N), N - 1, 0, (N - 1) ^ (1 << rnd.randrange(n))])
            yv = rnd.choice([rnd.randrange(N), N - 1 - xv, N - xv if xv else 0, rnd.randrange(N)])
            for lab, (m, x, y, o) in ms.items():
                rd = run(m, {x: xv, y: yv})
                assert (rd(y), rd(o)) == want[lab](xv, yv) and rd(x) == xv, (n, lab, xv, yv)
    ok(f"{reps} inputs each at n = 16, 33, 64 (carry-chain-heavy): all six routines exact")


def test_modadd():
    section("cla_modadd / cla_modsub: exhaustive for p = 7, 11, 13, 61")
    for p in (7, 11, 13, 61):
        n = p.bit_length()
        ma, ms = Machine("and"), Machine("and")
        xa, ya = ma.alloc(n, "x"), ma.alloc(n, "y")
        xs, ys = ms.alloc(n, "x"), ms.alloc(n, "y")
        C.cla_modadd(ma, xa, ya, p)
        C.cla_modsub(ms, xs, ys, p)
        for xv in range(p):
            for yv in range(p):
                rd = run(ma, {xa: xv, ya: yv})
                assert rd(ya) == (xv + yv) % p and rd(xa) == xv, (p, xv, yv)
                rd = run(ms, {xs: xv, ys: yv})
                assert rd(ys) == (yv - xv) % p and rd(xs) == xv, (p, xv, yv)
    rnd = random.Random(0x106)
    for p in (2**61 - 1, 2**64 - 59):
        n = p.bit_length()
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        C.cla_modadd(m, x, y, p)
        for _ in range(60):
            xv = rnd.choice([rnd.randrange(p), p - 1, 0])
            yv = rnd.choice([rnd.randrange(p), p - 1 - xv, p - xv if xv else 0])
            assert run(m, {x: xv, y: yv})(y) == (xv + yv) % p, (p, xv, yv)
    ok("every x, y < p: (y + x) mod p and (y - x) mod p, flags and borrow cleared; "
       "and sampled at p = 2^61 - 1, 2^64 - 59 around the reduction boundary")


def test_superposition():
    section("on genuine superpositions (Statevector): no phase, no entangled ancilla")
    from test_ec_quantum import _check_superposition
    n = 3
    m = Machine("and")
    x, y, c = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "c")
    C.cla_add(m, x, y, c[0])
    _check_superposition(m, x + y + c, x + y + c, range(64),
                         lambda v: (v & 7) | (((v & 7) + (v >> 3)) << 3), "cla_add, n = 3")
    m = Machine("and")
    x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
    C.cla_lt(m, x, y, o[0])
    _check_superposition(m, x + y + o, x + y + o, range(64),
                         lambda v: v | (int((v & 7) < (v >> 3)) << 6), "cla_lt, n = 3")
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    C.cla_modadd(m, x, y, 7)
    _check_superposition(m, x + y, x + y, [a | b << 3 for a in range(7) for b in range(7)],
                         lambda v: (v & 7) | ((((v & 7) + (v >> 3)) % 7) << 3),
                         "cla_modadd, p = 7")


def test_cost():
    section("cla_cost: the closed forms against the built circuits")
    sizes = list(range(1, 70)) + [96, 127, 128, 129] + ([200, 256] if FULL else [])
    for n in sizes:
        cc = C.cla_cost(n)
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        C.cla_add(m, x, y)
        assert cost(m) == (cc["add"]["toffoli"], cc["add"]["depth"],
                           2 * n + cc["add"]["ancillas"]), (n, cost(m), cc["add"])
        m = Machine("and")
        x, y, c = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "c")
        C.cla_add(m, x, y, c[0])
        assert cost(m) == (cc["add_carry"]["toffoli"], cc["add_carry"]["depth"],
                           2 * n + 1 + cc["add_carry"]["ancillas"]), (n, cost(m))
        m = Machine("and")
        x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
        C.cla_carry_out(m, x, y, o[0])
        assert cost(m) == (cc["carry_out"]["toffoli"], cc["carry_out"]["depth"],
                           2 * n + 1 + cc["carry_out"]["ancillas"]), (n, cost(m))
    ok(f"Toffolis, Toffoli depth and qubits exact for n in 1..69, 96, 127..129"
       f"{', 200, 256' if FULL else ''}")


def test_depth():
    section("Toffoli depth: O(log n) against the ripple adders' O(n)")
    rows, prev = [], None
    for n in (8, 16, 32, 64, 128):
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        C.cla_add(m, x, y)
        cla = cost(m)
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        A.gidney_add(m.ctx, x, y, m.anc(n - 1, "a"))
        gid = cost(m)
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        A.cdkm_add(m.ctx, x, y, m.anc(1, "c")[0])
        cdkm = cost(m)
        m = Machine("and")
        x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
        C.cla_carry_out(m, x, y, o[0])
        cco = cost(m)
        m = Machine("and")
        x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(1, "o")
        A.carry_out(m.ctx, x, y, o[0], m.anc(n, "a"))
        gco = cost(m)
        rows.append([n, *cla, *gid, *cdkm, *cco, *gco])
        assert cla[1] <= 4 * math.log2(n) + 3, (n, cla)
        assert cco[1] == math.ceil(math.log2(n)) + 1, (n, cco)
        assert gid[1] == n - 1 and cdkm[1] == 2 * n
        if prev is not None:
            assert cla[1] - prev <= 4, (n, cla, prev)     # +4 per doubling
        prev = cla[1]
        if n >= 32:                     # crossover: n = 16 ties, 8 loses
            assert cla[1] < gid[1] and cco[1] < gco[1], (n, cla, gid)
        if n >= 128:
            assert 4 * cla[1] < gid[1], (n, cla, gid)
    hdr = ["n", "CLA Tof", "depth", "qubits", "Gidney Tof", "depth", "qubits",
           "CDKM Tof", "depth", "qubits", "CLA cmp", "depth", "qubits",
           "Gid cmp", "depth", "qubits"]
    print(CO.table(rows, hdr))
    ok("in-place CLA: depth ~4 log n (+4 per doubling), 27 at n = 128 vs Gidney's "
       "127; shallower from n = 32 on (ties at 16); comparator ceil(log n) + 1")


def test_modadd_depth():
    section("cla_modadd vs ec_modarith.modadd: Toffoli depth")
    rows = []
    for n in (32, 64):
        p = random.Random(n).randrange(1 << (n - 1), 1 << n) | 1
        out = []
        for fn in (C.cla_modadd, MA.modadd):
            m = Machine("and")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            fn(m, x, y, p)
            out.append(cost(m))
        (tc, dc, qc), (tg, dg, qg) = out
        rows.append([n, tc, dc, qc, tg, dg, qg])
        assert dc < dg / 2, (n, out)
    print(CO.table(rows, ["n", "CLA Tof", "depth", "qubits",
                          "modadd Tof", "depth", "qubits"]))
    ok("CLA modular addition: under half the Toffoli depth at n = 32, a quarter at 64")


def test_hier():
    section("hier: the CLA builders capture as boundaries, counts unchanged")
    import hier as H
    tg = [("ec_cla", ["cla_add", "cla_sub", "cla_carry_out", "cla_lt", "cla_geq_const",
                      "cla_modadd"])]
    for n, p in ((16, 65521), (64, 2**64 - 59)):
        flat = Machine("and")
        x, y = flat.alloc(n, "x"), flat.alloc(n, "y")
        C.cla_modsub(flat, x, y, p)
        with H.tracing(tg):
            hm = H.HierMachine("and")
            x, y = hm.alloc(n, "x"), hm.alloc(n, "y")
            C.cla_modsub(hm, x, y, p)
            hc = H.count(hm)
        fc = CO.count(flat)
        for k in ("qubits", "toffoli_paper", "and", "and_dg", "toffoli"):
            assert fc[k] == hc[k], (n, k, fc[k], hc[k])
    ok("n = 16, 64: cla_modsub through `hier.tracing` = flat, field for field")


def main():
    test_add()
    test_compare()
    test_random()
    test_modadd()
    test_superposition()
    test_cost()
    test_depth()
    test_modadd_depth()
    test_hier()


if __name__ == "__main__":
    main()
    print("\ntest_ec_cla: all passed")
