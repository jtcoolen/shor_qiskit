"""Adders, comparators, modular arithmetic and multiplication.

Exhaustive wherever the input space allows; the point of exhaustive is that a
reversible circuit can be right on 99% of inputs and still be useless.
"""
from _ec_util import ok, rng, scope, section

import ec_adders as A
import ec_modarith as MA
import ec_mult as MU
from ec_sim import Machine, run


def main():
    r = rng()

    section("adders and comparators")
    for mode in ("and", "toffoli"):
        for n in range(1, scope(8, 11)):
            m = Machine(mode)
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            a = m.anc(max(n - 1, 1), "a")
            A.gidney_add(m.ctx, x, y, a)
            m.free(a)
            for u in range(2**n):
                for v in range(2**n):
                    rd = run(m, {x: u, y: v})
                    assert rd(y) == (u + v) % 2**n and rd(x) == u
    ok("gidney_add exhaustive, both AND and Toffoli modes")

    for n in range(1, scope(7, 9)):
        m = Machine("toffoli")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        c = m.anc(1, "c")
        A.cdkm_add(m.ctx, x, y, c[0])
        m.free(c)
        for u in range(2**n):
            for v in range(2**n):
                assert run(m, {x: u, y: v})(y) == (u + v) % 2**n
    ok("cdkm_add exhaustive (delegates to this package's existing rc_adder)")

    for n in range(1, scope(6, 8)):
        m = Machine("and")
        c, x, y = m.alloc(1, "c"), m.alloc(n, "x"), m.alloc(n, "y")
        cp, a = m.anc(n, "cp"), m.anc(max(n - 1, 1), "a")
        A.cadd(m.ctx, c[0], x, y, cp, a)
        m.free(cp, a)
        for ct in (0, 1):
            for u in range(2**n):
                for v in range(2**n):
                    rd = run(m, {c: ct, x: u, y: v})
                    assert rd(y) == ((v + u) % 2**n if ct else v)
    ok("cadd via copy-then-add (2026/106 Fig. 4b) exhaustive")

    for n in range(1, scope(6, 7)):
        for k in range(-1, 2**n + 2):
            m = Machine("and")
            x, o = m.alloc(n, "x"), m.alloc(1, "o")
            cr, a = m.anc(n, "cr"), m.anc(n, "a")
            A.lt_const(m.ctx, x, k, o[0], cr, a)
            m.free(cr, a)
            for v in range(2**n):
                assert run(m, {x: v})(o) == int(v < k)
    ok("lt_const exhaustive including the k<=0 and k>=2^n boundaries")

    section("modular arithmetic over GF(p)")
    PR = [3, 5, 7, 11, 13, 17, 31, 61] + ([127, 251] if scope(0, 1) else [])
    for p in PR:
        n = p.bit_length()

        m = Machine("and"); x, y = m.alloc(n, "x"), m.alloc(n, "y")
        MA.modadd(m, x, y, p)
        for u in range(p):
            for v in range(p):
                rd = run(m, {x: u, y: v})
                assert rd(y) == (u + v) % p and rd(x) == u

        m = Machine("and"); c = m.alloc(1, "c")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        MA.cmodadd(m, c[0], x, y, p)
        for ct in (0, 1):
            for u in range(p):
                for v in range(p):
                    assert run(m, {c: ct, x: u, y: v})(y) == ((u + v) % p if ct else v)

        m = Machine("and"); x, y = m.alloc(n, "x"), m.alloc(n, "y")
        MA.modsub(m, x, y, p)
        for u in range(p):
            for v in range(p):
                assert run(m, {x: u, y: v})(y) == (v - u) % p

        for fn, ref in ((MA.moddbl, lambda v: 2 * v % p),
                        (MA.modhalf, lambda v: v * pow(2, -1, p) % p),
                        (MA.modneg, lambda v: (-v) % p)):
            m = Machine("and"); x = m.alloc(n, "x")
            fn(m, x, p)
            for v in range(p):
                assert run(m, {x: v})(x) == ref(v), (p, fn.__name__, v)

        for fn, ref in ((MA.cmoddbl, lambda v: 2 * v % p),
                        (MA.cmodhalf, lambda v: v * pow(2, -1, p) % p),
                        (MA.cmodneg, lambda v: (-v) % p)):
            m = Machine("and"); c, x = m.alloc(1, "c"), m.alloc(n, "x")
            fn(m, c[0], x, p)
            for ct in (0, 1):
                for v in range(p):
                    assert run(m, {c: ct, x: v})(x) == (ref(v) if ct else v)

        for k in range(p):
            m = Machine("and"); y = m.alloc(n, "y")
            MA.modadd_const(m, y, k, p)
            for v in range(p):
                assert run(m, {y: v})(y) == (v + k) % p
    ok(f"modadd/cmodadd/modsub/moddbl/modhalf/modneg/cmod* /const exhaustive over p in {PR}")

    section("multiplication")
    for p in [3, 5, 7, 11, 13]:
        n = p.bit_length()
        m = Machine("and")
        x, y, z = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "z")
        MU.modmul(m, x, y, z, p)
        for u in range(p):
            for v in range(p):
                rd = run(m, {x: u, y: v})
                assert rd(z) == u * v % p and rd(x) == u and rd(y) == v

        m = Machine("and"); x, z = m.alloc(n, "x"), m.alloc(n, "z")
        MU.modsqr(m, x, z, p)
        for u in range(p):
            assert run(m, {x: u})(z) == u * u % p

        for fn, ref in ((MU.modmul_add, lambda w, u, v: (w + u * v) % p),
                        (MU.modmul_sub, lambda w, u, v: (w - u * v) % p)):
            m = Machine("and")
            x, y, ac = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "a")
            fn(m, x, y, ac, p)
            for u in range(p):
                for v in range(p):
                    for w in range(p):
                        assert run(m, {x: u, y: v, ac: w})(ac) == ref(w, u, v)

        m = Machine("and"); x, y, ac = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "a")
        MU.modmul_xor(m, x, y, ac, p)
        for u in range(p):
            for v in range(p):
                assert run(m, {x: u, y: v, ac: u * v % p})(ac) == 0

        for k in range(p):
            m = Machine("and"); x, z = m.alloc(n, "x"), m.alloc(n, "z")
            MU.modmul_const(m, x, k, z, p)
            for v in range(p):
                assert run(m, {x: v})(z) == k * v % p
    ok("modmul/modsqr/modmul_add/sub/xor/modmul_const exhaustive")

    section("width: the fast simulator is exact far past statevector range")
    for n, p in ((64, (1 << 61) - 1), (128, (1 << 127) - 1)):
        m = Machine("and"); x, y = m.alloc(n, "x"), m.alloc(n, "y")
        MA.modadd(m, x, y, p)
        for _ in range(5):
            u, v = r.randrange(p), r.randrange(p)
            assert run(m, {x: u, y: v})(y) == (u + v) % p
        ok(f"modadd verified at n={n} ({m.qc.num_qubits} qubits)")


if __name__ == "__main__":
    main()
    print("\ntest_ec_arith: all passed")
