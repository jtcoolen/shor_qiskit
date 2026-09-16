"""Kaliski modular inversion and division (2026/106 Alg. 2, [HJN+20], [RNSL17])."""
from _ec_util import ok, scope, section

import ec_classical as C
import ec_kaliski as K
from ec_sim import Machine, run


def main():
    section("one round against the classical model")
    for p in (3, 5, 7, 11, 13, 17, 31):
        n = p.bit_length()
        m = Machine("and")
        x, r = m.alloc(n, "x"), m.alloc(n, "r")
        recs, u, v, s, cnt = K.kaliski_pseudo_inverse(m, x, r, p)
        for xv in range(1, p):
            rd = run(m, {x: xv})
            rr, kk, uu, vv, ss = C.kaliski(xv, p, rounds=2 * n, conditional=True)
            assert rd(r) == rr % p, (p, xv, rd(r), rr % p)
            assert rd(u) == uu and rd(v) == vv and rd(s) == ss % p
            assert rd(cnt) == kk, (p, xv, rd(cnt), kk)
            assert rd(x) == xv
    ok("r, u, v, s and the active-round counter all match, every x, every p")

    section("full inversion with the 2^-k and sign corrections")
    for p in (3, 5, 7, 11, 13, 17, 31) + ((61, 127) if scope(0, 1) else ()):
        n = p.bit_length()
        m = Machine("and")
        x, o = m.alloc(n, "x"), m.alloc(n, "o")
        K.mod_inv(m, x, o, p)
        for xv in range(1, p):
            rd = run(m, {x: xv})
            assert rd(o) == pow(xv, -1, p), (p, xv, rd(o))
            assert rd(x) == xv
    ok("mod_inv exact for every invertible x")

    section("division, with every ancilla returned to |0>")
    for p in (3, 5, 7, 11, 13) + ((17,) if scope(0, 1) else ()):
        n = p.bit_length()
        m = Machine("and")
        c = m.alloc(1, "c")
        x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "o")
        K.mod_div(m, c[0], x, y, o, p)
        for ct in (0, 1):
            for xv in range(1, p):
                for yv in range(p):
                    rd = run(m, {c: ct, x: xv, y: yv})
                    assert rd(o) == ((yv * pow(xv, -1, p)) % p if ct else 0)
                    assert rd(x) == xv and rd(y) == yv
    ok("mod_div exact both branches; the ancilla checks in `run` prove the "
       "per-round records and counter all come back clean")

    section("the control is pushed into the inversion's setup")
    p, n = 31, 5
    m = Machine("and")
    c, x, o = m.alloc(1, "c"), m.alloc(n, "x"), m.alloc(n, "o")
    K.mod_inv(m, x, o, p, ctrl=c[0])
    for xv in range(1, p):
        assert run(m, {c: 0, x: xv})(o) == 0
    ok("with ctrl=0 the whole inversion is inert -- which is what keeps a point "
       "addition's q=0 branch from inverting the junk in its x register")


if __name__ == "__main__":
    main()
    print("\ntest_ec_kaliski: all passed")
