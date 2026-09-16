"""The classical reference models -- every circuit here is checked against these,
so they get checked first, and independently."""
from _ec_util import ok, rng, scope, section

import ec_classical as C


def main():
    r = rng()
    section("curve arithmetic")
    for cu, G, k, Q in [(C.CLASSIQ, C.CLASSIQ_G, 4, C.CLASSIQ_Q)]:
        assert cu.on_curve(G) and cu.on_curve(Q)
        assert cu.mul(k, G) == Q
        assert cu.point_order(G) == 5
    ok("Classiq tutorial curve y^2=x^3+5x+4 mod 7: G=(0,5) has order 5, [4]G=(0,2)")
    assert C.TOY11.point_order(C.TOY11_G) == 13
    ok("toy curve y^2=x^3+x+6 mod 11: G=(2,7) has order 13")

    section("Kaliski (2026/106 Alg. 2)")
    for p in (7, 11, 13, 101, 257, 65537):
        for _ in range(scope(20, 200)):
            x = r.randrange(1, p)
            rr, k, u, v, s = C.kaliski(x, p)
            assert u == 1 and v == 0 and s % p == 0
            assert rr == (-pow(x, -1, p) * pow(2, k, p)) % p
            assert C.mod_inverse_kaliski(x, p) == pow(x, -1, p)
    ok("pseudo-inverse invariant r == -x^-1 2^k mod p, u=1, s=p on termination")

    for n, p in ((7, 101), (17, 65537), (31, 2147483647)):
        for _ in range(scope(10, 100)):
            X = r.randrange(1, p)
            rr, k = C.mod_inverse_kaliski_unconditional(C.to_mont(X, p, n), p, n)
            assert rr == (-pow(X, -1, p) * pow(2, n, p)) % p
    ok("unconditional 2n rounds land on the Montgomery inverse with no counter")

    section("Montgomery (2026/106 Alg. 1b)")
    for n, w, p in ((16, 4, 65521), (16, 8, 65521), (32, 8, 4294967291)):
        for _ in range(scope(20, 200)):
            a, b = r.randrange(p), r.randrange(p)
            assert C.mont_mul_wordlevel(a, b, p, w, n) == a * b * pow(2, -n, p) % p
    ok("word-level Montgomery with the precomputed d == a*b*2^-n mod p")

    section("Euclidean dialog and Bezout replay (2026/1128 Alg. 2-4)")
    for p in (101, 257, 65537, 4294967291):
        n = p.bit_length()
        it = C.eea_iterations(n)
        for _ in range(scope(20, 200)):
            x, y = r.randrange(1, p), r.randrange(p)
            bits, u, v = C.eea_dialog(p, x, it)
            assert u == 1 and v == 0
            rr, s = C.bezout_replay(bits, y, 0, p)
            assert rr == 0 and s == y * x % p
            r2, s2 = C.bezout_replay_inv(bits, 0, y, p)
            assert s2 == 0 and r2 == y * pow(x, -1, p) % p
            assert C.bezout_inverse_classical(bits, p) == pow(x, -1, p)
            assert C.inplace_mul_ref(x, y, p) == x * y % p
            assert C.inplace_div_ref(x, y, p) == y * pow(x, -1, p) % p
    ok("replay reversed multiplies by x; the inverse replay divides by x")

    seen = set()
    trips = [(0, 0), (1, 0), (1, 1)]
    for a in trips:
        for b in trips:
            for c in trips:
                v = C.compress_triple([a, b, c])
                assert v < 32 and C.decompress_triple(v) == [a, b, c]
                seen.add(v)
    assert len(seen) == 27
    ok("Fig. 1 compression is injective: 27 patterns into 5 bits")

    section("Jacobian-affine addition (2026/106 Alg. 4)")
    cu = C.TOY11
    for i in range(1, 12):
        for j in range(1, 12):
            P, Q = cu.mul(i, C.TOY11_G), cu.mul(j, C.TOY11_G)
            if P.inf or Q.inf or P.x == Q.x:
                continue
            for Z in range(1, cu.p):
                X3, Y3, Z3 = C.jacobian_add_ref(
                    cu, P.x * Z * Z % cu.p, P.y * Z**3 % cu.p, Z, Q.x, Q.y)
                assert C.jacobian_to_affine(cu, X3, Y3, Z3) == cu.add(P, Q)
    ok("11-multiplication mixed addition agrees with the affine law, "
       "for every projective representative")

    section("zig-zag schedule (2026/106 Fig. 10)")
    for na in (1, 3, 6, 10, 15, 36, 100):
        tape, m, resid = C.zigzag_schedule(na)
        live, peak = {0}, 0
        for op, j in tape:
            live.add(j) if op == "add" else live.discard(j)
            peak = max(peak, len(live) - 1)
        assert sum(1 for o, _ in tape if o == "add") == na
        assert peak <= m and m * (m + 1) // 2 >= na
    assert C.n_point_additions(192, 11) == 36
    assert C.zigzag_registers(36) == 8
    ok("m(m+1)/2 >= additions, peak live <= m; paper's n=192 w=11 case gives 36 -> m=8")


if __name__ == "__main__":
    main()
    print("\ntest_ec_classical: all passed")
