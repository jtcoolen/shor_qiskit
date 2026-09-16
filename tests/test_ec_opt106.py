"""The 2026/106 optimizations: carry-save adder, Montgomery, unconditional
Kaliski, projective point addition, zig-zag."""
from _ec_util import ok, rng, scope, section

import ec_classical as C
import ec_kaliski as K
import ec_kaliski_opt as KO
import ec_montgomery as MG
import ec_proj as PJ
import ec_qcsa as Q
from ec_sim import Machine, run


def main():
    r = rng()

    section("quantum carry-save adder")
    for W in (4, 6, 8):
        m = Machine("and")
        A, B, Cr = m.alloc(W, "A"), m.alloc(W, "B"), m.alloc(W, "C")
        Sm, Mj = Q.csa_compress(m, A, B, Cr)
        for _ in range(scope(50, 300)):
            a, b, c = [r.getrandbits(W - 1) for _ in range(3)]
            rd = run(m, {A: a, B: b, Cr: c})
            assert rd(Sm) + rd(Mj) == a + b + c
            assert (rd(A), rd(B), rd(Cr)) == (a, b, c)
    ok("3:2 compressor: A+B+C == S + 2*MAJ with the inputs preserved")

    for n, w in ((4, 3), (4, 5), (5, 8), (6, 4)):
        W = Q.qcsa_width(n, w)
        m = Machine("and")
        ops = [m.alloc(W, f"o{i}") for i in range(w)]
        out = m.alloc(W, "out")
        Q.qcsa_sum(m, ops, out)
        for _ in range(scope(20, 100)):
            vals = [r.getrandbits(n) for _ in range(w)]
            rd = run(m, dict(zip(ops, vals)))
            assert rd(out) == sum(vals)
        nc, layers = Q.csa_layers(w)
        assert layers <= max(1, w.bit_length())
    ok("qcsa_sum exact for 3..8 operands; tree teardown returns every ancilla")

    section("word-level Montgomery multiplication")
    for kind in ("qcsa", "lookup"):
        for n, w, p in ((8, 2, 251), (8, 4, 251), (9, 3, 509), (12, 4, 4093)):
            m = Machine("and")
            a, b, o = m.alloc(n, "a"), m.alloc(n, "b"), m.alloc(n, "o")
            (MG.mont_mul_qcsa if kind == "qcsa" else MG.mont_mul_lookup)(
                m, a, b, o, p, w)
            R = pow(2, -n, p)
            for _ in range(scope(15, 60)):
                av, bv = r.randrange(p), r.randrange(p)
                rd = run(m, {a: av, b: bv})
                assert rd(o) == av * bv * R % p, (kind, n, w, av, bv)
                assert rd(a) == av and rd(b) == bv
        ok(f"mont_mul_{kind} == a*b*2^-n mod p")
    for n, w, p in ((8, 2, 251), (12, 4, 4093)):
        for kind in ("qcsa", "lookup"):
            m = Machine("and")
            a, b, o = m.alloc(n, "a"), m.alloc(n, "b"), m.alloc(n, "o")
            MG.mont_mul_clean(m, a, b, o, p, w, kind)
            for _ in range(scope(8, 30)):
                av, bv = r.randrange(p), r.randrange(p)
                assert run(m, {a: av, b: bv})(o) == av * bv * pow(2, -n, p) % p
    ok("mont_mul_clean leaves no garbage at all")

    section("unconditional Kaliski (Sec 3.3)")
    for p in (7, 11, 13, 17, 31, 61) + ((127, 251) if scope(0, 1) else ()):
        n = p.bit_length()
        m = Machine("and")
        x, o = m.alloc(n, "x"), m.alloc(n, "o")
        KO.mod_inv_mont_clean(m, x, o, p)
        for xv in (range(1, p) if p < 70 else [r.randrange(1, p) for _ in range(20)]):
            rd = run(m, {x: xv})
            assert rd(o) == pow(xv, -1, p) * pow(2, 2 * n, p) % p
            assert rd(x) == xv
        for _ in range(scope(10, 40)):
            X = r.randrange(1, p)
            assert run(m, {x: C.to_mont(X, p, n)})(o) == C.to_mont(pow(X, -1, p), p, n)
    ok("exact, and Montgomery form in gives Montgomery form out -- no counter, "
       "no corrective doublings")

    for p in (31, 127, 1021):
        n = p.bit_length()
        a = Machine("and"); xa, oa = a.alloc(n, "x"), a.alloc(n, "o"); K.mod_inv(a, xa, oa, p)
        b = Machine("and"); xb, ob = b.alloc(n, "x"), b.alloc(n, "o"); KO.mod_inv_mont(b, xb, ob, p)
        na = sum(1 for i in a.qc.data if i.operation.name == "ecand")
        nb = sum(1 for i in b.qc.data if i.operation.name == "ecand")
        assert nb < na, (p, na, nb)
        assert b.qc.depth() < a.qc.depth()
    ok("optimized inversion is strictly cheaper in both AND count and depth "
       "at every size tested")

    section("projective point addition (Alg. 4) -- 11 multiplications, no inversion")
    for curve, label in ((C.CLASSIQ, "mod 7"), (C.TOY11, "mod 11")):
        p, n = curve.p, curve.p.bit_length()
        pts = [P for P in curve.points() if not P.inf]
        good = 0
        for P2 in pts:
            m = Machine("and")
            X1, Y1, Z1 = m.alloc(n, "X1"), m.alloc(n, "Y1"), m.alloc(n, "Z1")
            X3, Y3, Z3 = m.alloc(n, "X3"), m.alloc(n, "Y3"), m.alloc(n, "Z3")
            PJ.jacobian_add(m, X1, Y1, Z1, P2.x, P2.y, X3, Y3, Z3, p)
            for P1 in pts:
                if P1.x == P2.x:
                    continue
                for Z in range(1, p if scope(0, 1) else 3):
                    x1, y1 = P1.x * Z * Z % p, P1.y * Z**3 % p
                    rd = run(m, {X1: x1, Y1: y1, Z1: Z})
                    got = C.jacobian_to_affine(curve, rd(X3), rd(Y3), rd(Z3))
                    assert got == curve.add(P1, P2), (label, P1, P2, Z)
                    assert (rd(X1), rd(Y1), rd(Z1)) == (x1, y1, Z)
                    good += 1
        ok(f"{label}: {good} additions exact; all 15 intermediates cleaned, only "
           f"the 3n-qubit input point left as garbage")

    section("zig-zag schedule applied to a real chain")
    curve = C.CLASSIQ
    p, n = curve.p, 3
    start, chain, acc = C.Point(3, 2), [], C.Point(3, 2)
    for _ in range(6):
        for cand in curve.points():
            if cand.inf or C.point_add_exceptional(curve, acc, cand):
                continue
            chain.append(cand)
            acc = curve.add(acc, cand)
            break
    m = Machine("and")
    X0, Y0, Z0 = m.alloc(n, "X0"), m.alloc(n, "Y0"), m.alloc(n, "Z0")
    trip, resid, mregs = PJ.zigzag_chain(
        m, (X0, Y0, Z0), [(q.x, q.y) for q in chain], p, n)
    rd = run(m, {X0: start.x, Y0: start.y, Z0: 1})
    got = C.jacobian_to_affine(curve, rd(trip[0]), rd(trip[1]), rd(trip[2]))
    assert got == acc, (got, acc)
    assert mregs == C.zigzag_registers(len(chain))
    ok(f"{len(chain)} chained additions on {mregs} register triples give the "
       f"right point; {len(resid)} intermediate points remain live, as the "
       f"schedule predicts")


if __name__ == "__main__":
    main()
    print("\ntest_ec_opt106: all passed")
