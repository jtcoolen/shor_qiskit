"""The 2026/1128 optimizations: Euclidean dialog, Bezout replay, in-place
multiplication, approximate and pseudo-Mersenne arithmetic, windowed addition."""
from _ec_util import ok, rng, scope, section

import ec_approx as AX
import ec_classical as C
import ec_eea as E
import ec_modarith as MA
import ec_window as W
from ec_sim import Machine, SimError, run


def ands(m):
    return sum(1 for i in m.qc.data if i.operation.name == "ecand")


def main():
    r = rng()

    section("the dialog (Alg. 2)")
    for q in (7, 11, 13, 31, 61, 127):
        n = q.bit_length()
        it = C.eea_iterations(n)
        m = Machine("and")
        x = m.alloc(n, "x")
        recs, u, v = E.dialog(m, x, q, it)
        for xv in range(1, q):
            rd = run(m, {x: xv})
            bits, uu, vv = C.eea_dialog(q, xv, it)
            assert rd(u) == uu and rd(v) == vv
            assert [(rd(a), rd(b)) for a, b in recs] == bits
    ok("decision bits match the classical model exactly; u=1, v=0 at the end, "
       "so x's own register comes back clean")

    section("in-place multiplication (Alg. 3 + 4)")
    for q in (7, 11, 13, 31) + ((61,) if scope(0, 1) else ()):
        n = q.bit_length()
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        E.inplace_mul(m, x, y, q)
        for xv in range(1, q):
            for yv in range(q):
                rd = run(m, {x: xv, y: yv})
                assert rd(y) == xv * yv % q, (q, xv, yv)
                assert rd(x) == xv
    ok("|x,y> -> |x, xy mod q>, x restored, every ancilla and every record "
       "qubit back to |0> -- one pass does the inversion and the multiplication")

    for q in (7, 11, 13, 31):
        n = q.bit_length()
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        E.inplace_div(m, x, y, q)
        for xv in range(1, q):
            for yv in range(q):
                rd = run(m, {x: xv, y: yv})
                assert rd(y) == yv * pow(xv, -1, q) % q
                assert rd(x) == xv
    ok("the SAME circuit run backwards divides -- which is why the point "
       "addition needs one multiplier, not a multiplier and an inverter")

    section("Fig. 1 compression")
    perm = E.compression_permutation()
    assert sorted(perm) == list(range(64)) and len(set(perm.values())) == 64
    m = Machine("and")
    recs = [(m.alloc(1, f"b0_{i}"), m.alloc(1, f"b1_{i}")) for i in range(3)]
    freed = E.compress_records(m, recs)
    from ec_sim import simulate
    idx = {qb: i for i, qb in enumerate(m.qc.qubits)}
    trips = [(0, 0), (1, 0), (1, 1)]
    for a in trips:
        for b in trips:
            for c in trips:
                init = {}
                for (p0, p1), pr in zip(recs, (a, b, c)):
                    init[p0[0]], init[p1[0]] = pr
                bits = simulate(m.qc, init)
                qs = [recs[0][0][0], recs[0][1][0], recs[1][0][0],
                      recs[1][1][0], recs[2][0][0], recs[2][1][0]]
                val = sum(bits[idx[qb]] << i for i, qb in enumerate(qs))
                assert val == C.compress_triple([a, b, c])
                assert bits[idx[freed[0]]] == 0
    ok("all 27 valid patterns compress correctly and free one qubit per 3 rounds")
    w = E.dialog_width_compressed(4096) / 4096
    assert 2.35 < w < 2.45, w
    ok(f"asymptotically 2.355n record qubits (measured {w:.3f}n at n=4096)")

    section("approximate arithmetic (Alg. 6, 7, 9, 10)")
    print("      these circuits are SUPPOSED to fail sometimes -- the question")
    print("      is whether the failure rate tracks 2^-msbs, and it does:")

    def rate_dbl(fn, q, **kw):
        n = q.bit_length()
        m = Machine("and")
        x = m.alloc(n, "x")
        fn(m, x, q, **kw)
        bad = 0
        for v in range(q):
            try:
                if run(m, {x: v})(x) != 2 * v % q:
                    bad += 1
            except SimError:
                bad += 1
        return bad / q, ands(m)

    for q in (251, 1021, 4093):
        n = q.bit_length()
        me = Machine("and"); xe = me.alloc(n, "x"); MA.moddbl(me, xe, q)
        prev = None
        for msbs in (2, 4, 6):
            rt, na = rate_dbl(AX.moddbl_approx, q, msbs=msbs)
            if prev is not None:
                assert rt <= prev + 1e-9, (q, msbs, rt, prev)
            prev = rt
            print(f"      Alg 6  q={q:>5} msbs={msbs}: failure {rt*100:>6.2f}%  "
                  f"{na:>3} ANDs (exact Alg 5: {ands(me)})")
        rt_exact, _ = rate_dbl(AX.moddbl_approx, q, msbs=n)
        assert rt_exact == 0.0
    ok("Alg 6 failure falls monotonically with msbs and reaches 0 at msbs=n")

    for q in (251, 1021, 65521):
        pm = AX.pseudo_mersenne(q)
        if not pm:
            continue
        u, f = pm
        n = q.bit_length()
        me = Machine("and"); xe = me.alloc(n, "x"); MA.moddbl(me, xe, q)
        lsbs = max(2, f.bit_length() + 3)
        rt, na = rate_dbl(AX.moddbl_pm, q, lsbs=lsbs)
        assert na < ands(me) / 2, (q, na, ands(me))
        print(f"      Alg 7  q={q}=2^{u}-{f}: failure {rt*100:>5.2f}%  "
              f"{na} ANDs vs {ands(me)} exact")
    ok("Alg 7 pseudo-Mersenne doubling costs under half the exact circuit")

    section("approximate arithmetic composed into the dialog multiplier")
    print("      per-operation error compounds over the replay's ~1.4n")
    print("      iterations, which is the whole reason 1128 picks msbs so large:")
    for q in (11, 31):
        n = q.bit_length()
        variants = [("exact", None, None)]
        for msbs in (2, n):
            variants.append(
                (f"approx msbs={msbs}",
                 (lambda ms: (lambda mm, reg: AX.moddbl_approx(mm, reg, q, ms)))(msbs),
                 (lambda ms: (lambda mm, c, a, b: AX.cmodadd_approx(mm, c, a, b, q, ms)))(msbs)))
        for label, dbl, cadd in variants:
            m = Machine("and")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            E.inplace_mul(m, x, y, q, dbl=dbl, cadd=cadd)
            bad = tot = 0
            for xv in range(1, q):
                for yv in range(0, q, 2):
                    tot += 1
                    try:
                        if run(m, {x: xv, y: yv})(y) != xv * yv % q:
                            bad += 1
                    except SimError:
                        bad += 1
            rate = bad / tot
            if label == "exact" or "msbs=%d" % n in label:
                assert rate == 0.0, (q, label, rate)
            print(f"      q={q:>3} {label:>16}: {100*rate:>5.1f}% failure, "
                  f"{ands(m):>5} ANDs")
    ok("full-width comparison is exact end to end; a 2-bit comparison fails "
       "~12% per operation, which compounds to ~80% over the whole multiplier "
       "-- at n=256 the paper's 40-50 bits give ~2^-40 per operation, so even "
       "360 iterations stay negligible")

    section("windowed point addition (Alg. 1)")
    curve = C.CLASSIQ
    p, n = curve.p, 3
    for wd in (2,):
        pts = W.window_points(curve, C.CLASSIQ_G, wd)
        m = Machine("and")
        addr = m.alloc(wd, "i")
        x2, y2 = m.alloc(n, "x2"), m.alloc(n, "y2")
        W.windowed_point_add(m, addr, x2, y2, pts, p)
        good = skip = zero = 0
        for R in curve.points():
            if R.inf:
                continue
            for i in range(1 << wd):
                Pi = pts[i]
                if Pi.inf and R.x == 0:
                    skip += 1
                    continue
                if not Pi.inf and C.point_add_exceptional(curve, R, Pi):
                    skip += 1
                    continue
                want = R if Pi.inf else curve.add(R, Pi)
                if want.inf:
                    skip += 1
                    continue
                rd = run(m, {addr: i, x2: R.x, y2: R.y})
                assert (rd(x2), rd(y2)) == (want.x, want.y), (R, i, Pi)
                assert rd(addr) == i
                good += 1
                zero += int(Pi.inf)
        ok(f"w={wd}: {good} windowed additions exact ({zero} of them adding the "
           f"point at infinity with no special case), {skip} excluded")
    assert W.n_windowed_additions(256, 16) == 34
    ok("2*ceil((n+1)/w) additions: 34 at n=256, w=16, against 514 unwindowed")


if __name__ == "__main__":
    main()
    print("\ntest_ec_opt1128: all passed")
