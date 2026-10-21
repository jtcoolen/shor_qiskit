"""Signed odd-digit windows ([HJN+20] Sec 5.1, [106] Sec 6.1): ec_signedwin.

  * the table has 2^(w-1) entries, none of them O, and the sign/complement
    rule applied to it adds [i] B + Delta for every address i (classically);
  * the circuit does exactly that on every (accumulator, address) pair of an
    odd-order curve, exceptional pairs excluded and counted, ancillas clean;
    the measurement-based configuration is phase-correct run literally;
  * lookups halve: against the full 2^w table (masked, so the two differ only
    in the table) the ten recompute lookups drop by exactly 10 * 2^w Toffolis
    and the MBU ones to the (w-1)-bit formula, paid back by two y-negations;
  * a run of windows computes start + [k] P + Delta for every k, classically
    and through the circuit, with W + 1 negations instead of 2W.
"""
import random

from _ec_util import FULL, ok, section

import ec_classical as C
import ec_cost as CO
import ec_mbu as MB
import ec_modarith as MA
import ec_signedwin as S
import ec_window as W
from ec_sim import Machine, Reg, run

# prime-order curves, so every point but O generates and ord is odd
CU13 = C.Curve(13, 0, 2, "p13-ord19")
CU31 = C.Curve(31, 0, 3, "p31-ord43")
CU61 = C.Curve(61, 0, 2, "p61-ord61")


def affine(curve):
    return [P for P in curve.points() if not P.inf]


def build(curve, w, table, cfg=S.SIGNED, neg=None, signed=True, mk=Machine):
    n = curve.p.bit_length()
    m = mk("and")
    a, x, y = m.alloc(w, "a"), m.alloc(n, "x"), m.alloc(n, "y")
    if signed:
        S.windowed_point_add_signed(m, a, x, y, table, curve.p, cfg, neg)
    else:
        W.windowed_point_add_cfg(m, a, x, y, table, curve.p, cfg)
    return m, a, x, y


def cases(curve, B, w, table, delta):
    """(R, i, want) over every affine R and address i; want None if the inner
    addition (+-R) + T is exceptional."""
    for R in affine(curve):
        for i in range(1 << w):
            neg, j = S.signed_digit(i, w)
            Rin = curve.neg(R) if neg else R
            if C.point_add_exceptional(curve, Rin, table[j]):
                yield R, i, None
                continue
            yield R, i, curve.add(curve.add(R, curve.mul(i, B)), delta)


def test_table():
    section("the table: 2^(w-1) odd multiples of B/2, no O, adds [i] B + Delta")
    checked = 0
    for curve, ws in ((CU13, (2, 3, 4)), (CU31, (2, 3, 4, 5)), (CU61, (2, 3, 4, 5))):
        pts = affine(curve)
        r = curve.point_order(pts[0])
        h = (r + 1) // 2
        for B in pts[:3]:
            for w in ws:
                T, delta = S.signed_window_points(curve, B, w)
                assert len(T) == 1 << (w - 1)
                assert not any(P.inf for P in T)
                assert all(T[j] == curve.mul((2 * j + 1) * h % r, B) for j in range(len(T)))
                assert delta == curve.mul((1 - (1 << w)) * h % r, B)
                for R in [C.O] + pts:
                    for i in range(1 << w):
                        got = S.signed_add_ref(curve, R, i, T, w)
                        assert got == curve.add(curve.add(R, curve.mul(i, B)), delta)
                        checked += 1
    ok(f"{checked} (R, i) pairs: -+T[l~] is exactly [i] B + Delta; tables O-free")

    # 2 must be invertible mod ord(B): a point of order 2 (y = 0) is refused
    two = next((cu, P) for p in (7, 11, 13, 17) for a in range(p) for b in range(p)
               if (4 * a**3 + 27 * b * b) % p
               for cu in [C.Curve(p, a, b)] for P in affine(cu) if P.y == 0)
    try:
        S.signed_window_points(two[0], two[1], 2)
        raise AssertionError("an even-order base was accepted")
    except AssertionError as e:
        assert "odd" in str(e), e
    ok("an even-order base is refused (signed odd digits need ord(B) odd)")


def test_single_window():
    section("one signed window: every (accumulator, address) pair")
    runs = [(CU31, 2, "recompute", S.SIGNED, None),
            (CU31, 3, "recompute", S.SIGNED, None),
            (CU31, 2, "IonQ lookups", S.SIGNED_IONQ, None),
            (CU31, 3, "IonQ lookups", S.SIGNED_IONQ, None),
            (CU61, 2, "recompute, exact cmodneg", S.SIGNED, MA.cmodneg)]
    if FULL:
        runs += [(CU61, 3, "recompute", S.SIGNED, None),
                 (CU61, 3, "IonQ lookups", S.SIGNED_IONQ, None),
                 (CU13, 4, "recompute", S.SIGNED, None)]
    for curve, w, label, cfg, neg in runs:
        B = affine(curve)[3]
        T, delta = S.signed_window_points(curve, B, w)
        m, a, x, y = build(curve, w, T, cfg, neg)
        good = skipped = 0
        for R, i, want in cases(curve, B, w, T, delta):
            if want is None:
                skipped += 1
                continue
            rd = run(m, {a: i, x: R.x, y: R.y})
            assert (rd(x), rd(y), rd(a)) == (want.x, want.y, i), (curve.name, w, label, R, i)
            good += 1
        c = CO.count(m)
        print(f"      {curve.name} w={w} {label:<25} {c['qubits']:>3} qubits "
              f"{c['toffoli_paper']:>5} Toffoli-eq  ({good} pairs, {skipped} exceptional)")
    ok("R + [i] B + Delta on every non-exceptional pair, address restored, "
       "ancillas |0>")

    section("the measurement-based configuration is phase-correct, run literally")
    B = affine(CU13)[0]
    T, delta = S.signed_window_points(CU13, B, 2)
    m, a, x, y = build(CU13, 2, T, S.SIGNED_IONQ)
    inputs = []
    for R, i, want in cases(CU13, B, 2, T, delta):
        if want is not None and len(inputs) < 10:
            inputs.append(({a: i, x: R.x, y: R.y}, want))
    want_of = {id(d): S_ for d, S_ in inputs}

    def expect(inp, br):
        S_ = want_of[id(inp)]
        assert (br.value(x), br.value(y)) == (S_.x, S_.y)
    MB.live_coherent(m.qc, [d for d, _ in inputs],
                     [MB.random_outcomes(s) for s in range(3)],
                     checks=m.checks, expect=expect)
    ok(f"{len(inputs)} inputs x 3 outcome sequences: same phase, correct sums")


def neg_cost(n, p, fn):
    m = Machine("and")
    c, y = m.alloc(1, "c"), m.alloc(n, "y")
    fn(m, c[0], y, p)
    return CO.count(m)["toffoli_paper"]


def mbu_lookups(aw):
    """[1128]/IonQ: three merged loads and one repair, on an aw-bit address."""
    fix = MB.best_split(aw, MB.phase_ancillas(aw, aw // 2, False), False)[1]
    return 3 * MB.walk_ands(aw, False) + fix


def test_costs():
    section("cost: half the table, two cheap negations (p = 61, n = 6)")
    curve, p, n = CU61, 61, 6
    B = affine(curve)[3]
    ny, nx = neg_cost(n, p, S.cneg_y), neg_cost(n, p, MA.cmodneg)
    assert ny == n - 1, ny
    print(f"      y-negation: cneg_y {ny} Toffoli-eq (cmodneg {nx}: the zero guard "
          f"is dead weight on an odd-order curve)")
    rows = []
    for w in (2, 3, 4, 5):
        plain = W.window_points(curve, B, w)
        masked, _ = W.masked_window_points(curve, B, w, random.Random(w))
        T, _ = S.signed_window_points(curve, B, w)
        for label, cu, cm, cs in (
                ("recompute", W.PointAddCfg(), W.PointAddCfg(offsets=True), S.SIGNED),
                ("IonQ", W.PointAddCfg(lookup="mbu", merge_xy=True, free_xy1=True),
                 W.IONQ_LOOKUPS, S.SIGNED_IONQ)):
            u = CO.count(build(curve, w, plain, cu, signed=False)[0])
            mk = CO.count(build(curve, w, masked, cm, signed=False)[0])
            sg = CO.count(build(curve, w, T, cs)[0])
            if label == "recompute":
                # same addition, same cfg: the difference IS the lookups and signs
                full, half = 10 * 2 * ((1 << w) - 1), 10 * 2 * ((1 << (w - 1)) - 1)
                assert mk["toffoli_paper"] - sg["toffoli_paper"] == full - half - 2 * ny
            else:
                full, half = mk["mbu_toffoli"], sg["mbu_toffoli"]
                assert (full, half) == (mbu_lookups(w), mbu_lookups(w - 1)), (w, full, half)
            assert sg["qubits"] <= mk["qubits"]
            rows.append([w, label, full, half, u["toffoli_paper"], mk["toffoli_paper"],
                         sg["toffoli_paper"], u["qubits"], mk["qubits"], sg["qubits"]])
    print(CO.table(rows, ["w", "lookups", "lk 2^w", "lk 2^(w-1)", "plain", "masked",
                          "signed", "q plain", "q mask", "q sign"]))
    ok("recompute: masked - signed = 10*2^w - 2 negations exactly; MBU "
       "lookups at the (w-1)-bit formula; qubits never more")

    section("secp256k1, w = 16: one window built at n = 256 (hier)")
    import ec_gcd as G
    import ec_square as SQ
    import hier as H
    curve, R_, n, w, msbs = C.SECP256K1, C.SECP256K1_N, 256, 16, 48
    p = curve.p
    pm = G.PM(p, msbs=msbs)
    cfg_m = W.PointAddCfg(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True,
                          mul=G.CondInv(arith=pm, cmp_msbs=msbs, c_pad=2.3, replay="ci"),
                          square=lambda m, c, s_, a_, q: SQ.csub_square_pm(
                              m, c, s_, a_, q, msbs=msbs))
    rng = random.Random(1)
    B = curve.mul(rng.randrange(1, R_), C.SECP256K1_G)
    masked, _ = W.masked_window_points(curve, B, w, rng)
    T, _ = S.signed_window_points(curve, B, w, order=R_)
    got = {}
    for label, tab, signed in (("masked 2^16", masked, False), ("signed 2^15", T, True)):
        with H.tracing():
            m, _, _, _ = build(curve, w, tab, cfg_m, signed=signed, mk=H.HierMachine)
            got[label] = H.count(m)
    mk_, sg = got["masked 2^16"], got["signed 2^15"]
    ny = neg_cost(n, p, S.cneg_y)
    assert (mk_["mbu_toffoli"], sg["mbu_toffoli"]) == (mbu_lookups(w), mbu_lookups(w - 1))
    assert mk_["toffoli_paper"] - sg["toffoli_paper"] == \
        mk_["mbu_toffoli"] - sg["mbu_toffoli"] - 2 * ny
    assert sg["qubits"] <= mk_["qubits"]
    for label, c in got.items():
        print(f"      {label}: {c['toffoli_paper']:>9,} Toffoli-eq ({c['mbu_toffoli']:,} "
              f"in lookups), {c['qubits']} qubits")
    wins = 28                                     # [1128] eq. (1), [Luo] Sec 6.4
    saved = wins * (mk_["mbu_toffoli"] - sg["mbu_toffoli"]) - (wins + 1) * ny
    print(f"      over {wins} windows (W+1 = {wins + 1} negations of {ny}): "
          f"{saved:,} Toffoli-eq saved")
    ok("the lookup term of [1128] eq. (1) halves, 3*2^w -> 3*2^(w-1), at the "
       "same qubit count")


def test_accumulation():
    section("a run of windows: start + [k] P + Delta, for every k")
    n_k = 0
    for curve, w, nwin in ((CU61, 2, 3), (CU61, 3, 2), (CU31, 2, 3), (CU13, 3, 2)):
        P = affine(curve)[1]
        _, Delta = S.signed_window_tables(curve, P, w, nwin)
        for Sp in (C.O, affine(curve)[0], affine(curve)[-1]):
            for k in range(1 << (w * nwin)):
                R = S.signed_accumulate(curve, P, k, w, nwin, start=Sp)
                want = curve.add(curve.add(Sp, curve.mul(k, P)), Delta)
                assert R == want, (curve.name, w, nwin, Sp, k)
                n_k += 1
    ok(f"{n_k} (curve, start, k): the digits telescope to 2k + 1 - 2^(wW) and "
       f"the half-point table turns that into [k] P + Delta")

    section("the same run through the circuit: two windows, W + 1 negations")
    curve, w, nwin = CU31, 2, 2
    p, n = curve.p, curve.p.bit_length()
    P = affine(curve)[1]
    tables, Delta = S.signed_window_tables(curve, P, w, nwin)
    for label, cfg in (("recompute", S.SIGNED), ("IonQ lookups", S.SIGNED_IONQ)):
        m = Machine("and")
        kr, x, y = m.alloc(w * nwin, "k"), m.alloc(n, "x"), m.alloc(n, "y")
        addrs = [Reg(list(kr[J * w:(J + 1) * w]), f"w{J}") for J in range(nwin)]
        S.signed_windows(m, addrs, x, y, tables, p, cfg)
        good = skipped = 0
        starts = affine(curve)[::7]
        for Sp in starts:
            for k in range(1 << (w * nwin)):
                trace = []
                R = S.signed_accumulate(curve, P, k, w, nwin, start=Sp, trace=trace)
                if any(C.point_add_exceptional(curve, curve.neg(Ri) if ng else Ri, T)
                       for Ri, ng, T in trace):
                    skipped += 1
                    continue
                assert R == curve.add(curve.add(Sp, curve.mul(k, P)), Delta)
                rd = run(m, {kr: k, x: Sp.x, y: Sp.y})
                assert (rd(x), rd(y), rd(kr)) == (R.x, R.y, k), (label, Sp, k)
                good += 1
        # W + 1 negations: the run costs W single windows less W - 1 negations
        one = CO.count(build(curve, w, tables[0], cfg)[0])["toffoli_paper"]
        run_cost = CO.count(m)["toffoli_paper"]
        ny = neg_cost(n, p, S.cneg_y)
        assert run_cost == nwin * one - (nwin - 1) * ny, (run_cost, one, ny)
        print(f"      {label:<13} {good} (start, k) exact, {skipped} with an "
              f"exceptional step; {run_cost} Toffoli-eq = {nwin} x {one} - "
              f"{nwin - 1} x {ny}")
    ok("start + [k] P + Delta through the circuit; merged negations save one "
       "per window boundary")


def main():
    test_table()
    test_single_window()
    test_costs()
    test_accumulation()


if __name__ == "__main__":
    main()
    print("\ntest_ec_signedwin: all passed")
