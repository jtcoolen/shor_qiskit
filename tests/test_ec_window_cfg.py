"""windowed_point_add_cfg: the 2026 refinements to [1128] Algorithm 1.

  * the default configuration IS windowed_point_add, gate for gate;
  * every configuration adds correctly on every (accumulator, address) pair of
    a p = 61 curve, exceptional pairs excluded and counted;
  * lookups cost what IonQ/1128 say: three loads, one repair;
  * the measurement-based configurations are phase-correct when their
    measurements are actually performed (ec_mbu.run_live);
  * SELECT-SWAP loads cost 2^(w-k) - 2 + (2^k - 1) b and clear their junk by
    measurement into the same merged repair.
"""
import dataclasses
import itertools
import random

from _ec_util import FULL, ok, random_curve, random_generator, section

import ec_classical as C
import ec_cost as CO
import ec_mbu as MB
import ec_window as W
from ec_sim import Machine, run


def ops(m):
    idx = {q: i for i, q in enumerate(m.qc.qubits)}
    return [(ci.operation.name, tuple(idx[q] for q in ci.qubits)) for ci in m.qc.data]


def build(curve, pts, p, cfg=None):
    n, w = p.bit_length(), (len(pts) - 1).bit_length()
    m = Machine("and")
    addr, x, y = m.alloc(w, "a"), m.alloc(n, "x"), m.alloc(n, "y")
    if cfg is None:
        W.windowed_point_add(m, addr, x, y, pts, p)
    else:
        W.windowed_point_add_cfg(m, addr, x, y, pts, p, cfg)
    return m, addr, x, y


def cases(curve, pts, offsets):
    for R in curve.points():
        if R.inf:
            continue
        for i, T in enumerate(pts):
            if T.inf:
                if R.x == 0:            # adding O on x = 0: the (0,0) encoding fails
                    yield R, i, None
                    continue
                yield R, i, R
                continue
            if C.point_add_exceptional(curve, R, T):
                yield R, i, None
                continue
            S = curve.add(R, T)
            yield R, i, (None if S.inf else S)


CONFIGS = {
    "default": W.PointAddCfg(),
    "mbu": W.PointAddCfg(lookup="mbu"),
    "mbu+merge": W.PointAddCfg(lookup="mbu", merge_xy=True),
    "mbu+merge+free": W.PointAddCfg(lookup="mbu", merge_xy=True, free_xy1=True),
    "mbu+serial": W.PointAddCfg(lookup="mbu", serial_load=True),
    "recompute+free": W.PointAddCfg(free_xy1=True),
    "IonQ (mbu+merge+free+offsets)": W.IONQ_LOOKUPS,
    "IonQ + SELECT-SWAP k = 1": dataclasses.replace(W.IONQ_LOOKUPS, select_swap=1),
    "IonQ + SELECT-SWAP on 3x, k = 2": dataclasses.replace(W.IONQ_LOOKUPS, select_swap=(0, 2)),
    "mbu+serial + SELECT-SWAP k = 1": W.PointAddCfg(lookup="mbu", serial_load=True, select_swap=1),
}


def main():
    rnd = random.Random(61)
    curve, pts_all = random_curve(rnd, pmax=61, pmin=61)
    G = random_generator(rnd, curve, pts_all, min_order=8)
    p, n = curve.p, curve.p.bit_length()

    section("default configuration = windowed_point_add, gate for gate")
    for w in (1, 2, 3):
        plain = W.window_points(curve, G, w)
        a = build(curve, plain, p)[0]
        b = build(curve, plain, p, W.PointAddCfg())[0]
        assert ops(a) == ops(b)
        assert [c[0] for c in a.checks] == [c[0] for c in b.checks]
    ok("w = 1, 2, 3: identical instruction streams and ancilla checks")

    section(f"every configuration adds correctly on {curve.name} (p={p})")
    ws = (2, 3) if FULL else (2,)
    for w in ws:
        plain = W.window_points(curve, G, w)
        masked, mu = W.masked_window_points(curve, G, w, random.Random(w))
        for label, cfg in CONFIGS.items():
            pts = masked if cfg.offsets else plain
            m, addr, x, y = build(curve, pts, p, cfg)
            good = skipped = 0
            for R, i, S in cases(curve, pts, cfg.offsets):
                if S is None:
                    skipped += 1
                    continue
                rd = run(m, {addr: i, x: R.x, y: R.y})
                assert (rd(x), rd(y), rd(addr)) == (S.x, S.y, i), (label, R, i)
                good += 1
            c = CO.count(m)
            print(f"      w={w} {label:<31} {c['qubits']:>4} qubits "
                  f"{c['toffoli_paper']:>6} Toffoli-eq  ({good} pairs, "
                  f"{skipped} exceptional)")
    ok("all configurations exact on every non-exceptional pair; with offsets "
       "the x_R = 0 exclusion is gone")

    section("lookup cost: 3 loads + 1 merged repair instead of 10 recomputes")
    w = 4
    plain = W.window_points(curve, G, w)
    masked, _ = W.masked_window_points(curve, G, w, random.Random(4))
    base = CO.count(build(curve, plain, p)[0])
    ionq = CO.count(build(curve, masked, p, W.IONQ_LOOKUPS)[0])
    L = 1 << w
    fix = MB.best_split(w, MB.phase_ancillas(w, w // 2, False), False)[1]
    lookup_ionq = 3 * MB.walk_ands(w, False) + fix
    lookup_base = 10 * 2 * (L - 1)
    saved = base["toffoli_paper"] - ionq["toffoli_paper"]
    print(f"      w={w}: lookups {lookup_base} -> {lookup_ionq} Toffoli-eq; "
          f"whole addition {base['toffoli_paper']} -> {ionq['toffoli_paper']} "
          f"({saved} saved), qubits {base['qubits']} -> {ionq['qubits']}")
    assert ionq["mbu_toffoli"] == lookup_ionq
    assert base["qubits"] - ionq["qubits"] >= 2 * n
    ok("measured lookup cost matches 3 (L-2) + one sqrt(L) repair exactly")

    section("SELECT-SWAP: 2^(w-k) - 2 ANDs + (2^k - 1) b Fredkins per load, junk measured")
    for kp, k3 in ((1, 0), (0, 1), (2, 3), (4, 4)):
        c = CO.count(build(curve, masked, p, dataclasses.replace(W.IONQ_LOOKUPS,
                                                                 select_swap=(kp, k3)))[0])
        want = ionq["toffoli_paper"]
        for k, b, times in ((kp, 2 * n, 2), (k3, n, 1)):
            if k:
                want -= times * (MB.walk_ands(w, False) - MB.select_swap_ands(w, k, b))
        print(f"      w={w} k=({kp}, {k3}): {c['toffoli_paper']} Toffoli-eq, {c['qubits']} qubits, "
              f"{c['measure'] - ionq['measure']} more measurements")
        assert c["toffoli_paper"] == want, (kp, k3, c["toffoli_paper"], want)
        assert c["measure"] - ionq["measure"] == (
            2 * ((1 << kp) - 1) * 2 * n + ((1 << k3) - 1) * n)
    ok("the measured cost is the formula exactly; each junk word costs one "
       "X-measurement per qubit and no repair of its own")

    section("measurement-based configurations are phase-correct, run literally")
    cu13 = C.Curve(13, 1, 6, "p13")
    pts13 = [P for P in cu13.points() if not P.inf]
    G13 = max(pts13, key=cu13.point_order)
    for label in ("mbu+merge+free", "IonQ (mbu+merge+free+offsets)", "IonQ + SELECT-SWAP k = 1",
                  "IonQ + SELECT-SWAP on 3x, k = 2"):
        cfg = CONFIGS[label]
        tab = (W.masked_window_points(cu13, G13, 2, random.Random(1))[0]
               if cfg.offsets else W.window_points(cu13, G13, 2))
        m, addr, x, y = build(cu13, tab, 13, cfg)
        inputs = []
        for R, i, S in cases(cu13, tab, cfg.offsets):
            if S is not None and len(inputs) < 10:
                inputs.append(({addr: i, x: R.x, y: R.y}, S))
        want = {id(d): S for d, S in inputs}

        def expect(inp, br):
            S = want[id(inp)]
            assert (br.value(x), br.value(y)) == (S.x, S.y)
        MB.live_coherent(m.qc, [d for d, _ in inputs],
                         [MB.random_outcomes(s) for s in range(3)],
                         checks=m.checks, expect=expect)
        print(f"      {label}: {m.qc.num_qubits} qubits, {len(inputs)} inputs x "
              f"3 outcome sequences, every AND-dagger and unlookup measured")
    ok("same phase on every input, correct sums: the merged repair works")


if __name__ == "__main__":
    main()
    print("\ntest_ec_window_cfg: all passed")
