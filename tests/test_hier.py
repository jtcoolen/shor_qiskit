"""hier: memoised hierarchical building gives exactly the flat counts.

Every field `ec_cost.count` reports -- Toffolis, ANDs and their measured
uncomputes, Fredkins, the carried costs of measurement-based gates,
measurements, and the qubit count -- must agree between the flat circuit and
the hierarchical one, on the full windowed point addition across GCD backends
and lookup styles.  That is what licenses running the same builders at n = 256.
And outside `tracing()` nothing changes: the wrapped builders are restored.
"""
import random

from _ec_util import ok, random_curve, random_generator, section

import ec_adders as A
import ec_cost as CO
import ec_gcd as G
import ec_square as SQ
import ec_window as W
import hier as H
from ec_sim import Machine

FIELDS = ("qubits", "toffoli_paper", "and", "and_dg", "toffoli", "cswap",
          "mbu_toffoli", "measure")


def build(mk, cfg, tab, q):
    m = mk("and")
    a, x, y = m.alloc(2, "a"), m.alloc(6, "x"), m.alloc(6, "y")
    W.windowed_point_add_cfg(m, a, x, y, tab, q, cfg)
    return m


def main():
    section("hierarchical = flat, field for field")
    rnd = random.Random(61)
    curve, pts = random_curve(rnd, pmax=61, pmin=61)
    G0 = random_generator(rnd, curve, pts, 8)
    tab_m = W.masked_window_points(curve, G0, 2, random.Random(2))[0]
    tab_p = W.window_points(curve, G0, 2)
    q = 61
    cfgs = {
        "default": (W.PointAddCfg(), tab_p),
        "IonQ lookups": (W.IONQ_LOOKUPS, tab_m),
        "CondInv PM + squarer": (W.PointAddCfg(
            lookup="mbu", merge_xy=True, offsets=True, free_xy1=True,
            mul=G.CondInv(arith=G.PM(q, msbs=6), cmp_msbs=6, c_pad=2.3, replay="ci"),
            square=lambda m, c, s, a, p: SQ.csub_square_pm(m, c, s, a, p, msbs=6)), tab_m),
        "Dialog share + Fig. 1": (W.PointAddCfg(
            lookup="mbu", offsets=True,
            mul=G.Dialog(fused_cmp=True, c_pad=2.3, share=True, compress="fig1")), tab_m),
        "PingPong": (W.PointAddCfg(lookup="mbu", offsets=True, mul=G.PingPong(rounds=17)), tab_m),
        "Jump2": (W.PointAddCfg(lookup="mbu", offsets=True, mul=G.Jump2(steps=9)), tab_m),
    }
    for lab, (cfg, tab) in cfgs.items():
        flat = CO.count(build(Machine, cfg, tab, q))
        with H.tracing():
            hm = build(H.HierMachine, cfg, tab, q)
            h = H.count(hm)
        diff = {k: (flat[k], h[k]) for k in FIELDS if flat[k] != h[k]}
        assert not diff, (lab, diff)
        print(f"      {lab:<24} {flat['toffoli_paper']:>6} Toffoli-eq, {flat['qubits']} qubits; "
              f"{len(build(Machine, cfg, tab, q).qc.data)} flat ops -> "
              f"{h['top_level_ops']} top-level, {h['cached_nodes']} cached")
    ok("identical on every field, six configurations")

    try:
        from qualtran.resource_counting import QECGatesCost, get_cost_value
    except ImportError:
        print("      (qualtran not installed: export check skipped)")
    else:
        for lab, (cfg, tab) in cfgs.items():
            with H.tracing():
                hm = build(H.HierMachine, cfg, tab, q)
                ours = H.count(hm)["toffoli_paper"]
                gc = get_cost_value(H.to_qualtran(hm, lab), QECGatesCost())
            assert gc.total_toffoli_only() == ours, (lab, gc, ours)
        ok("exported to Qualtran bloqs, QECGatesCost counts exactly the same Toffolis")

    assert not getattr(A.add, "__boundary__", False)
    assert "__boundary__" not in vars(G.Dialog)["_round"].__dict__
    ok("outside tracing() the builders are the originals")


def test_hier():
    main()


if __name__ == "__main__":
    main()
    print("\ntest_hier: all passed")
