"""The qubit/Toffoli frontier of one windowed point addition at n = 256.

Each row adds one space option to the IonQ-style configuration and is built
at n = 256 on secp256k1 (w = 16) through `hier.tracing()`: exact counts from
the same builders the tests verify.  Writes bench/ec_space_256.json.
"""
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "bench"))

import ec_gcd as G
import ec_space as SP
import ec_square as SQ
import ec_window as W
import hier as H
from ec_hier_256 import CURVE, GEN, MSBS, N, ORDER, P, WBITS


def rows():
    pm, pms = G.PM(P, msbs=MSBS), G.PMSpace(P, msbs=MSBS)
    sq = lambda m, c, s, a, p: SQ.csub_square_pm(m, c, s, a, p, msbs=MSBS)
    base = dict(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True, square=sq)
    pml = G.PMSpace(P, msbs=MSBS, lean=True)
    lean = dict(base, add=pms, square=lambda m, c, s, a, p: SP.csub_square_pm_space(
        m, c, s, a, p, msbs=MSBS))
    lean_all = dict(base, add=pml, square=lambda m, c, s, a, p: SP.csub_square_pm_space(
        m, c, s, a, p, msbs=MSBS, lean=True))

    def squarer(budget):
        return dict(lean_all, square=lambda m, c, s, a, p: SP.csub_square_pm_space(
            m, c, s, a, p, msbs=MSBS, sqr_space=budget, lean=True))

    def dialog(arith, **kw):
        return G.Dialog(arith=arith, fused_cmp=True, cmp_msbs=MSBS, c_pad=2.3, share=True,
                        compress="fig1", **kw)

    def ci(**kw):
        return W.PointAddCfg(**base, mul=G.CondInv(cmp_msbs=MSBS, c_pad=2.3, **kw))
    return {
        "IonQ-style (cond.-inverted, PM, IonQ replay)": ci(arith=pm, replay="ci"),
        "+ Fig. 1 record packing": ci(arith=pm, replay="ci", compress="fig1"),
        "+ replay in x's qubits": ci(arith=pm, replay="ci", compress="fig1", reuse_x=True),
        "+ CDKM replay arithmetic (PMSpace)": ci(arith=pms, replay="ci", compress="fig1",
                                                 reuse_x=True),
        "+ CDKM walk adder": ci(arith=pms, replay="ci", compress="fig1", reuse_x=True,
                                walk_space=True),
        "same, dialog replay instead of IonQ's": ci(arith=pms, replay="standard",
                                                    compress="fig1", reuse_x=True,
                                                    walk_space=True),
        "[1128] dialog + sharing + Fig. 1, PMSpace": W.PointAddCfg(
            **base, mul=G.Dialog(arith=pms, fused_cmp=True, cmp_msbs=MSBS, c_pad=2.3,
                                 share=True, compress="fig1")),
        "+ CDKM square-subtract and point-add adders": W.PointAddCfg(
            **lean, mul=dialog(pms)),
        "+ CDKM dialog walk": W.PointAddCfg(**lean, mul=dialog(pms, walk_space=True)),
        "+ lean replay cells (CDKM compare, chunked all-ones, borrowed increment)":
            W.PointAddCfg(**lean_all, mul=dialog(pml, walk_space=True)),
        "+ CDKM squarer": W.PointAddCfg(**squarer(True), mul=dialog(pml, walk_space=True)),
        "+ Gidney where there is headroom (walk 0.94n, squarer 0.78n)": W.PointAddCfg(
            **squarer(200), mul=dialog(pml, walk_space=240)),
        "  (side: cond.-inverted walk, same cells -- its record is not shared)":
            W.PointAddCfg(**squarer(200), mul=G.CondInv(
                arith=pml, cmp_msbs=MSBS, c_pad=2.3, replay="standard", compress="fig1",
                reuse_x=True, walk_space=True)),
    }


def main():
    rng = random.Random(1)
    B = CURVE.mul(rng.randrange(1, ORDER), GEN)
    table, _ = W.masked_window_points(CURVE, B, WBITS, rng)
    out = {}
    print(f"one windowed point addition, secp256k1, n = {N}, w = {WBITS}:")
    only = sys.argv[1:]
    for name, cfg in rows().items():
        if only and not any(o in name for o in only):
            continue
        t = time.time()
        with H.tracing():
            m = H.HierMachine("and", "padd256")
            a, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
            W.windowed_point_add_cfg(m, a, x, y, table, P, cfg)
            c = H.count(m)
        out[name] = {"toffoli": c["toffoli_paper"], "qubits": c["qubits"]}
        print(f"  {name:<46} {c['qubits']:>5} qubits  {c['toffoli_paper']:>10,} Toffolis  "
              f"({time.time() - t:.0f} s)", flush=True)
    print("  published (+16 window qubits here): [1128] space-optimised 1,192 + 16 "
          "qubits / 2^21.19 = 2.39M (secp256k1); IonQ ~1,457 qubits / 1.196M + 3 lookups")
    path = ROOT / "bench" / "ec_space_256.json"
    if only and path.exists():          # a partial run updates its rows only
        out = {**json.loads(path.read_text()), **out}
        out = {k: out[k] for k in rows() if k in out}
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
