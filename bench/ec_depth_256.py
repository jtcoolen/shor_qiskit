"""Depth against qubits and Toffolis at n = 256: the carry-lookahead configurations.

Every other configuration here ripples, so its Toffoli depth is ~90% of its
Toffoli count.  These rows put `ec_cla`'s log-depth adders into the replay
(`ec_depth.CLAArith`), the GCD walk (`Dialog(walk_cla=True)`) and the squarer
(`ec_depth.csub_square_cla`), and report what that buys and costs.  Built at
n = 256 through `hier.tracing()`; depth is `hier.exact_depth` (the flat circuit's Toffoli depth).

    ./venv/bin/python bench/ec_depth_256.py         # writes bench/ec_depth_256.json
"""
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "bench"))

import ec_depth as DP
import ec_gcd as G
import ec_square as SQ
import ec_window as W
import hier as H
from ec_hier_256 import CURVE, GEN, CMP, MSBS, N, ORDER, P, WBITS


def rows():
    pm = G.PM(P, msbs=MSBS)
    cla = DP.CLAArith(P)
    sq = lambda m, c, s, a, p: SQ.csub_square_pm(m, c, s, a, p, msbs=MSBS)
    sqc = lambda m, c, s, a, p: DP.csub_square_cla(m, c, s, a, p)
    base = dict(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True)

    def dlg(arith, **kw):
        return G.Dialog(arith=arith, fused_cmp=True, cmp_msbs=CMP, c_pad=2.3, **kw)
    return {
        "dialog, PM replay (ripple adders)": W.PointAddCfg(**base, square=sq, mul=dlg(pm)),
        "+ carry-lookahead replay": W.PointAddCfg(**base, square=sq, mul=dlg(cla)),
        "+ carry-lookahead walk": W.PointAddCfg(**base, square=sq, mul=dlg(cla, walk_cla=True)),
        "+ carry-lookahead squarer": W.PointAddCfg(**base, square=sqc,
                                                   mul=dlg(cla, walk_cla=True)),
        "+ register sharing, Fig. 1": W.PointAddCfg(
            **base, square=sqc, mul=dlg(cla, walk_cla=True, share=True, compress="fig1")),
    }


def main():
    rng = random.Random(1)
    B = CURVE.mul(rng.randrange(1, ORDER), GEN)
    table, _ = W.masked_window_points(CURVE, B, WBITS, rng)
    out = {"lookup_depth_floor": 3 * (1 << WBITS)}
    for name, cfg in rows().items():
        t = time.time()
        with H.tracing():
            m = H.HierMachine("and", "padd256")
            a, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
            W.windowed_point_add_cfg(m, a, x, y, table, P, cfg)
            c = H.count(m)
            c["toffoli_depth"] = H.exact_depth(m)
        out[name] = {"toffoli": c["toffoli_paper"], "expected": round(c["toffoli_expected"]),
                     "qubits": c["qubits"], "toffoli_depth": c["toffoli_depth"]}
        print(f"  {name:<40} {c['qubits']:>5} qubits {c['toffoli_paper']:>11,} Toffolis "
              f"depth {c['toffoli_depth']:>10,} ({time.time() - t:.0f} s)", flush=True)
    (ROOT / "bench" / "ec_depth_256.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
