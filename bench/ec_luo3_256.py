"""[Luo26] Sec 5-6 at n = 256: the point addition on three field registers.

Same builders the tests verify (`ec_luo3`), built at n = 256 on secp256k1
through `hier.tracing()`, with the lean pseudo-Mersenne cells for the
multiplications and the constant steps:

  blocks    the in-place division and multiplication (two EEAs and three
            multiplications each, y vented once), and [Luo26] Fig. 14's
            controlled addition of a classical point
  windowed  one signed windowed addition (w = 16), the looked-up point loaded
            one coordinate at a time into the one free register
  full      the whole ECDLP-256 circuit on it: 28 additions after a first
            lookup.  Its depth is not scheduled (2.4G Toffolis): the additions
            run one after another on the same registers, so 28 x one
            addition's exact depth bounds it.
  gidney    the same blocks on `ec_cqadd.GidneyArith` -- [Gid25b]'s constant-
            workspace classical-quantum adder, exact for any odd p -- and its
            cells alone (the 3n and 4n adders, a modular addition, a doubling)

    ./venv/bin/python bench/ec_luo3_256.py        # writes bench/ec_luo3_256.json
"""
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "bench"))

import ec_cost as CO
import ec_cqadd as CQ
import ec_gcd as G
import ec_luo3 as L3
import ec_shor as S
import ec_signedwin as SW
import hier as H
from ec_hier_256 import CURVE, GEN, MSBS, N, ORDER, P, WBITS


def main():
    t0 = time.time()
    ar = G.PMSpace(P, msbs=MSBS, lean=True)
    be = L3.Luo3(arith=ar)
    out = {"n": N, "w": WBITS, "formula_3reg": L3.point_add_qubits(N),
           "published": {"Luo26": {"qubits_formula": L3.point_add_qubits(N),
                                   "inversion_toffoli": 15.35e6,
                                   "point_addition_toffoli": 70.29e6,
                                   "ecdlp_toffoli_log2": 30.88},
                         "ECDSA.Fail": {"qubits": [825, 851]}}}
    rng = random.Random(5)
    T = CURVE.mul(rng.randrange(1, ORDER), GEN)

    def build(label, fn, depth=True):
        t = time.time()
        with H.tracing():
            m = H.HierMachine("and", "luo3")
            fn(m)
            c = H.count(m)
            d = H.exact_depth(m) if depth else None
        out[label] = {"qubits": c["qubits"], "toffoli": c["toffoli_paper"],
                      "measure": c["measure"], "toffoli_depth": d}
        print(f"  {label:<28} {c['qubits']:>5} qubits {c['toffoli_paper']:>13,} Toffolis "
              f"{c['measure']:>6} measurements" + (f", depth {d:,}" if d else "") +
              f" ({time.time() - t:.0f} s)", flush=True)
        return m

    def blk(name):
        def fn(m):
            x, y = m.alloc(N, "x"), m.alloc(N, "y")
            getattr(be, name)(m, x, y, P)
        return fn
    build("division", blk("div"), depth=False)
    build("multiplication", blk("mul"), depth=False)

    def padd(m):
        c, x, y = m.alloc(1, "c"), m.alloc(N, "x"), m.alloc(N, "y")
        L3.point_add_ctrl_luo(m, c[0], x, y, T.x, T.y, P, be)
    build("controlled addition", padd, depth=False)

    cfg = L3.windowed_cfg(P, ar)
    B = CURVE.mul(random.Random(1).randrange(1, ORDER), GEN)
    stab, _ = SW.signed_window_points(CURVE, B, WBITS, order=ORDER)

    def win(m):
        a, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
        SW.windowed_point_add_signed(m, a, x, y, stab, P, cfg, neg=L3.neg_lean)
    build("signed windowed addition", win)

    # --- the same on Gidney's classical-quantum adder (any odd p)
    from ec_sim import Machine
    cells = {}
    for label, fn in (
            ("classical-quantum adder, n - 1 dirty", lambda m, x, y: CQ.cq_add_dirty(m, x, P, y[:N - 1])),
            ("classical-quantum adder, 3 clean", lambda m, x, y: CQ.cq_add(m, x, P)),
            ("modular addition b += c a", lambda m, x, y: CQ.GidneyArith(P).cadd(m, y[0], x, y[1:])),
            ("modular doubling, dirty-assisted",
             lambda m, x, y: CQ.GidneyArith(P).dbl(m, x, dirty=list(y)))):
        m = Machine("and")
        x, y = m.alloc(N, "x"), m.alloc(N + 1, "y")
        fn(m, x, y)
        c = CO.count(m)
        cells[label] = {"toffoli": c["toffoli_paper"], "per_n": round(c["toffoli_paper"] / N, 3),
                        "clean": c["qubits"] - 2 * N - 1, "measure": c["measure"]}
        print(f"  {label:<40} {c['toffoli_paper']:>6} Toffolis ({c['toffoli_paper'] / N:.2f} n), "
              f"{c['qubits'] - 2 * N - 1} clean", flush=True)
    out["gidney_cells"] = cells
    beg = L3.Luo3(arith=CQ.GidneyArith(P))
    out_g = {}

    def blk_g(name):
        def fn(m):
            x, y = m.alloc(N, "x"), m.alloc(N, "y")
            getattr(beg, name)(m, x, y, P)
        return fn
    saved = dict(out)
    build("division", blk_g("div"), depth=False)
    out_g["division"] = out["division"]

    def padd_g(m):
        c, x, y = m.alloc(1, "c"), m.alloc(N, "x"), m.alloc(N, "y")
        L3.point_add_ctrl_luo(m, c[0], x, y, T.x, T.y, P, beg)
    build("controlled addition", padd_g, depth=False)
    out_g["controlled addition"] = out["controlled addition"]
    cfg_g = L3.windowed_cfg(P, CQ.GidneyArith(P))

    def win_g(m):
        a, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
        SW.windowed_point_add_signed(m, a, x, y, stab, P, cfg_g, neg=L3.neg_lean)
    build("signed windowed addition", win_g, depth=False)
    out_g["signed windowed addition"] = out["signed windowed addition"]
    for k in ("division", "controlled addition", "signed windowed addition"):
        out[k] = saved[k]
    out["gidney"] = out_g

    t = time.time()
    r3 = random.Random(3)
    Q = CURVE.mul(r3.randrange(1, ORDER), GEN)
    S0 = CURVE.mul(r3.randrange(1, ORDER), GEN)
    with H.tracing():
        m, info = S.ecdlp_windowed(CURVE, GEN, Q, ORDER, WBITS, m_bits=N, offset=S0, drop=3,
                                   oracle="arith", cfg=cfg, first_lookup=True, seed=7,
                                   signed=True, sign_neg=L3.neg_lean)
        c = H.count(m)
    semi = c["qubits"] - info["bits_k"] - info["bits_l"] + WBITS
    out["full_algorithm"] = {"toffoli": c["toffoli_paper"], "expected": round(c["toffoli_expected"]),
                             "additions": info["additions"], "qubits_semiclassical": semi,
                             "toffoli_depth_bound": info["additions"] *
                             out["signed windowed addition"]["toffoli_depth"]}
    print(f"  whole ECDLP-256 ({info['additions']} additions + first lookup): "
          f"{c['toffoli_paper']:,} Toffolis on {semi} qubits ({time.time() - t:.0f} s)")
    (ROOT / "bench" / "ec_luo3_256.json").write_text(json.dumps(out, indent=2))
    print(f"wrote bench/ec_luo3_256.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
