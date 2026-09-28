"""Build the windowed point addition -- and the whole ECDLP circuit -- at n = 256.

Not a projection: the same builders the tests verify at toy sizes, run on
secp256k1 with 16-bit windows, through `hier.tracing()` so that repeated
sub-circuits of one shape are built once and reused (counts are exact: the
hierarchical count equals the flat one, `tests/test_hier.py`).

    ./venv/bin/python bench/ec_hier_256.py            # one addition, several configs
    ./venv/bin/python bench/ec_hier_256.py --full     # also the 28-addition circuit

Writes bench/ec_hier_256.json.
"""
import json
import math
import pathlib
import random
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))

import ec_classical as C
import ec_gcd as G
import ec_shor as S
import ec_square as SQ
import ec_window as W
import hier as H

CURVE, GEN, ORDER = C.SECP256K1, C.SECP256K1_G, C.SECP256K1_N
P = CURVE.p
N, WBITS, MSBS = 256, 16, 48
# top-bit comparisons in the GCD see registers padded by the width schedule
# (2.3 sqrt(n) bits of leading zeros), so they need 40 significant bits *plus*
# the padding: IonQ Table X, "comparison bits in gcd: 40 + 2.3 sqrt(n)" = 77
CMP = 40 + math.ceil(2.3 * math.sqrt(N))


def configs():
    pm = G.PM(P, msbs=MSBS)
    sq = lambda m, c, s, a, p: SQ.csub_square_pm(m, c, s, a, p, msbs=MSBS)
    base = dict(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True)
    return {
        "[1128] dialog, exact arithmetic (MBU lookups, masks)":
            W.PointAddCfg(**base),
        "[1128] dialog, fused cmp77, schedule, PM":
            W.PointAddCfg(**base, mul=G.Dialog(arith=pm, fused_cmp=True, cmp_msbs=CMP, c_pad=2.3)),
        "+ dedicated squarer":
            W.PointAddCfg(**base, mul=G.Dialog(arith=pm, fused_cmp=True, cmp_msbs=CMP, c_pad=2.3),
                          square=sq),
        "IonQ: cond.-inverted walk + replay, squarer":
            W.PointAddCfg(**base, mul=G.CondInv(arith=pm, cmp_msbs=CMP, c_pad=2.3, replay="ci"),
                          square=sq),
        "ECDSA.Fail ping-pong (704 rounds), squarer":
            W.PointAddCfg(**base, mul=G.PingPong(arith=pm, rounds=704), square=sq),
        "space: dialog + register sharing + Fig. 1 packing, squarer":
            W.PointAddCfg(**base, mul=G.Dialog(arith=pm, fused_cmp=True, cmp_msbs=CMP,
                                               c_pad=2.3, share=True, compress="fig1"),
                          square=sq),
    }


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def one_addition(cfg, table):
    m = H.HierMachine("and", "padd256")
    addr, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
    W.windowed_point_add_cfg(m, addr, x, y, table, P, cfg)
    c = H.count(m)
    c["toffoli_depth"] = H.exact_depth(m)
    return c


def main():
    full = "--full" in sys.argv
    out = {"n": N, "w": WBITS, "msbs": MSBS, "curve": "secp256k1", "addition": {}}
    t0 = time.time()
    rng = random.Random(1)
    B = CURVE.mul(rng.randrange(1, ORDER), GEN)
    table, _ = W.masked_window_points(CURVE, B, WBITS, rng)
    print(f"table of 2^{WBITS} points: {time.time() - t0:.0f} s", flush=True)
    for name, cfg in configs().items():
        t = time.time()
        with H.tracing():
            c = one_addition(cfg, table)
        c.update(seconds=round(time.time() - t, 1))
        out["addition"][name] = c
        print(f"  {name:<52} {c['toffoli_paper']:>10,} Toffoli-eq  {c['qubits']:>5} qubits  "
              f"({c['seconds']:.0f} s, {c['cached_nodes']} nodes, "
              f"{c['top_level_ops']:,} top-level ops, peak RSS {rss_gb():.1f} GB)", flush=True)

    if full:
        name = "IonQ: cond.-inverted walk + replay, squarer"
        cfg = configs()[name]
        k = rng.randrange(1, ORDER)
        Q = CURVE.mul(k, GEN)
        S0 = CURVE.mul(rng.randrange(1, ORDER), GEN)
        t = time.time()
        with H.tracing():
            m, info = S.ecdlp_windowed(CURVE, GEN, Q, ORDER, WBITS, m_bits=N, offset=S0,
                                       drop=3, masks=True, oracle="arith", cfg=cfg,
                                       first_lookup=True, seed=7)
            c = H.count(m)
            c["toffoli_depth"] = H.exact_depth(m)
        semi = c["qubits"] - info["bits_k"] - info["bits_l"] + WBITS
        c.update(seconds=round(time.time() - t, 1), additions=info["additions"],
                 bits_k=info["bits_k"], bits_l=info["bits_l"], qubits_semiclassical=semi)
        try:
            from qualtran.resource_counting import QECGatesCost, get_cost_value
            gc = get_cost_value(H.to_qualtran(m, "ecdlp256"), QECGatesCost())
            c["qualtran"] = {"toffoli": int(gc.total_toffoli_only()),
                             "clifford": int(gc.clifford), "measurement": int(gc.measurement)}
            assert c["qualtran"]["toffoli"] == c["toffoli_paper"]
        except ImportError:
            pass
        out["full_algorithm"] = {"config": name, **c}
        print(f"\nwhole ECDLP-256 circuit ({info['additions']} windowed additions + first "
              f"lookup, {info['bits_k']}+{info['bits_l']} exponent bits): "
              f"{c['toffoli_paper']:,} Toffoli-eq; {c['qubits']} qubits with full exponent "
              f"registers, {semi} with {WBITS} recycled control qubits "
              f"[{c['seconds']:.0f} s, peak RSS {rss_gb():.1f} GB]")
        if "qualtran" in c:
            q_ = c["qualtran"]
            print(f"  Qualtran QECGatesCost on the exported tree: {q_['toffoli']:,} Toffolis, "
                  f"{q_['clifford']:,} Cliffords, {q_['measurement']:,} measurements")
        print("published: [1128] 2^26.1 = 7.3e7; Babbush <= 7e7 / 9e7; IonQ 3.9e7")

    (ROOT / "bench" / "ec_hier_256.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote bench/ec_hier_256.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
