"""Closing the Toffoli gap to IonQ at n = 256, and where the Toffolis go.

Same builders the tests verify, built at n = 256 on secp256k1 with 16-bit windows
through `hier.tracing()`.  Three things are measured:

  cells       the replay's signed modular addition and halving, alone
  rows        one windowed point addition per configuration: worst-case
              Toffolis, expected-executed Toffolis (a measurement-based repair
              counted with the probability it fires -- IonQ's convention), qubits
  components  the same additions split by component (walk, replay, lookups...)
  full        the whole 28-addition ECDLP-256 circuit on IonQ's cells

    ./venv/bin/python bench/ec_toffoli_256.py          # writes bench/ec_toffoli_256.json
"""
import collections
import functools
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "bench"))

import ec_approx as AX
import ec_cost as CO
import ec_gcd as G
import ec_mbu as MB
import ec_modarith as MA
import ec_shor as S
import ec_square as SQ
import ec_window as W
import hier as H
from ec_hier_256 import CURVE, GEN, CMP, MSBS, N, ORDER, P, WBITS
from ec_sim import Machine

ZERO_STEPS = 37                      # IonQ Table X: "rounds with x + y = p"


def cells():
    out = {}
    for lab, fn in (
            ("signed add, Alg 11 + 0<->q swaps (before)",
             lambda m, e, x, y: AX.csignadd_pm(m, e, x, y, P, None, MSBS)),
            ("signed add, careful (first 37 steps)",
             lambda m, e, x, y: AX.csignadd_pm_phase(m, e, x, y, P, None, None, True, MSBS)),
            ("signed add, phase-approximate (IonQ Alg 2)",
             lambda m, e, x, y: AX.csignadd_pm_phase(m, e, x, y, P))):
        m = Machine("and")
        e, x, y = m.alloc(1, "e"), m.alloc(N, "x"), m.alloc(N, "y")
        fn(m, e[0], x, y)
        c = CO.count(m)
        out[lab] = {"toffoli": c["toffoli_paper"], "expected": c["toffoli_expected"]}
    for lab, lsbs in (("halving, 74-bit correction (before)", None),
                      ("halving, kappa = 65", AX.phase_kappa(P))):
        m = Machine("and")
        x = m.alloc(N, "x")
        AX.modhalf_pm(m, x, P, lsbs)
        c = CO.count(m)
        out[lab] = {"toffoli": c["toffoli_paper"], "expected": c["toffoli_expected"]}
    return out


def rows():
    pm, ph = G.PM(P, msbs=MSBS), G.PMPhase(P, msbs=MSBS)
    sq = lambda m, c, s, a, p: SQ.csub_square_pm(m, c, s, a, p, msbs=MSBS)
    base = dict(lookup="mbu", merge_xy=True, offsets=True, free_xy1=True, square=sq)
    neg = lambda m, c, x, p: AX.cmodneg_approx(m, c, x, p)

    def ci(arith, **kw):
        return G.CondInv(arith=arith, cmp_msbs=CMP, c_pad=2.3, replay="ci", **kw)
    return {
        "IonQ-style, Alg 11 replay cells (before)": W.PointAddCfg(**base, mul=ci(pm)),
        "+ phase-approximate adder in the replay, 37 careful steps":
            W.PointAddCfg(**base, mul=ci(ph, zero_steps=ZERO_STEPS)),
        "+ the same adder in the point addition, approximate negation":
            W.PointAddCfg(**base, add=ph, neg=neg, mul=ci(ph, zero_steps=ZERO_STEPS)),
        "+ Fig. 1 packing, replay in x's qubits":
            W.PointAddCfg(**base, add=ph, neg=neg,
                          mul=ci(ph, zero_steps=ZERO_STEPS, compress="fig1", reuse_x=True)),
        "+ signed windows (2^15-entry tables), no Fig. 1":
            ("signed", W.PointAddCfg(**base, add=ph, neg=neg,
                                     mul=ci(ph, zero_steps=ZERO_STEPS))),
    }


def table():
    rng = random.Random(1)
    B = CURVE.mul(rng.randrange(1, ORDER), GEN)
    return W.masked_window_points(CURVE, B, WBITS, rng)[0]


# --- components: wrap builders for the duration of one build -------------------
def _toffolis(ops):
    t = collections.Counter()
    for ci_ in ops:
        op = ci_.operation
        if isinstance(op, H.NodeGate):
            t.update(H.node_tally(op))
        else:
            t[op.name] += 1
            c = getattr(op, "ec_cost", None)
            if c:
                t["mbu"] += c.get("toffoli", 0)
    return t["ecand"] + t["ccx"] + t["mcx"] + t["cswap"] + t["mbu"] + sum(
        v for k, v in t.items() if k.startswith("cost:") and k.endswith("|toffoli"))


def components(cfg, tab):
    tot, stack, saved = collections.Counter(), [], []

    def wrap(owner, name, label):
        orig = getattr(owner, name)

        @functools.wraps(orig)
        def w(*a, **k):
            m = next((v for v in list(a) + list(k.values())
                      if hasattr(v, "qc") and hasattr(v, "anc")), None)
            if m is None or label in stack:
                return orig(*a, **k)
            mark = len(m.qc.data)
            stack.append(label)
            try:
                return orig(*a, **k)
            finally:
                stack.pop()
                tot[label] += _toffolis(m.qc.data[mark:])
        saved.append((owner, name, orig))
        setattr(owner, name, w)

    targets = [(mod, [x for x in names if x != "windowed_point_add_cfg"])
               for mod, names in H.DEFAULT_TARGETS]
    with H.tracing(targets):
        wrap(MB, "lookup", "table lookups and repair")
        wrap(MB, "phase_fix", "table lookups and repair")
        be = type(cfg.mul)
        wrap(be, "record", "GCD walk and its undoing")
        wrap(be, "unrecord", "GCD walk and its undoing")
        wrap(be, "_replay_div", "Bezout replay")
        wrap(be, "_replay_mul", "Bezout replay")
        wrap(type(cfg.mul.arith), "signadd", "  of which signed additions")
        import types
        import dataclasses
        holder = types.SimpleNamespace(square=cfg.square)
        wrap(holder, "square", "square-subtract")
        cfg = dataclasses.replace(cfg, square=holder.square)
        try:
            m = H.HierMachine("and", "padd256")
            a, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
            W.windowed_point_add_cfg(m, a, x, y, tab, P, cfg)
            total = H.count(m)["toffoli_paper"]
        finally:
            for owner, name, orig in reversed(saved):
                setattr(owner, name, orig)
    out = dict(tot)
    out["everything else"] = total - sum(v for k, v in out.items() if not k.startswith("  "))
    out["total"] = total
    return out


def main():
    t0 = time.time()
    out = {"n": N, "w": WBITS, "msbs": MSBS, "zero_steps": ZERO_STEPS,
           "kappa": AX.phase_kappa(P), "delta": 32,
           "published": {"IonQ": {"toffoli": 1392608, "qubits": 1462,
                                  "toffoli_without_lookups": 1196000}}}
    out["cells"] = cells()
    for k, v in out["cells"].items():
        print(f"  {k:<48} {v['toffoli']:>5} worst / {v['expected']:>6.1f} expected")

    tab = table()
    out["rows"] = {}
    import ec_signedwin as SW
    rng = random.Random(1)
    B = CURVE.mul(rng.randrange(1, ORDER), GEN)
    stab, _ = SW.signed_window_points(CURVE, B, WBITS, order=ORDER)
    for name, cfg in rows().items():
        t = time.time()
        with H.tracing():
            m = H.HierMachine("and", "padd256")
            a, x, y = m.alloc(WBITS, "a"), m.alloc(N, "x"), m.alloc(N, "y")
            if isinstance(cfg, tuple):                    # a signed window
                SW.windowed_point_add_signed(m, a, x, y, stab, P, cfg[1])
            else:
                W.windowed_point_add_cfg(m, a, x, y, tab, P, cfg)
            c = H.count(m)
            c["toffoli_depth"] = H.exact_depth(m)
        out["rows"][name] = {"toffoli": c["toffoli_paper"],
                             "expected": round(c["toffoli_expected"]), "qubits": c["qubits"],
                             "toffoli_depth": c["toffoli_depth"]}
        print(f"  {name:<62} {c['toffoli_paper']:>10,} worst {c['toffoli_expected']:>12,.0f} "
              f"expected {c['qubits']:>5} qubits ({time.time() - t:.0f} s)", flush=True)

    names = list(rows())
    out["components"] = {}
    for name in (names[0], names[2]):
        out["components"][name] = components(rows()[name], tab)
        print(f"  components of {name}:")
        for k, v in out["components"][name].items():
            print(f"      {k:<34} {v:>10,}")

    name = names[2]
    rng = random.Random(3)
    k = rng.randrange(1, ORDER)
    Q = CURVE.mul(k, GEN)
    S0 = CURVE.mul(rng.randrange(1, ORDER), GEN)
    t = time.time()
    with H.tracing():
        m, info = S.ecdlp_windowed(CURVE, GEN, Q, ORDER, WBITS, m_bits=N, offset=S0,
                                   drop=3, masks=True, oracle="arith", cfg=rows()[name],
                                   first_lookup=True, seed=7)
        c = H.count(m)
        c["toffoli_depth"] = H.exact_depth(m)
    semi = c["qubits"] - info["bits_k"] - info["bits_l"] + WBITS
    out["full_algorithm"] = {"config": name, "toffoli": c["toffoli_paper"],
                             "expected": round(c["toffoli_expected"]),
                             "toffoli_depth": c["toffoli_depth"],
                             "additions": info["additions"], "qubits_semiclassical": semi}
    print(f"  whole ECDLP-256 ({info['additions']} additions + first lookup): "
          f"{c['toffoli_paper']:,} worst, {c['toffoli_expected']:,.0f} expected, "
          f"{semi} qubits ({time.time() - t:.0f} s)")
    print("  published: IonQ 28 x (1.196M + 3 x 2^16) = 39.0M, 1,462 qubits")
    (ROOT / "bench" / "ec_toffoli_256.json").write_text(json.dumps(out, indent=2))
    print(f"wrote bench/ec_toffoli_256.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
