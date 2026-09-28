"""Point-addition costs at n = 256 (secp256k1), assembled from measured parts.

A full 256-bit point addition is millions of gates -- too many to build as one
Qiskit circuit here -- but every construction in ec_gcd is a fixed setup, a
walk of L rounds whose cost depends only on the working width, and a replay of
L identical steps.  So each part is *built at n = 256 and counted*:

    one walk round at two widths      -> its cost is affine in the width
    one replay step                   -> identical every round
    the whole multiplier at L = 3, 4  -> setup + teardown, by difference

and the parts are combined with the paper's round budget and width schedule.
Nothing is fitted: every coefficient is a gate count of a circuit this package
builds (and tests exactly at toy sizes).

Writes bench/ec_project_256.json.
"""
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))

import ec_cost as CO
import ec_gcd as G
import ec_mbu as MB
import ec_square as SQ
import ec_window as W
from ec_classical import eea_iterations
from ec_sim import Machine, Reg

P256K1 = 2**256 - 2**32 - 977
N = 256
MSBS = 48            # [1128]: 40-50 comparison bits at n = 256 (modular arithmetic)
CMP = 40 + math.ceil(2.3 * math.sqrt(256))   # GCD: 40 + the schedule's padding (IonQ Table X)


def toff(m):
    return CO.count(m)["toffoli_paper"]


def qubits(m):
    return m.qc.num_qubits


# --- walk rounds -----------------------------------------------------------------
def dialog_round(backend, w):
    m = Machine("and")
    U, V = m.alloc(w, "u"), m.alloc(w, "v")
    rec = (m.anc(1, "b0"), m.anc(1, "b1"))
    backend._round(m, U, V, rec)
    return toff(m)


def condinv_round(backend, w):
    m = Machine("and")
    U, V = m.alloc(w + 1, "u"), m.alloc(w + 1, "v")
    rec = (m.anc(1, "a"), m.anc(1, "m"))
    backend._round(m, U, V, rec, w)
    return toff(m)


def pingpong_round(W_):
    m = Machine("and")
    R0, R1 = m.alloc(W_, "r0"), m.alloc(W_, "r1")
    G.PingPong._walk_body(m, R0, R1, [m.anc(1, "e")])
    return toff(m)


def jump2_round(backend, w):
    m = Machine("and")
    U, V = m.alloc(w, "u"), m.alloc(w, "v")
    recs = [tuple(m.anc(1, nm) for nm in ("b", "s", "s2")) for _ in range(2)]
    t1 = m.anc(1, "t1")
    backend._walk_body(m, U, V, recs, t1)        # step 0 + one ordinary step
    m0 = Machine("and")
    U0, V0 = m0.alloc(w, "u"), m0.alloc(w, "v")
    recs0 = [tuple(m0.anc(1, nm) for nm in ("b", "s", "s2"))]
    backend._walk_body(m0, U0, V0, recs0, m0.anc(1, "t1"))
    return toff(m) - toff(m0)                    # one ordinary macro-step


# --- replay steps ------------------------------------------------------------------
def replay_step(kind, arith, q=P256K1, n=N):
    m = Machine("and")
    r, s = m.alloc(n, "r"), m.alloc(n, "s")
    b0, b1 = m.alloc(1, "b0"), m.alloc(1, "b1")
    if kind == "dialog":
        arith.dbl(m, s)
        arith.cadd(m, b0[0], r, s)
        for a, b in zip(r, s):
            m.ctx.cswap(b1[0], a, b)
    elif kind == "ci":
        arith.signadd(m, b0[0], r, s)
        arith.half(m, s)
        for a, b in zip(r, s):
            m.ctx.cswap(b1[0], a, b)
    elif kind == "pingpong":
        arith.signadd(m, b0[0], r, s)
        arith.half(m, s)
    elif kind == "jump2":
        import ec_modarith as MA
        arith.cadd(m, b0[0], r, s)
        for a, b in zip(r, s):
            m.ctx.cswap(b1[0], a, b)
        arith.dbl(m, s)
        MA.cmoddbl(m, b0[0], s, q)
    return toff(m)


# --- fixed overhead, by difference ------------------------------------------------------
def overhead(make, per_round, L1=3, L2=4):
    def total(L):
        m = Machine("and")
        x, y = m.alloc(N, "x"), m.alloc(N, "y")
        make(L).mul(m, x, y, P256K1)
        return toff(m), qubits(m)
    t1, q1 = total(L1)
    t2, q2 = total(L2)
    rnd = t2 - t1
    return t1 - L1 * rnd, rnd, q1, (q2 - q1)


def main():
    t0 = time.time()
    pm = G.PM(P256K1, msbs=MSBS)
    out = {"n": N, "q": "secp256k1", "msbs": MSBS, "variants": {}}
    it = eea_iterations(N)
    sched = G.Dialog(c_pad=2.3).widths(N)

    rows = []

    def add(name, walk_rounds, replay_cost, L, over, q_base, q_per, note):
        # multiplier: walk forward + walk backward + one replay pass
        mul = over + 2 * sum(walk_rounds) + L * replay_cost
        rec = {"in_place_mul": mul, "walk_round_avg": sum(walk_rounds) / L,
               "replay_step": replay_cost, "rounds": L,
               "qubits_mul": q_base + (L - 3) * q_per, "note": note}
        out["variants"][name] = rec
        rows.append((name, mul, rec["qubits_mul"]))
        print(f"  {name:<46} {mul:>9,} Toffoli-eq per in-place mul  "
              f"({L} rounds; walk {rec['walk_round_avg']:.0f}/round, replay {replay_cost}/step)",
              flush=True)

    print("in-place multiplication at n = 256 (div costs the same):")
    # 1. [1128] as the repo builds it today: exact arithmetic, full width
    d0 = G.Dialog()
    w0 = dialog_round(d0, N + 1)
    over, _, qb, qp = overhead(lambda L: G.Dialog(iters=L), None)
    add("[1128] Dialog, exact, full width (repo today)", [w0] * it,
        replay_step("dialog", G.Exact(P256K1)), it, over, qb, qp,
        "ec_eea.inplace_mul")

    # 2. [1128] with every refinement: fused + truncated compare, schedule, PM
    d1 = G.Dialog(fused_cmp=True, cmp_msbs=CMP, c_pad=2.3)
    a, b = dialog_round(d1, 200), dialog_round(d1, 100)
    walk = [b + (a - b) * (w - 100) // 100 for w in sched]
    over, _, qb, qp = overhead(lambda L: G.Dialog(fused_cmp=True, cmp_msbs=CMP,
                                                  arith=pm, iters=L), None)
    add("[1128] Dialog, fused cmp77, c_pad 2.3, PM", walk,
        replay_step("dialog", pm), it, over, qb, qp, "1128 Sec 3-4")

    # 3. IonQ conditionally-inverted walk, both replays
    c1 = G.CondInv(cmp_msbs=CMP, c_pad=2.3)
    a, b = condinv_round(c1, 200), condinv_round(c1, 100)
    walk = [b + (a - b) * (w - 100) // 100 for w in sched]
    for rk in ("standard", "ci"):
        over, _, qb, qp = overhead(lambda L, rk=rk: G.CondInv(arith=pm, cmp_msbs=CMP,
                                                               replay=rk, iters=L), None)
        add(f"IonQ CondInv, cmp77, c_pad 2.3, PM, replay={rk}", walk,
            replay_step("dialog" if rk == "standard" else "ci", pm), it, over,
            qb, qp, "IonQ Sec VI")

    # 4. ECDSA.Fail ping-pong, 704 rounds
    L = 704
    wr = pingpong_round(N + 2)
    over, _, qb, qp = overhead(lambda L_: G.PingPong(arith=pm, rounds=L_), None)
    add("ECDSA.Fail ping-pong, 704 rounds, PM", [wr] * L,
        replay_step("pingpong", pm), L, over, qb, qp, "ECDSA.Fail Alg 5")

    # 5. ECDSA.Fail Jump-2, 261 macro-steps
    L = 261
    j = G.Jump2()
    wr = jump2_round(j, N + 1)
    over, _, qb, qp = overhead(lambda L_: G.Jump2(arith=pm, steps=L_), None)
    add("ECDSA.Fail Jump-2, 261 steps, PM", [wr] * L,
        replay_step("jump2", pm), L, over, qb, qp, "ECDSA.Fail Alg 4")

    # --- the rest of a point addition ---------------------------------------------
    print("\nthe rest of a windowed point addition at n = 256, w = 16:")
    ms = Machine("and")
    s_, a_ = ms.alloc(N, "s"), ms.alloc(N, "a")
    W._csub_square(ms, None, s_, a_, P256K1)
    sq_general = toff(ms)
    ms = Machine("and")
    s_, a_ = ms.alloc(N, "s"), ms.alloc(N, "a")
    SQ.csub_square_pm(ms, None, s_, a_, P256K1, msbs=MSBS)
    sq_pm = toff(ms)
    w = 16
    L = 1 << w
    lookups_old = 10 * 2 * (L - 1)
    fix = MB.best_split(w, MB.phase_ancillas(w, w // 2, False), False)[1]
    lookups_new = 3 * MB.walk_ands(w, False) + fix
    out["square_sub"] = {"general": sq_general, "pm_fold": sq_pm}
    out["lookups_w16"] = {"ten_recomputed": lookups_old, "three_mbu": lookups_new}
    print(f"  square-subtract: general {sq_general:,}  ->  dedicated squarer + PM fold {sq_pm:,}")
    print(f"  lookups: 10 recomputed {lookups_old:,}  ->  3 MBU loads + 1 repair {lookups_new:,}")

    # a handful of modular additions/subtractions: steps 3, 4, 8, 12, 14, 15
    import ec_modarith as MA
    mm = Machine("and")
    x_, y_ = mm.alloc(N, "x"), mm.alloc(N, "y")
    MA.modadd(mm, x_, y_, P256K1)
    madd = toff(mm)
    other_old = 5 * madd
    out["modadd"] = madd

    V = out["variants"]
    today = 2 * V["[1128] Dialog, exact, full width (repo today)"]["in_place_mul"] \
        + sq_general + lookups_old + other_old
    best_name = min(V, key=lambda k: V[k]["in_place_mul"])
    best = 2 * V[best_name]["in_place_mul"] + sq_pm + lookups_new + other_old
    out["point_addition"] = {"today": today, "best": best, "best_gcd": best_name,
                             "published": {"1128": 2**21.19 + 3 * 2**16,
                                           "IonQ": 1.196e6 + 3 * 2**16}}
    print(f"\none windowed point addition, n = 256, w = 16:")
    print(f"  as built today           {today:>12,.0f} Toffoli-eq")
    print(f"  with every refinement    {best:>12,.0f}   (GCD: {best_name})")
    print(f"  published: [1128] {2**21.19 + 3*2**16:,.0f}, IonQ {1.196e6 + 3*2**16:,.0f}")
    adds_old, adds_new = 34, 28
    out["full_algorithm"] = {"today": adds_old * today, "best": adds_new * best,
                             "additions": [adds_old, adds_new]}
    print(f"whole algorithm: {adds_old} x today = {adds_old*today:,.0f};  "
          f"{adds_new} x best = {adds_new*best:,.0f}")
    (ROOT / "bench" / "ec_project_256.json").write_text(json.dumps(out, indent=2))
    print(f"\n({time.time() - t0:.0f} s)  wrote bench/ec_project_256.json")


if __name__ == "__main__":
    main()
