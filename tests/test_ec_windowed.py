"""The windowed ECDLP circuit end to end (E1, E3, E4, E17).

Distributions are computed *exactly* with sparse_sim (every measurement branch
enumerated) and held to the closed form -- equal registers, or unequal once
windows are dropped.  The arithmetic oracle is checked on every basis state.
"""
import math
import random

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from _ec_util import FULL, ok, section

import ec_classical as C
import ec_cost as CO
import ec_shor as S
import ec_window as W
import shor_stats as ST
import sparse_sim as SS
from ec_sim import run
from semiclassical import semiclassical_iqft, semiclassical_iqft_windowed
from shor_essentials import qft


def pairs_dist(dist, bk, bl):
    P = np.zeros((1 << bk, 1 << bl))
    for key, p in dist.items():
        P[key & ((1 << bk) - 1), (key >> bk) & ((1 << bl) - 1)] += p
    return P


def test_windowed_qft():
    section("semiclassical_iqft_windowed = the full inverse QFT")
    rnd = random.Random(1)
    worst = 0.0
    for w in (1, 2, 3):
        for nw in (1, 2, 3):
            t = w * nw
            if t > 7:
                continue
            for _ in range(3):
                th = [rnd.uniform(0, 2 * math.pi) for _ in range(t)]
                c, tg, out = QuantumRegister(t, "c"), QuantumRegister(1, "t"), ClassicalRegister(t)
                full = QuantumCircuit(c, tg, out)
                full.x(tg[0])
                full.h(c)
                for k in range(t):
                    full.cp(th[k], c[k], tg[0])
                full.append(qft(t).inverse(), list(c))
                full.measure(c, out)
                r, out2 = QuantumRegister(w, "r"), ClassicalRegister(t)
                sc = QuantumCircuit(r, tg, out2)
                sc.x(tg[0])
                wins = [(lambda J: lambda qc, a: [qc.cp(th[J * w + b], a[b], tg[0])
                                                  for b in range(w)])(J) for J in range(nw)]
                semiclassical_iqft_windowed(sc, list(r), out2, wins)
                a = SS.simulate_sparse(full).distribution()
                b = SS.simulate_sparse(sc).distribution()
                worst = max(worst, max(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)))
    assert worst < 1e-12, worst
    ok(f"w = 1..3, up to 3 windows, arbitrary phases: max |dP| = {worst:.1e}")


CURVES = [("CLASSIQ", C.CLASSIQ, C.CLASSIQ_G, 3), ("TOY11", C.TOY11, C.TOY11_G, 5)]


def test_distributions():
    section("windowed ECDLP: exact output distribution")
    for name, curve, P, k in CURVES:
        order = curve.point_order(P)
        Q = curve.mul(k, P)
        for masks in (False, True):
            for one in (False, True):
                qc, info = S.ecdlp_windowed(curve, P, Q, order, 2, masks=masks,
                                            one_control=one)
                bk, bl = info["bits_k"], info["bits_l"]
                got = pairs_dist(SS.simulate_sparse(qc).distribution(), bk, bl)
                want = ST.ecdlp_probs(order, k, bk)
                d = float(np.abs(got - want).max())
                assert d < 1e-10, (name, masks, one, d)
                print(f"      {name} r={order} w=2 masks={masks!s:<5} "
                      f"{'1 x w ctrl' if one else 'full regs':<10} {info['qubits']:>3} "
                      f"qubits: max |dP| = {d:.1e}")
    ok("full registers and w recycled qubits, with and without IonQ masks: "
       "the closed-form distribution exactly")


def test_dropped():
    section("dropped windows: unequal registers, recovered by search")
    name, curve, P, k = CURVES[1]
    order = curve.point_order(P)
    Q = curve.mul(k, P)
    qc, info = S.ecdlp_windowed(curve, P, Q, order, 2, drop=1, one_control=True)
    bk, bl = info["bits_k"], info["bits_l"]
    got = pairs_dist(SS.simulate_sparse(qc).distribution(), bk, bl)
    want = ST.ecdlp_probs2(order, k, bk, bl)
    assert float(np.abs(got - want).max()) < 1e-10
    ok(f"{name}: {bk} + {bl} bits (one window of Q dropped), {info['qubits']} "
       f"qubits: distribution = ecdlp_probs2 exactly")

    def success(P2):
        s = 0.0
        for j1 in range(1 << bk):
            for j2 in range(1 << bl):
                if P2[j1, j2] < 1e-15:
                    continue
                c = C.ecdlp_postprocess_short({(j1, j2): 1}, order, bk, bl, curve, P, Q)
                if c and c[0][0] == k:
                    s += P2[j1, j2]
        return s
    real = success(want)
    flat = success(np.full_like(want, 1 / want.size))
    print(f"      single-shot success: {100*real:.1f}% on the real distribution, "
          f"{100*flat:.1f}% on flat noise")
    assert real > flat + 0.15, (real, flat)
    ok("the search recovers k far more often than noise does")


def test_multikey():
    section("multi-key (Litinski): one P-half, then one Q-half per key")
    name, curve, P, _ = CURVES[1]
    order = curve.point_order(P)
    keys = [5, 11, 2]
    Qs = [curve.mul(kk, P) for kk in keys]
    qc, info = S.ecdlp_multikey_1c(curve, P, Qs, order, 2)
    b = info["bits"]
    dist = SS.simulate_sparse(qc).distribution()
    for j, kk in enumerate(keys):
        P2 = np.zeros((1 << b, 1 << b))
        for key, p in dist.items():
            j1 = key & ((1 << b) - 1)
            j2 = (key >> (b * (1 + j))) & ((1 << b) - 1)
            P2[j1, j2] += p
        d = float(np.abs(P2 - ST.ecdlp_probs(order, kk, b)).max())
        assert d < 1e-10, (j, d)
    ok(f"{len(keys)} keys on {info['qubits']} qubits: every (j1, j2_j) marginal "
       f"is the single-key distribution -- the P-half is shared")


def expected_acc(curve, S0, tables, addrs, masks):
    acc = S0
    for T, a in zip(tables, addrs):
        R = T[a]
        if R.inf:
            if not masks and acc.x == 0:
                return None
            continue
        if C.point_add_exceptional(curve, acc, R):
            return None
        acc = curve.add(acc, R)
        if acc.inf:
            return None
    return acc


def test_arith():
    section("arithmetic oracle, windowed: basis states on a p = 61 curve")
    from _ec_util import random_curve, random_generator
    rnd = random.Random(61)
    curve, pts = random_curve(rnd, pmax=61, pmin=61)
    P = max(pts, key=curve.point_order)
    order = curve.point_order(P)
    k = rnd.randrange(2, order)
    Q = curve.mul(k, P)
    w = 2
    samples = 60 if not FULL else 200
    for masks, first in ((False, False), (True, False), (True, True)):
        m, info = S.ecdlp_windowed(curve, P, Q, order, w, masks=masks,
                                   oracle="arith", first_lookup=first, seed=3)
        kr, lr, px, py = info["regs"]
        tP, tQ = info["tables"]
        S0 = info["offset"]
        good = exc = 0
        for _ in range(samples):
            u, v = rnd.randrange(1 << info["bits_k"]), rnd.randrange(1 << info["bits_l"])
            addrs = [(u >> (J * w)) & 3 for J in range(len(tP))] + \
                    [(v >> (J * w)) & 3 for J in range(len(tQ))]
            tabs = tP + tQ
            if first:
                start = curve.add(S0, tP[0][addrs[0]])
                acc = expected_acc(curve, start, tabs[1:], addrs[1:], masks)
            else:
                acc = expected_acc(curve, S0, tabs, addrs, masks)
            if acc is None:
                exc += 1
                continue
            rd = run(m, {kr: u, lr: v})
            assert (rd(px), rd(py)) == (acc.x, acc.y), (masks, first, u, v)
            good += 1
        c = CO.count(m)
        print(f"      {curve.name} r={order} masks={masks!s:<5} first_lookup={first!s:<5} "
              f"{info['additions']} additions, {c['qubits']} qubits, "
              f"{c['toffoli_paper']} Toffoli-eq: {good} exact, {exc} exceptional")
        assert good >= samples // 2, (masks, first, good)
    ok("S + [u]P + [v]Q (+ the mask shift) on every sampled non-exceptional input")


def main():
    test_windowed_qft()
    test_distributions()
    test_dropped()
    test_multikey()
    test_arith()


if __name__ == "__main__":
    main()
    print("\ntest_ec_windowed: all passed")
