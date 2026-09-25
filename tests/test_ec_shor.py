"""Shor's ECDLP end to end.

Two oracles, two jobs:
  arith  the real arithmetic, verified on every basis state (which is the whole
         truth for a permutation, but cannot be run in superposition at width)
  table  the same circuit shape with a permutation oracle, run through Aer with
         genuine superposition, real QFT and real measurement statistics

Every Aer run, full-register and semiclassical alike, is also held to the exact
output distribution (shor_stats.ecdlp_probs): the measured histogram must sit
within shot noise of it, not merely put the right k on top.
"""
from _ec_util import ok, section

import ec_classical as C
import ec_shor as S
import shor_stats as ST
from ec_sim import run


def against_theory(counts, r, k, mb, shots):
    """(TVD from the exact distribution, shot-noise bound, single-shot success
    measured, single-shot success exact)."""
    q = 1 << mb
    exact = ST.ecdlp_probs(r, k, mb)
    emp = ST.empirical(S.pair_counts(counts), (q, q))
    hit = ST.ecdlp_success_mask(r, k, mb)
    return (ST.tvd(emp, exact), ST.tvd_null(exact, shots),
            float(emp[hit].sum()), float(exact[hit].sum()))


def main():
    section("arithmetic oracle: every basis state gives S + [u]P + [v]Q")
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    m, info = S.ecdlp_circuit(curve, P, Q, order, oracle="arith")
    kr, lr, px, py = info["regs"]
    mb, S0 = info["m_bits"], info["offset"]
    good = exc = 0
    for u in range(1 << mb):
        for v in range(1 << mb):
            acc, bad = S0, False
            rungs = ([curve.mul(1 << i, P) for i in range(mb) if (u >> i) & 1] +
                     [curve.mul(1 << i, Q) for i in range(mb) if (v >> i) & 1])
            for R in rungs:
                if C.point_add_exceptional(curve, acc, R):
                    bad = True
                    break
                acc = curve.add(acc, R)
            if bad:
                exc += 1
                continue
            rd = run(m, {kr: u, lr: v})
            assert (rd(px), rd(py)) == (acc.x, acc.y), (u, v)
            assert rd(kr) == u and rd(lr) == v
            good += 1
    ok(f"{good} basis states exact on {info['qubits']} qubits "
       f"({len(m.qc.data)} gates), {exc} hit an exceptional addition")

    section("table oracle: full algorithm, real superposition, real measurement")
    for label, cu, G in (("y^2=x^3+5x+4 mod 7", C.CLASSIQ, C.CLASSIQ_G),
                         ("y^2=x^3+x+6 mod 11", C.TOY11, C.TOY11_G)):
        r = cu.point_order(G)
        worst, fit, got, want = 1.0, 0.0, [], []
        for k in range(1, r):
            Qt = cu.mul(k, G)
            counts, cands, inf = S.run_ecdlp(cu, G, Qt, r, shots=4096)
            top = cands[0][0]
            total = sum(w for _, w in cands)
            share = dict(cands)[k] / total
            assert top == k, (label, k, cands[:4])
            assert C.verify_dlog(cu, G, Qt, top)
            worst = min(worst, share)
            d, bound, s_got, s_want = against_theory(counts, r, k, inf["m_bits"], 4096)
            assert d <= bound, (label, k, d, bound)
            fit = max(fit, d / bound)
            got.append(s_got)
            want.append(s_want)
        ok(f"{label}: recovered k for all {r-1} logarithms on {inf['qubits']} "
           f"qubits; correct k is the TOP candidate every time, holding at "
           f"least {worst*100:.0f}% of the vote (uniform would be {100/r:.0f}%)")
        ok(f"{label}: histogram within shot noise of the exact distribution for "
           f"every k (worst TVD at {fit*100:.0f}% of the bound); one shot alone "
           f"gives k with P = {sum(want)/len(want):.3f} exact, "
           f"{sum(got)/len(got):.3f} measured")

    section("semiclassical variant: one control qubit, recycled")
    for label, cu, G in (("y^2=x^3+5x+4 mod 7", C.CLASSIQ, C.CLASSIQ_G),
                         ("y^2=x^3+x+6 mod 11", C.TOY11, C.TOY11_G)):
        r = cu.point_order(G)
        fit, got, want = 0.0, [], []
        for k in range(1, r):
            Qt = cu.mul(k, G)
            counts, cands, inf = S.run_ecdlp(cu, G, Qt, r, shots=2048,
                                             one_control=True)
            assert cands[0][0] == k, (label, k, cands[:4])
            d, bound, s_got, s_want = against_theory(counts, r, k, inf["m_bits"], 2048)
            assert d <= bound, (label, k, d, bound)
            fit = max(fit, d / bound)
            got.append(s_got)
            want.append(s_want)
        full = S.ecdlp_circuit(cu, G, cu.mul(1, G), r, oracle="table")[1]["qubits"]
        ok(f"{label}: all {r-1} logarithms recovered on {inf['qubits']} qubits "
           f"instead of {full} -- the two control registers cost one qubit "
           f"between them, which is why both papers report only the arithmetic")
        ok(f"{label}: and it samples the SAME distribution as the full circuit -- "
           f"within shot noise of the exact one for every k (worst TVD at "
           f"{fit*100:.0f}% of the bound); one shot gives k with "
           f"P = {sum(want)/len(want):.3f} exact, {sum(got)/len(got):.3f} measured")

    section("the signal is real, not an artefact of the post-processing")
    import random
    rnd = random.Random(1)
    tally = {}
    for _ in range(4096):
        pr = (rnd.randrange(8), rnd.randrange(8))
        tally[pr] = tally.get(pr, 0) + 1
    cands = C.ecdlp_postprocess(tally, 5, 3)
    share = cands[0][1] / sum(w for _, w in cands)
    assert share < 0.25, share
    ok(f"flat measurements give a top share of {share*100:.1f}% against a "
       f"uniform 20% -- no peak, so the peaks above come from the circuit")
    exact = ST.ecdlp_probs(5, 4, 3)
    d, bound = ST.tvd(ST.empirical(tally, (8, 8)), exact), ST.tvd_null(exact, 4096)
    assert d > 3 * bound, (d, bound)
    ok(f"and the distribution test sees it: flat noise sits at TVD {d:.3f} from "
       f"the exact distribution, {d/bound:.0f}x the shot-noise bound of {bound:.3f}")


if __name__ == "__main__":
    main()
    print("\ntest_ec_shor: all passed")
