"""Ekera-Hastad short-DL factoring (R1) at toy size.

  * the circuit's exact output (sparse_sim, every branch) is the closed form
    for z = x - e y, full registers and one recycled control qubit alike;
  * one shot factors N with high probability using only the short-logarithm
    relation -- no order, no "g^d == A" filter -- far above flat noise;
  * the Gidney-Ekera and Gidney 2025 choices of A fail at toy N, because
    their logarithm exceeds ord(g) there: only the centred D is short enough.
"""
import numpy as np

import ekera_hastad as EH
import shor_stats as ST
import sparse_sim as SS


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def order(g, N):
    return ST.multiplicative_order(g, N)


def exact(qc, inst):
    dist = SS.simulate_sparse(qc).distribution()
    P = np.zeros((1 << inst.lx, 1 << inst.ly))
    for key, p in dist.items():
        P[key & ((1 << inst.lx) - 1), key >> inst.lx] += p
    return P


def success(P, inst, p, q, shots=1):
    """Exact probability that `shots` independent runs, post-processed
    together, give the right factors."""
    outs = [((j, k), P[j, k]) for j in range(P.shape[0]) for k in range(P.shape[1])
            if P[j, k] > 1e-15]
    want = (min(p, q), max(p, q))
    good = 0.0
    if shots == 1:
        for jk, w in outs:
            d = EH.eh_postprocess({jk: 1}, inst)[0][0]
            good += w * (EH.factors_from_d(inst, d) == want)
        return good
    for jk1, w1 in outs:
        for jk2, w2 in outs:
            samp = {jk1: 1}
            samp[jk2] = samp.get(jk2, 0) + 1
            d = EH.eh_postprocess(samp, inst)[0][0]
            good += w1 * w2 * (EH.factors_from_d(inst, d) == want)
    return good


CASES = [(15, 2, 3, 5), (21, 2, 3, 7)]


def main():
    for N, g, p, q in CASES:
        inst = EH.EHInstance(N, g, "cfs")
        r = order(g, N)
        c = (-inst.e) % r
        for one in (False, True):
            qc = EH.eh_circuit_1c(inst) if one else EH.eh_circuit(inst)
            P = exact(qc, inst)
            want = ST.ecdlp_probs2(r, c, inst.lx, inst.ly)
            d = float(np.abs(P - want).max())
            assert d < 1e-10, (N, one, d)
        noise = np.full_like(want, 1 / want.size)
        s1, f1 = success(want, inst, p, q), success(noise, inst, p, q)
        s2, f2 = success(want, inst, p, q, 2), success(noise, inst, p, q, 2)
        print(f"      N={N} g={g} (ord {r}): D = {inst.true_d(p, q)}, exponent "
              f"{inst.lx}+{inst.ly} = {inst.lx + inst.ly} qubits (order finding: "
              f"{2 * inst.n}); success 1 run {100*s1:.1f}% (noise {100*f1:.1f}%), "
              f"2 runs {100*s2:.1f}% (noise {100*f2:.1f}%)")
        assert s1 > 2 * f1 and s2 > 2 * f2 and s2 >= s1 - 1e-12, (N, s1, f1, s2, f2)
    ok("circuit output = closed form (full and one-control); runs factor N from "
       "the short logarithm alone, far above noise, better with two runs")

    for N, g, p, q in CASES:
        rates = {}
        for var in ("cfs", "ge", "g25"):
            inst = EH.EHInstance(N, g, var)
            r = order(g, N)
            P = ST.ecdlp_probs2(r, (-inst.e) % r, inst.lx, inst.ly)
            rates[var] = success(P, inst, p, q)
        print(f"      N={N}: one-run success  cfs {100*rates['cfs']:.1f}%  "
              f"ge {100*rates['ge']:.1f}%  g25 {100*rates['g25']:.1f}%")
        inst = EH.EHInstance(N, g, "ge")
        if inst.true_d(p, q) > (1 << inst.m):
            assert rates["ge"] == 0.0, rates        # outside the short range
    ok("reported for all three choices of A; Gidney-Ekera's d = p + q lies "
       "outside the 2^m search range at toy N and never succeeds, the other two "
       "are pinned only modulo ord(g) and succeed as often as their ties allow")

    for N, g, p, q in CASES:
        inst = EH.EHInstance(N, g, "cfs")
        assert EH.factors_from_d(inst, inst.true_d(p, q)) == (p, q)
    ok("factors_from_d recovers (p, q) from the true D")


def test_eh():
    main()


if __name__ == "__main__":
    main()
    print("\ntest_eh: all passed")
