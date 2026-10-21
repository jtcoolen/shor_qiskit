"""Same-task feasibility check for a bounded-step affine point adder.

This is a classical, noiseless analysis, not an Aer run. It traces both
divisions of every controlled addition in the toy ECDLP oracle, x1-x2 for
lambda and x2-x3 for its uncomputation, and models a clean complete-oracle
failure contract: a failed addition is skipped and its failure flag is retained
and measured with the output. A circuit implementing that contract must
reproduce this distribution, so a TVD above the sampling bound rules out a
same-task E2E pass for that step limit.
"""

import json
import pathlib

import numpy as np

import ec_classical as C
import ec_divsteps_reference as DR
import ec_shor as S
import shor_stats as ST


OUT = pathlib.Path(__file__).with_suffix(".json")
SHOTS = 4096


def analyse(curve, base, target, order, mb, limit):
    p = curve.p
    steps = {w: DR.walk(w, p)[1] for w in range(1, p)}
    offset = S._pick_offset(curve, base, target, order, mb, strict=True)
    rungs = S._rungs(curve, base, target, mb)
    adds = failures = 0
    div1, div2, outcomes = set(), set(), {}
    for u in range(1 << mb):
        for v in range(1 << mb):
            acc, flags = offset, []
            for bits, points in zip((u, v), rungs):
                for i, point in enumerate(points):
                    failed = False
                    if (bits >> i) & 1:
                        new = curve.add(acc, point)
                        first, second = (acc.x - point.x) % p, (point.x - new.x) % p
                        div1.add(first)
                        div2.add(second)
                        failed = max(steps[first], steps[second]) > limit
                        adds += 1
                        failures += failed
                        if not failed:
                            acc = new
                    flags.append(failed)
            outcomes.setdefault(((acc.x, acc.y), tuple(flags)), []).append((u, v))
    q = 1 << mb
    probs = np.zeros((q, q))
    for members in outcomes.values():
        amplitudes = np.zeros((q, q))
        for u, v in members:
            amplitudes[u, v] = 1
        probs += np.abs(np.fft.fft2(amplitudes)) ** 2
    probs /= probs.sum()
    k = next(i for i in range(order) if curve.mul(i, base) == target)
    exact = ST.ecdlp_probs(order, k, mb)
    return {
        "p": p, "order": order, "k": k, "m_bits": mb, "step_limit": limit,
        "exact_steps": max(steps.values()),
        "div1_steps": max(steps[w] for w in div1),
        "div2_steps": max(steps[w] for w in div2),
        "failed_additions": failures, "additions": adds,
        "tvd_vs_exact": round(ST.tvd(probs, exact), 6),
        "tvd_limit": round(ST.tvd_null(exact, SHOTS), 6),
    }


def measure():
    p5 = C.Curve(5, 0, 1, "p5")
    base5 = next(P for P in p5.points()
                 if not P.inf and p5.point_order(P) == 3)
    p7, base7, target7 = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    rows = []
    for curve, base, target, order in ((p5, base5, p5.mul(2, base5), 3),
                                       (p7, base7, target7, 5)):
        exact = max(DR.walk(w, curve.p)[1] for w in range(1, curve.p))
        for limit in (exact - 1, exact):
            row = analyse(curve, base, target, order, 2, limit)
            rows.append(row)
            print(f"p={row['p']} limit={limit}/{exact}: "
                  f"{row['failed_additions']}/{row['additions']} failed adds, "
                  f"TVD {row['tvd_vs_exact']:.3f} vs {row['tvd_limit']:.3f}",
                  flush=True)
    return {
        "scope": "classical same-task model of a skip-on-failure affine adder "
                 "with retained per-addition failure flags; not an Aer run",
        "shots_for_bound": SHOTS, "rows": rows,
    }


if __name__ == "__main__":
    result = measure()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")
