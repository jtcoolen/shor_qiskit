"""Output distributions at toy size, against the answer we know.

Each circuit here is a permutation on basis states, so its whole output
distribution follows from what it computes on every basis input (the Fourier
transform of each level set; `shor_stats.ecdlp_probs_oracle`,
`rns_classical.eh_distribution`).  Knowing the discrete logarithm (or the
factors), that distribution is compared with the ideal one:

  ecdlp_signed   the signed-window ECDLP circuit (arithmetic oracle, run on
                 every (u, v)) on a curve of odd prime order: total variation
                 to the ideal, and the one-run success probability against the
                 ideal and IonQ's bound (sqrt(P0) - 2 p_f)^2
  rsa            Gidney 2025's approximate, masked exponentiation (the model
                 the circuit equals bit for bit) inside Ekera-Hastad, N = 241 x
                 251, s = 2: total variation to the exact oracle and the mass it
                 puts on the exact oracle's peaks

    ./venv/bin/python bench/toy_distributions.py   # writes bench/toy_distributions.json
"""
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np

import ec_shor as S
import ec_window as W
import ekera_hastad as EH
import rns_classical as RC
import shor_stats as ST
from ec_sim import SimError, run


def ecdlp_signed():
    from test_ec_windowed import _prime_odd_curve
    curve, P, order = _prime_odd_curve(pmin=37, pmax=61)   # large enough that most
                                                           # additions are not exceptional
    k = 7 % order
    Q = curve.mul(k, P)
    m, info = S.ecdlp_windowed(curve, P, Q, order, 2, oracle="arith", signed=True,
                               first_lookup=True, seed=3, cfg=W.PointAddCfg(offsets=True))
    kr, lr, px, py = info["regs"]
    bk, bl = info["bits_k"], info["bits_l"]
    f, bad = [], 0
    for u in range(1 << bk):
        row = []
        for v in range(1 << bl):
            ideal = curve.add(curve.add(curve.add(info["offset"], curve.mul(u, P)),
                                        curve.mul(v, Q)), info["shift"])
            try:
                rd = run(m, {kr: u, lr: v})
                got = (rd(px), rd(py))
            except SimError:
                got = ("dirty", u, v)
            bad += ideal.inf or got != (ideal.x, ideal.y)
            row.append(got)
        f.append(row)
    pf = bad / (1 << (bk + bl))
    Pi, Pr = ST.ecdlp_probs2(order, k, bk, bl), ST.ecdlp_probs_oracle(f, bk, bl)
    P0 = ST.success_probability(Pi, order, k, bk, bl, curve, P, Q)
    Pt = ST.success_probability(Pr, order, k, bk, bl, curve, P, Q)
    return {"curve": curve.name, "order": order, "k": k, "wrong_fraction": pf,
            "tvd": ST.tvd(Pr, Pi), "success": Pt, "ideal_success": P0,
            "ionq_bound": max(0.0, math.sqrt(P0) - 2 * pf) ** 2}


def rsa():
    N, g = 241 * 251, 3
    inst = EH.EHInstance(N, g, "cfs", s=2)
    lx, ly = inst.lx, inst.ly
    bases = RC.eh_bases(N, g, inst.A, lx, ly)
    par = RC.Params(n=16, s=2, ell=8, w1=6, w3=2, w4=3, f=13, m=lx + ly)
    conf = RC.make_config(N, g, par, random.Random(2), bases=bases, mask_bits=7, pool_bits=6)
    r = ST.multiplicative_order(g, N)
    Pe = RC.eh_distribution(conf, lx, ly, approx=False)
    closed = float(np.abs(Pe - ST.ecdlp_probs2(r, (-inst.e) % r, lx, ly)).max())
    out = {"N": N, "s": 2, "primes": len(conf.primes), "L_mod_N": conf.L % N, "f": par.f,
           "deviation_bound": RC.deviation_bound(conf),
           "worst_deviation": max(RC.deviation(conf, e) for e in range(1 << (lx + ly))),
           "exact_vs_closed_form": closed, "masks": []}
    top = np.argsort(Pe, axis=None)[::-1]
    top = top[:np.searchsorted(np.cumsum(Pe.flatten()[top]), 0.9) + 1]
    for mb in (5, 7, 9):
        Pa = RC.eh_distribution(conf, lx, ly, approx=True, mask_bits=mb)
        out["masks"].append({"mask_bits": mb, "window": (1 << mb) << conf.t, "period": r,
                             "tvd": ST.tvd(Pa, Pe), "peak_mass": float(Pa.flatten()[top].sum()),
                             "exact_peak_mass": float(Pe.flatten()[top].sum())})
    return out


def main():
    path = ROOT / "bench" / "toy_distributions.json"
    if sys.argv[1:] == ["ecdlp"] and path.exists():          # recompute one part only
        out = json.loads(path.read_text())
    else:
        out = {"rsa": rsa()}
        print(json.dumps(out["rsa"], indent=1))
    out["ecdlp_signed"] = ecdlp_signed()
    print(json.dumps(out["ecdlp_signed"], indent=1))
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
