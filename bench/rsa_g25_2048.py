"""Gidney 2025's RSA-2048 shot, built at size.

`rns.py` is checked exhaustively against `rns_classical.reference` at toy N.
Here the same builders run with [G25] Table 5's parameters.  A shot is |P|
iterations of one prime's loops (loop1 .. unloop2) plus a final loop1 and one
phaseup per exponent window; every iteration is the same circuit up to table
contents, and no gate count depends on the contents (lookups by unary
iteration / ANF, adders, fixed-width comparisons) -- which is checked by
building the iteration for three different primes.  So

    Toffolis per shot = |P| x (one iteration) + (the ends),

with |P| from a real random residue system of l-bit primes with L >= N^W1.

    ./venv/bin/python bench/rsa_g25_2048.py [n ...]      # default: 1024 1536 2048
Writes bench/rsa_g25.json.
"""
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))

import ec_cost as CO
import hier as H
import ec_mbu as MB
import rns
import rns_classical as RC
from ec_sim import Machine


def ends(conf):
    """The final loop1 (dlog back to 0) and the merged loop1 phaseups."""
    m = Machine("and", "ends")
    e = m.alloc(conf.par.m, "e")
    dlog = m.alloc(conf.D, "dlog")
    rns.loop1(m, conf, e, dlog, len(conf.primes))
    for j in range(conf.par.W1):
        MB.phase_fix(m, ("rns-l1", j))
    return CO.count(m)


def one_size(n, seed=1):
    t0 = time.time()
    par = RC.params_table5(n)
    rng = random.Random(seed)
    N = rng.getrandbits(n) | (1 << (n - 1)) | 1
    g = 3
    P, L, pool = RC.residue_system_size(N, par, rng)
    M = RC.multipliers(N, g, par)
    built = [p for p in pool if all(x % p for row in M for x in row)][:3]
    conf = RC.make_config(N, g, par, rng, primes=built, L=L)
    iters = []
    for i in (1, 2):
        mi = rns.one_prime(conf, i)
        c = CO.count(mi)
        c["toffoli_depth"] = H.exact_depth(mi)
        iters.append({k: c[k] for k in ("toffoli_paper", "toffoli_expected", "qubits", "measure",
                                        "toffoli_depth")})
    assert iters[0] == iters[1], iters
    it = iters[0]
    e = ends(conf)
    shot = P * it["toffoli_paper"] + e["toffoli_paper"]
    shot_exp = P * it["toffoli_expected"] + e["toffoli_expected"]
    s_, ell, w1, w3, w4, f, m, pdev, shots, pub_t, pub_q = RC.TABLE5[n]
    tal = RC.tallies(par, P)
    out = {"n": n, "params": {"s": s_, "ell": ell, "w1": w1, "w3": w3, "w4": w4, "f": f, "m": m},
           "primes": P, "iteration": it, "ends": e["toffoli_paper"],
           "per_shot": shot, "per_shot_expected": shot_exp,
           "depth_per_shot": P * it["toffoli_depth"],
           "shots": shots, "per_factoring_expected": shot_exp * shots,
           "qubits": it["qubits"], "tally_per_shot": tal["per_shot"],
           "tally_qubits": tal["qubits"],
           "published": {"toffoli_per_factoring": pub_t, "qubits": pub_q},
           "seconds": round(time.time() - t0, 1)}
    print(f"n={n}: |P| = {P}, one iteration {it['toffoli_paper']:,} Toffolis "
          f"({it['toffoli_expected']:,.0f} expected), {it['qubits']} qubits; per shot "
          f"{shot_exp:.3e} expected -> per factoring {shot_exp * shots:.2e} "
          f"(paper {pub_t:.1e}, tally {tal['per_shot'] * shots:.2e}); qubits "
          f"{it['qubits']} (paper {pub_q}) [{out['seconds']} s]", flush=True)
    return out


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [1024, 1536, 2048]
    path = ROOT / "bench" / "rsa_g25.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for n in sizes:
        data[str(n)] = one_size(n)
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
