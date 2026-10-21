"""rns / rns_classical: Gidney 2025's approximate residue-number exponentiation.

  * the symbolic tallies reproduce [G25] Table 5 (Toffolis per factoring within
    10%, the qubit formula exactly);
  * the approximation stays inside [G25]'s deviation bound for every exponent;
  * the circuit's output equals the bit-exact model on every exponent, with
    every ancilla clean, at two toy moduli;
  * one prime's iteration costs the same whatever the prime (so a shot is |P|
    copies of it -- what bench/rsa_g25_2048.py builds at n = 2048).
"""
import random

from _ec_util import FULL, ok, section

import ec_cost as CO
import rns
import rns_classical as RC
from ec_sim import SimError, run

TOY = [(1003, 2, RC.Params(n=10, s=4, ell=7, w1=2, w3=2, w4=3, f=7, m=6)),
       (3127, 3, RC.Params(n=12, s=4, ell=7, w1=3, w3=2, w4=2, f=8, m=6))]


def test_tallies():
    section("[G25] Table 5 from the symbolic tallies")
    for n in sorted(RC.TABLE5):
        par = RC.params_table5(n)
        t = RC.tallies(par)
        pub = RC.TABLE5[n]
        tf = RC.per_factoring(par)
        assert abs(tf / pub[9] - 1) < 0.10, (n, tf, pub[9])
        assert t["qubits"] == par.m + 3 * par.f + 2 * par.ell + par.len_m
        print(f"      n={n}: {tf:.2e} Toffolis per factoring (paper {pub[9]:.1e}), "
              f"{t['qubits']} qubits (paper {pub[10]})")
    ok("within 10% of the paper's Toffolis; qubits = m + 3f + 2l + len m (the "
       "paper's own formula; its Table 5 column is ~2% lower)")


def test_classical():
    section("residue systems and the approximation")
    rng = random.Random(3)
    for N, g, par in TOY + [(58621, 2, RC.Params(n=16, s=4, ell=8, w1=2, w3=3, w4=3, f=12, m=8))]:
        conf = RC.make_config(N, g, par, rng)
        assert conf.L >= N ** par.W1 and conf.L % N < max(1, N >> par.f)
        worst = 0.0
        for e in range(1 << par.m):
            for S, Sc, V in RC.residues(conf, e):
                pass
            worst = max(worst, RC.deviation(conf, e))
        assert worst <= RC.deviation_bound(conf), (N, worst)
        print(f"      N={N}: |P|={len(conf.primes)}, worst deviation {worst:.4f} "
              f"(bound {RC.deviation_bound(conf):.3f})")
    for v in range(300):
        for mod in (12, 30, 66):
            reg, r = RC.loop2_model(v, mod, 10, mod.bit_length())
            assert r == v % mod
    ok("L > N^W1 with L mod N small; every exponent within [G25]'s bound; loop2 "
       "is long division")


def test_circuit():
    section("the circuit = the model, every exponent, ancillas clean")
    rng = random.Random(4)
    for N, g, par in TOY:
        conf = RC.make_config(N, g, par, rng)
        m, e, acc = rns.build(conf)
        for ev in range(1 << par.m):
            mask = rng.randrange(conf.trunc)
            try:
                got = run(m, {e: ev, acc: mask})(acc)
            except SimError as ex:
                raise AssertionError((N, ev, str(ex)))
            assert got == RC.reference(conf, ev, mask), (N, ev)
        c = CO.count(m)
        print(f"      N={N}: {len(conf.primes)} primes, {c['qubits']} qubits, "
              f"{c['toffoli_paper']} Toffolis ({c['toffoli_expected']:.0f} executed)")
    ok("every exponent, random masks: output register bit-exact")


def test_prime_independence():
    section("one prime's iteration costs the same for every prime")
    N, g, par = TOY[1]
    conf = RC.make_config(N, g, par, random.Random(5))
    costs = [CO.count(rns.one_prime(conf, i)) for i in range(1, len(conf.primes))]
    keys = ("toffoli_paper", "toffoli_expected", "qubits", "measure")
    assert all({k: c[k] for k in keys} == {k: costs[0][k] for k in keys} for c in costs)
    ok(f"{len(costs)} primes, identical counts: a shot is |P| copies")


def test_distribution():
    section("the output distribution of a shot, against the exact oracle")
    import numpy as np
    import ekera_hastad as EH
    import shor_stats as ST
    N, g = 241 * 251, 3
    inst = EH.EHInstance(N, g, "cfs", s=2)
    lx, ly = inst.lx, inst.ly
    bases = RC.eh_bases(N, g, inst.A, lx, ly)
    par = RC.Params(n=16, s=2, ell=8, w1=6, w3=2, w4=3, f=13, m=lx + ly)
    conf = RC.make_config(N, g, par, random.Random(2), bases=bases, mask_bits=7, pool_bits=6)
    assert all(N % p for p in conf.primes)
    r = ST.multiplicative_order(g, N)
    Pe = RC.eh_distribution(conf, lx, ly, approx=False)
    assert np.abs(Pe - ST.ecdlp_probs2(r, (-inst.e) % r, lx, ly)).max() < 1e-12
    Pa = RC.eh_distribution(conf, lx, ly, approx=True)
    top = np.argsort(Pe, axis=None)[::-1]
    top = top[:np.searchsorted(np.cumsum(Pe.flatten()[top]), 0.9) + 1]
    mass_e, mass_a = Pe.flatten()[top].sum(), Pa.flatten()[top].sum()
    print(f"      N = {N}, s = 2: {len(conf.primes)} primes (none dividing N), L mod N = "
          f"{conf.L % N}, f = {par.f}, masks < 2^{conf.mask_bits}: TVD {ST.tvd(Pa, Pe):.3f}, "
          f"mass on the exact oracle's peaks {mass_a:.3f} (exact {mass_e:.3f})")
    assert mass_a > 0.95 * mass_e
    ok("the exact oracle's distribution is the closed form; the approximate, "
       "masked shot puts the same mass on the same peaks")


def main():
    test_tallies()
    test_classical()
    test_circuit()
    test_prime_independence()
    test_distribution()


if __name__ == "__main__":
    main()
    print("\ntest_rns: all passed")
