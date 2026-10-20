"""Shared helpers for the ECDLP test suite."""
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "tests"))

FULL = os.environ.get("SHOR_EC_FULL", "") not in ("", "0")


def primes(upto):
    return [p for p in range(3, upto + 1)
            if all(p % d for d in range(2, int(p**0.5) + 1))]


def scope(small, big):
    """Pick the test scope: `big` only under SHOR_EC_FULL=1."""
    return big if FULL else small


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def section(msg):
    print(f"\n=== {msg} ===", flush=True)


def rng(seed=0xEC):
    return random.Random(seed)


def random_curve(rnd, pmax=97, pmin=5):
    """A random non-singular short Weierstrass curve with at least one point.

    Rejection sampling on (p, a, b): the discriminant condition rules out a
    little under 1/p of the (a, b) pairs, so this terminates immediately.
    """
    from ec_classical import Curve
    ps = [p for p in primes(pmax) if p >= pmin]
    while True:
        p = rnd.choice(ps)
        a, b = rnd.randrange(p), rnd.randrange(p)
        if (4 * a**3 + 27 * b**2) % p == 0:
            continue
        cu = Curve(p, a, b, f"rand-p{p}-a{a}-b{b}")
        pts = [P for P in cu.points() if not P.inf]
        if pts:
            return cu, pts


def random_generator(rnd, curve, pts, min_order=3):
    """A point of order at least `min_order`, or None."""
    cands = [P for P in pts if curve.point_order(P) >= min_order]
    return rnd.choice(cands) if cands else None


def aer_e2e(label, shots=4000, seed=5, **kw):
    """Run the arithmetic ECDLP circuit on Aer (GPU if available) and check it.

    Toy curve, full QFT and measurement, genuine superposition.  Asserts the
    discrete log is recovered and the histogram sits within shot noise of the
    exact Shor distribution.  `kw` is passed to `ec_shor.ecdlp_circuit`.
    """
    import ec_classical as C
    import ec_shor as S
    import shor_stats as ST
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    k = next(k for k in range(order) if curve.mul(k, P) == Q)
    counts, cands, info = S.run_ecdlp_arith(
        curve, P, Q, order, shots=shots, seed=seed, **kw)
    mb = info["m_bits"]
    q = 1 << mb
    exact = ST.ecdlp_probs(order, k, mb)
    emp = ST.empirical(S.pair_counts(counts), (q, q))
    tvd, null = ST.tvd(emp, exact), ST.tvd_null(exact, shots)
    assert cands and cands[0][0] == k, (label, cands[:3], k)
    assert tvd <= null, (label, tvd, null)
    ok(f"{label}: {info['qubits']}-qubit arithmetic circuit on {info['aer']}, "
       f"k = {k} recovered, TVD {tvd:.3f} <= shot noise {null:.3f}")
    return info
