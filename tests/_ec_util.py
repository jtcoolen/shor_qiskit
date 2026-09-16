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
