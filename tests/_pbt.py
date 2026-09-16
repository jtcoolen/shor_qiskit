"""A small property-based testing harness with shrinking.

`hypothesis` is not a dependency of this package, and adding one for the test
suite alone seemed a poor trade -- so this is the part of it that matters:
generate cases from a seeded RNG, run a property, and on failure shrink the case
toward something small enough to read.

A property returns None to pass, SKIP to reject the case (the curve was
singular, the addition was exceptional), or raises to fail.  Rejections are
counted, so a generator that quietly rejects almost everything shows up as a low
"kept" number rather than as a green test that checked nothing.

Two things the shrinker has to get right, and a naive one does not:

* Shrinking must stay inside the generator's domain.  Walking a prime modulus
  down through composite values makes the circuit raise for reasons that have
  nothing to do with the bug, and the shrinker happily reports one of those as
  the minimal case.  So each shrinkable field declares its *domain*, not just a
  lower bound.
* The shrunk case must be re-checked.  If it does not reproduce the failure, the
  original is reported instead.  A confidently wrong minimal case is worse than
  no minimal case.
"""
import random

SKIP = object()


class Failed(AssertionError):
    pass


def _fails(prop, case):
    try:
        res = prop(**case)
    except Exception:
        return True
    return res is not SKIP and res is not None


def _domain(spec, case, key):
    """The candidate values for a field, smallest first."""
    if isinstance(spec, (list, tuple, range)):
        return sorted(set(spec))
    lo = spec
    cur = case[key]
    return list(range(lo, cur)) if isinstance(cur, int) else []


def shrink(case, shrink_keys, prop, budget=400):
    """Greedy coordinate-wise shrink, staying in each field's declared domain."""
    best, spent = dict(case), 0
    for key, spec in shrink_keys:
        if key not in best:
            continue
        for cand in _domain(spec, best, key):
            if spent >= budget:
                break
            if cand == best[key]:
                continue
            spent += 1
            trial = dict(best, **{key: cand})
            if _fails(prop, trial):
                best = trial
                break                     # smallest first: take it and move on
    return best if _fails(prop, best) else dict(case)


def check(name, gen, prop, n=50, seed=0xEC, shrink_keys=(), verbose=True):
    """Run `prop` on `n` cases from `gen(rnd)`.  Returns (kept, skipped).

    gen         : rnd -> dict of keyword arguments
    prop        : **case -> None (pass) | SKIP (reject) | raises (fail)
    shrink_keys : [(field, domain)] where domain is a lower bound or an
                  explicit list of legal values
    """
    rnd = random.Random(seed)
    kept = skipped = 0
    for i in range(n):
        case = gen(rnd)
        try:
            res = prop(**case)
        except Exception as exc:
            small = shrink(case, shrink_keys, prop)
            note = "" if small != case else "   (could not shrink further)"
            raise Failed(
                f"PROPERTY FAILED: {name}\n"
                f"  case      : {case}\n"
                f"  shrunk    : {small}{note}\n"
                f"  seed      : {seed}, iteration {i}\n"
                f"  exception : {type(exc).__name__}: {exc}\n"
                f"  reproduce : prop(**{small!r})"
            ) from exc
        if res is SKIP:
            skipped += 1
        else:
            kept += 1
    if verbose:
        pct = 100 * kept / max(1, kept + skipped)
        print(f"  ok  {name}: {kept} cases checked, {skipped} rejected "
              f"({pct:.0f}% kept)", flush=True)
    assert kept > 0, f"{name}: every case was rejected -- the generator is broken"
    return kept, skipped
