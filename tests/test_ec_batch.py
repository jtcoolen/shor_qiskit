"""Batch inversion (Montgomery's trick, [Lit23] Fig. 9b): ec_batch.

  * batch_mod_inv: every nonzero input at p = 7, 11, 13 for k = 2 (sampled
    at 31), and k = 3 exhaustively at p = 7 (sampled at 11, 13, 31; under
    SHOR_EC_FULL=1 all but k = 3 at p = 31 are exhaustive); inputs preserved,
    every intermediate back to |0> (the simulator checks each free);
  * with controls, an idle x_i may hold anything -- zero included -- and its
    output stays 0; batch_mod_div likewise, and clear=True undoes it;
  * point_add_ctrl_batch agrees with k point_add_ctrl's on sampled pairs,
    including idle accumulators an ungated inversion would choke on;
  * cost: exactly 2 Inv + 5(k-1) Mul for inverses and 2 Inv + (7k-6) Mul +
    (5k-3) n for quotients; separate against batched at toy n and built at
    n = 16, 32 (flat) and 64, 256 (hier, which agrees with flat).
"""
import itertools
import random

from _ec_util import FULL, ok, section

import ec_batch as BA
import ec_classical as C
import ec_cost as CO
import ec_mult as MU
import ec_pointadd as PA
from ec_sim import Machine, run


def pack(bits):
    return sum(b << i for i, b in enumerate(bits))


def inv_machine(p, k, ctrl=False):
    n = p.bit_length()
    m = Machine("and")
    cs = m.alloc(k, "c") if ctrl else None
    xs = [m.alloc(n, f"x{i}") for i in range(k)]
    outs = [m.alloc(n, f"o{i}") for i in range(k)]
    BA.batch_mod_inv(m, xs, outs, p, list(cs) if ctrl else None)
    return m, cs, xs, outs


def test_batch_inv():
    section("batch_mod_inv: k inverses, one inversion")
    total = 0
    for p in (7, 11, 13, 31):
        for k in (2, 3):
            m, _, xs, outs = inv_machine(p, k)
            combos = list(itertools.product(range(1, p), repeat=k))
            exhaustive = (k == 2 and (p < 31 or FULL)) or p == 7 or (FULL and p <= 13)
            if not exhaustive:
                combos = random.Random(p * k).sample(combos, 250 if k == 2 else 100)
            for xv in combos:
                rd = run(m, dict(zip(xs, xv)))
                assert [rd(o) for o in outs] == [pow(v, -1, p) for v in xv], (p, xv)
                assert [rd(x) for x in xs] == list(xv)
            total += len(combos)
            print(f"      p={p:>2} k={k}: {len(combos):>4} inputs "
                  f"{'(all)' if exhaustive else '(sampled)'}")
    ok(f"{total} inputs: outs = x^-1, xs preserved, prefix products, "
       f"back-substitution and the Kaliski records all back to |0>")

    section("controlled: an idle factor is 1, whatever its register holds")
    for p in (7, 11):
        m, cs, xs, outs = inv_machine(p, 2, ctrl=True)
        cases = [(c, xv) for c in itertools.product((0, 1), repeat=2)
                 for xv in itertools.product(*[range(1, p) if ci else range(p) for ci in c])]
        if p > 7 and not FULL:
            cases = random.Random(p).sample(cases, 150)
        for c, xv in cases:
            rd = run(m, {cs: pack(c), xs[0]: xv[0], xs[1]: xv[1]})
            want = [pow(v, -1, p) if ci else 0 for v, ci in zip(xv, c)]
            assert [rd(o) for o in outs] == want, (p, c, xv)
        print(f"      p={p}: {len(cases)} (ctrl, x) inputs, idle x = 0 included")
    ok("outs = ctrl * x^-1; a zero on an idle element no longer poisons the batch")


def test_batch_div():
    section("batch_mod_div: quotients, and clear=True returns them to |0>")
    for p, k, samples in ((7, 2, 200), (13, 2, 200), (11, 3, 100)):
        n = p.bit_length()
        rnd = random.Random(p)
        for clear in (False, True):
            m = Machine("and")
            cs = m.alloc(k, "c")
            xs = [m.alloc(n, f"x{i}") for i in range(k)]
            ys = [m.alloc(n, f"y{i}") for i in range(k)]
            outs = [m.alloc(n, f"o{i}") for i in range(k)]
            BA.batch_mod_div(m, list(cs), xs, ys, outs, p, clear=clear)
            for _ in range(samples):
                c = [rnd.randrange(2) for _ in range(k)]
                xv = [rnd.randrange(1, p) if ci else rnd.randrange(p) for ci in c]
                yv = [rnd.randrange(p) for _ in range(k)]
                want = [y * pow(x, -1, p) % p if ci else 0 for x, y, ci in zip(xv, yv, c)]
                inp = {cs: pack(c), **dict(zip(xs, xv)), **dict(zip(ys, yv))}
                if clear:
                    inp.update(zip(outs, want))
                rd = run(m, inp)
                assert [rd(o) for o in outs] == ([0] * k if clear else want), (p, c, xv, yv)
                assert [rd(x) for x in xs] == xv and [rd(y) for y in ys] == yv
        print(f"      p={p:>2} k={k}: {samples} samples each way")

    p, n, k = 11, 4, 2                                  # no controls at all
    for clear in (False, True):
        m = Machine("and")
        xs = [m.alloc(n, f"x{i}") for i in range(k)]
        ys = [m.alloc(n, f"y{i}") for i in range(k)]
        outs = [m.alloc(n, f"o{i}") for i in range(k)]
        BA.batch_mod_div(m, None, xs, ys, outs, p, clear=clear)
        rnd = random.Random(clear)
        for xv in itertools.product(range(1, p), repeat=k):
            yv = [rnd.randrange(p) for _ in range(k)]
            want = [y * pow(x, -1, p) % p for x, y in zip(xv, yv)]
            inp = {**dict(zip(xs, xv)), **dict(zip(ys, yv))}
            if clear:
                inp.update(zip(outs, want))
            rd = run(m, inp)
            assert [rd(o) for o in outs] == ([0] * k if clear else want), (xv, yv)
    print(f"      p={p} k={k}, ctrls=None: every nonzero x, each way")
    ok("outs = ctrl * y / x, and its exact inverse; inputs preserved, ancillas |0>")


def _padd_cases(curve, adds, q, rnd, pts, forced=None):
    """Accumulators for k additions: non-exceptional where active; idle ones
    random, or `forced` (points an ungated inversion would fail on)."""
    acc = []
    for i, (P2, qi) in enumerate(zip(adds, q)):
        if not qi and forced and forced[i]:
            acc.append(rnd.choice(forced[i]))
            continue
        while True:
            R = rnd.choice(pts)
            if not qi or not C.point_add_exceptional(curve, R, P2):
                break
        acc.append(R)
    return acc


def test_point_add_batch():
    section("point_add_ctrl_batch: k additions, two inversions in all")
    rnd = random.Random(11)
    for curve in (C.TOY11, C.Curve(13, 0, 2, "p13-ord19")):
        p, n = curve.p, curve.p.bit_length()
        pts = [P for P in curve.points() if not P.inf]
        for k, tuples, per in ((1, 6, 8), (2, 6, 10), (3, 3, 8)):
            good = idle_hard = 0
            for _ in range(tuples):
                adds = [rnd.choice(pts) for _ in range(k)]
                # idle accumulators where step 3 or step 8 would divide by 0
                forced = [[R for R in pts if R.x == P2.x or (R.x + 2 * P2.x) % p == 0]
                          for P2 in adds]
                m = Machine("and")
                qs = m.alloc(k, "q")
                X = [m.alloc(n, f"x{i}") for i in range(k)]
                Y = [m.alloc(n, f"y{i}") for i in range(k)]
                BA.point_add_ctrl_batch(m, list(qs), X, Y, [(P.x, P.y) for P in adds], p)
                ref = None
                if k == 1:
                    ref = Machine("and")
                    rq, rX, rY = ref.alloc(1, "q"), ref.alloc(n, "x"), ref.alloc(n, "y")
                    PA.point_add_ctrl(ref, rq[0], rX, rY, adds[0].x, adds[0].y, p)
                for t in range(per):
                    q = [rnd.randrange(2) for _ in range(k)]
                    acc = _padd_cases(curve, adds, q, rnd, pts, forced if t % 2 else None)
                    idle_hard += sum(1 for i in range(k) if not q[i] and acc[i] in forced[i])
                    inp = {qs: pack(q)}
                    for i in range(k):
                        inp[X[i]], inp[Y[i]] = acc[i].x, acc[i].y
                    rd = run(m, inp)
                    for i in range(k):
                        want = curve.add(acc[i], adds[i]) if q[i] else acc[i]
                        assert (rd(X[i]), rd(Y[i])) == (want.x, want.y), (k, q, acc, adds)
                    if ref is not None:
                        rr = run(ref, {rq: q[0], rX: acc[0].x, rY: acc[0].y})
                        assert (rr(rX), rr(rY)) == (rd(X[0]), rd(Y[0]))
                    good += 1
            c = CO.count(m)
            print(f"      {curve.name} k={k}: {good} controlled k-tuples exact "
                  f"({idle_hard} idle accumulators with x1 = x2 or x1 = -2 x2); "
                  f"{c['toffoli_paper']} Toffoli-eq, {c['qubits']} qubits")
    ok("each accumulator gets + P2 when its q is set and is untouched otherwise; "
       "k = 1 agrees with point_add_ctrl")


def _count(fn, *regs_n, **kw):
    m = Machine("and")
    regs = [m.alloc(w, f"r{i}") for i, w in enumerate(regs_n)]
    fn(m, *regs, **kw)
    return CO.count(m)["toffoli_paper"]


def test_costs():
    section("cost identities: 2 Inv + 5(k-1) Mul, and 2 Inv + (7k-6) Mul + (5k-3) n")
    for p in (11, 31, 251):
        n = p.bit_length()
        mul = _count(lambda m, a, b, z: MU.modmul(m, a, b, z, p), n, n, n)
        imul = _count(lambda m, a, b, z: m.emit_inverse(MU.modmul, m, a, b, z, p), n, n, n)
        assert mul == imul
        inv1 = BA.batch_costs(n, 1, "inv", p)["batched"]["toffoli"]
        for k in (2, 3, 4):
            bi = BA.batch_costs(n, k, "inv", p)["batched"]["toffoli"]
            bd = BA.batch_costs(n, k, "div", p)["batched"]["toffoli"]
            assert bi == inv1 + 5 * (k - 1) * mul, (p, k, bi, inv1, mul)
            assert bd == inv1 + (7 * k - 6) * mul + (5 * k - 3) * n, (p, k, bd)
        print(f"      n={n}: clean inversion {inv1} = {inv1 / mul:.2f} Mul "
              f"(batching pays when it exceeds 5 Mul)")
    ok("measured counts are the formulas exactly: nothing else rides along")

    section("separate against batched (Toffoli-eq, qubits)")
    rows = []

    def add(n, kind, k, how, p=None, hier=False):
        c = BA.batch_costs(n, k, kind, p, separate=how, hier=hier)
        s, b = c["separate"], c["batched"]
        assert k == 1 or b["toffoli"] < s["toffoli"], (n, kind, k)
        rows.append([n, kind, k, s["toffoli"], b["toffoli"],
                     f"{b['toffoli'] / s['toffoli']:.3f}", s["qubits"], b["qubits"]])
        return c

    for n, p in ((4, 11), (5, 31)):                     # toy
        for kind in ("inv", "div", "padd"):
            for k in (1, 2, 3):
                c = add(n, kind, k, "build" if n == 4 else "scale", p)
                if n == 4 and k > 1:                    # scaling = building all k
                    assert c == BA.batch_costs(n, k, kind, p, separate="scale"), (n, kind, k)
    flat = [(16, "inv", 2), (16, "div", 2), (16, "padd", 2), (32, "inv", 2)]
    for n, kind, k in flat:                             # built flat
        c = add(n, kind, k, "scale")
        if k == 2:                                      # hier agrees with flat
            assert c == BA.batch_costs(n, k, kind, separate="scale", hier=True)
    for n, p, ks in ((16, None, (4,)), (32, None, (2, 4, 8)), (64, None, (1, 2, 4, 8)),
                     (256, 2**256 - 2**32 - 977, (1, 4))):   # through hier
        for kind in ("inv", "div", "padd"):
            for k in ks:
                if (n, kind, k) not in flat:
                    add(n, kind, k, "scale", p, hier=True)
    rows.sort(key=lambda r: (r[0], ("inv", "div", "padd").index(r[1]), r[2]))
    print(CO.table(rows, ["n", "op", "k", "separate", "batched", "ratio",
                          "q sep", "q batch"]))
    ok("batched is cheaper from k = 2 at every width, and costs qubits: "
       "2(k-1) held registers (a_i, b_i) plus the k live lambdas")


def main():
    test_batch_inv()
    test_batch_div()
    test_point_add_batch()
    test_costs()


if __name__ == "__main__":
    main()
    print("\ntest_ec_batch: all passed")
