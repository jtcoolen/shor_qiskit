"""Twisted Edwards curves and the extended-coordinate out-of-place addition of
2026/106 Sec 4.4 (Ed25519), against ec_proj's Jacobian addition (Sec 4.2)."""
import random
import time

from _ec_util import FULL, ok, section

import ec_classical as C
import ec_cost as CO
import ec_edwards as ED
import ec_mult as MU
import ec_proj as PJ
from ec_classical import Point
from ec_sim import Machine, getv, setv, simulate


# --- helpers -----------------------------------------------------------------
def quad(m, n, tag):
    return tuple(m.alloc(n, tag + c) for c in "XYZT")


def sim(m, inputs, keep):
    """Run, and assert every qubit outside `keep` ends |0> (not just at frees)."""
    init = {}
    for reg, v in inputs.items():
        setv(init, reg, v)
    bits = simulate(m.qc, init, m.checks)
    kept = {q for r in keep for q in r}
    idx = {q: i for i, q in enumerate(m.qc.qubits)}
    assert all(bits[idx[q]] == 0 for q in m.qc.qubits if q not in kept), \
        "a scratch qubit is dirty"
    return lambda reg: getv(bits, m.qc, reg)


def rd4(rd, P):
    return tuple(rd(r) for r in P)


def toffolis(m):
    """ec_cost's toffoli_paper without its (slow) depth pass."""
    t = {}
    for ci in m.qc.data:
        t[ci.operation.name] = t.get(ci.operation.name, 0) + 1
    return t.get("ecand", 0) + t.get("ccx", 0) + t.get("mcx", 0) + t.get("cswap", 0)


_ORD = {}


def order(E, P):
    k = (E.p, E.d, P)
    if k not in _ORD:
        _ORD[k] = E.point_order(P)
    return _ORD[k]


def kinds(E, P1, P2):
    """Which of the cases a non-complete law would have to exclude this pair hits."""
    I = E.identity
    out = set()
    if P1 == I or P2 == I:
        out.add("identity")
    if P1 == P2:
        out.add("doubling")
    if P1 == E.neg(P2):
        out.add("P+(-P)")
    if any(order(E, P) in (2, 4) for P in (P1, P2)):
        out.add("order 2/4")
    return out


def is_prime(n):
    """Deterministic Miller-Rabin below 3.3e24."""
    if n < 2:
        return False
    B = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for q in B:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in B:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def generic_curve(n):
    """p = the largest prime = 1 mod 4 below 2^n, d the least non-square, and
    a random point: a real complete a = -1 curve at cryptographic-style width."""
    p = next(q for q in range((1 << n) - 3, 0, -4) if is_prime(q))
    d = next(k for k in range(2, p) if not ED._is_square(k, p))
    E = ED.TwistedEdwards(p, -1, d, f"ed-{n}")
    assert E.complete()
    return E


def random_point(E, rnd):
    while True:
        y = rnd.randrange(E.p)
        x = E.recover_x(y, rnd.randrange(2))
        if x is not None:
            return Point(x, y)


class MulCounter:
    """Count the multiplications a build emits: forward, and inside undone bodies.

    Patches ec_mult.modmul / modmul_const to record the instruction span each
    call emitted, and Machine.step / undo to find which spans an undo reverses.
    Undoing does not re-call the builder, so this is the only way to see it.
    """

    def __enter__(self):
        self.spans, self.steps = [], {}
        self.fwd = {"M": 0, "c": 0}
        self.undone = {"M": 0, "c": 0}
        self._orig = (MU.modmul, MU.modmul_const, Machine.step, Machine.undo)
        om, oc, ostep, oundo = self._orig

        def wrap(fn, kind):
            def w(m, *a, **k):
                s = len(m.qc.data)
                fn(m, *a, **k)
                self.spans.append((s, len(m.qc.data), kind))
                self.fwd[kind] += 1
            return w

        def step(m, fn, *a, **k):
            mark = len(m.qc.data)
            out, body = ostep(m, fn, *a, **k)
            self.steps[id(body)] = (mark, mark + len(body), body)
            return out, body

        def undo(m, body):
            s, e, _ = self.steps[id(body)]
            for cs, ce, kind in self.spans:
                if s <= cs and ce <= e:
                    self.undone[kind] += 1
            oundo(m, body)

        MU.modmul, MU.modmul_const = wrap(om, "M"), wrap(oc, "c")
        Machine.step, Machine.undo = step, undo
        return self

    def __exit__(self, *exc):
        MU.modmul, MU.modmul_const, Machine.step, Machine.undo = self._orig

    def result(self):
        return ((self.fwd["M"], self.fwd["c"]), (self.undone["M"], self.undone["c"]))


# --- builders, one per circuit -----------------------------------------------
def build_jacobian(W, n, P2):
    m = Machine("and")
    X1, Y1, Z1 = m.alloc(n, "X1"), m.alloc(n, "Y1"), m.alloc(n, "Z1")
    X3, Y3, Z3 = m.alloc(n, "X3"), m.alloc(n, "Y3"), m.alloc(n, "Z3")
    PJ.jacobian_add(m, X1, Y1, Z1, P2.x, P2.y, X3, Y3, Z3, W.p)
    return m, (X1, Y1, Z1), (X3, Y3, Z3)


def build_const(E, n, P2):
    m = Machine("and")
    P1, P3 = quad(m, n, "1"), quad(m, n, "3")
    ED.edwards_add_const(m, P1, P2.x, P2.y, P3, E.p, E.d)
    return m, P1, P3


def build_niels(E, n):
    m = Machine("and")
    P1, P3 = quad(m, n, "1"), quad(m, n, "3")
    N = tuple(m.alloc(n, t) for t in ("ymx", "ypx", "kt"))
    ED.edwards_add_niels(m, P1, N, P3, E.p)
    return m, P1, N, P3


def build_qq(E, n):
    m = Machine("and")
    P1, P2, P3 = quad(m, n, "1"), quad(m, n, "2"), quad(m, n, "3")
    ED.edwards_add(m, P1, P2, P3, E.p, E.d)
    return m, P1, P2, P3


# =============================================================================
# Classical
# =============================================================================
def test_complete():
    section("complete unified law on every pair of every toy curve")
    for E in ED.ED_TOY:
        assert E.complete(), E
        pts = E.points()
        I = E.identity
        assert I in pts and Point(0, E.p - 1) in pts
        seen = set()
        for P in pts:
            assert E.add(P, I) == P and E.add(I, P) == P
            assert E.add(P, E.neg(P)) == I
            for Q in pts:
                R = E.add(P, Q)
                assert E.is_on_curve(R) and R == E.add(Q, P)
                seen |= kinds(E, P, Q)
        assert seen == {"identity", "doubling", "P+(-P)", "order 2/4"}
        # Lagrange: every order divides #E; [ord P] P = O; [-1] P = -P
        for P in pts:
            o = E.point_order(P)
            assert len(pts) % o == 0 and E.mul(o, P) == I and E.mul(-1, P) == E.neg(P)
        ok(f"{E.name}: a={E.a} d={E.d}, #E = {len(pts)}, all {len(pts)**2} pairs "
           f"incl. identity, doubling, P+(-P), order 2/4 -- no exceptions")

    r = random.Random(7)
    for E in ED.ED_TOY:
        pts = E.points()
        for _ in range(200):
            P, Q, R = (r.choice(pts) for _ in range(3))
            assert E.add(E.add(P, Q), R) == E.add(P, E.add(Q, R))
            k, l = r.randrange(-50, 50), r.randrange(-50, 50)
            assert E.add(E.mul(k, P), E.mul(l, P)) == E.mul(k + l, P)
    ok("associativity and [k]P + [l]P = [k+l]P, 200 random triples per curve")

    # completeness is not free: with d a square the law has real exceptions
    E = ED.TwistedEdwards(13, -1, 4, "d-square")
    assert not E.complete()
    bad = 0
    for P in E.points():
        for Q in E.points():
            try:
                E.add(P, Q)
            except ZeroDivisionError:
                bad += 1
    assert bad > 0
    ok(f"control: a=-1, d=4 (a square) mod 13 has {bad} exceptional pairs -- "
       f"the non-square d is what buys completeness")


def test_extended():
    section("extended-coordinate formulas (HWCD08) against the affine law")
    r = random.Random(11)
    for E in ED.ED_TOY:
        p, pts = E.p, E.points()
        for P in pts:
            for Q in pts:
                R = E.add(P, Q)
                e1 = E.to_extended(P, r.randrange(1, p))
                e2 = E.to_extended(Q, r.randrange(1, p))
                outs = [ED.ext_add_general_ref(E, e1, e2), E.add_extended(e1, e2)]
                if E.a == p - 1:
                    outs += [ED.ext_add_ref(E, e1, e2),
                             ED.ext_madd_ref(E, e1, Q.x, Q.y),
                             ED.ext_add_niels_ref(E, e1, E.niels(Q))]
                for o in outs:
                    assert E.on_curve_extended(*o), (E, P, Q)   # Z3 != 0 always
                    assert E.from_extended(*o) == R, (E, P, Q)
    ok("hwcd (any a), hwcd-3 (a=-1), the [106] mixed form and the Niels form: "
       "every pair, random Z, Z3 never 0, XY = ZT holds")

    for E in [E for E in ED.ED_TOY if E.a == 1]:
        E2, fwd, back = E.to_a_minus_1()
        assert E2.a == E.p - 1 and E2.complete()
        pts = E.points()
        assert sorted(map(fwd, pts), key=repr) == sorted(E2.points(), key=repr)
        for P in pts:
            assert back(fwd(P)) == P
            for Q in pts[::3]:
                assert fwd(E.add(P, Q)) == E2.add(fwd(P), fwd(Q))
    ok("a = 1 curves are a = -1 curves (d -> -d) under x -> sqrt(-1) x")


def test_weierstrass():
    section("birational map to short Weierstrass is a group isomorphism")
    for E in ED.ED_TOY:
        W = E.weierstrass()
        pts, wpts = E.points(), W.points()
        img = [E.to_weierstrass(P) for P in pts]
        assert len(set(img)) == len(pts) == len(wpts) and set(img) == set(wpts)
        assert all(E.from_weierstrass(E.to_weierstrass(P)) == P for P in pts)
        assert all(E.from_montgomery(E.to_montgomery(P)) == P for P in pts)
        for P in pts:
            for Q in pts:
                assert E.to_weierstrass(E.add(P, Q)) == W.add(E.to_weierstrass(P),
                                                                E.to_weierstrass(Q))
        # an Edwards ECDLP is the same ECDLP on the Weierstrass side
        G = max(pts, key=E.point_order)
        o = E.point_order(G)
        assert W.point_order(E.to_weierstrass(G)) == o
        for k in range(o):
            Q = E.mul(k, G)
            assert C.verify_dlog(W, E.to_weierstrass(G), E.to_weierstrass(Q), k)
        ok(f"{E.name} -> y^2 = x^3 + {W.a}x + {W.b}: bijective, homomorphic on all "
           f"{len(pts)**2} pairs, dlog preserved for all k < ord G = {o}")


def test_ed25519():
    section("Ed25519 parameters")
    E, B, L = ED.ED25519, ED.ED25519_B, ED.ED25519_L
    p = E.p
    assert p == 2**255 - 19 and E.a == p - 1
    assert E.d * 121666 % p == (-121665) % p
    assert E.complete()                           # -1 square (p = 1 mod 4), d not
    assert E.is_on_curve(B)
    assert B.y == 4 * pow(5, -1, p) % p and E.recover_x(B.y, 0) == B.x
    assert is_prime(L) and E.mul(L, B) == E.identity and E.mul(L - 1, B) == E.neg(B)
    assert E.on_curve_extended(*ED.ext_add_ref(E, E.to_extended(B, 7), E.to_extended(B, 3)))
    ok("p = 2^255 - 19, d = -121665/121666, B = (x, 4/5) with x even; "
       "[l]B = O for l = 2^252 + 27742317777372353535851937790883648493")

    A, Bm = E.montgomery()
    assert A == 486662 and E.to_montgomery(B).x == 9
    W = E.weierstrass()
    # Wei25519 (draft-ietf-lwig-curve-representations) is the B = 1 scaling
    wa = 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa984914a144
    wb = 0x7b425ed097b425ed097b425ed097b425ed097b425ed097b4260b5e9c7710c864
    assert W.a * Bm * Bm % p == wa and W.b * pow(Bm, 3, p) % p == wb
    WB = E.to_weierstrass(B)
    assert W.on_curve(WB) and W.mul(L, WB).inf and E.from_weierstrass(WB) == B
    ok("Montgomery A = 486662 and u(B) = 9 (Curve25519); Weierstrass form is "
       "Wei25519 up to u -> u/B; [l] phi(B) = O there too")


# =============================================================================
# Quantum
# =============================================================================
def test_quantum_const():
    section("edwards_add_const: [106] Sec 4.4, classical addend -- every pair")
    r = random.Random(3)
    for E in (ED.ED13, ED.ED17):
        p, n, pts = E.p, E.p.bit_length(), E.points()
        reps = 3 if FULL else (2 if E is ED.ED13 else 1)
        runs, seen = 0, {}
        for P2 in pts:
            m, P1, P3 = build_const(E, n, P2)
            for P1v in pts:
                for j in range(reps):
                    e1 = E.to_extended(P1v, 1 if j == 0 and reps > 1 else r.randrange(1, p))
                    rd = sim(m, dict(zip(P1, e1)), P1 + P3)
                    out = rd4(rd, P3)
                    assert out == ED.ext_madd_ref(E, e1, P2.x, P2.y), (E, P1v, P2)
                    assert E.on_curve_extended(*out)
                    assert E.from_extended(*out) == E.add(P1v, P2)
                    assert rd4(rd, P1) == e1
                    runs += 1
                for k in kinds(E, P1v, P2):
                    seen[k] = seen.get(k, 0) + 1
        assert set(seen) == {"identity", "doubling", "P+(-P)", "order 2/4"}
        ok(f"{E.name}: {runs} additions exact, all ancillas |0>, input point kept; "
           f"no exclusions -- covers " + ", ".join(f"{k} x{v}" for k, v in sorted(seen.items())))


def test_quantum_general():
    section("edwards_add: two quantum points, add-2008-hwcd-3")
    r = random.Random(5)
    for E in (ED.ED13, ED.ED17):
        p, n, pts = E.p, E.p.bit_length(), E.points()
        m, P1, P2, P3 = build_qq(E, n)
        pairs = [(a, b) for a in pts for b in pts]
        if not (FULL or E is ED.ED13):             # keep every special pair
            pairs = [pq for pq in pairs if kinds(E, *pq)] + r.sample(pairs, 60)
        for P1v, P2v in pairs:
            e1 = E.to_extended(P1v, r.randrange(1, p))
            e2 = E.to_extended(P2v, r.randrange(1, p))
            inp = dict(zip(P1, e1))
            inp.update(zip(P2, e2))
            rd = sim(m, inp, P1 + P2 + P3)
            out = rd4(rd, P3)
            assert out == ED.ext_add_ref(E, e1, e2), (E, P1v, P2v)
            assert E.from_extended(*out) == E.add(P1v, P2v)
            assert rd4(rd, P1) == e1 and rd4(rd, P2) == e2
        ok(f"{E.name}: {len(pairs)} pairs (both inputs at random Z) exact; both "
           f"inputs preserved, the 2n-qubit Niels scratch of P2 cleaned")


def test_quantum_ctrl_window():
    section("controlled and windowed additions: the identity needs no special case")
    E = ED.ED13
    p, n, pts = E.p, 4, E.points()
    r = random.Random(9)
    for P2 in r.sample(pts, 5) + [E.identity]:
        m = Machine("and")
        c = m.alloc(1, "c")
        P1, P3 = quad(m, n, "1"), quad(m, n, "3")
        ED.edwards_add_const_ctrl(m, c[0], P1, P2.x, P2.y, P3, p, E.d)
        for P1v in pts:
            e1 = E.to_extended(P1v, r.randrange(1, p))
            for cv in (0, 1):
                inp = dict(zip(P1, e1))
                inp[c] = cv
                rd = sim(m, inp, (c,) + P1 + P3)
                want = E.add(P1v, P2) if cv else P1v
                assert E.from_extended(*rd4(rd, P3)) == want and rd(c) == cv
    ok("edwards_add_const_ctrl: P1 + [c]P2 for c in {0,1} on 6 addends x all P1; "
       "the control is CNOT loads of Niels(P2) or Niels(O) = (1,1,0)")

    G = max(pts, key=E.point_order)
    for w in (1, 2, 3):
        table = ED.niels_window(E, G, w)
        assert table[0] == ED.NIELS_IDENTITY
        m = Machine("and")
        addr = m.alloc(w, "i")
        P1, P3 = quad(m, n, "1"), quad(m, n, "3")
        ED.edwards_add_window(m, addr, P1, table, P3, p)
        for i in range(1 << w):
            for P1v in pts:
                e1 = E.to_extended(P1v, r.randrange(1, p))
                inp = dict(zip(P1, e1))
                inp[addr] = i
                rd = sim(m, inp, (addr,) + P1 + P3)
                assert E.from_extended(*rd4(rd, P3)) == E.add(P1v, E.mul(i, G))
    ok("edwards_add_window, w = 1..3: P1 + [i]G for every i (i = 0 included, no "
       "offset) and every P1")


def test_quantum_inverse():
    section("PA-dagger: the reverse addition clears the output")
    E = ED.ED13
    p, n, pts = E.p, 4, E.points()
    r = random.Random(13)
    for P2 in r.sample(pts, 4):
        m = Machine("and")
        P1, P3 = quad(m, n, "1"), quad(m, n, "3")
        ED.edwards_add_const_inv(m, P1, P2.x, P2.y, P3, p, E.d)
        for P1v in pts:
            e1 = E.to_extended(P1v, r.randrange(1, p))
            inp = dict(zip(P1, e1))
            inp.update(zip(P3, ED.ext_madd_ref(E, e1, P2.x, P2.y)))
            rd = sim(m, inp, P1)
            assert rd4(rd, P3) == (0, 0, 0, 0) and rd4(rd, P1) == e1
    m = Machine("and")
    P1, P2, P3 = quad(m, n, "1"), quad(m, n, "2"), quad(m, n, "3")
    ED.edwards_add_inv(m, P1, P2, P3, p, E.d)
    for _ in range(40):
        e1 = E.to_extended(r.choice(pts), r.randrange(1, p))
        e2 = E.to_extended(r.choice(pts), r.randrange(1, p))
        inp = dict(zip(P1, e1))
        inp.update(zip(P2, e2))
        inp.update(zip(P3, ED.ext_add_ref(E, e1, e2)))
        rd = sim(m, inp, P1 + P2)
        assert rd4(rd, P3) == (0, 0, 0, 0)
    ok("edwards_add_const_inv and edwards_add_inv hand the register back clean")


def test_vs_weierstrass_exceptions():
    section("the same group through ec_proj.jacobian_add: doubling breaks it")
    E = ED.ED13
    W, n = E.weierstrass(), 4
    fail = 0
    for P in E.points():
        if P == E.identity:
            continue                        # O has no affine encoding for Alg. 4
        Q = E.to_weierstrass(P)
        m, (X1, Y1, Z1), (X3, Y3, Z3) = build_jacobian(W, n, Q)
        rd = sim(m, {X1: Q.x, Y1: Q.y, Z1: 1}, (X1, Y1, Z1, X3, Y3, Z3))
        got = C.jacobian_to_affine(W, rd(X3), rd(Y3), rd(Z3))
        fail += got != W.add(Q, Q)
    doublings = len(E.points()) - 1
    assert fail == doublings - 1            # only 2*(0,-1) = O survives: Z3 = 0
    ok(f"{fail} of {doublings} Weierstrass doublings wrong (H = 0 gives Z3 = 0); "
       f"the Edwards circuits got every doubling right above")


def test_counts():
    section("operation counts, Toffolis and qubits vs ec_proj.jacobian_add")
    rows = []
    E = ED.ED13
    with MulCounter() as mc:
        build_jacobian(E.weierstrass(), 4, E.to_weierstrass(E.points()[3]))
    got = {"jac": mc.result()}
    with MulCounter() as mc:
        build_const(E, 4, E.points()[3])
    got["const"] = mc.result()
    with MulCounter() as mc:
        build_niels(E, 4)
    got["niels"] = mc.result()
    with MulCounter() as mc:
        build_qq(E, 4)
    got["qq"] = mc.result()
    want = list(ED.OPS.values())
    assert [got[k] for k in ("jac", "const", "niels", "qq")] == want, got
    for (lab, ((fm, fc), (um, uc))) in zip(ED.OPS, want):
        print(f"      {lab:52s} forward {fm}M+{fc}c = {fm + fc:2d}, "
              f"uncompute {um}M+{uc}c = {um + uc:2d}")
    ok("measured off the builds: 11 forward multiplications (Weierstrass) vs 7 "
       "(Edwards, [106]'s count); uncompute 10 vs 3")

    def measure(E, n, big):
        rnd = random.Random(n)
        P2 = random_point(E, rnd)
        W = E.weierstrass()
        res, t0 = {}, time.time()
        for lab, build in (
                ("jacobian_add", lambda: build_jacobian(W, n, E.to_weierstrass(P2))[0]),
                ("edwards_add_const", lambda: build_const(E, n, P2)[0]),
                ("edwards_add_niels", lambda: build_niels(E, n)[0]),
                ("edwards_add", lambda: build_qq(E, n)[0])):
            if big and n > 32 and lab == "edwards_add" and not FULL:
                continue
            m = build()
            tof = toffolis(m)
            if not big:
                assert tof == CO.count(m)["toffoli_paper"]
            res[lab] = (m.qc.num_qubits, tof, m)
        base = res["jacobian_add"]
        for lab, (q, t, _) in res.items():
            rows.append([E.p if not big else f"~2^{n}", n, lab, q, t,
                         f"{q / base[0]:.2f}", f"{t / base[1]:.2f}"])
        assert res["edwards_add_const"][0] < base[0] and res["edwards_add_const"][1] < base[1]
        assert res["edwards_add_niels"][1] < base[1]
        return res, time.time() - t0

    for E in (ED.ED13, ED.ED17):
        measure(E, E.p.bit_length(), False)
    rnd = random.Random(1)
    for n in (32, 64):
        E = generic_curve(n)
        res, dt = measure(E, n, True)
        if n == 32:
            # one full-width basis state through the 32-bit circuit, as a spot check
            P1v, P2v = random_point(E, rnd), random_point(E, rnd)
            m, P1, P3 = build_const(E, n, P2v)
            e1 = E.to_extended(P1v, rnd.randrange(1, E.p))
            rd = sim(m, dict(zip(P1, e1)), P1 + P3)
            assert E.from_extended(*rd4(rd, P3)) == E.add(P1v, P2v)
    print(CO.table(rows, ["p", "n", "circuit", "qubits", "Toffoli", "q/jac", "T/jac"]))
    ok("Edwards (classical addend) beats Jacobian-affine at every size in both "
       "Toffolis and qubits; a 32-bit Edwards addition simulated exactly")


def main():
    t = time.time()
    test_complete()
    test_extended()
    test_weierstrass()
    test_ed25519()
    test_quantum_const()
    test_quantum_general()
    test_quantum_ctrl_window()
    test_quantum_inverse()
    test_vs_weierstrass_exceptions()
    test_counts()
    print(f"\n      ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
    print("\ntest_ec_edwards: all passed")
