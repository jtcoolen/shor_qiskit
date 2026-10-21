"""Toy complete twisted-Edwards (extended coordinates) arithmetic and ECDLP.

Standard HWCD 2008 / RFC 8032 Sec 5.1.4 a = -1 law on tiny fields -- not
Kim 2026/106 Sec 4.4 and not Ed25519 scale (see `ec_edwards` docstring).

  classical   the extended law against the affine law, every pair, every
              projective scaling; XY = ZT preserved; curve validation
  ext add     the reversible out-of-place controlled addition, exhaustively on
              basis states: output, retained input intact, scratch clean
  affine add  the in-place (forward + inverse-of-minus-R) controlled addition
  oracle      |u,v,0> -> |u,v,S+uP+vQ>, every (u, v), all scratch clean, both
              the in-place and the Bennett (retained-input) styles
  Aer         the full circuit (H, oracle, inverse QFT, measure) on Aer CPU
              MPS, in pure-Toffoli and Gidney-AND modes; scratch is measured
              in superposition and must read zero; histogram within shot noise
              of the exact Shor distribution
"""
import itertools

from _ec_util import FULL, ok, section

import ec_classical as C
import ec_cost
import ec_edwards as E
import shor_stats as ST
from ec_sim import Machine, getv, setv, simulate


def check_classical():
    section("classical: extended law == affine law, complete")
    curves = [E.EDW5, E.EDW13, E.EdwardsCurve(17, 3), E.EdwardsCurve(29, 2)]
    for c in curves:
        pts = c.points()
        lams = range(1, c.p) if c.p <= 13 else (1, 2, c.p - 1)
        for P, Q in itertools.product(pts, pts):
            ref = c.add(P, Q)
            assert c.on_curve(ref)
            for l1 in lams:
                R = c.ext_add(c.to_ext(P, l1), c.to_ext(Q, 3 % c.p or 1))
                assert c.ext_valid(R), (c.name, P, Q, l1, R)
                assert c.from_ext(R) == ref
        g = E.generator(c)
        assert c.mul(c.point_order(g), g) == c.identity
        ok(f"p={c.p} d={c.d}: {len(pts)} points, all {len(pts)**2} pairs incl. "
           f"identity/doubling/inverse, XY=ZT kept, no exceptional input")
    for bad in ((7, 3), (13, 3), (5, 4)):
        try:
            E.EdwardsCurve(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted non-complete parameters {bad}")
    ok("p = 3 mod 4 and square d are rejected (law would not be complete)")

    c = E.EDW5
    aw, bw, f = c.weierstrass()
    W = C.Curve(c.p, aw, bw)
    pts = c.points()
    w = lambda P: C.O if f(P) is None else C.Point(*f(P))
    assert len({repr(w(P)) for P in pts}) == len(W.points()) == len(pts)
    assert all(W.add(w(P), w(Q)) == w(c.add(P, Q)) for P in pts for Q in pts)
    ok(f"EDW5 is isomorphic to y^2 = x^3 + {aw}x + {bw} (the baseline curve)")


def check_ext_add(c, lams, rs=None):
    n = c.p.bit_length()
    pts = c.points()
    tof = None
    for R in (rs or pts):
        m = Machine("and")
        ctrl = m.alloc(1, "c")
        P1 = tuple(m.alloc(n, s) for s in "XYZT")
        P3 = tuple(m.alloc(n, s + "3") for s in "XYZT")
        E.ext_add_ctrl(m, ctrl[0], P1, R, c, P3)
        tof = ec_cost.count(m)["toffoli_paper"]
        for P, lam, cb in itertools.product(pts, lams, (0, 1)):
            init, e = {}, c.to_ext(P, lam)
            for r, v in zip(P1, e):
                setv(init, r, v)
            setv(init, ctrl, cb)
            b = simulate(m.qc, init, m.checks)
            out = tuple(getv(b, m.qc, r) for r in P3)
            assert tuple(getv(b, m.qc, r) for r in P1) == e, "input not retained"
            assert c.ext_valid(out), (P, R, lam, cb, out)
            assert c.from_ext(out) == (c.add(P, R) if cb else P)
    return m, tof


def check_reversible():
    section("controlled extended addition: exhaustive basis states")
    c = E.EDW5
    m, tof = check_ext_add(c, range(1, c.p))
    ok(f"p=5: all 8 addends x 8 points x 4 scalings x ctrl -- output is a valid "
       f"(X:Y:Z:T) of P+R (or P), input retained, scratch freed clean; "
       f"{m.qc.num_qubits} qubits incl. 8n I/O, {tof} Toffoli-paper per add")
    c13 = E.EDW13
    lams = range(1, 13) if FULL else (1, 5, 12)
    m, tof = check_ext_add(c13, lams, None if FULL else c13.points()[:3])
    ok(f"p=13: same checks ({len(lams)} scalings), {tof} Toffoli-paper per add")

    section("in-place controlled affine addition (no retained input)")
    for c in (E.EDW5,) + ((E.EDW13,) if FULL else ()):
        n = c.p.bit_length()
        for R in c.points():
            m = Machine("and")
            ctrl = m.alloc(1, "c")
            x, y = m.anc(n, "x"), m.anc(n, "y")
            nx, ny = E.affine_add_ctrl(m, ctrl[0], x, y, R, c)
            for P, cb in itertools.product(c.points(), (0, 1)):
                init = {}
                setv(init, x, P[0])
                setv(init, y, P[1])
                setv(init, ctrl, cb)
                b = simulate(m.qc, init, m.checks)
                assert (getv(b, m.qc, nx), getv(b, m.qc, ny)) == (
                    c.add(P, R) if cb else P)
                assert getv(b, m.qc, x) == getv(b, m.qc, y) == 0
        ok(f"p={c.p}: every point, addend and control -- old (x, y) cleared by "
           f"the inverse of '+(-R)'; {m.qc.num_qubits} qubits, "
           f"{ec_cost.count(m)['toffoli_paper']} Toffoli-paper per add")


def check_oracle():
    section("ECDLP oracle: every (u, v), full group, no offset")
    c = E.EDW5
    P = E.generator(c)
    k, mb = 5, 3
    Q = c.mul(k, P)
    for style in ("inplace", "bennett"):
        m, info = E.ecdlp_oracle(c, P, Q, mb, style=style)
        kr, lr, ox, oy = info["regs"]
        keep = set(list(kr) + list(lr) + list(ox) + list(oy))
        for u, v in itertools.product(range(1 << mb), repeat=2):
            init = {}
            setv(init, kr, u)
            setv(init, lr, v)
            b = simulate(m.qc, init, m.checks)
            assert (getv(b, m.qc, ox), getv(b, m.qc, oy)) == c.mul(u + k * v, P)
            assert all(b[i] == 0 for i, q in enumerate(m.qc.qubits)
                       if q not in keep), (style, u, v)
        ops = set(m.qc.count_ops())
        assert ops <= {"x", "cx", "ccx", "swap", "cswap", "mcx", "barrier"}, ops
        cnt = ec_cost.count(m)
        ok(f"{style}: all 64 (u, v) incl. identity and doubling, scratch clean; "
           f"{m.qc.num_qubits} qubits, {cnt['toffoli']} Toffoli, retained "
           f"projective input peak {info['retained_qubits_peak']} qubits")


def check_aer():
    section("Aer CPU: complete Edwards ECDLP end to end")
    c = E.EDW5
    P = E.generator(c)
    order = c.point_order(P)
    for mode, k in (("toffoli", 5), ("and", 3)):
        Q = c.mul(k, P)
        shots = 2000
        pairs, scr, cands, info = E.run_ecdlp(
            c, P, Q, order, shots=shots, seed=11, mode=mode)
        mb = info["m_bits"]
        exact = ST.ecdlp_probs(order, k, mb)
        emp = ST.empirical(pairs, (1 << mb, 1 << mb))
        tvd, null = ST.tvd(emp, exact), ST.tvd_null(exact, shots)
        assert set(scr) == {0}, scr
        assert cands and cands[0][0] == k and c.mul(cands[0][0], P) == Q
        assert tvd <= null, (tvd, null)
        wrong = ST.tvd(emp, ST.ecdlp_probs(order, (k + 2) % order, mb))
        assert wrong > 3 * null, (wrong, null)
        ok(f"{mode}: {info['qubits']} qubits on {info['aer']}; k={k} recovered; "
           f"{info['scratch_qubits']} scratch qubits measured |0> in all "
           f"{shots} shots; TVD {tvd:.3f} <= noise {null:.3f} "
           f"(vs {wrong:.3f} for a wrong k)")


def main():
    check_classical()
    check_reversible()
    check_oracle()
    check_aer()


if __name__ == "__main__":
    main()
    print("\ntest_ec_edwards: all passed")
