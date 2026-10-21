"""Kim ePrint 2026/106 Sec 4.4 mixed addition with a windowed 3-constant QROM.

Toy scale only (p = 5, 13; see the `ec_edwards_kim` docstring).

  qrom        one unary-iteration walk XORs (y-x, y+x, 2d x y) of W[a] into
              three registers for every address, is self-inverse, and keeps
              the address superposition's relative phases (statevector)
  mixed add   the 7-product out-of-place law, both register layouts, every
              accumulator x projective scaling x addend constants held in L:
              output a valid (X:Y:Z:T) of the sum, P1 and L restored, clean
  windowed    QROM load + mixed add + QROM unload for w = 1..3, every address
  affine      the in-place window step (old point cleared by "+(-W[a])")
  oracle      |u,v,0> -> |u,v,[u]P+[v]Q>, every (u, v), scratch clean, for
              every style x layout x w x first_lookup (FULL: also "and" mode)
  selector    build_oracle("horner") is the `ec_edwards` baseline,
              build_oracle("kim-qrom") the QROM path with L1..L3 registers
  Aer         the full Shor circuit on Aer CPU MPS (63 qubits), scratch
              measured |0> in every shot, histogram within shot noise
"""
import itertools

import numpy as np
from qiskit.quantum_info import Statevector

from _ec_util import FULL, ok, section

import ec_cost
import ec_edwards as E
import ec_edwards_kim as K
import shor_stats as ST
from ec_sim import Machine, getv, setv, simulate


def run(m, init):
    b = simulate(m.qc, init, m.checks)
    return b, (lambda r: getv(b, m.qc, r))


def check_qrom():
    section("QROM: three constants per address, one walk, self-inverse")
    c = E.EDW5
    n = c.p.bit_length()
    G = E.generator(c)
    for w in (1, 2, 3):
        W = K.window_points(c, G, 0, w)
        assert W[0] == c.identity and W == [c.mul(a, G) for a in range(1 << w)]
        tabs = K.mixed_tables(c, W)
        assert [t[0] for t in tabs] == [1, 1, 0]
        for twice in (False, True):
            m = Machine("and")
            a = m.alloc(w, "a")
            L = [m.alloc(n, f"L{i}") for i in range(3)]
            K.qrom_load(m, a, L, tabs)
            if twice:
                K.qrom_load(m, a, L, tabs)
            for x in range(1 << w):
                init = {}
                setv(init, a, x)
                b, rd = run(m, init)
                want = [0, 0, 0] if twice else [t[x] for t in tabs]
                assert [rd(r) for r in L] == want and rd(a) == x
                assert sum(b) == bin(x).count("1") + sum(
                    bin(v).count("1") for v in want), "QROM ancilla left dirty"
        cnt = ec_cost.count(m)
        ok(f"w={w}: all {1 << w} addresses load (y-x, y+x, 2d x y) of W[a] "
           f"(W[0] = O -> (1, 1, 0)); load+unload = identity; "
           f"{cnt['toffoli_paper']} AND/Toffoli for the pair, "
           f"{max(w - 1, 0)} QROM ancilla(s)")

    for w in (1, 2, 3):
        m = Machine("toffoli")
        a = m.alloc(w, "a")
        L = [m.alloc(n, f"L{i}") for i in range(3)]
        tabs = K.mixed_tables(c, K.window_points(c, G, 0, w))
        K.qrom_load(m, a, L, tabs)
        nq = m.qc.num_qubits
        idx = lambda reg, v: sum(((v >> i) & 1) << m.qc.find_bit(q).index
                                 for i, q in enumerate(reg))
        rng = np.random.default_rng(w)
        amp = rng.normal(size=1 << w) + 1j * rng.normal(size=1 << w)
        amp /= np.linalg.norm(amp)
        psi, want = np.zeros(1 << nq, complex), np.zeros(1 << nq, complex)
        for x in range(1 << w):
            psi[idx(a, x)] = amp[x]
            want[idx(a, x) | sum(idx(r, t[x]) for r, t in zip(L, tabs))] = amp[x]
        fid = abs(np.vdot(want, Statevector(psi).evolve(m.qc).data))
        assert abs(fid - 1) < 1e-9, fid
    ok("w=1..3: random-phase address superposition -> exact entangled "
       "sum_a c_a |a>|W[a] constants> (statevector fidelity 1, no stray phase)")


def mixed_case(c, layout, lams, addends):
    n = c.p.bit_length()
    m = Machine("and")
    P1 = tuple(m.alloc(n, s) for s in "XYZT")
    L = tuple(m.alloc(n, s) for s in ("L1", "L2", "L3"))
    P3 = tuple(m.alloc(n, s + "3") for s in "XYZT")
    K.mixed_add(m, P1, L, c, P3, layout)
    io = set(q for r in P1 + L + P3 for q in r)
    for R, P, lam in itertools.product(addends, c.points(), lams):
        init, e, k = {}, c.to_ext(P, lam), c.addend_constants(R)
        for r, v in zip(P1 + L, e + k):
            setv(init, r, v)
        b, rd = run(m, init)
        out = tuple(rd(r) for r in P3)
        assert tuple(rd(r) for r in P1 + L) == e + k, "input not restored"
        assert c.ext_valid(out) and c.from_ext(out) == c.add(P, R), (P, R, lam)
        assert not any(b[i] for i, q in enumerate(m.qc.qubits) if q not in io)
    return m


def check_mixed_add():
    section("mixed add: seven products, constants from quantum registers")
    for layout in K.LAYOUTS:
        c = E.EDW5
        m = mixed_case(c, layout, range(1, c.p), c.points())
        cnt = ec_cost.count(m)
        ok(f"p=5 {layout}: 8 addends x 8 points x 4 scalings -- valid (X:Y:Z:T) "
           f"of P+R, P1 and L restored, scratch clean; {m.qc.num_qubits} "
           f"qubits incl. 11n I/O, {cnt['toffoli_paper']} Toffoli-paper")
        c = E.EDW13
        lams = range(1, 13) if FULL else (1, 7, 12)
        adds = c.points() if FULL else c.points()[:4] + [c.identity]
        mixed_case(c, layout, lams, adds)
        ok(f"p=13 {layout}: {len(adds)} addends x all {len(c.points())} points "
           f"x {len(lams)} scalings")


def check_windowed():
    section("windowed mixed add: QROM load, 7 products, QROM unload")
    c = E.EDW5
    n = c.p.bit_length()
    G = E.generator(c)
    for layout, w in itertools.product(K.LAYOUTS, (1, 2, 3)):
        W = K.window_points(c, G, 1 if w < 3 else 0, w)
        m = Machine("and")
        a = m.alloc(w, "a")
        P1 = tuple(m.alloc(n, s) for s in "XYZT")
        P3 = tuple(m.alloc(n, s + "3") for s in "XYZT")
        K.windowed_mixed_add(m, a, P1, W, c, P3, layout)
        io = set(q for r in (a,) + P1 + P3 for q in r)
        for x, P, lam in itertools.product(range(1 << w), c.points(), (1, 3)):
            init, e = {}, c.to_ext(P, lam)
            setv(init, a, x)
            for r, v in zip(P1, e):
                setv(init, r, v)
            b, rd = run(m, init)
            out = tuple(rd(r) for r in P3)
            assert c.from_ext(out) == c.add(P, W[x]) and rd(a) == x
            assert tuple(rd(r) for r in P1) == e
            assert not any(b[i] for i, q in enumerate(m.qc.qubits) if q not in io)
        if layout == "compact":
            ok(f"w={w}: all {1 << w} addresses x 8 points x 2 scalings "
               f"(incl. W[0] = O); {ec_cost.count(m)['toffoli_paper']} "
               f"Toffoli-paper, {m.qc.num_qubits} qubits")
    ok("paper layout: same checks for w = 1..3")

    section("in-place affine window step (no retained input)")
    for w in (1, 2):
        W = K.window_points(c, G, 1, w)
        m = Machine("and")
        a = m.alloc(w, "a")
        x, y = m.anc(n, "x"), m.anc(n, "y")
        nx, ny = K.affine_add_window(m, a, x, y, W, c)
        for xa, P in itertools.product(range(1 << w), c.points()):
            init = {}
            setv(init, a, xa)
            setv(init, x, P[0])
            setv(init, y, P[1])
            b, rd = run(m, init)
            assert (rd(nx), rd(ny)) == c.add(P, W[xa])
            assert rd(x) == rd(y) == 0
        ok(f"w={w}: every address x point -- old (x, y) cleared by the inverse "
           f"of '+(-W[a])' (negated QROM tables); {m.qc.num_qubits} qubits")


def oracle_ok(c, P, Q, k, mb, m, info):
    kr, lr, ox, oy = info["regs"]
    keep = set(list(kr) + list(lr) + list(ox) + list(oy))
    for u, v in itertools.product(range(1 << mb), repeat=2):
        init = {}
        setv(init, kr, u)
        setv(init, lr, v)
        b, rd = run(m, init)
        assert (rd(ox), rd(oy)) == c.mul(u + k * v, P), (u, v)
        assert not any(b[i] for i, q in enumerate(m.qc.qubits) if q not in keep)


def check_oracle():
    section("ECDLP oracle: every (u, v), every variant")
    c = E.EDW5
    P = E.generator(c)
    k, mb = 5, 3
    Q = c.mul(k, P)
    modes = ("toffoli", "and") if FULL else ("toffoli",)
    for style, layout, w, fl, mode in itertools.product(
            K.STYLES, K.LAYOUTS, (1, 2, 3), (True, False), modes):
        m, info = K.ecdlp_oracle(c, P, Q, mb, w, mode, style, layout, fl)
        oracle_ok(c, P, Q, k, mb, m, info)
        assert info["mixed_additions"] == len(info["windows"]) - int(fl)
    ok(f"{2 * 2 * 3 * 2 * len(modes)} variants (style x layout x w x "
       f"first_lookup x {'/'.join(modes)}): all 64 (u, v), scratch clean")

    section("selector: baseline vs QROM path")
    mh, ih = K.build_oracle("horner", c, P, Q, mb)
    mk, ik = K.build_oracle("kim-qrom", c, P, Q, mb, w=2)
    names = lambda m: {r.name.split("_")[0] for r in m.qc.qregs}
    # pooled ancillas keep their first name, so L1 may reuse an older register
    qrom_regs = {"L2", "L3", "qrom"}
    assert qrom_regs <= names(mk) and not qrom_regs & names(mh), (
        names(mk), names(mh))
    oracle_ok(c, P, Q, k, mb, mh, ih)
    oracle_ok(c, P, Q, k, mb, mk, ik)
    th, tk = (ec_cost.count(x)["toffoli_paper"] for x in (mh, mk))
    ok(f"horner: {mh.qc.num_qubits} qubits, {th} Toffoli-paper (6 controlled "
       f"additions, no constant registers); kim-qrom w=2: {mk.qc.num_qubits} "
       f"qubits, {tk} Toffoli-paper, constants coherently loaded by QROM")


def check_aer():
    section("Aer CPU: QROM-windowed Edwards ECDLP end to end")
    c = E.EDW5
    P = E.generator(c)
    order = c.point_order(P)
    cases = [("toffoli", 5, "bennett", 3)]
    if FULL:
        cases += [("and", 3, "inplace", 3), ("toffoli", 3, "inplace", 2)]
    for mode, k, style, w in cases:
        Q = c.mul(k, P)
        shots = 2000
        pairs, scr, cands, info = K.run_ecdlp(
            c, P, Q, order, w, shots=shots, seed=11, mode=mode, style=style)
        mb = info["m_bits"]
        exact = ST.ecdlp_probs(order, k, mb)
        emp = ST.empirical(pairs, (1 << mb, 1 << mb))
        tvd, null = ST.tvd(emp, exact), ST.tvd_null(exact, shots)
        assert set(scr) == {0}, scr
        assert cands and cands[0][0] == k and c.mul(cands[0][0], P) == Q
        assert tvd <= null, (tvd, null)
        wrong = ST.tvd(emp, ST.ecdlp_probs(order, (k + 2) % order, mb))
        assert wrong > 3 * null, (wrong, null)
        ok(f"{style} w={w} {mode}: {info['qubits']} qubits on {info['aer']}; "
           f"k={k} recovered; {info['scratch_qubits']} scratch qubits |0> in "
           f"all {shots} shots; TVD {tvd:.3f} <= noise {null:.3f} "
           f"(vs {wrong:.3f} for a wrong k)")


def main():
    check_qrom()
    check_mixed_add()
    check_windowed()
    check_oracle()
    check_aer()


if __name__ == "__main__":
    main()
    print("\ntest_ec_edwards_kim: all passed")
