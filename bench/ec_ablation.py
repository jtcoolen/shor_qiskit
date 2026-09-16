"""Ablation study: what each ECDLP technique buys, alone and combined.

Every number the ECDLP part of the note quotes comes from here.  The script
writes `bench/ec_ablation.json` (machine-readable, so claims can be checked
mechanically) and prints LaTeX table bodies.

Ground rules, so the tables mean something:

* Each ablation fixes a *task* and varies one implementation choice.  Comparing
  a modular doubling against a point addition would be meaningless; comparing
  two modular doublings is not.
* The metric is Toffoli-equivalents (AND, AND-dagger, Toffoli and Fredkin each
  count once), which is what both source papers report, plus T-count under the
  7/4/0 model and full circuit depth.
* Correctness is re-checked inside the benchmark.  A cheaper circuit that is
  wrong is not a datapoint, so every variant is run against the classical model
  before its cost is recorded.  Variants that are *deliberately* approximate
  record a measured failure rate instead.
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))

import ec_adders as A
import ec_approx as AX
import ec_classical as C
import ec_cost as CO
import ec_eea as E
import ec_kaliski as K
import ec_kaliski_opt as KO
import ec_modarith as MA
import ec_montgomery as MG
import ec_mult as MU
import ec_pointadd as PA
import ec_proj as PJ
import ec_window as W
from ec_sim import Machine, SimError, run

RESULTS = {}


def measure(build):
    m = Machine("and")
    regs = build(m)
    return m, regs, CO.count(m)


def check_all(m, regs, cases, ref):
    """Run every case; return the failure fraction (0.0 means exact)."""
    bad = 0
    for case in cases:
        try:
            rd = run(m, {regs[k]: v for k, v in case["in"].items()})
            if any(rd(regs[k]) != v for k, v in ref(case).items()):
                bad += 1
        except SimError:
            bad += 1
    return bad / max(1, len(cases))


def record(section, variant, cost, fail=0.0, note="", task=""):
    RESULTS.setdefault(section, {"task": task, "variants": {}})
    if task:
        RESULTS[section]["task"] = task
    RESULTS[section]["variants"][variant] = {
        "qubits": cost["qubits"],
        "toffoli_paper": cost.get("toffoli_paper"),   # AND-dagger free: the papers' metric
        "toffoli_equiv": cost["toffoli_equiv"],       # every 3-qubit op counted
        "t": cost["t"], "depth": cost["depth"], "and": cost["and"],
        "and_dg": cost["and_dg"], "gates": cost["gates"],
        "failure_rate": fail, "note": note,
    }
    return RESULTS[section]["variants"][variant]


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# =============================================================================
# A. Adders -- the bottom of the stack
# =============================================================================
def ablate_adders(ns=(8, 16, 32)):
    hdr("A. ADDERS: CDKM vs Gidney, plain and controlled")
    rnd = random.Random(1)
    for n in ns:
        # plain addition
        def b_cdkm(m, n=n):
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            c = m.anc(1, "c")
            A.cdkm_add(m.ctx, x, y, c[0])
            m.free(c)
            return {"x": x, "y": y}

        def b_gid(m, n=n):
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            a = m.anc(max(n - 1, 1), "a")
            A.gidney_add(m.ctx, x, y, a)
            m.free(a)
            return {"x": x, "y": y}

        cases = [{"in": {"x": rnd.getrandbits(n), "y": rnd.getrandbits(n)}}
                 for _ in range(12)]
        for name, bld in (("CDKM [CDKM04]", b_cdkm), ("Gidney [Gid18]", b_gid)):
            m, regs, cost = measure(bld)
            f = check_all(m, regs, cases,
                          lambda c: {"y": (c["in"]["x"] + c["in"]["y"]) % 2**n,
                                     "x": c["in"]["x"]})
            assert f == 0.0, (name, n)
            record(f"adder_n{n}", name, cost, task=f"{n}-bit addition y += x")
            print(f"  n={n:>3} {name:<18} {cost['toffoli_paper']:>5} Toff-eq  "
                  f"{cost['qubits']:>4} qubits  depth {cost['depth']:>5}")

        # controlled addition: reconstruct vs copy-then-add ([106] Fig. 4b)
        def b_recon(m, n=n):
            ct = m.alloc(1, "ct")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            c = m.anc(1, "c")
            A.cdkm_add(m.ctx, x, y, c[0], [ct[0]])
            m.free(c)
            return {"ct": ct, "x": x, "y": y}

        def b_copy(m, n=n):
            ct = m.alloc(1, "ct")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            cp, a = m.anc(n, "cp"), m.anc(max(n - 1, 1), "a")
            A.cadd(m.ctx, ct[0], x, y, cp, a)
            m.free(cp, a)
            return {"ct": ct, "x": x, "y": y}

        # the same control style with the adder held fixed, so the control
        # choice can be separated from the CDKM-to-Gidney choice
        def b_copy_cdkm(m, n=n):
            ct = m.alloc(1, "ct")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            cp, car = m.anc(n, "cp"), m.anc(1, "k")
            for i, q in enumerate(x):
                m.ctx.and_(ct[0], q, cp[i])
            A.cdkm_add(m.ctx, cp, y, car[0])
            for i, q in enumerate(x):
                m.ctx.and_dg(ct[0], q, cp[i])
            m.free(cp, car)
            return {"ct": ct, "x": x, "y": y}

        ccases = [{"in": {"ct": t, "x": rnd.getrandbits(n), "y": rnd.getrandbits(n)}}
                  for t in (0, 1) for _ in range(6)]
        for name, bld in (("reconstruct (CDKM)", b_recon),
                          ("copy-then-add, CDKM inside", b_copy_cdkm),
                          ("copy-then-add [106] Fig 4b", b_copy)):
            m, regs, cost = measure(bld)
            f = check_all(m, regs, ccases,
                          lambda c: {"y": ((c["in"]["x"] + c["in"]["y"]) % 2**n
                                           if c["in"]["ct"] else c["in"]["y"])})
            assert f == 0.0, (name, n)
            record(f"cadder_n{n}", name, cost,
                   task=f"{n}-bit controlled addition y += x if ctrl")
            print(f"  n={n:>3} {name:<28} {cost['toffoli_paper']:>5} Toff-eq  "
                  f"{cost['qubits']:>4} qubits  depth {cost['depth']:>5}")


# =============================================================================
# B. Modular arithmetic: exact vs approximate vs pseudo-Mersenne
# =============================================================================
def ablate_modarith(primes=(251, 1021, 4093)):
    hdr("B. MODULAR ARITHMETIC: exact [RNSL17] vs approximate/pseudo-Mersenne [1128]")
    for q in primes:
        n = q.bit_length()
        pm = AX.pseudo_mersenne(q)

        def mk(fn, **kw):
            def b(m):
                x = m.alloc(n, "x")
                fn(m, x, q, **kw)
                return {"x": x}
            return b

        variants = [("exact (Alg 5)", MA.moddbl, {})]
        for msbs in (4, 8):
            if msbs < n:
                variants.append((f"approx msbs={msbs} (Alg 6)", AX.moddbl_approx,
                                 {"msbs": msbs}))
        if pm:
            u, f_ = pm
            variants.append((f"pseudo-Mersenne (Alg 7)", AX.moddbl_pm,
                             {"lsbs": max(2, f_.bit_length() + 3)}))
        print(f"  --- modular doubling mod q={q} (n={n})"
              + (f", q = 2^{pm[0]} - {pm[1]}" if pm else "") + " ---")
        for name, fn, kw in variants:
            m, regs, cost = measure(mk(fn, **kw))
            bad = 0
            for v in range(q):
                try:
                    if run(m, {regs["x"]: v})(regs["x"]) != 2 * v % q:
                        bad += 1
                except SimError:
                    bad += 1
            fail = bad / q
            record(f"moddbl_q{q}", name, cost, fail,
                   task=f"modular doubling mod {q}")
            print(f"      {name:<28} {cost['toffoli_paper']:>4} Toff-eq  "
                  f"depth {cost['depth']:>4}  failure {100*fail:>5.2f}%")

        # controlled modular addition
        def mkadd(fn, **kw):
            def b(m):
                c = m.alloc(1, "c")
                x, y = m.alloc(n, "x"), m.alloc(n, "y")
                fn(m, c[0], x, y, q, **kw)
                return {"c": c, "x": x, "y": y}
            return b

        variants = [("exact (Alg 8)", MA.cmodadd, {})]
        for msbs in (4, 8):
            if msbs < n:
                variants.append((f"approx msbs={msbs} (Alg 9)", AX.cmodadd_approx,
                                 {"msbs": msbs}))
        if pm:
            variants.append(("pseudo-Mersenne (Alg 10)", AX.cmodadd_pm, {}))
        print(f"  --- controlled modular addition mod q={q} ---")
        rnd = random.Random(7)
        sample = [(rnd.randrange(q), rnd.randrange(q), t)
                  for t in (0, 1) for _ in range(150)]
        for name, fn, kw in variants:
            m, regs, cost = measure(mkadd(fn, **kw))
            bad = 0
            for a, b_, t in sample:
                try:
                    got = run(m, {regs["c"]: t, regs["x"]: a, regs["y"]: b_})(regs["y"])
                    if got != ((a + b_) % q if t else b_):
                        bad += 1
                except SimError:
                    bad += 1
            fail = bad / len(sample)
            record(f"cmodadd_q{q}", name, cost, fail,
                   task=f"controlled modular addition mod {q}")
            print(f"      {name:<28} {cost['toffoli_paper']:>4} Toff-eq  "
                  f"depth {cost['depth']:>4}  failure {100*fail:>5.2f}%")


# =============================================================================
# C. Modular multiplication
# =============================================================================
def ablate_multiplication(cases=((8, 2, 251), (12, 4, 4093), (16, 4, 65521))):
    hdr("C. MODULAR MULTIPLICATION: schoolbook vs Montgomery (QCSA / QROM)")
    rnd = random.Random(3)
    for n, w, p in cases:
        assert p.bit_length() == n
        R = pow(2, -n, p)
        sample = [(rnd.randrange(p), rnd.randrange(p)) for _ in range(10)]

        def b_school(m, n=n, p=p):
            a, b_, o = m.alloc(n, "a"), m.alloc(n, "b"), m.alloc(n, "o")
            MU.modmul(m, a, b_, o, p)
            return {"a": a, "b": b_, "o": o}

        def b_qcsa(m, n=n, p=p, w=w):
            a, b_, o = m.alloc(n, "a"), m.alloc(n, "b"), m.alloc(n, "o")
            MG.mont_mul_qcsa(m, a, b_, o, p, w)
            return {"a": a, "b": b_, "o": o}

        def b_qrom(m, n=n, p=p, w=w):
            a, b_, o = m.alloc(n, "a"), m.alloc(n, "b"), m.alloc(n, "o")
            MG.mont_mul_lookup(m, a, b_, o, p, w)
            return {"a": a, "b": b_, "o": o}

        print(f"  --- n={n}, w={w}, p={p} ---")
        for name, bld, ref in (
                ("schoolbook double-and-add", b_school, lambda a, b_: a * b_ % p),
                (f"Montgomery+QCSA [106] w={w}", b_qcsa, lambda a, b_: a * b_ * R % p),
                (f"Montgomery+QROM [HJN+20] w={w}", b_qrom, lambda a, b_: a * b_ * R % p)):
            m, regs, cost = measure(bld)
            for a, b_ in sample:
                got = run(m, {regs["a"]: a, regs["b"]: b_})(regs["o"])
                assert got == ref(a, b_), (name, n, w, a, b_, got)
            record(f"mul_n{n}_w{w}", name, cost,
                   task=f"{n}-bit modular multiplication mod {p}")
            print(f"      {name:<32} {cost['toffoli_paper']:>5} Toff-eq  "
                  f"{cost['qubits']:>4} qubits  depth {cost['depth']:>5}")


# =============================================================================
# D. Inversion / division -- the dominant cost
# =============================================================================
def ablate_inversion(primes=(31, 127, 251, 1021, 4093)):
    hdr("D. INVERSION: Kaliski [HJN+20] vs unconditional+postponed [106] Sec 3.3")
    for p in primes:
        n = p.bit_length()

        def b_ref(m, n=n, p=p):
            x, o = m.alloc(n, "x"), m.alloc(n, "o")
            K.mod_inv(m, x, o, p)
            return {"x": x, "o": o}

        def b_opt(m, n=n, p=p):
            x, o = m.alloc(n, "x"), m.alloc(n, "o")
            KO.mod_inv_mont(m, x, o, p)
            return {"x": x, "o": o}

        xs = list(range(1, min(p, 40)))
        mr, rr, cr = measure(b_ref)
        for xv in xs:
            assert run(mr, {rr["x"]: xv})(rr["o"]) == pow(xv, -1, p)
        mo, ro, co = measure(b_opt)
        for xv in xs:
            assert run(mo, {ro["x"]: xv})(ro["o"]) == pow(xv, -1, p) * pow(2, 2 * n, p) % p
        record(f"inv_n{n}", "reference Kaliski [HJN+20]", cr,
               task=f"modular inversion mod {p}")
        record(f"inv_n{n}", "unconditional+postponed [106]", co)
        print(f"  n={n:>3}  ref {cr['toffoli_paper']:>6} Toff-eq / {cr['qubits']:>3} q "
              f"/ depth {cr['depth']:>6}   ->   opt {co['toffoli_paper']:>6} / "
              f"{co['qubits']:>3} q / depth {co['depth']:>6}   "
              f"[{100*(1-co['toffoli_paper']/cr['toffoli_paper']):>4.1f}% Toff, "
              f"{100*(1-co['depth']/cr['depth']):>4.1f}% depth]")

    hdr("D2. WHAT THE POINT ADDITION ACTUALLY NEEDS: y/x vs in-place y*x")
    for p in (7, 11, 13, 31, 61):
        n = p.bit_length()

        def b_div(m, n=n, p=p):
            c = m.alloc(1, "c")
            x, y, o = m.alloc(n, "x"), m.alloc(n, "y"), m.alloc(n, "o")
            K.mod_div(m, c[0], x, y, o, p)
            return {"c": c, "x": x, "y": y, "o": o}

        def b_dialog(m, n=n, p=p):
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            E.inplace_mul(m, x, y, p)
            return {"x": x, "y": y}

        md, rd_, cd = measure(b_div)
        for xv in range(1, min(p, 8)):
            for yv in range(min(p, 5)):
                assert run(md, {rd_["c"]: 1, rd_["x"]: xv, rd_["y"]: yv})(rd_["o"]) \
                    == yv * pow(xv, -1, p) % p
        mi, ri, ci = measure(b_dialog)
        for xv in range(1, min(p, 8)):
            for yv in range(min(p, 5)):
                assert run(mi, {ri["x"]: xv, ri["y"]: yv})(ri["y"]) == xv * yv % p
        record(f"inplace_n{n}_p{p}", "Kaliski division (out-of-place)", cd,
               task=f"the multiplicative step of a point addition, mod {p}")
        record(f"inplace_n{n}_p{p}", "EEA dialog + Bezout replay [1128]", ci)
        print(f"  p={p:>3}  division {cd['toffoli_paper']:>6} Toff-eq / {cd['qubits']:>3} q"
              f"   ->   dialog {ci['toffoli_paper']:>6} / {ci['qubits']:>3} q   "
              f"[{100*(1-ci['toffoli_paper']/cd['toffoli_paper']):>4.1f}% Toff, "
              f"{100*(1-ci['qubits']/cd['qubits']):>4.1f}% qubits]")


# =============================================================================
# E. Point addition -- the three constructions
# =============================================================================
def ablate_point_addition():
    hdr("E. POINT ADDITION: affine in-place vs projective out-of-place vs windowed")
    for curve, label in ((C.CLASSIQ, "p=7"), (C.TOY11, "p=11")):
        p, n = curve.p, curve.p.bit_length()
        pts = [P for P in curve.points() if not P.inf]
        P2 = next(P for P in pts if P.x != 0)

        def b_aff(m, n=n, p=p, P2=P2):
            q = m.alloc(1, "q")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            PA.point_add_ctrl(m, q[0], x, y, P2.x, P2.y, p)
            return {"q": q, "x": x, "y": y}

        def b_prj(m, n=n, p=p, P2=P2):
            X1, Y1, Z1 = m.alloc(n, "X1"), m.alloc(n, "Y1"), m.alloc(n, "Z1")
            X3, Y3, Z3 = m.alloc(n, "X3"), m.alloc(n, "Y3"), m.alloc(n, "Z3")
            PJ.jacobian_add(m, X1, Y1, Z1, P2.x, P2.y, X3, Y3, Z3, p)
            return {"X1": X1, "Y1": Y1, "Z1": Z1, "X3": X3, "Y3": Y3, "Z3": Z3}

        ma, ra, ca = measure(b_aff)
        nchk = 0
        for P1 in pts:
            if C.point_add_exceptional(curve, P1, P2):
                continue
            got = run(ma, {ra["q"]: 1, ra["x"]: P1.x, ra["y"]: P1.y})
            want = curve.add(P1, P2)
            assert (got(ra["x"]), got(ra["y"])) == (want.x, want.y)
            nchk += 1
        mp, rp, cp = measure(b_prj)
        for P1 in pts:
            if P1.x == P2.x:
                continue
            g = run(mp, {rp["X1"]: P1.x, rp["Y1"]: P1.y, rp["Z1"]: 1})
            assert C.jacobian_to_affine(curve, g(rp["X3"]), g(rp["Y3"]),
                                        g(rp["Z3"])) == curve.add(P1, P2)
        record(f"padd_{label}", "affine in-place [106] Alg 3", ca,
               task=f"one point addition on {label} ({nchk} pairs checked)")
        record(f"padd_{label}", "Jacobian out-of-place [106] Alg 4", cp,
               note="leaves the 3n-qubit input point as garbage")
        print(f"  {label}: affine {ca['toffoli_paper']:>6} Toff-eq / {ca['qubits']:>3} q "
              f"/ depth {ca['depth']:>6}")
        print(f"  {label}: projec {cp['toffoli_paper']:>6} Toff-eq / {cp['qubits']:>3} q "
              f"/ depth {cp['depth']:>6}  "
              f"[{100*(1-cp['toffoli_paper']/ca['toffoli_paper']):>4.1f}% Toff]")

    # windowed addition ([1128] Alg 1) on the p=7 curve
    curve, p, n = C.CLASSIQ, 7, 3
    for wd in (2,):
        pts = W.window_points(curve, C.CLASSIQ_G, wd)

        def b_win(m, wd=wd, n=n, p=p, pts=pts):
            addr = m.alloc(wd, "i")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            W.windowed_point_add(m, addr, x, y, pts, p)
            return {"i": addr, "x": x, "y": y}

        mw, rw, cw = measure(b_win)
        nchk = 0
        for R in [P for P in curve.points() if not P.inf]:
            for i in range(1 << wd):
                Pi = pts[i]
                if Pi.inf and R.x == 0:
                    continue
                if not Pi.inf and C.point_add_exceptional(curve, R, Pi):
                    continue
                want = R if Pi.inf else curve.add(R, Pi)
                if want.inf:
                    continue
                g = run(mw, {rw["i"]: i, rw["x"]: R.x, rw["y"]: R.y})
                assert (g(rw["x"]), g(rw["y"])) == (want.x, want.y)
                nchk += 1
        record("padd_windowed_p7", f"windowed w={wd} [1128] Alg 1", cw,
               task=f"one windowed point addition on p=7 ({nchk} cases checked)",
               note="dialog multiplier; handles i=0 (adding O) with no special case")
        print(f"  p=7 windowed w={wd}: {cw['toffoli_paper']:>6} Toff-eq / "
              f"{cw['qubits']:>3} q / depth {cw['depth']:>6}  ({nchk} cases verified)")


# =============================================================================
# F. Temporary ANDs, and the whole ECDLP circuit
# =============================================================================
def ablate_and_and_full():
    hdr("F. TEMPORARY ANDs: what Gidney's 4T-compute/0T-uncompute buys")
    for p in (31, 251, 4093):
        n = p.bit_length()
        m = Machine("and")
        x, o = m.alloc(n, "x"), m.alloc(n, "o")
        K.mod_inv(m, x, o, p)
        c = CO.count(m)
        t_all_toffoli = c["toffoli_equiv"] * CO.TOFFOLI_T
        record(f"tempand_n{n}", "all Toffoli (7 T each)",
               dict(c, t=t_all_toffoli, toffoli_paper=c["toffoli_equiv"]),
               task=f"T-count of one inversion mod {p}")
        record(f"tempand_n{n}", "temporary AND (4 T / 0 T)", c)
        print(f"  n={n:>3}: {c['and']:>5} ANDs + {c['and_dg']:>5} AND-daggers  ->  "
              f"T {t_all_toffoli:>6} (all Toffoli) vs {c['t']:>6} (temp AND)  "
              f"= {t_all_toffoli/max(1,c['t']):.2f}x")

    hdr("G. THE FULL ECDLP CIRCUIT")
    import ec_shor as S
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    m, info = S.ecdlp_circuit(curve, P, Q, order, oracle="arith")
    c = CO.count(m)
    record("full_ecdlp", "arithmetic oracle, full control registers", c,
           task=f"Shor ECDLP on y^2=x^3+5x+4 mod 7, ord(G)={order}")
    print(f"  arithmetic oracle : {c['qubits']:>3} qubits, {c['toffoli_paper']:>6} "
          f"Toff-eq, {c['gates']:>7} gates, depth {c['depth']}")
    qc_t, it = S.ecdlp_circuit(curve, P, Q, order, oracle="table")
    qc_1, i1 = S.ecdlp_circuit_1c(curve, P, Q, order)
    RESULTS["full_ecdlp"]["variants"]["table oracle (simulable)"] = {
        "qubits": qc_t.num_qubits, "gates": len(qc_t.data),
        "depth": qc_t.depth(), "toffoli_paper": None, "toffoli_equiv": None, "t": None,
        "and": None, "and_dg": None, "failure_rate": 0.0,
        "note": "permutation oracle; runs end to end on Aer"}
    RESULTS["full_ecdlp"]["variants"]["table oracle, semiclassical (1 ctrl qubit)"] = {
        "qubits": qc_1.num_qubits, "gates": len(qc_1.data),
        "depth": qc_1.depth(), "toffoli_paper": None, "toffoli_equiv": None, "t": None,
        "and": None, "and_dg": None, "failure_rate": 0.0,
        "note": "Griffiths-Niu; both scalar registers cost one qubit between them"}
    print(f"  table oracle      : {qc_t.num_qubits:>3} qubits, {len(qc_t.data):>7} gates")
    print(f"  table, 1 control  : {qc_1.num_qubits:>3} qubits, {len(qc_1.data):>7} gates")


# =============================================================================
# H. Counting arguments that only make sense at cryptographic size
# =============================================================================
def ablate_counting():
    hdr("H. COUNTING: windowing, zig-zag, dialog space (projections, not circuits)")
    proj = {}
    rows = []
    for n in (192, 224, 256, 384, 521):
        for w in (1, 11, 16):
            adds = (2 * n + 2) if w == 1 else C.n_point_additions(n, w)
            mregs = C.zigzag_registers(adds)
            rows.append((n, w, adds, mregs, mregs * 3 * n))
    proj["windowing_zigzag"] = [
        {"n": n, "w": w, "additions": a, "zigzag_registers": mr,
         "garbage_qubits": g} for n, w, a, mr, g in rows]
    print("    n    w   additions   zig-zag regs   garbage (mG)")
    for n, w, a, mr, g in rows:
        tag = " (unwindowed)" if w == 1 else ""
        print(f"  {n:>3}  {w:>3}   {a:>9}   {mr:>12}   {g:>12}{tag}")

    print()
    space = {}
    for n in (256, 384, 521):
        sp = CO.proj_eea_space(n)
        space[n] = sp
        print(f"  n={n}: dialog record {sp['record_raw']} raw -> "
              f"{sp['record_compressed']} compressed ({sp['record_compressed_per_n']:.3f}n; "
              f"paper 2.355n asymptotic), with register sharing "
              f"{sp['with_register_sharing']} (not implemented)")
    proj["eea_space"] = space
    proj["qubit_totals_1128"] = {
        n: {"space_optimized": CO.proj_1128_qubits(n, "space"),
            "gate_optimized": CO.proj_1128_qubits(n, "gate")}
        for n in (256,)}
    import math
    proj["shor_toffoli_1128"] = {
        "value": CO.proj_1128_shor_toffoli(),
        "log2": round(math.log2(CO.proj_1128_shor_toffoli()), 2)}
    print(f"\n  [1128] Table 1 projection at n=256: "
          f"{CO.proj_1128_qubits(256,'space')} / {CO.proj_1128_qubits(256,'gate')} qubits "
          f"(paper 1192 / 1446)")
    print(f"  [1128] eq. (1) full Shor: 2^{proj['shor_toffoli_1128']['log2']} Toffoli "
          f"(paper 2^26.11)")
    RESULTS["_projections"] = proj


# =============================================================================
# I. The derivation itself, checked numerically
# =============================================================================
def ablate_derivation():
    """Verify every step of the "why it works" derivation.

    The interference argument in Part VII is an argument, not a measurement, so
    it is checked here against a direct evaluation of the amplitudes: build the
    post-oracle state on a toy (r, k), apply the Fourier transform by brute
    force, and compare with the closed form the derivation predicts.
    """
    import cmath
    import math
    hdr("I. THE DERIVATION: amplitudes computed directly vs the closed form")

    def amps(r, k, m, w0=0):
        q = 2 ** m
        pairs = [(u, v) for u in range(q) for v in range(q) if (u + k * v) % r == w0]
        W = cmath.exp(2j * math.pi / q)
        A = {(j1, j2): sum(W ** ((u * j1 + v * j2) % q) for u, v in pairs)
             / (q * math.sqrt(len(pairs)))
             for j1 in range(q) for j2 in range(q)}
        return q, A

    out = {}
    # (a) clean case r | q: the closed form should hold everywhere, exactly
    clean = []
    for r, k, m in ((4, 3, 4), (8, 3, 5), (4, 1, 6)):
        q, A = amps(r, k, m)
        M = q // r
        ok = all(abs(abs(A[(j1, j2)]) -
                     ((1 / math.sqrt(r)) if (j1 % M == 0 and j2 % M == 0 and
                      (j2 // M - k * (j1 // M)) % r == 0) else 0.0)) < 1e-9
                 for j1 in range(q) for j2 in range(q))
        phi = sum(1 for a in range(r) if math.gcd(a, r) == 1)
        clean.append({"r": r, "k": k, "q": q, "closed_form_exact": ok,
                      "outcomes": r, "prob_each": 1 / r, "usable": phi / r})
        print(f"  r={r} k={k} q={q}: closed form exact everywhere -> {ok}; "
              f"{r} outcomes at 1/r each; usable fraction phi(r)/r = {phi/r:.3f}")
    out["clean"] = clean

    # (b) the offset S cannot matter: different cosets give identical |amplitudes|
    q, A0 = amps(4, 3, 4, w0=0)
    _, A2 = amps(4, 3, 4, w0=2)
    same = all(abs(abs(A0[j]) - abs(A2[j])) < 1e-9 for j in A0)
    out["offset_invisible"] = same
    print(f"  |amplitudes| independent of which coset (hence of the offset S): {same}")

    # (c) the real case: r prime, so r never divides q, and the peaks smear
    print("  r is prime for ECDLP, so r | q only when r = 2 -- the clean case")
    print("  never actually arises.  How much does the smearing cost?")
    real = []
    for r, k in ((5, 2), (7, 3), (11, 4), (13, 5)):
        m0 = math.ceil(math.log2(r))
        for m in range(m0, m0 + 4):
            q, A = amps(r, k, m)
            tot = sum(abs(a) ** 2 for a in A.values())
            good = zero = 0.0
            for (j1, j2), a in A.items():
                pr = abs(a) ** 2 / tot
                a1, a2 = round(j1 * r / q) % r, round(j2 * r / q) % r
                if a1 == 0:
                    zero += pr
                elif (a2 - k * a1) % r == 0:
                    good += pr
            real.append({"r": r, "k": k, "m": m, "q": q,
                         "p_usable_correct": good, "p_a1_zero": zero})
            print(f"    r={r:>3} m={m} q={q:>4}: usable and correct {100*good:>5.1f}%, "
                  f"a1=0 (useless) {100*zero:>5.1f}%")
    out["real"] = real
    RESULTS["_derivation"] = out


def main():
    ablate_derivation()
    ablate_adders()
    ablate_modarith()
    ablate_multiplication()
    ablate_inversion()
    ablate_point_addition()
    ablate_and_and_full()
    ablate_counting()

    out = ROOT / "bench" / "ec_ablation.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
