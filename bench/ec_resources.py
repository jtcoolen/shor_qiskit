"""Resource measurements for the ECDLP circuits.

Everything printed under MEASURED is counted off a circuit this package builds
and the test suite verifies.  Everything under PROJECTED is the papers' own
formula evaluated at cryptographic n, and is not verified by anything here.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))

import ec_classical as C
import ec_cost as CO
import ec_eea as E
import ec_kaliski as K
import ec_kaliski_opt as KO
import ec_montgomery as MG
import ec_mult as MU
import ec_pointadd as PA
import ec_proj as PJ
from ec_sim import Machine


def build(fn):
    m = Machine("and")
    fn(m)
    return m


def hdr(t):
    print(f"\n{t}\n{'=' * len(t)}")


def main():
    hdr("MEASURED -- modular inversion: [HJN+20] vs [106] Sec 3.3")
    rows = []
    for p in (31, 127, 251, 1021, 4093, 65521):
        n = p.bit_length()
        a = build(lambda m: K.mod_inv(m, m.alloc(n, "x"), m.alloc(n, "o"), p))
        b = build(lambda m: KO.mod_inv_mont(m, m.alloc(n, "x"), m.alloc(n, "o"), p))
        ca, cb = CO.count(a), CO.count(b)
        rows.append([n, ca["toffoli_equiv"], cb["toffoli_equiv"],
                     f"{100*(1-cb['toffoli_equiv']/ca['toffoli_equiv']):.0f}%",
                     ca["qubits"], cb["qubits"],
                     ca["depth"], cb["depth"],
                     f"{100*(1-cb['depth']/ca['depth']):.0f}%"])
    print(CO.table(rows, ["n", "ref Toff", "opt Toff", "saved",
                          "ref qb", "opt qb", "ref depth", "opt depth", "saved"]))
    print("  [106] reports 58-60% T-depth improvement for division at n=192-521;")
    print("  the trend here is the same, against a different baseline and at toy n.")

    hdr("MEASURED -- modular multiplication")
    rows = []
    for n, w, p in ((8, 2, 251), (8, 4, 251), (12, 4, 4093), (16, 4, 65521), (16, 8, 65521)):
        sb = build(lambda m: MU.modmul(m, m.alloc(n, "a"), m.alloc(n, "b"),
                                       m.alloc(n, "o"), p))
        mq = build(lambda m: MG.mont_mul_qcsa(m, m.alloc(n, "a"), m.alloc(n, "b"),
                                              m.alloc(n, "o"), p, w))
        ml = build(lambda m: MG.mont_mul_lookup(m, m.alloc(n, "a"), m.alloc(n, "b"),
                                                m.alloc(n, "o"), p, w))
        for lab, mm in (("schoolbook", sb), ("Montgomery+QCSA [106]", mq),
                        ("Montgomery+QROM [HJN+20]", ml)):
            c = CO.count(mm)
            rows.append([n, w, lab, c["qubits"], c["toffoli_equiv"], c["t"], c["depth"]])
    print(CO.table(rows, ["n", "w", "multiplier", "qubits", "Toffoli-eq", "T", "depth"]))
    print("  The QCSA multiplier trades qubits and gates for DEPTH, which is")
    print("  [106]'s stated objective; at these toy widths the depth win is the")
    print("  only one visible, because the carry-save tree has almost nothing to")
    print("  flatten when there are two or four words.")

    hdr("MEASURED -- point addition, all three constructions")
    rows = []
    for curve in (C.CLASSIQ, C.TOY11):
        p, n = curve.p, curve.p.bit_length()
        aff = build(lambda m: PA.point_add_ctrl(
            m, m.alloc(1, "q")[0], m.alloc(n, "x"), m.alloc(n, "y"), 2, 1, p))
        prj = build(lambda m: PJ.jacobian_add(
            m, m.alloc(n, "X1"), m.alloc(n, "Y1"), m.alloc(n, "Z1"),
            2, 1, m.alloc(n, "X3"), m.alloc(n, "Y3"), m.alloc(n, "Z3"), p))
        for lab, mm in (("affine in-place [106] Alg 3", aff),
                        ("Jacobian out-of-place [106] Alg 4", prj)):
            c = CO.count(mm)
            rows.append([f"p={p}", lab, c["qubits"], c["toffoli_equiv"],
                         c["t"], c["depth"]])
    print(CO.table(rows, ["curve", "construction", "qubits", "Toffoli-eq", "T", "depth"]))
    print("  The projective addition is far cheaper and needs no inversion, but")
    print("  leaves its 3n-qubit input behind; the affine one is in place.")

    hdr("MEASURED -- in-place multiplication: division vs Euclidean dialog")
    rows = []
    for q in (7, 11, 13, 31, 61):
        n = q.bit_length()
        dv = build(lambda m: K.mod_div(m, m.alloc(1, "c")[0], m.alloc(n, "x"),
                                       m.alloc(n, "y"), m.alloc(n, "o"), q))
        ip = build(lambda m: E.inplace_mul(m, m.alloc(n, "x"), m.alloc(n, "y"), q))
        cd, ci = CO.count(dv), CO.count(ip)
        rows.append([n, cd["qubits"], cd["toffoli_equiv"], ci["qubits"],
                     ci["toffoli_equiv"],
                     f"{100*(1-ci['toffoli_equiv']/cd['toffoli_equiv']):.0f}%"])
    print(CO.table(rows, ["n", "div qb", "div Toff", "dialog qb", "dialog Toff",
                          "dialog saves"]))
    print("  [1128]'s point is structural rather than arithmetic: one dialog pass")
    print("  does the inversion AND the multiplication, and the same circuit run")
    print("  backwards divides.  Two of the affine addition's expensive steps")
    print("  collapse into one reusable component.")

    hdr("MEASURED -- AND vs Toffoli, i.e. what the temporary AND buys")
    rows = []
    for p in (31, 251, 4093):
        n = p.bit_length()
        ma = build(lambda m: K.mod_inv(m, m.alloc(n, "x"), m.alloc(n, "o"), p))
        c = CO.count(ma)
        toff_only = c["toffoli_equiv"] * CO.TOFFOLI_T
        rows.append([n, c["and"], c["and_dg"], c["t"], toff_only,
                     f"{toff_only / max(1, c['t']):.2f}x"])
    print(CO.table(rows, ["n", "AND", "AND-dg", "T (with temp AND)",
                          "T (all Toffoli)", "saving"]))

    hdr("MEASURED -- Shor ECDLP, both oracles")
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    import ec_shor as S
    m, info = S.ecdlp_circuit(curve, P, Q, order, oracle="arith")
    c = CO.count(m)
    print(f"  arithmetic oracle : {c['qubits']} qubits, {c['toffoli_equiv']} "
          f"Toffoli-eq, {c['gates']} gates, depth {c['depth']}")
    qc, info2 = S.ecdlp_circuit(curve, P, Q, order, oracle="table")
    print(f"  table oracle      : {qc.num_qubits} qubits, {len(qc.data)} gates "
          f"-- simulable end to end")

    hdr("PROJECTED -- the papers' formulas at cryptographic sizes (NOT verified here)")
    rows = []
    for n in (192, 224, 256, 384, 521):
        w = 11 if n <= 256 else 13
        adds, mregs = CO.proj_106_additions(n, w)
        rows.append([n, w, adds, mregs, mregs * 3 * n])
    print(CO.table(rows, ["n", "w", "windowed additions", "zig-zag registers",
                          "garbage qubits (mG)"]))
    print("  [106] Sec 4.2: 2*ceil((n+1)/w) additions, m(m+1)/2 >= that.")
    print("  Its own worked example (n=192, w=11 -> 36 additions, m=8) is asserted")
    print("  in tests/test_ec_classical.py.")

    rows = []
    for n in (256,):
        sp = CO.proj_eea_space(n)
        rows.append([n, sp["record_raw"], sp["record_compressed"],
                     f"{sp['record_compressed_per_n']:.3f}n",
                     sp["with_register_sharing"],
                     CO.proj_1128_qubits(n, "space"), CO.proj_1128_qubits(n, "gate")])
    print()
    print(CO.table(rows, ["n", "record raw", "record compressed", "per n",
                          "w/ reg sharing", "total (space-opt)", "total (gate-opt)"]))
    print("  [1128] Table 1 reports 1192 and 1446 qubits for secp256k1.  Note the")
    print("  totals come from the paper's closed form 4.355n + O(sqrt n), NOT from")
    print("  summing the columns to their left -- those carry the O(sqrt n) margin")
    print("  explicitly and give a larger figure.  Neither checks the other.")
    print("  'w/ reg sharing' is the Sec 3.1 optimization this package does NOT")
    print("  implement: its failure mode is a register overflow rather than a wrong")
    print("  answer, which the exact-simulation harness cannot meaningfully check.")
    import math
    print(f"  [1128] eq. (1) full Shor Toffoli count: "
          f"2^{math.log2(CO.proj_1128_shor_toffoli()):.2f} -- the paper's own equation")
    print("  evaluated at the paper's own per-addition figure, restated for scale.")
    print("  Nothing here measures a 256-bit point addition.")


if __name__ == "__main__":
    main()
