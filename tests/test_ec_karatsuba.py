"""Symmetric and Karatsuba square-subtract [ECDSA.Fail Sec 5.3.4].

Exhaustive on toy primes: X -= ctrl * Z^2 mod p is exact for every (ctrl, Z,
X) with both forms, the wide fold matches the per-bit modular apply, the
windowed ECDLP circuit is exact on every basis state with each square
backend, and the full circuit recovers the logarithm on Aer.
"""
from _ec_util import aer_e2e, ok, section

import ec_classical as C
import ec_cost as CO
import ec_karatsuba as K
import ec_shor as S
import ec_window as W
from ec_sim import Machine, run

FORMS = (("reference", W._csub_square), ("symmetric", K.csub_square_symmetric),
         ("karatsuba", K.csub_square_karatsuba))


def build(f, p):
    n = p.bit_length()
    m = Machine("and")
    c, z, x = m.alloc(1, "c"), m.alloc(n, "z"), m.alloc(n, "x")
    f(m, c[0], z, x, p)
    return m, c, z, x


def main():
    section("square-subtract: exact on every (ctrl, z, x)")
    for p in (5, 7, 11, 13, 31):
        for name, f in FORMS:
            m, c, z, x = build(f, p)
            for cv in (0, 1):
                for zv in range(p):
                    for xv in range(p):
                        r = run(m, {c: cv, z: zv, x: xv})
                        assert (r(z), r(x)) == (zv, (xv - cv * zv * zv) % p), \
                            (name, p, cv, zv, xv)
        ok(f"p={p}: reference, symmetric, karatsuba exact")

    section("wide fold agrees with the per-bit modular apply")
    for p in (7, 13, 31):
        n = p.bit_length()
        for coeff in (1, 3, p - 1):
            for apply in (K._apply_bitwise, K._apply_wide):
                m = Machine("and")
                w, a = m.alloc(2 * n, "w"), m.alloc(n, "a")
                apply(m, w, a, coeff, p)
                for wv in range(0, 1 << (2 * n), 7):
                    for av in range(p):
                        r = run(m, {w: wv, a: av})
                        assert (r(w), r(a)) == (wv, (av - coeff * wv) % p)
    ok("fold == bitwise for p in 7, 13, 31")

    section("cost: symmetric halves the partial products")
    for p in (251, 65521, 2**31 - 1):
        row = []
        for name, f in FORMS:
            k = CO.count(build(f, p)[0])
            row.append(f"{name} {k['toffoli_paper']}/{k['qubits']}")
        ok(f"n={p.bit_length()}: " + ", ".join(row))
    ref = CO.count(build(W._csub_square, 2**31 - 1)[0])["toffoli_paper"]
    sym = CO.count(build(K.csub_square_symmetric, 2**31 - 1)[0])["toffoli_paper"]
    assert sym < 0.7 * ref, (sym, ref)

    section("end to end: Shor's ECDLP circuit with each square backend")
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    for square in ("symmetric", "karatsuba"):
        qc, info = S.ecdlp_circuit(
            curve, P, Q, order, oracle="arith", arithmetic="windowed",
            window_size=2, m_bits=2, square=square)
        mach = info["machine"]
        kr, lr, px, py = info["regs"]
        for u in range(1 << info["m_bits"]):
            for v in range(1 << info["m_bits"]):
                want = curve.add(curve.add(info["offset"], curve.mul(u, P)),
                                 curve.mul(v, Q))
                rd = run(mach, {kr: u, lr: v})
                assert (rd(px), rd(py)) == (want.x, want.y), (square, u, v)
        ok(f"{square}: all 16 basis states exact on {info['qubits']} qubits")
    try:
        S.ecdlp_circuit(curve, P, Q, order, oracle="arith", square="bogus",
                        arithmetic="windowed", window_size=2, m_bits=2)
        raise AssertionError("bad square accepted")
    except ValueError:
        pass

    section("Aer end to end: superposition, QFT, measurement (GPU if present)")
    for square in ("symmetric", "karatsuba"):
        for gcd in ("dialog", "jump2"):
            aer_e2e(f"{square}+{gcd} w=2 m=3", arithmetic="windowed",
                    window_size=2, m_bits=3, gcd=gcd, square=square)


if __name__ == "__main__":
    main()
    print("\ntest_ec_karatsuba: all passed")
