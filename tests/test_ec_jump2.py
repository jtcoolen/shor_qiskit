"""Jump-2 Euclidean dialog and base-5 codec [ECDSA.Fail Sec 5.3.1-5.3.2].

Exhaustive on toy primes: the classical macro-step walk and replay multiply
correctly, the codec is a bijection on the reachable alphabet that clears its
inputs, the circuit multiplies and divides exactly on every (x != 0, y) with
and without the codec, and the windowed ECDLP circuit is exact end to end.
"""
import itertools

from _ec_util import aer_e2e, ok, section

import ec_classical as C
import ec_cost as CO
import ec_eea as E
import ec_jump2 as J
import ec_shor as S
from ec_sim import Machine, run


def build(p, codec=False, div=False):
    n = p.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    (J.j2_div if div else J.j2_mul)(m, x, y, p, codec=codec)
    return m, x, y


def main():
    section("classical macro-steps: exact replay, ~1.3n steps")
    for p in (5, 7, 11, 13, 31, 61, 127, 251, 1021):
        L = J.j2_rounds(p)
        for x in range(1, p):
            (u, v), tau = J.j2_walk(x, p, L)
            assert (u, v) == (1, 0)
            for sym in tau:
                assert sym[:3] in J.SYMBOLS, sym
            for y in (0, 1, p - 1, x):
                assert J.j2_replay_mul(tau, y, p) == (0, x * y % p)
    big = 131101
    assert J.j2_rounds(big) == 35            # n = 18: ceil(1.5n) + 8
    for x in (1, 2, 3, big - 1, 65536, 12345):
        assert J.j2_rounds_needed(x, big) <= J.j2_rounds(big)
    ok(f"every unit for p <= 1021; every symbol in the 5-letter alphabet; "
       f"L(1021) = {J.j2_rounds(1021)} macro-steps")

    section("base-5 codec: 9 qubits -> 7, inputs returned clean")
    m = Machine("and")
    syms = [m.alloc(3, f"s{k}") for k in range(3)]
    z = m.alloc(7, "z")
    J.encode_triple(m, syms, z)
    for t in itertools.product(J.SYMBOLS, repeat=3):
        rd = run(m, {syms[k]: t[k][0] | t[k][1] << 1 | t[k][2] << 2
                     for k in range(3)})
        assert rd(z) == sum(J.digit(*t[k]) * 5 ** k for k in range(3))
        assert all(rd(s) == 0 for s in syms)
    assert sorted(J.digit(*s) for s in J.SYMBOLS) == [0, 1, 2, 3, 4]
    ok("all 125 reachable triples encode to d0 + 5 d1 + 25 d2 < 128")

    section("circuit: x*y and y/x on every basis state, ancillas clean")
    for p in (5, 7, 11, 13, 31):
        for codec in (False, True):
            mm, x, y = build(p, codec)
            md, xd, yd = build(p, codec, div=True)
            for xv in range(1, p):
                inv = pow(xv, -1, p)
                for yv in range(p):
                    rd = run(mm, {x: xv, y: yv})
                    assert (rd(x), rd(y)) == (xv, xv * yv % p), (p, xv, yv)
                    rd = run(md, {xd: xv, yd: yv})
                    assert (rd(xd), rd(yd)) == (xv, yv * inv % p)
            ok(f"p={p} {'codec' if codec else 'raw  '}: "
               f"{(p - 1) * p} products and quotients exact")

    section("cost against the [1128] Alg. 3 dialog")
    for p in (61, 251):
        c = CO.count(build(p)[0])
        n = p.bit_length()
        m = Machine("and")
        x, y = m.alloc(n, "x"), m.alloc(n, "y")
        E.inplace_mul(m, x, y, p)
        d = CO.count(m)
        assert c["toffoli_paper"] < d["toffoli_paper"]
        assert c["qubits"] <= d["qubits"]
        ok(f"p={p}: Jump-2 {c['toffoli_paper']} Toffolis / {c['qubits']} "
           f"qubits, dialog {d['toffoli_paper']} / {d['qubits']}")

    section("end to end: Shor's ECDLP circuit with gcd='jump2[-codec]'")
    curve, P, Q = C.CLASSIQ, C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    for gcd in ("jump2", "jump2-codec"):
        qc, info = S.ecdlp_circuit(
            curve, P, Q, order, oracle="arith", arithmetic="windowed",
            window_size=2, m_bits=2, gcd=gcd)
        mach = info["machine"]
        kr, lr, px, py = info["regs"]
        for u in range(1 << info["m_bits"]):
            for v in range(1 << info["m_bits"]):
                want = curve.add(curve.add(info["offset"], curve.mul(u, P)),
                                 curve.mul(v, Q))
                rd = run(mach, {kr: u, lr: v})
                assert (rd(px), rd(py)) == (want.x, want.y), (gcd, u, v)
        assert qc.count_ops()["measure"] == 2 * info["m_bits"]
        ok(f"{gcd}: all 16 basis states exact on {info['qubits']} qubits")

    section("Aer end to end: superposition, QFT, measurement (GPU if present)")
    for gcd in ['jump2', 'jump2-codec']:
        for ws, mbits in ((2, 2), (2, 3)):
            aer_e2e(f"{gcd} w={ws} m={mbits}", arithmetic="windowed",
                    window_size=ws, m_bits=mbits, gcd=gcd)


if __name__ == "__main__":
    main()
    print("\ntest_ec_jump2: all passed")
