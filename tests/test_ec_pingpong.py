"""Comparison-free ping-pong dialog-GCD [ECDSA.Fail, arXiv:2609.09582 Sec 5.3.3].

Exhaustive on toy primes: the classical walk inverts every unit, the circuit
multiplies and divides exactly on every (x != 0, y) with every ancilla clean,
the windowed point addition built on it is exact, and the whole ECDLP circuit
with `gcd="pingpong"` gives S + [u]P + [v]Q on every basis state.
"""
import math

from _ec_util import aer_e2e, ok, section

import ec_classical as C
import ec_cost as CO
import ec_eea as E
import ec_pingpong as PP
import ec_shor as S
import ec_window as W
from ec_sim import Machine, run


def build(fn, p):
    n = p.bit_length()
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    fn(m, x, y, p)
    return m, x, y


def main():
    section("classical walk: converges and inverts every unit")
    for p in (3, 5, 7, 11, 13, 31, 61, 127, 251):
        L = PP.pp_rounds(p)
        for x in range(1, p):
            a, b = PP.pp_inverse_classical(x, p, L)
            assert a == b == pow(x, -1, p), (p, x)
    # the paper's own example (Sec 5.3.3): p = 13, x = 6 needs 8 rounds
    (r0, r1), tau = PP.pp_walk(6, 13, PP.pp_rounds(13))
    assert abs(r0) == abs(r1) == 1
    # regression: the paper's average-case 2.75n is too small here
    assert PP.pp_rounds_needed(9, 16411) == 43 > math.ceil(2.75 * 15)
    assert PP.pp_rounds(16411) == 54
    m, x, y = build(PP.pp_mul, 16411)
    rd = run(m, {x: 9, y: 1})
    assert (rd(x), rd(y)) == (9, 9)
    ok("p = 16411: the exact budget (54) covers x = 9, which needs 43 rounds "
       "-- more than the paper's 2.75n = 42")
    ok("every x in [1, p) for p <= 251 is inverted by the round budget; "
       f"L(13) = {PP.pp_rounds(13)}")

    section("circuit: x*y and y/x on every basis state, ancillas clean")
    for p in (5, 7, 11, 13, 31):
        mm, x, y = build(PP.pp_mul, p)
        md, xd, yd = build(PP.pp_div, p)
        for xv in range(1, p):
            inv = pow(xv, -1, p)
            for yv in range(p):
                rd = run(mm, {x: xv, y: yv})
                assert (rd(x), rd(y)) == (xv, xv * yv % p), (p, xv, yv)
                rd = run(md, {xd: xv, yd: yv})
                assert (rd(xd), rd(yd)) == (xv, yv * inv % p), (p, xv, yv)
        ok(f"p={p}: {(p - 1) * p} products and quotients exact")

    section("cost against the [1128] Alg. 3 dialog")
    for p in (13, 61, 251):
        c = CO.count(build(PP.pp_mul, p)[0])
        d = CO.count(build(E.inplace_mul, p)[0])
        assert c["toffoli_paper"] < d["toffoli_paper"], (p, c, d)
        ok(f"p={p}: ping-pong {c['toffoli_paper']} Toffolis / {c['qubits']} "
           f"qubits, dialog {d['toffoli_paper']} / {d['qubits']}")

    section("windowed point addition on ping-pong")
    curve = C.CLASSIQ
    p, n = curve.p, curve.p.bit_length()
    wd = 2
    pts = W.window_points(curve, C.CLASSIQ_G, wd)
    m = Machine("and")
    addr = m.alloc(wd, "i")
    x2, y2 = m.alloc(n, "x2"), m.alloc(n, "y2")
    W.windowed_point_add(m, addr, x2, y2, pts, p, gcd="pingpong")
    good = 0
    for R in curve.points():
        if R.inf:
            continue
        for i in range(1 << wd):
            Pi = pts[i]
            if Pi.inf and R.x == 0:
                continue
            if not Pi.inf and C.point_add_exceptional(curve, R, Pi):
                continue
            want = R if Pi.inf else curve.add(R, Pi)
            if want.inf:
                continue
            rd = run(m, {addr: i, x2: R.x, y2: R.y})
            assert (rd(x2), rd(y2)) == (want.x, want.y), (R, i)
            assert rd(addr) == i
            good += 1
    ok(f"w={wd}: {good} windowed additions exact")

    section("end to end: Shor's ECDLP circuit with gcd='pingpong'")
    P, Q = C.CLASSIQ_G, C.CLASSIQ_Q
    order = curve.point_order(P)
    for ws, mbits in ((1, 1), (2, 2)):
        qc, info = S.ecdlp_circuit(
            curve, P, Q, order, oracle="arith", arithmetic="windowed",
            window_size=ws, m_bits=mbits, gcd="pingpong")
        mach = info["machine"]
        kr, lr, px, py = info["regs"]
        for u in range(1 << info["m_bits"]):
            for v in range(1 << info["m_bits"]):
                want = curve.add(curve.add(info["offset"], curve.mul(u, P)),
                                 curve.mul(v, Q))
                rd = run(mach, {kr: u, lr: v})
                assert (rd(px), rd(py)) == (want.x, want.y), (u, v)
                assert rd(kr) == u and rd(lr) == v
        assert qc.count_ops()["measure"] == 2 * info["m_bits"]
        ok(f"w={ws}, m={mbits}: all {1 << 2 * mbits} basis states exact on "
           f"{info['qubits']} qubits")
    try:
        S.ecdlp_circuit(curve, P, Q, order, oracle="arith", gcd="pingpong")
    except ValueError:
        pass
    else:
        raise AssertionError("gcd selection accepted for affine arithmetic")
    ok("gcd selection is rejected outside the windowed path")

    section("Aer end to end: superposition, QFT, measurement (GPU if present)")
    for gcd in ['pingpong']:
        for ws, mbits in ((2, 2), (2, 3)):
            aer_e2e(f"{gcd} w={ws} m={mbits}", arithmetic="windowed",
                    window_size=ws, m_bits=mbits, gcd=gcd)


if __name__ == "__main__":
    main()
    print("\ntest_ec_pingpong: all passed")
