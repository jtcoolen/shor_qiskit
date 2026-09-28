"""g25_arith: Gidney 2025's subtract-and-underflow modular addition."""
import depth as D
import ec_cost as CO
import ec_mbu as MB
import g25_arith as GA
from ec_sim import Machine, SimError, run
from qiskit.circuit import QuantumCircuit, QuantumRegister
from rc_adder import rc_c_add_mod


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def main():
    for p in (13, 61):
        n = p.bit_length()
        costs = {}
        for mode in ("static", "mbu"):
            m = Machine("and")
            y, t2 = m.alloc(n, "y"), m.alloc(n, "t2")
            GA.sub_mod_underflow(m, y, t2, p, mode)
            for yv in range(p):
                for tv in range(1, p + 1):
                    rd = run(m, {y: yv, t2: tv})
                    assert (rd(y), rd(t2)) == ((yv - tv) % p, tv), (p, mode, yv, tv)
            c = CO.count(m)
            costs[mode] = (c["toffoli_paper"], c["mbu_toffoli"])
            if mode == "mbu":
                inputs = [{y: yv, t2: tv} for yv in range(0, p, 3) for tv in range(1, p + 1, 4)]
                MB.live_coherent(m.qc, inputs,
                                 [MB.all_ones_outcome(), MB.random_outcomes(1),
                                  MB.random_outcomes(2)], checks=m.checks)
        qc = QuantumCircuit(QuantumRegister(2, "c"), QuantumRegister(n, "y"),
                            QuantumRegister(2, "sf"), QuantumRegister(n + 2, "a"))
        rc_c_add_mod(qc, [qc.qubits[0], qc.qubits[1]], 5, qc.qubits[2:2 + n],
                     qc.qubits[2 + n], qc.qubits[3 + n], p, qc.qubits[4 + n:])
        old = D.profile(qc)["toffoli"]
        st, (mb, fix) = costs["static"][0], costs["mbu"]
        print(f"      p={p}: static {st}, measured {mb} worst / {mb - fix / 2:.0f} "
              f"expected Toffoli-eq; five-block ripple-carry adder {old}")
        assert st < old / 3
    ok("y - t2 mod p exact on every input (0 < t2 <= p), both forms; the measured "
       "form is phase-correct run literally; 4x cheaper than the five-block adder")

    m = Machine("and")
    p = 13
    y, t = m.alloc(4, "y"), m.alloc(4, "t")
    GA.add_mod_lookup(m, y, t, p)
    for yv in range(p):
        for tv in range(p):
            assert run(m, {y: yv, t: tv})(y) == (yv + tv) % p
    ok("add_mod_lookup: y + t mod p through the flipped value")


def test_windowed():
    import math
    import sparse_sim as SS
    from shor_stats import order_finding_probs
    for p, k, w in ((13, 5, 2), (21, 4, 2), (33, 7, 3)):
        n = p.bit_length()
        for ctl in (False, True):
            m = Machine("and")
            c = m.alloc(1, "c")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            GA.windowed_mult_add(m, c[0] if ctl else None, x, y, k, p, w)
            for cv in ((0, 1) if ctl else (1,)):
                for xv in range(p):
                    for yv in range(0, p, 3):
                        rd = run(m, {c: cv, x: xv, y: yv})
                        assert rd(y) == (yv + cv * k * xv) % p and rd(x) == xv
    ok("windowed multiply-add (flipped tables, control in the address, MBU "
       "unloads): exact for p = 13, 21, 33")
    for N, k in ((15, 7), (21, 2)):
        n = N.bit_length()
        m = Machine("and")
        c, x = m.alloc(1, "c"), m.alloc(n, "x")
        GA.c_mult_inplace(m, c[0], x, k, N, 2)
        for cv in (0, 1):
            for xv in range(N):
                if math.gcd(xv, N) == 1:
                    assert run(m, {c: cv, x: xv})(x) == pow(k, cv, N) * xv % N
    ok("in-place controlled multiplication: exact on the unit group, N = 15, 21")
    for A in (7, 2):
        m, info = GA.order_circuit_g25(A, 15)
        dist = SS.simulate_sparse(m.qc, checks=m.checks).distribution()
        ref = order_finding_probs(A, 15, info["t"])
        assert max(abs(dist.get(y, 0) - ref[y]) for y in range(1 << info["t"])) < 1e-10
    ok(f"order finding N = 15 on the G25 multiplier ({info['qubits']} qubits, "
       f"{CO.count(m)['toffoli_paper']} Toffoli-eq): the closed-form distribution")


def test_g25_arith():
    main()
    test_windowed()


if __name__ == "__main__":
    main()
    test_windowed()
    print("\ntest_g25_arith: all passed")
