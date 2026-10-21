"""The coset representation, completed: encoding circuit and order finding."""
import numpy as np

import coset as CS
import ec_cost as CO
import sparse_sim as SS
from ec_sim import Machine
from shor_stats import order_finding_probs


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def main():
    for N, cpad, k in ((15, 3, 1), (15, 3, 7), (21, 2, 5), (33, 2, 10)):
        W = CS.coset_width(N, cpad)
        m = Machine("and")
        r = m.alloc(W, "r")
        CS.encode_circuit(m, r, k, N, cpad)
        br = SS.simulate_sparse(m.qc, checks=m.checks).only()
        v = np.zeros(1 << W, complex)
        for (val,), a in br.amplitudes(r).items():
            v[val] = a
        assert abs(abs(np.vdot(CS.coset_state(k, N, cpad), v)) ** 2 - 1) < 1e-12
    ok("encode_circuit prepares the coset state exactly (replaces initialize), "
       "every ancilla clean")

    prev = 1.0
    for cpad in (2, 3, 4):
        m, info = CS.order_circuit_coset(7, 15, cpad)
        dist = SS.simulate_sparse(m.qc, checks=m.checks).distribution()
        ref = order_finding_probs(7, 15, info["t"])
        tvd = 0.5 * sum(abs(dist.get(y, 0) - ref[y]) for y in range(1 << info["t"]))
        bound = CS.deviation_bound(info["additions"], cpad)
        print(f"      N=15 cpad={cpad}: TVD {tvd:.3f} (bound {bound:.2f}), "
              f"{info['qubits']} qubits, {CO.count(m)['toffoli_paper']} Toffoli-eq")
        assert tvd <= min(bound, 1.0) and tvd < prev
        prev = tvd
    ok("coset order finding: plain additions only, deviation falls with the "
       "padding and stays within Gidney-Ekera's bound")


def test_coset_order():
    main()


if __name__ == "__main__":
    main()
    print("\ntest_coset_order: all passed")
