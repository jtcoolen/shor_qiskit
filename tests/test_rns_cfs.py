"""Tests for the CFS ePrint 2024/222 low-bit RNS toy (shor_qiskit/rns_cfs.py)."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from rns_cfs import (CfsConfig, CfsFactoringToy, cfs_factoring_circuit,
                     cfs_multiproduct, cfs_order_circuit, eh_postprocess, eh_score,
                     ideal_distribution, ideal_order_distribution,
                     order_postprocess, register_values, simulate_basis,
                     split_counts)

GOOD = dict(modulus=15, base=2, primes=(7, 11, 13, 19), out_bits=2, beta=1)
BAD = dict(modulus=15, base=2, primes=(3, 5, 7, 11), out_bits=2, beta=1)
N21 = CfsConfig(21, [pow(2, 1 << i, 21) for i in range(5)],
                (5, 7, 11, 13, 29), 2, 2)
MPS = AerSimulator(method="matrix_product_state", device="CPU")


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def test_parameters_and_eq27_cofactors():
    toy = CfsFactoringToy(**GOOD)
    c = toy.config()
    assert (toy.n, toy.half, toy.ell, toy.a_bits, toy.b_bits) == (4, 2, 2, 4, 2)
    assert toy.A == pow(2, 7 - 2, 15)
    assert c.multipliers == (2, 4, 1, 1, 8, 4)    # G^{2^i}, A^{-2^j}
    assert c.M == 7 * 11 * 13 * 19 and c.M > 2 * 4 * 8 * 4
    assert c.u == (49 + 121 + 169 + 361 - 1).bit_length()          # Lemma 5
    assert c.t_N == (2 * 4 * 8 * 4 * 15 - 1).bit_length()          # Prop. 1
    assert (c.qm_width, c.qn_width) == (c.u + c.q_bits + 1, c.u_prime + 2)
    for p in c.primes:
        assert c.Mp(p) * c.wp(p) % p == 1
        assert c.B(p) == c.Mp(p) * c.wp(p) % 4
        for i in range(p.bit_length()):
            assert c.C(p, i) == (2**i * c.wp(p) * (2**c.u // p)) % 2**c.qm_width
            assert c.Dp(p, i) == (2**i * c.Mp(p) * c.wp(p) * (2**c.t_N // 15)
                                  >> (c.t_N - c.u_prime)) % 2**c.qn_width
    for x in range(64):                              # Lemma 7 residues
        for j, p in enumerate(c.primes):
            assert c.residue(j, x) == c.multi_product(x) % p


VALIDATION = [
    (dict(modulus=16, multipliers=[3], primes=(5, 7)), "odd"),
    (dict(modulus=15, multipliers=[3], primes=(5, 7)), "Lemma 6"),
    (dict(modulus=15, multipliers=[7], primes=(7, 11)), "zero-residue"),
    (dict(modulus=15, multipliers=[2, 4, 8], primes=(3, 5)), "Remark 1"),
    (dict(modulus=15, multipliers=[2], primes=(2, 5)), "odd primes"),
    (dict(modulus=15, multipliers=[2], primes=(5, 5)), "odd primes"),
    (dict(modulus=15, multipliers=[2], primes=(9, 5)), "odd primes"),
    (dict(modulus=15, multipliers=[2], primes=(3, 5, 7, 11, 13, 17, 19, 23,
                                                29, 31, 37, 41)), "t_N"),
    ]


def raises(exc, match, fn):
    try:
        fn()
    except exc as e:
        assert match in str(e), (match, str(e))
        return
    raise AssertionError(f"{exc.__name__} not raised")


def test_validation():
    for kwargs, msg in VALIDATION:
        raises(ValueError, msg, lambda: CfsConfig(out_bits=2, **kwargs))


def test_bad_beta_and_type():
    raises(ValueError, "beta", lambda: CfsFactoringToy(**{**GOOD, "beta": 4}))
    raises(ValueError, "beta", lambda: CfsFactoringToy(**{**GOOD, "beta": 0}))
    raises(TypeError, "CfsConfig", lambda: cfs_multiproduct(GOOD))


def test_failure_classification():
    good = CfsFactoringToy(**GOOD).config()
    assert good.failures(internal=True) == {}
    bad = CfsFactoringToy(**BAD).config()
    out = bad.failures()
    assert sorted(x for x, w in out.items() if w == "heuristic3") == \
        [0, 4, 8, 12, 17, 21, 25, 29, 34, 38, 42, 46]
    assert sorted(x for x, w in out.items() if w == "lemma5") == [51, 55, 59, 63]
    internal = bad.failures(internal=True)
    assert set(internal) - set(out) == {19, 23, 27, 31, 49, 50, 53, 54, 57,
                                        58, 61, 62}
    for x in out:
        assert bad.algorithm1(x)["res"] != bad.exact_output(x)
    # nu = 2 hides the Heuristic 3 output errors in this instance.
    nu2 = CfsFactoringToy(**{**BAD, "nu": 2}).config()
    assert all(w == "lemma5" for w in nu2.failures(internal=True).values())
    # Lemma 5 is tight: shrinking u breaks qM.
    tiny = CfsConfig(15, good.multipliers, good.primes, 2, 1, u_override=4)
    assert set(tiny.failures(internal=True).values()) == {"lemma5"}


def check_circuit_exhaustive_clean(kwargs):
    """Every basis input: Res == Algorithm 1 (including its failures), x
    preserved, every scratch register returned to |0>."""
    conf = CfsFactoringToy(**kwargs).config()
    qc, regs, tally = cfs_multiproduct(conf)
    lanes = 1 << conf.m
    inputs = {q: [(x >> i) & 1 for x in range(lanes)]
              for i, q in enumerate(regs["x"])}
    state = simulate_basis(qc, inputs)
    assert register_values(state, regs["x"], lanes) == list(range(lanes))
    assert register_values(state, regs["res"], lanes) == \
        [conf.algorithm1(x)["res"] for x in range(lanes)]
    for name in ("s", "residue", "qm", "qn", "anc"):
        assert register_values(state, regs[name], lanes) == [0] * lanes, name
    assert len(regs["res"]) == conf.out_bits
    assert (len(regs["qm"]), len(regs["qn"])) == (conf.qm_width, conf.qn_width)


def test_circuit_exhaustive_clean():
    check_circuit_exhaustive_clean(GOOD)
    check_circuit_exhaustive_clean(BAD)


def test_simulate_basis_rejects_non_classical():
    qc = QuantumCircuit(1)
    qc.h(0)
    raises(ValueError, "non-classical",
           lambda: simulate_basis(qc, {qc.qubits[0]: [0]}))


def test_resource_tally():
    conf = CfsFactoringToy(**GOOD).config()
    qc, regs, tally = cfs_multiproduct(conf)
    P = len(conf.primes)
    assert tally.residue_computations == 2 * P
    assert tally.lookups == 4 * P                     # Sec. 6.1: 4 |P|
    assert tally.lookup_entries == 4 * sum(conf.s_max(j) + 1 for j in range(P))
    assert tally.constant_additions == 2
    assert qc.num_qubits == 56
    assert {"x", "cx", "ccx"} <= set(qc.count_ops())
    assert not any(i.operation.name == "measure" for i in qc.data)


def test_coherent_uncompute_interference():
    """H, oracle, oracle^-1, H returns |0..0> with certainty: the oracle is a
    coherent permutation with clean scratch (no garbage phases)."""
    conf = CfsFactoringToy(**GOOD).config()
    oracle, regs, _ = cfs_multiproduct(conf)
    qc = QuantumCircuit(*oracle.qregs)
    qc.h(regs["x"])
    qc.compose(oracle, inplace=True)
    qc.z(regs["res"][0])                    # phase oracle (-1)^{Res_0}
    qc.compose(oracle.inverse(), inplace=True)
    qc.h(regs["x"])
    qc.measure_all()
    counts = MPS.run(transpile(qc, MPS, optimization_level=0), shots=400,
                     seed_simulator=3).result().get_counts()
    f = np.array([conf.algorithm1(x)["res"] & 1 for x in range(64)])
    expect0 = abs(np.sum((-1.0) ** f) / 64) ** 2
    zeros = "0" * qc.num_qubits
    # all scratch is zero in every shot; x-register distribution is the WHT.
    nx = conf.m
    assert all(k[:-nx] == zeros[:-nx] for k in counts)
    assert abs(counts.get(zeros, 0) / 400 - expect0) < 0.1


def test_eh_algorithm2_end_to_end_n15():
    toy = CfsFactoringToy(**GOOD)
    qc, regs, tally = cfs_factoring_circuit(toy)
    assert qc.num_qubits == 57
    assert [c.name for c in qc.cregs] == ["j", "k"]
    counts = MPS.run(transpile(qc, MPS, optimization_level=0), shots=2000,
                     seed_simulator=7).result().get_counts()
    assert all(len(k.split()) == 2 and len(k.split()[0]) == toy.b_bits
               and len(k.split()[1]) == toy.a_bits for k in counts)
    outcomes = split_counts(counts)
    ideal = ideal_distribution(toy)
    emp = np.zeros_like(ideal)
    for (j, k), c in outcomes.items():
        emp[j, k] += c / 2000
    assert _tv(ideal, emp) < 0.05
    assert set(outcomes) <= {tuple(i) for i in np.argwhere(ideal > 1e-9)}
    assert abs(ideal[0, 0] - 0.625) < 1e-9
    # Theorem 3: P(0) <= 1/2 + eps + P_uncompressed(0)/2, here 1/4.
    assert ideal[0, 0] <= 0.5 + 0.01 + 0.25 / 2
    # The good-pair criterion singles out d = 1 on the measured data: every
    # non-zero shot is a good pair for it, every other d scores strictly less.
    scores = {d: eh_score(toy, outcomes, d) for d in range(-1, 1 << toy.half)}
    assert scores[1] == 2000 - outcomes.get((0, 0), 0)
    assert all(v < scores[1] for d, v in scores.items() if d != 1)
    assert eh_postprocess(toy, outcomes) == (3, 5, 1)
    # ... and outcomes that are no good pair for d = 1 are rejected.
    assert eh_postprocess(toy, {(2, 0): 1}) is None
    assert eh_postprocess(toy, {(0, 0): 9}) is None


def test_eh_ideal_matches_exact_when_alg1_exact():
    toy = CfsFactoringToy(**GOOD)
    assert np.allclose(ideal_distribution(toy, True),
                       ideal_distribution(toy, False))
    bad = CfsFactoringToy(**BAD)
    assert not np.allclose(ideal_distribution(bad, True),
                           ideal_distribution(bad, False))


def test_order_finding_n21_end_to_end():
    """Toy Shor order finding of 2 mod 21 through the compressed CFS oracle
    (deviation: not Algorithm 2).  61 qubits, Aer MPS."""
    assert N21.failures(internal=True) == {}
    qc, regs, _ = cfs_order_circuit(N21, beta=1)
    assert qc.num_qubits == 61
    counts = MPS.run(transpile(qc, MPS, optimization_level=0), shots=2000,
                     seed_simulator=7).result().get_counts()
    emp = np.zeros(32)
    for k, c in counts.items():
        emp[int(k, 2)] += c / 2000
    assert _tv(ideal_order_distribution(N21, 1), emp) < 0.06
    assert order_postprocess(counts, 2, 21, 5) == 6


def test_order_postprocess_reduces_multiple():
    assert order_postprocess({"01001": 5}, 2, 21, 5) == 6   # 9/32~5/18 -> 6
    assert order_postprocess({"11011": 5, "00101": 1}, 2, 21, 5) == 6
    assert order_postprocess({"00000": 9}, 2, 21, 5) == 0


def main():
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"ok {name} ({time.time() - t:.1f}s)")
    print("test_rns_cfs: all passed")


if __name__ == "__main__":
    main()
