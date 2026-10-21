"""Gidney arXiv:2505.15917 App. A.1 approx_modexp toy on N=15, g=2, m=3."""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from rns_gidney import (ApproxModexpConfig, approx_modexp,
                        approx_period_circuit, fig3_reference,
                        ideal_joint_distribution, loop2_reference,
                        split_counts, unloop2_reference)
from shor_essentials import order_from_counts

SIM = AerSimulator(method="statevector", device="CPU")


def toy(mask_bits=0):
    return ApproxModexpConfig(15, 2, 3, (17, 23), 2, window1=2,
                              mask_bits=mask_bits)


def test_classical_tables():
    conf = toy()
    assert conf.crt_modulus == 391 and conf.truncated_modulus == 3
    assert [conf.product(e) for e in range(8)] == [pow(2, e, 15)
                                                  for e in range(8)]
    for i, p in enumerate(conf.primes):
        assert all(conf.residue(i, e) == pow(2, e, 15) % p for e in range(8))
    assert [conf.approximate(e) for e in range(8)] == [1, 0, 1, 2] * 2
    assert all(conf.deviation(e) <= 0.25 for e in range(8))
    wide = ApproxModexpConfig(15, 2, 3, (17, 23, 31), 2)
    assert [wide.approximate(e) for e in range(8)] == [
        fig3_reference(2, e, 15, [17, 23, 31], 2) >> 2 for e in range(8)]
    for i, p in enumerate(conf.primes):
        assert all(unloop2_reference(loop2_reference(v, p - 1, 5, 7),
                                     p - 1, 5, 7) == v for v in range(128))
        assert max(conf.dlog_sum(i, e) for e in range(8)) <= 2 * (p - 2)
        assert all(loop2_reference(d, p - 1, 5, 7) % 32 == d % (p - 1)
                   for d in range(2 * (p - 2) + 1))
    for bad in (dict(primes=(17, 19)), dict(primes=(17, 17)),
                dict(primes=(16, 23)), dict(kept_bits=3),
                dict(mask_bits=2)):
        args = dict(modulus=15, base=2, exponent_bits=3, primes=(17, 23),
                    kept_bits=2, window1=2)
        args.update(bad)
        try:
            ApproxModexpConfig(**args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {bad}")
    print("classical: Eq 20/51 mirrors, Fig 3, loop2 roundtrip, validation ok")


def run(qc, shots, seed, **kwargs):
    return SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots,
                   seed_simulator=seed, **kwargs).result()


def test_exhaustive_oracle():
    conf = toy()
    oracle, regs, _ = approx_modexp(conf)
    for e in range(1 << conf.exponent_bits):
        qc = QuantumCircuit(*oracle.qregs, *oracle.cregs)
        for b in range(conf.exponent_bits):
            if e >> b & 1:
                qc.x(regs["exponent"][b])
        qc.compose(oracle, inplace=True)
        for name, reg in regs.items():
            out = ClassicalRegister(len(reg), "o_" + name)
            qc.add_register(out)
            qc.measure(reg, out)
        seen = {tuple(sorted((k, v) for k, v in fields.items()
                             if k.startswith("o_")))
                for fields, _ in split_counts(run(qc, 12, e).get_counts(), qc)}
        want = {(("o_dlog", 0), ("o_exponent", e),
                 ("o_residue", 0), ("o_result", conf.approximate(e)))}
        assert seen == want, (e, seen, want)
    print("oracle: 8/8 basis inputs exact, wrap/dlog/residue cleared "
          "over random measurement branches")


def test_phase_cleanup():
    for mask_bits in (0, 1):
        conf = toy(mask_bits)
        oracle, regs, _ = approx_modexp(conf)
        qc = QuantumCircuit(*oracle.qregs, *oracle.cregs)
        qc.h(regs["exponent"])
        qc.compose(oracle, inplace=True)
        qc.save_statevector(pershot=True)
        shots = 16
        result = run(qc, shots, 7 + mask_bits)
        states = result.data(0)["statevector"]
        fired = split_counts(result.get_counts(), qc)
        assert any(f["wrap_mx"] for f, _ in fired), "no wrap phase fix fired"
        assert any(f["residue_mx"] for f, _ in fired), "no unloop3 fix fired"
        m, trunc = conf.exponent_bits, conf.truncated_modulus
        ideal = np.zeros(1 << qc.num_qubits, dtype=complex)
        for e in range(1 << m):
            for s in range(1 << mask_bits):
                v = (s + conf.approximate(e)) % trunc
                ideal[e | v << m] += 1
        ideal /= np.linalg.norm(ideal)
        fids = [abs(np.vdot(ideal, np.asarray(st))) ** 2 for st in states]
        assert len(fids) == shots and min(fids) > 1 - 1e-9, fids
        print(f"coherence mask_bits={mask_bits}: {shots} shot branches, "
              f"min fidelity {min(fids):.12f} with Eq 26 state")


def recover(conf, shots=256, seed=11):
    qc, _, _ = approx_period_circuit(conf)
    counts = run(qc, shots, seed).get_counts()
    joint, freq = {}, {}
    for fields, c in split_counts(counts, qc):
        key = (fields["approx"], fields["frequency"])
        joint[key] = joint.get(key, 0) + c
        bits = format(fields["frequency"], f"0{conf.exponent_bits}b")
        freq[bits] = freq.get(bits, 0) + c
    ideal = ideal_joint_distribution(conf)
    tv = 0.5 * sum(abs(joint.get(k, 0) / shots - ideal.get(k, 0))
                   for k in set(joint) | set(ideal))
    order = order_from_counts(freq, 2, 15, conf.exponent_bits)
    return order, tv, joint, freq, qc


def test_end_to_end():
    for mask_bits in (0, 1):
        conf = toy(mask_bits)
        order, tv, joint, freq, _ = recover(conf, seed=11 + mask_bits)
        assert order == 4, (order, freq)
        assert all(int(y, 2) % 2 == 0 for y in freq), freq
        assert tv < 0.12, tv
        print(f"Aer e2e mask_bits={mask_bits}: order {order}, "
              f"frequencies {dict(sorted(freq.items()))}, TV {tv:.3f}")


def main():
    test_classical_tables()
    test_exhaustive_oracle()
    test_phase_cleanup()
    test_end_to_end()


if __name__ == "__main__":
    main()
