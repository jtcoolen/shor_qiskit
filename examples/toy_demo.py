"""Toy RSA and toy elliptic-curve keys, broken by Shor on Aer.

    ./venv/bin/python examples/toy_demo.py              # RSA N=15, ECDLP mod 11
    ./venv/bin/python examples/toy_demo.py --N 21       # 1-2 min: shot by shot
    ./venv/bin/python examples/toy_demo.py --full       # full counting registers

Toy sizes, obviously: a 4-bit modulus and a 13-element group.  The point is
the whole pipeline running end to end -- key, circuit, measurement, classical
post-processing, private key -- with every measured histogram set beside the
exact distribution it should follow.  By default both attacks use the
semiclassical circuits: one recycled counting qubit instead of 2n.
"""
import argparse
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "shor_qiskit"))

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

import ec_classical as C
import ec_shor as S
import shor_stats as ST
from onectrl import order_circuit_1c
from shor_essentials import order_circuit, order_from_counts


def rsa(N, message, shots, seed, full):
    print(f"\n=== toy RSA, N = {N} ===")
    p = next(d for d in range(3, N) if N % d == 0)       # the key owner's secret
    phi = (p - 1) * (N // p - 1)
    e = next(e for e in range(3, phi, 2) if math.gcd(e, phi) == 1)
    c = pow(message, e, N)
    print(f"public key (N={N}, e={e});  ciphertext c = {message}^{e} mod {N} = {c}")

    n = math.ceil(math.log2(N))
    t = 2 * n
    pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
    sampler = Sampler(seed=seed)
    rnd = random.Random(seed)
    bases = [A for A in range(2, N - 1) if math.gcd(A, N) == 1]
    rnd.shuffle(bases)          # a base sharing a factor would give the answer
    for A in bases:             # by gcd alone -- skipped, to run the quantum part
        qc = order_circuit(A, N) if full else order_circuit_1c(A, N)
        print(f"\nbase A = {A}: order finding on {qc.num_qubits} qubits "
              f"({'full register' if full else 'one counting qubit'}), {shots} shots")
        counts = ST.order_counts(sampler.run([pm.run(qc)], shots=shots)
                                 .result()[0].data.out.get_counts())
        exact = ST.order_finding_probs(A, N, t)
        emp = ST.empirical(counts, 2**t)
        print(f"  {'y':>5} {'shots':>6} {'measured':>9} {'exact':>7}   y/2^t ~ s/r")
        for y, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
            fr = ST.nearest_fraction(y, t, N)
            print(f"  {y:5d} {cnt:6d} {emp[y]:9.4f} {exact[y]:7.4f}   "
                  f"{y / 2**t:.4f} ~ {fr.numerator}/{fr.denominator}")
        hit = ST.order_success_mask(A, N, t)
        print(f"  TVD from exact {ST.tvd(emp, exact):.3f} (shot-noise bound "
              f"{ST.tvd_null(exact, shots):.3f}); one shot yields r with "
              f"P = {exact[hit].sum():.3f} exact, {emp[hit].sum():.3f} measured")
        r = order_from_counts({format(y, f"0{t}b"): v for y, v in counts.items()},
                              A, N, t)
        if not r or r % 2:
            print(f"  r = {r}: odd or not found, next base")
            continue
        f = math.gcd(pow(A, r // 2, N) - 1, N)
        if not 1 < f < N:
            print(f"  r = {r}, but A^(r/2) = -1 mod N: trivial gcd, next base")
            continue
        q = N // f
        d = pow(e, -1, (f - 1) * (q - 1))
        m = pow(c, d, N)
        print(f"  r = {r}  ->  gcd({A}^{r//2} - 1, {N}) = {f}  ->  N = {f} x {q}")
        print(f"private exponent d = {e}^-1 mod {(f-1)*(q-1)} = {d};  "
              f"decrypted m = {c}^{d} mod {N} = {m}")
        assert m == message
        return
    raise RuntimeError("every base failed")


def ecdlp(curve, G, shots, seed, full):
    r = curve.point_order(G)
    ax = "x" if curve.a == 1 else f"{curve.a}x"
    print(f"\n=== toy ECDLP, {curve.name}: y^2 = x^3 + {ax} + {curve.b} "
          f"mod {curve.p} ===")
    k = random.Random(seed).randrange(1, r)              # the private key
    Q = curve.mul(k, G)
    print(f"generator G = ({G.x}, {G.y}) of order {r};  "
          f"public key Q = [k]G = ({Q.x}, {Q.y})")
    counts, cands, info = S.run_ecdlp(curve, G, Q, r, shots=shots, seed=seed,
                                      one_control=not full)
    mb, q = info["m_bits"], 1 << info["m_bits"]
    print(f"\nShor on {info['qubits']} qubits "
          f"({'full registers' if full else 'one counting qubit'}), {shots} shots")
    pairs = S.pair_counts(counts)
    exact = ST.ecdlp_probs(r, k, mb)
    emp = ST.empirical(pairs, (q, q))
    print(f"  {'(j1, j2)':>9} {'shots':>6} {'measured':>9} {'exact':>7}   votes for k =")
    for (j1, j2), cnt in sorted(pairs.items(), key=lambda kv: -kv[1])[:6]:
        v = C.ecdlp_postprocess({(j1, j2): 1}, r, mb, search=0)
        print(f"  {str((j1, j2)):>9} {cnt:6d} {emp[j1, j2]:9.4f} {exact[j1, j2]:7.4f}"
              f"   {v[0][0] if v else '-'}")
    hit = ST.ecdlp_success_mask(r, k, mb)
    print(f"  TVD from exact {ST.tvd(emp, exact):.3f} (shot-noise bound "
          f"{ST.tvd_null(exact, shots):.3f}); one shot yields k with "
          f"P = {exact[hit].sum():.3f} exact, {emp[hit].sum():.3f} measured")
    total = sum(w for _, w in cands)
    print("  candidates by vote: " + ", ".join(
        f"k={kk} ({w / total:.0%})" for kk, w in cands[:4]))
    found = next(kk for kk, _ in cands if C.verify_dlog(curve, G, Q, kk))
    print(f"private key recovered: k = {found}  (check: [{found}]G == Q)")
    assert found == k


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=15, help="RSA modulus: 15, 21, 33, 35")
    ap.add_argument("--message", type=int, default=2)
    ap.add_argument("--curve", choices=("p7", "p11"), default="p11")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--full", action="store_true",
                    help="full counting registers instead of one recycled qubit")
    a = ap.parse_args()
    rsa(a.N, a.message % a.N, a.shots, a.seed, a.full)
    curve, G = {"p7": (C.CLASSIQ, C.CLASSIQ_G), "p11": (C.TOY11, C.TOY11_G)}[a.curve]
    ecdlp(curve, G, a.shots, a.seed, a.full)
