"""The rest of the [106] and ECDSA.Fail catalogues, measured.

None of these moves a headline number; each is built, tested (tests/test_ec_*),
and measured here so Part VII can say what it costs:

  cla         [DKRS04] carry-lookahead adder and comparator (ec_cla): Toffolis,
              depth, ancillas against Gidney's ripple, n = 8 .. 256
  edwards     [106] Sec 4.4: twisted Edwards (Ed25519-shaped) addition against
              the Weierstrass Jacobian one (ec_edwards, ec_proj), n = 32, 64
  projective  [106] Alg 4 with a quantum addend (ec_proj_q) against the affine
              in-place addition, n = 32
  census      ECDSA.Fail's fire census (ec_opt): Toffolis and failure on fresh
              inputs against the sample size, on a toy dialog multiplier
  signed      odd/signed windows (ec_signedwin), when present
  batch       Montgomery's batch inversion (ec_batch), when present
  luo         the register-shared EEA (ec_luo), when present

    ./venv/bin/python bench/ec_catalogue.py      # writes bench/ec_catalogue.json
"""
import importlib.util
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "tests"))

import depth
import ec_adders as A
import ec_cla as CL
import ec_cost as CO
from ec_sim import Machine, SimError, run


def cla():
    out = []
    for n in (8, 32, 64, 128, 256):
        row = {"n": n}
        for lab, fn in (("cla", lambda m, x, y: CL.cla_add(m, x, y)),
                        ("gidney", lambda m, x, y: A.gidney_add(m.ctx, x, y, m.anc(n - 1, "g")))):
            m = Machine("and")
            x, y = m.alloc(n, "x"), m.alloc(n, "y")
            fn(m, x, y)
            c = CO.count(m)
            row[lab] = {"toffoli": c["toffoli_paper"], "qubits": c["qubits"],
                        "depth": depth.toffoli_depth(m.qc)}
        out.append(row)
    return out


def edwards():
    import ec_edwards as ED
    import ec_proj as PJ
    out = []
    for p in (2 ** 32 - 99, 2 ** 64 - 59):
        n = p.bit_length()
        d = (-121665 * pow(121666, -1, p)) % p
        row = {"n": n}

        def trip(m, k, nm):
            return tuple(m.alloc(n, f"{c}{nm}") for c in "XYZT"[:k])
        m = Machine("and")
        P1, P3 = trip(m, 4, "1"), trip(m, 4, "3")
        ED.edwards_add_const(m, P1, 3, 7, P3, p, d)
        c = CO.count(m)
        row["edwards_const"] = {"toffoli": c["toffoli_paper"], "qubits": c["qubits"]}
        m = Machine("and")
        P1, P3 = trip(m, 3, "1"), trip(m, 3, "3")
        PJ.jacobian_add(m, *P1, 3, 7, *P3, p)
        c = CO.count(m)
        row["jacobian_const"] = {"toffoli": c["toffoli_paper"], "qubits": c["qubits"]}
        out.append(row)
    return out


def projective():
    import ec_proj_q as PQ
    rows = PQ.cost_rows(2 ** 32 - 5)
    return [{"label": lab, "n": n, "qubits": q, "toffoli": t} for lab, n, q, t in rows]


def census():
    import ec_gcd as G
    import ec_opt as O
    rnd = random.Random(7)
    q, n = 61, 6
    m = Machine("and")
    x, y = m.alloc(n, "x"), m.alloc(n, "y")
    G.Dialog(fused_cmp=True).mul(m, x, y, q)
    everything = [(a, b) for a in range(1, q) for b in range(q)]
    fresh = rnd.sample(everything, 400)

    def rate(mk):
        bad = 0
        for a, b in fresh:
            try:
                rd = run(mk, {x: a, y: b})
                bad += (rd(x), rd(y)) != (a, a * b % q)
            except SimError:
                bad += 1
        return bad / len(fresh)
    out = {"base": CO.count(m)["toffoli_paper"]}
    full = O.strip_unfired(m, O.census(m, [{x: a, y: b} for a, b in everything]))
    out["all_inputs"] = CO.count(full)["toffoli_paper"]
    out["samples"] = []
    for k in (10, 30, 100, 300):
        mk = O.strip_unfired(m, O.census(m, [{x: a, y: b} for a, b in rnd.sample(everything, k)]))
        out["samples"].append({"k": k, "toffoli": CO.count(mk)["toffoli_paper"],
                               "fresh_wrong": rate(mk)})
    return out


def extra_batch():
    import ec_batch as EB
    out = []
    for n, k, kind in ((64, 2, "inv"), (64, 4, "inv"), (64, 8, "inv"), (256, 4, "inv"),
                       (256, 4, "padd")):
        c = EB.batch_costs(n, k, kind, hier=True)
        out.append({"n": n, "k": k, "kind": kind, **c})
    return out


def extra_luo():
    import ec_gcd as G
    import ec_luo as LU
    import ec_luo_classical as LC
    import hier as H
    out = []
    for n in (32, 64, 128, 256):
        import ec_batch as EB
        p = EB.prime_below(1 << n)
        row = {"n": n, "formula": LC.luo_qubits(n)}
        with H.tracing():
            for lab, fn in (("inverse", lambda m, x: LU.luo_inv(m, x, p)),
                            ("division", lambda m, x: LU.luo_div(m, x, m.alloc(n, "y"), p)),
                            ("dialog", lambda m, x: G.Dialog(fused_cmp=True).div(
                                m, x, m.alloc(n, "y"), p))):
                m = H.HierMachine("and")
                x = m.alloc(n, "x")
                fn(m, x)
                c = H.count(m)
                row[lab] = {"qubits": c["qubits"], "toffoli": c["toffoli_paper"]}
        out.append(row)
    return out


def optional(name):
    return importlib.util.find_spec(name) is not None


def main():
    data = {"cla": cla(), "edwards": edwards(), "projective": projective(), "census": census()}
    for mod, key in (("ec_signedwin", "signed"), ("ec_batch", "batch"), ("ec_luo", "luo")):
        fn = globals().get(f"extra_{key}")
        if optional(mod) and fn:
            data[key] = fn()
    (ROOT / "bench" / "ec_catalogue.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=1)[:3000])


if __name__ == "__main__":
    main()
