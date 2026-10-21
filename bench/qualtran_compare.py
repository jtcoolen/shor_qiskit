"""Our circuits at n = 256, counted by Qualtran, beside Qualtran's own model.

`hier.to_qualtran` turns the hierarchically built circuit into Qualtran bloqs,
so `QECGatesCost` counts exactly what this package builds (and adds the
Clifford and measurement counts `ec_cost` does not keep).  Qualtran's library
`ECAdd` -- Litinski 2023's construction -- is costed symbolically at n = 256 as
the reference point for "before 2026".

    ./venv/bin/python bench/qualtran_compare.py        (needs `pip install qualtran`)
"""
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shor_qiskit"))
sys.path.insert(0, str(ROOT / "bench"))

import sympy
from qualtran.bloqs.cryptography.ecc import ECAdd
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

import ec_hier_256 as E
import ec_window as W
import hier as H


def main():
    out = {"n": E.N, "w": E.WBITS}
    n, p = sympy.symbols("n p", positive=True, integer=True)
    lit = ECAdd(n=n, mod=p, window_size=4)
    tof = get_cost_value(lit, QECGatesCost()).total_toffoli_only()
    qb = get_cost_value(lit, QubitCount())
    out["qualtran_litinski_ecadd"] = {"toffoli": int(sympy.N(tof.subs(n, 256))),
                                      "qubits": int(sympy.N(qb.subs(n, 256))),
                                      "formula": str(sympy.simplify(tof))}
    print(f"Qualtran ECAdd (Litinski 2023), n=256: "
          f"{out['qualtran_litinski_ecadd']['toffoli']:,} Toffolis, "
          f"{out['qualtran_litinski_ecadd']['qubits']} qubits")

    rng = random.Random(1)
    B = E.CURVE.mul(rng.randrange(1, E.ORDER), E.GEN)
    table, _ = W.masked_window_points(E.CURVE, B, E.WBITS, rng)
    out["ours"] = {}
    for name, cfg in E.configs().items():
        t = time.time()
        with H.tracing():
            m = H.HierMachine("and", "padd256")
            addr, x, y = m.alloc(E.WBITS, "a"), m.alloc(E.N, "x"), m.alloc(E.N, "y")
            W.windowed_point_add_cfg(m, addr, x, y, table, E.P, cfg)
            ours = H.count(m)
            gc = get_cost_value(H.to_qualtran(m, "padd"), QECGatesCost())
        assert gc.total_toffoli_only() == ours["toffoli_paper"]
        out["ours"][name] = {"toffoli": int(gc.total_toffoli_only()), "qubits": ours["qubits"],
                             "clifford": int(gc.clifford), "measurement": int(gc.measurement)}
        print(f"  {name:<58} {gc.total_toffoli_only():>10,} Toffolis  {ours['qubits']:>5} qubits  "
              f"{gc.clifford:>12,} Cliffords  {gc.measurement:>9,} measurements  ({time.time()-t:.0f} s)")
    out["published"] = {"Litinski 2023 (per addition, one instance)": 8.34e6,
                        "[1128] Schrottenloher": 2**21.19 + 3 * 2**16,
                        "IonQ": 1.196e6 + 3 * 2**16,
                        "Babbush et al. (<=, low-gate variant)": 2.1e6}
    (ROOT / "bench" / "qualtran_compare.json").write_text(json.dumps(out, indent=2))
    print("wrote bench/qualtran_compare.json")


if __name__ == "__main__":
    main()
