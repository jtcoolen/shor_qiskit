"""Measure opt-in toy divsteps round cuts as flagged inversion components.

These are NOT complete affine oracles: the compact point adder cannot
consume a failure flag without separately specifying exceptional behavior.
"""

import json
from pathlib import Path

from ec_cost import count
from ec_divsteps import mod_inv
from ec_divsteps_reference import walk
from ec_sim import Machine, run


OUT = Path(__file__).with_suffix(".json")


def measure():
    cases = []
    for p in (5, 7, 11):
        full = max(walk(w, p)[1] for w in range(1, p))
        for steps in (full - 1, full):
            m = Machine("and")
            control, x, out, flag = (
                m.alloc(1, "control"), m.alloc(p.bit_length(), "input"),
                m.alloc(p.bit_length(), "inverse"), m.alloc(1, "failure"))
            mod_inv(m, x, out, p, ctrl=control[0],
                    history="arithmetic", round_limit=steps, failure=flag[0])
            failed = 0
            for value in range(1, p):
                expected_failure = walk(value, p)[1] > steps
                result = run(m, {control: 1, x: value})
                assert result(flag) == int(expected_failure)
                assert result(out) == (
                    0 if expected_failure else pow(value, -1, p))
                failed += expected_failure
            cost = count(m.qc)
            cases.append({
                "p": p, "rounds": steps, "exact_rounds": full,
                "units_tested": p - 1, "failures": failed,
                "failure_rate": failed / (p - 1),
                "qubits": cost["qubits"],
                "toffoli_paper": cost["toffoli_paper"],
                "depth": cost["depth"],
                "scope": "controlled inversion component with explicit failure wire",
            })
    return {"rows": cases,
            "note": "No truncated schedule is enabled in the affine oracle; "
                    "component metrics are excluded from the full-oracle frontier."}


if __name__ == "__main__":
    OUT.write_text(json.dumps(measure(), indent=2) + "\n")
    print(f"Wrote {OUT}")
