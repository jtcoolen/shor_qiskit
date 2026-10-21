"""The Pareto filter excludes exactly the dominated resource triples."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from ec_pareto import GCDS, SQUARES, frontier


def test_frontier():
    rows = [
        {"label": "A", "qubits": 10, "toffoli_paper": 100, "depth": 50},
        {"label": "B", "qubits": 10, "toffoli_paper": 100, "depth": 60},
        {"label": "C", "qubits": 12, "toffoli_paper": 80, "depth": 40},
        {"label": "D", "qubits": 11, "toffoli_paper": 120, "depth": 51},
    ]
    assert frontier(rows) == ["A", "C"]
    report = json.loads((ROOT / "bench" / "ec_pareto.json").read_text())
    assert report["frontier"] == frontier(report["rows"])
    assert {(row["gcd"], row["square"]) for row in report["rows"]} == {
        (gcd, square) for gcd in GCDS for square in SQUARES
    }
    assert all(row["tvd"] <= report["tvd_limit"] for row in report["rows"])
    assert report["device"] in report["available_devices"]

    affine = json.loads((ROOT / "bench" / "ec_gu_affine.json").read_text())
    assert {row["label"] for row in affine["rows"]} == {
        "kaliski", "kaliski-opt", "divsteps"}
    assert affine["frontier"] == frontier(affine["rows"])
    assert all(row["basis_cases"] == 16 for row in affine["rows"])
    for row in affine["rows"]:
        if row["tvd"] is None:
            assert row["aer"].startswith("not sampled:")
        else:
            assert row["tvd"] <= affine["tvd_limit"]
            assert row["aer"].startswith("Aer ")
    assert next(row for row in affine["rows"] if row["label"] == "divsteps")[
        "tvd"] is not None


if __name__ == "__main__":
    test_frontier()
    print("test_ec_pareto: all passed")
