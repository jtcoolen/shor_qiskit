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
        "kaliski", "kaliski-opt", "divsteps", "divsteps-decoded",
        "divsteps-arithmetic", "packed-euclid"}
    assert affine["frontier"] == frontier(affine["rows"])
    assert all(row["basis_cases"] == 16 for row in affine["rows"])
    for row in affine["rows"]:
        if row["tvd"] is None:
            assert row["aer"].startswith("not sampled:")
        else:
            assert row["tvd"] <= affine["tvd_limit"]
            assert row["aer"].startswith("Aer ")
    for backend in ("divsteps", "divsteps-decoded", "divsteps-arithmetic",
                    "packed-euclid"):
        assert next(row for row in affine["rows"] if row["label"] == backend)[
            "tvd"] is not None
    assert {row["backend"] for row in affine["additional_validation"]} == {
        "divsteps", "divsteps-decoded", "packed-euclid"}
    for extra in affine["additional_validation"]:
        assert extra["task"].startswith("affine arithmetic ECDLP p=7")
        assert extra["tvd"] <= extra["tvd_limit"]
        assert extra["aer"].startswith("Aer ")

    from plot_optimisation_frontiers import edwards_kim_rows
    kim = json.loads((ROOT / "bench" / "ec_edwards_kim.json").read_text())
    assert len(kim["oracles"]) == 26
    assert all(row["basis_verified_all_64"] for row in kim["oracles"])
    for mode in ("toffoli", "and"):
        rows, winners = edwards_kim_rows(mode)
        assert len(rows) == len(kim["oracles"])
        assert winners == set(frontier(rows))
        assert winners == {
            "Horner inplace", "QROM bennett compact w3 load",
            "QROM bennett paper w3 load",
        }
        for row in rows:
            if row["sampled"]:
                original = next(
                    variant for variant in kim["oracles"]
                    if variant[mode]["qubits"] == row["qubits"]
                    and variant[mode]["toffoli_paper"] == row["toffoli_paper"])
                shot = next(run for run in original["aer_runs"]
                            if run["mode"] == mode)
                assert shot["recovered_k"] == kim["curve"]["k"]
                assert shot["scratch_all_zero_shots"] == shot["shots"]
                assert shot["tvd"] <= shot["tvd_noise_bound"]
        assert not next(row for row in rows if row["label"] ==
                        "QROM bennett paper w3 load")["sampled"]


if __name__ == "__main__":
    test_frontier()
    print("test_ec_pareto: all passed")
