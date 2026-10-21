"""Regression pins: every existing ablation variant still costs what it cost.

Part VII of shor-complete.tex quotes these numbers, the presentation plots them
and the notebook prints them.  New optimisations are added *alongside* the
existing builders, never in place of them, so rebuilding every variant must
reproduce bench/ec_ablation.json exactly.  If this fails, either an existing
builder changed (don't) or the JSON is stale (re-run bench/ec_ablation.py and
regenerate Part VII).
"""
import contextlib
import io
import json

from _ec_util import ROOT, ok, section

import sys
sys.path.insert(0, str(ROOT / "bench"))
import ec_ablation as EA  # noqa: E402

PINNED = ("qubits", "toffoli_paper", "toffoli_equiv", "t", "and", "and_dg",
          "gates", "depth", "failure_rate")

ABLATIONS = ("ablate_adders", "ablate_modarith", "ablate_multiplication",
             "ablate_inversion", "ablate_point_addition",
             "ablate_and_and_full", "ablate_counting")


def main():
    ref = json.loads((ROOT / "bench" / "ec_ablation.json").read_text())

    section("rebuild every ablation variant")
    for fn in ABLATIONS:
        with contextlib.redirect_stdout(io.StringIO()):
            getattr(EA, fn)()
    got = {k: v for k, v in EA.RESULTS.items() if not k.startswith("_")}
    want = {k: v for k, v in ref.items() if not k.startswith("_")}
    ok(f"{len(got)} sections rebuilt")

    section("costs match bench/ec_ablation.json")
    assert set(got) == set(want), (
        f"sections differ: missing {sorted(set(want) - set(got))}, "
        f"extra {sorted(set(got) - set(want))}")
    diffs = []
    for sec in sorted(want):
        gv, wv = got[sec]["variants"], want[sec]["variants"]
        assert set(gv) == set(wv), (sec, sorted(set(gv) ^ set(wv)))
        for var in wv:
            for k in PINNED:
                if gv[var].get(k) != wv[var].get(k):
                    diffs.append((sec, var, k, gv[var].get(k), wv[var].get(k)))
    for d in diffs:
        print("  DIFF", *d)
    assert not diffs, f"{len(diffs)} pinned costs changed"
    n = sum(len(v["variants"]) for v in want.values())
    ok(f"{n} variants x {len(PINNED)} fields identical")


if __name__ == "__main__":
    main()
    print("\ntest_ec_regress: all passed")
