"""Confirm that Part VII of shor-complete.tex still says what the benchmark measured.

Rather than scraping numbers out of the .tex and hoping to recognise them, this
regenerates the whole part from `part7_template.tex` + `ec_ablation.json` and
diffs it against what is actually in the document.  If they agree, then every
number in Part VII came from a benchmark run, by construction -- there is no
way for a hand-edit to drift.

Exit 0 on agreement, 1 otherwise, printing a unified diff.
"""
import difflib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

START = "\\part*{Part VII"
END = "\\begin{thebibliography}"


def extract(doc):
    s = doc.read_text()
    i = s.index(START)
    # back up to the \clearpage that opens the part
    i = s.rindex("\\clearpage", 0, i)
    j = s.index(END, i)
    return s[i:j].rstrip("\n") + "\n"


def main():
    gen = subprocess.run([sys.executable, str(ROOT / "bench" / "ec_make_part7.py")],
                         capture_output=True, text=True)
    if gen.returncode != 0:
        print("generator failed:\n" + gen.stderr)
        return 1
    want = gen.stdout.strip("\n") + "\n"
    have = extract(ROOT / "shor-complete.tex").strip("\n") + "\n"

    if want == have:
        nums = sum(c.isdigit() for c in have)
        print(f"OK: Part VII matches the generated version exactly "
              f"({len(have.splitlines())} lines, {nums} digits, all traceable to "
              f"bench/ec_ablation.json)")
        return 0

    print("MISMATCH: Part VII has drifted from the benchmark output.\n")
    for line in difflib.unified_diff(want.splitlines(), have.splitlines(),
                                     "generated", "in shor-complete.tex",
                                     lineterm="", n=2):
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
