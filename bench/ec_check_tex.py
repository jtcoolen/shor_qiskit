"""Confirm that Parts VII and VIII of shor-complete.tex still say what the
benchmarks measured.

Rather than scraping numbers out of the .tex and hoping to recognise them, this
regenerates each part from its template (`part7_template.tex`,
`part8_template.tex`) and the benchmark JSON files and diffs it against what is
actually in the document.  If they agree, then every number in those parts
came from a benchmark run, by construction -- there is no way for a hand-edit
to drift.

Exit 0 on agreement, 1 otherwise, printing a unified diff.
"""
import difflib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PARTS = [("\\part*{Part VII", "ec_make_part7.py"),
         ("\\part*{Part VIII", "make_part8.py")]
END = "\\begin{thebibliography}"


def extract(doc, k):
    """Part k's text: from the \\clearpage that opens it to the one that opens
    the next part (or the bibliography)."""
    s = doc.read_text()
    i = s.rindex("\\clearpage", 0, s.index(PARTS[k][0]))
    if k + 1 < len(PARTS) and PARTS[k + 1][0] in s:
        j = s.rindex("\\clearpage", 0, s.index(PARTS[k + 1][0]))
    else:
        j = s.index(END, i)
    return s[i:j].rstrip("\n") + "\n"


def check(k):
    start, script = PARTS[k]
    name = start.split("{")[1]
    gen = subprocess.run([sys.executable, str(ROOT / "bench" / script)],
                         capture_output=True, text=True)
    if gen.returncode != 0:
        print(f"{name}: generator failed:\n" + gen.stderr)
        return 1
    want = gen.stdout.strip("\n") + "\n"
    have = extract(ROOT / "shor-complete.tex", k).strip("\n") + "\n"
    if want == have:
        nums = sum(c.isdigit() for c in have)
        print(f"OK: {name} matches the generated version exactly "
              f"({len(have.splitlines())} lines, {nums} digits, all traceable to "
              f"bench/*.json)")
        return 0
    print(f"MISMATCH: {name} has drifted from the benchmark output.\n")
    for line in difflib.unified_diff(want.splitlines(), have.splitlines(),
                                     "generated", "in shor-complete.tex",
                                     lineterm="", n=2):
        print(line)
    return 1


def main():
    text = (ROOT / "shor-complete.tex").read_text()
    return sum(check(k) for k in range(len(PARTS)) if PARTS[k][0] in text)


if __name__ == "__main__":
    sys.exit(main())
