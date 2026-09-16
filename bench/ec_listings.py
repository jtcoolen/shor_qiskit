"""Extract Qiskit listings for Part VII, verbatim from shor_qiskit/ec_*.py.

The document's standing promise is that every listing is copied from a file in
the package.  Copying by hand makes that a promise; extracting by AST makes it a
fact, and `ec_check_tex.py` then re-checks it on every run -- edit a function and
the document either follows or the check fails.

Two liberties are allowed, and both are visible in the output:

* `drop_doc` removes the docstring entirely (used where the prose around the
  listing already says what the docstring says, which in this document is most
  places);
* `elide` replaces a run of lines matching a marker with a bare `...`, the same
  convention the existing listings in Parts IV--VI use.

Nothing else is touched: indentation, comments and blank lines are as written.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "shor_qiskit"


def _node(mod, name):
    src = (SRC / f"{mod}.py").read_text()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n.name == name:
            return src.splitlines(), n
    raise KeyError(f"{mod}.{name} not found")


def code(mod, name, drop_doc=False, keep_doc=None, elide=(), tail=None):
    """The source of `mod.name` as a list of lines.

    drop_doc  remove the docstring
    keep_doc  keep only its first N lines, then `...`
    elide     [(first_line_substring, last_line_substring)] -> replaced by `...`
    tail      stop after the line containing this substring, then `...`
    """
    lines, node = _node(mod, name)
    body = lines[node.lineno - 1:node.end_lineno]

    doc = ast.get_docstring(node, clean=False)
    if doc is not None and (drop_doc or keep_doc is not None):
        d0 = node.body[0].lineno - node.lineno          # first docstring line
        d1 = node.body[0].end_lineno - node.lineno      # last docstring line
        indent = " " * (len(body[d0]) - len(body[d0].lstrip()))
        if drop_doc:
            body = body[:d0] + body[d1 + 1:]
        else:
            kept = body[d0:d0 + keep_doc]
            if keep_doc < (d1 - d0 + 1):
                kept = kept + [indent + '..."""']
            body = body[:d0] + kept + body[d1 + 1:]

    for first, last in elide:
        i = next((k for k, l in enumerate(body) if first in l), None)
        j = next((k for k, l in enumerate(body) if last in l and k >= (i or 0)), None)
        if i is None or j is None:
            raise KeyError(f"elide {first!r}..{last!r} not found in {mod}.{name}")
        indent = " " * (len(body[i]) - len(body[i].lstrip()))
        body = body[:i] + [indent + "..."] + body[j + 1:]

    if tail:
        i = next((k for k, l in enumerate(body) if tail in l), None)
        if i is None:
            raise KeyError(f"tail {tail!r} not found in {mod}.{name}")
        indent = " " * (len(body[i]) - len(body[i].lstrip()))
        body = body[:i + 1] + [indent + "..."]

    while body and not body[-1].strip():
        body.pop()
    return body


def listing(mod, name, **kw):
    """A complete lstlisting environment for `mod.name`."""
    return "\\begin{lstlisting}\n" + "\n".join(code(mod, name, **kw)) + "\n\\end{lstlisting}"


def snippet(mod, name, **kw):
    """Just the lines, for embedding in a hand-written listing."""
    return "\n".join(code(mod, name, **kw))


if __name__ == "__main__":
    import sys
    print(listing(*sys.argv[1:3], drop_doc="--drop" in sys.argv))
