"""Every name the notebook, presentation, examples and bench scripts use still exists.

Those consumers import functions from shor_qiskit/ by name -- some of them
private helpers -- so a rename breaks them silently until someone re-executes
them.  This parses each consumer, resolves which local module every
`Alias.name` and `from module import name` refers to, and asserts the name is
still there.  It imports nothing from the consumers themselves.
"""
import ast
import importlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL = {p.stem for d in ("shor_qiskit", "tests") for p in (ROOT / d).glob("*.py")}


def consumers():
    for pat in ("presentation/*.py", "examples/*.py", "bench/*.py"):
        for p in sorted(ROOT.glob(pat)):
            yield str(p.relative_to(ROOT)), p.read_text()
    nb = ROOT / "notebooks" / "shor_walkthrough.ipynb"
    cells = json.loads(nb.read_text())["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join("" if re.match(r"\s*[%!]", l) else l for l in src.splitlines())
        yield f"{nb.relative_to(ROOT)}[cell {i}]", src


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _local_bindings(fn):
    """Names a function binds itself (args, assignments), not via import."""
    bound = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    return bound


def uses(tree):
    """(module, name) pairs referenced through imports of local modules.

    Scope-aware in the one way that matters: a function that rebinds an alias
    (`A = {...}` where A is also `import ec_adders as A`) is not referring to
    the module when it writes `A.items()`.
    """
    alias, out = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in LOCAL:
                    alias[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module in LOCAL and not node.level:
            for a in node.names:
                if a.name != "*":
                    out.add((node.module, a.name))

    def visit(node, shadowed):
        if isinstance(node, SCOPES):
            shadowed = shadowed | _local_bindings(node)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in alias and node.value.id not in shadowed):
            out.add((alias[node.value.id], node.attr))
        for child in ast.iter_child_nodes(node):
            visit(child, shadowed)

    visit(tree, frozenset())
    return out


def main():
    missing, total = [], 0
    for where, src in consumers():
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"{where} does not parse: {e}")
        for mod, name in sorted(uses(tree)):
            total += 1
            if not hasattr(importlib.import_module(mod), name):
                missing.append(f"{where}: {mod}.{name}")
    for m in missing:
        print("  MISSING", m)
    assert not missing, f"{len(missing)} names used by consumers no longer exist"
    print(f"  ok  {total} name references resolve")


def test_api_surface():
    main()


if __name__ == "__main__":
    import sys
    sys.path[:0] = [str(ROOT / "shor_qiskit"), str(ROOT / "tests")]
    main()
    print("\ntest_api_surface: all passed")
