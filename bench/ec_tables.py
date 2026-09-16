"""Emit the LaTeX table bodies used by Part VII of shor-complete.tex.

The tables in the note are generated from `ec_ablation.json`, never typed by
hand, and `ec_check_tex.py` afterwards verifies that what is in the .tex still
matches what the benchmark measured.  Re-run the benchmark, re-run this, paste,
re-run the checker.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "bench" / "ec_ablation.json").read_text())


def g(sec, var, field):
    return D[sec]["variants"][var][field]


def num(v):
    """LaTeX thin-space thousands separators, matching the note's style."""
    if v is None:
        return "---"
    s = str(int(v))
    out = ""
    while len(s) > 3:
        out = "\\," + s[-3:] + out
        s = s[:-3]
    return s + out


def pct(new, old):
    return f"${100*(1-new/old):.0f}\\%$"


def emit(title, body):
    print(f"\n%%% ---- {title} ---- %%%")
    print(body.rstrip())


# ---------------------------------------------------------------- adders
rows = []
for n in (8, 16, 32):
    a, b = D[f"adder_n{n}"]["variants"], D[f"cadder_n{n}"]["variants"]
    rows.append(
        f"${n}$ & ${a['CDKM [CDKM04]']['toffoli_paper']}$ & "
        f"${a['Gidney [Gid18]']['toffoli_paper']}$ & "
        f"${b['reconstruct (CDKM)']['toffoli_paper']}$ & "
        f"${b['copy-then-add [106] Fig 4b']['toffoli_paper']}$ & "
        f"${a['CDKM [CDKM04]']['qubits']}$ & ${a['Gidney [Gid18]']['qubits']}$\\\\")
emit("adders", "\n".join(rows))

# ------------------------------------------------- modular arithmetic
rows = []
for q in (251, 1021, 4093):
    dv, cv = D[f"moddbl_q{q}"]["variants"], D[f"cmodadd_q{q}"]["variants"]

    def pick(v, key):
        for name in v:
            if key in name:
                return v[name]
        return None
    for label, key in (("exact (Alg.~5/8)", "exact"),
                       ("approx., 4 MSBs (Alg.~6/9)", "msbs=4"),
                       ("pseudo-Mersenne (Alg.~7/10)", "pseudo-Mersenne")):
        dd, cc = pick(dv, key), pick(cv, key)
        if dd is None:
            continue
        rows.append(
            f"${q}$ & {label} & ${dd['toffoli_paper']}$ & "
            f"${100*dd['failure_rate']:.1f}\\%$ & "
            f"${cc['toffoli_paper']}$ & ${100*cc['failure_rate']:.1f}\\%$\\\\")
emit("modular arithmetic (exact / approx / pseudo-Mersenne)", "\n".join(rows))

# ------------------------------------------------------- multiplication
rows = []
for sec, n, w, p in (("mul_n8_w2", 8, 2, 251), ("mul_n12_w4", 12, 4, 4093),
                     ("mul_n16_w4", 16, 4, 65521)):
    v = D[sec]["variants"]
    sb = v["schoolbook double-and-add"]
    qc = v[f"Montgomery+QCSA [106] w={w}"]
    qr = v[f"Montgomery+QROM [HJN+20] w={w}"]
    rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{$n={n}$, $w={w}$, $p={p}$}}}}\\\\")
    for nm, c in (("schoolbook double-and-add", sb),
                  ("Montgomery over a carry-save tree", qc),
                  ("Montgomery over a QROM lookup", qr)):
        speed = "---" if c is sb else f"${sb['depth']/c['depth']:.1f}\\times$"
        rows.append(f"\\quad {nm} & ${num(c['toffoli_paper'])}$ & ${c['qubits']}$ & "
                    f"${num(c['depth'])}$ & {speed}\\\\")
emit("multiplication", "\n".join(rows))

# ------------------------------------------------------------ inversion
rows = []
for n, p in ((5, 31), (7, 127), (8, 251), (10, 1021), (12, 4093)):
    v = D[f"inv_n{n}"]["variants"]
    r, o = v["reference Kaliski [HJN+20]"], v["unconditional+postponed [106]"]
    rows.append(
        f"${n}$ & ${num(r['toffoli_paper'])}$ & ${num(o['toffoli_paper'])}$ & "
        f"{pct(o['toffoli_paper'], r['toffoli_paper'])} & "
        f"${num(r['depth'])}$ & ${num(o['depth'])}$ & "
        f"{pct(o['depth'], r['depth'])} & ${r['qubits']}$ & ${o['qubits']}$\\\\")
emit("inversion: reference vs unconditional+postponed", "\n".join(rows))

# ------------------------------------- division vs dialog (the real task)
rows = []
for key, n, p in (("inplace_n3_p7", 3, 7), ("inplace_n4_p11", 4, 11),
                  ("inplace_n5_p31", 5, 31), ("inplace_n6_p61", 6, 61)):
    v = D[key]["variants"]
    d, e = v["Kaliski division (out-of-place)"], v["EEA dialog + Bezout replay [1128]"]
    rows.append(
        f"${p}$ & ${num(d['toffoli_paper'])}$ & ${d['qubits']}$ & "
        f"${num(e['toffoli_paper'])}$ & ${e['qubits']}$ & "
        f"{pct(e['toffoli_paper'], d['toffoli_paper'])} & "
        f"{pct(e['qubits'], d['qubits'])}\\\\")
emit("division vs Euclidean dialog", "\n".join(rows))

# -------------------------------------------------------- point addition
rows = []
for key, label in (("padd_p=7", "$p=7$"), ("padd_p=11", "$p=11$")):
    v = D[key]["variants"]
    a = v["affine in-place [106] Alg 3"]
    j = v["Jacobian out-of-place [106] Alg 4"]
    rows.append(f"{label} & affine, in place (Alg.~3) & ${num(a['toffoli_paper'])}$ & "
                f"${a['qubits']}$ & ${num(a['depth'])}$ & ---\\\\")
    rows.append(f"      & Jacobian, out of place (Alg.~4) & ${num(j['toffoli_paper'])}$ & "
                f"${j['qubits']}$ & ${num(j['depth'])}$ & "
                f"{pct(j['toffoli_paper'], a['toffoli_paper'])}\\\\")
w = D["padd_windowed_p7"]["variants"]["windowed w=2 [1128] Alg 1"]
a7 = D["padd_p=7"]["variants"]["affine in-place [106] Alg 3"]
rows.append(f"$p=7$ & windowed $w=2$, dialog mult.\\ (Alg.~1) & ${num(w['toffoli_paper'])}$ & "
            f"${w['qubits']}$ & ${num(w['depth'])}$ & "
            f"{pct(w['toffoli_paper'], a7['toffoli_paper'])}\\\\")
emit("point addition: three constructions", "\n".join(rows))

# --------------------------------------------------------- temporary AND
rows = []
for n, p in ((5, 31), (8, 251), (12, 4093)):
    v = D[f"tempand_n{n}"]["variants"]
    allt, tmp = v["all Toffoli (7 T each)"], v["temporary AND (4 T / 0 T)"]
    rows.append(f"${n}$ & ${num(tmp['and'])}$ & ${num(tmp['and_dg'])}$ & "
                f"${num(allt['t'])}$ & ${num(tmp['t'])}$ & "
                f"${allt['t']/tmp['t']:.2f}\\times$\\\\")
emit("temporary ANDs", "\n".join(rows))

# ------------------------------------------------------- windowing/zigzag
proj = D["_projections"]["windowing_zigzag"]
rows = []
for n in (192, 256, 384, 521):
    got = {r["w"]: r for r in proj if r["n"] == n}
    r1, r11, r16 = got[1], got[11], got[16]
    rows.append(f"${n}$ & ${num(r1['additions'])}$ & ${r11['additions']}$ & "
                f"${r16['additions']}$ & ${r1['zigzag_registers']}$ & "
                f"${r11['zigzag_registers']}$ & ${r16['zigzag_registers']}$\\\\")
emit("windowing and zig-zag (counting, not circuits)", "\n".join(rows))

# --------------------------------------------------------- full circuit
v = D["full_ecdlp"]["variants"]
ar = v["arithmetic oracle, full control registers"]
tb = v["table oracle (simulable)"]
sc = v["table oracle, semiclassical (1 ctrl qubit)"]
emit("full ECDLP circuit", "\n".join([
    f"arithmetic oracle, two $\\lceil\\log_2 r\\rceil$-qubit registers & ${ar['qubits']}$ & "
    f"${num(ar['toffoli_paper'])}$ & ${num(ar['gates'])}$ & ${num(ar['depth'])}$\\\\",
    f"permutation oracle, same register layout & ${tb['qubits']}$ & --- & "
    f"${num(tb['gates'])}$ & ${num(tb['depth'])}$\\\\",
    f"permutation oracle, semiclassical QFT & ${sc['qubits']}$ & --- & "
    f"${num(sc['gates'])}$ & ${num(sc['depth'])}$\\\\"]))

# ------------------------------------------------------------ dialog space
sp = D["_projections"]["eea_space"]
rows = []
for n in ("256", "384", "521"):
    s = sp[n]
    rows.append(f"${n}$ & ${num(s['record_raw'])}$ & ${num(s['record_compressed'])}$ & "
                f"${s['record_compressed_per_n']:.3f}n$ & "
                f"${num(s['with_register_sharing'])}$\\\\")
emit("dialog space", "\n".join(rows))
