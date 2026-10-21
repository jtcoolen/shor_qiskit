"""Generate Part VII of shor-complete.tex, with every number pulled from
`ec_ablation.json`.  Run this, paste the output into the document, then run
`ec_check_tex.py` to confirm the document still matches the benchmark.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
D = json.loads((ROOT / "bench" / "ec_ablation.json").read_text())
PARETO = json.loads((ROOT / "bench" / "ec_pareto.json").read_text())
GU_AFFINE = json.loads((ROOT / "bench" / "ec_gu_affine.json").read_text())
V = lambda s: D[s]["variants"]

import ec_listings as L


def expand_listings(text):
    """Replace %%LST module.func [opts]%% with the function, verbatim from source.

    Options: `drop` (no docstring), `doc=N` (keep N docstring lines),
    `elide=A..B` (replace that run with `...`), `tail=S` (stop after S).
    """
    def one(m):
        spec = m.group(1).split()
        mod, fn = spec[0].split(".")
        kw, elide = {}, []
        for opt in spec[1:]:
            if opt == "drop":
                kw["drop_doc"] = True
            elif opt.startswith("doc="):
                kw["keep_doc"] = int(opt[4:])
            elif opt.startswith("elide="):
                a, b = opt[6:].split("..")
                elide.append((a.replace("~", " "), b.replace("~", " ")))
            elif opt.startswith("tail="):
                kw["tail"] = opt[5:].replace("~", " ")
        if elide:
            kw["elide"] = elide
        return L.listing(mod, fn, **kw)
    return re.sub(r"%%LST ([^%]+)%%", one, text)


def num(v):
    if v is None:
        return "---"
    s, out = str(int(v)), ""
    while len(s) > 3:
        out, s = "\\," + s[-3:] + out, s[:-3]
    return s + out


def rows_gu_affine():
    rows = []
    for row in GU_AFFINE["rows"]:
        sample = (f"TVD ${row['tvd']:.3f}$"
                  if row["tvd"] is not None else "not sampled")
        rows.append(
            f"{row['label']} & ${row['qubits']}$ & "
            f"${num(row['toffoli_paper'])}$ & ${num(row['depth'])}$ & "
            f"{sample} & "
            f"{'yes' if row['label'] in GU_AFFINE['frontier'] else 'no'}\\\\")
    return "\n".join(rows)


def pct(new, old):
    return f"{100*(1-new/old):.0f}\\%"


def rows_deriv_clean():
    out = []
    for c in D["_derivation"]["clean"]:
        out.append(f"${c['r']}$ & ${c['k']}$ & ${c['q']}$ & ${c['outcomes']}$ & "
                   f"$1/{c['r']}$ & \\checkmark & ${c['usable']:.2f}$\\\\")
    return "\n".join(out)


def rows_deriv_real():
    out = []
    by_r = {}
    for e in D["_derivation"]["real"]:
        by_r.setdefault(e["r"], []).append(e)
    for r, es in sorted(by_r.items()):
        cells = " & ".join(f"${100*e['p_usable_correct']:.0f}\\%$" for e in es)
        out.append(f"${r}$ & " + cells + f" & ${100*es[-1]['p_a1_zero']:.0f}\\%$\\\\")
    return "\n".join(out)


def rows_adders():
    out = []
    for n in (8, 16, 32):
        a, b = V(f"adder_n{n}"), V(f"cadder_n{n}")
        out.append(f"${n}$ & ${a['CDKM [CDKM04]']['toffoli_paper']}$ & "
                   f"${a['Gidney [Gid18]']['toffoli_paper']}$ & "
                   f"${b['reconstruct (CDKM)']['toffoli_paper']}$ & "
                   f"${b['copy-then-add, CDKM inside']['toffoli_paper']}$ & "
                   f"${b['copy-then-add [106] Fig 4b']['toffoli_paper']}$ & "
                   f"${a['CDKM [CDKM04]']['qubits']}$ & "
                   f"${a['Gidney [Gid18]']['qubits']}$\\\\")
    return "\n".join(out)


def rows_modarith():
    out = []
    for q in (251, 1021, 4093):
        dv, cv = V(f"moddbl_q{q}"), V(f"cmodadd_q{q}")
        pick = lambda v, k: next((v[nm] for nm in v if k in nm), None)
        for label, key in (("exact (Alg.~5 / Alg.~8)", "exact"),
                           ("top 4 bits only (Alg.~6 / Alg.~9)", "msbs=4"),
                           ("pseudo-Mersenne (Alg.~7 / Alg.~10)", "pseudo-Mersenne")):
            dd, cc = pick(dv, key), pick(cv, key)
            if dd is None:
                continue
            out.append(f"${q}$ & {label} & ${dd['toffoli_paper']}$ & "
                       f"${100*dd['failure_rate']:.1f}\\%$ & "
                       f"${cc['toffoli_paper']}$ & ${100*cc['failure_rate']:.1f}\\%$\\\\")
    return "\n".join(out)


def rows_mul():
    out = []
    for sec, n, w, p in (("mul_n8_w2", 8, 2, 251), ("mul_n12_w4", 12, 4, 4093),
                         ("mul_n16_w4", 16, 4, 65521)):
        v = V(sec)
        sb = v["schoolbook double-and-add"]
        out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{$n={n}$, $w={w}$, $p={p}$}}}}\\\\")
        for nm, c in (("schoolbook double-and-add", sb),
                      ("Montgomery over a carry-save tree", v[f"Montgomery+QCSA [106] w={w}"]),
                      ("Montgomery over a QROM lookup", v[f"Montgomery+QROM [HJN+20] w={w}"])):
            sp = "---" if c is sb else f"${sb['depth']/c['depth']:.1f}\\times$"
            out.append(f"\\quad {nm} & ${num(c['toffoli_paper'])}$ & ${c['qubits']}$ & "
                       f"${num(c['depth'])}$ & {sp}\\\\")
    return "\n".join(out)


def rows_inv():
    out = []
    for n in (5, 7, 8, 10, 12):
        v = V(f"inv_n{n}")
        r, o = v["reference Kaliski [HJN+20]"], v["unconditional+postponed [106]"]
        out.append(f"${n}$ & ${num(r['toffoli_paper'])}$ & ${num(o['toffoli_paper'])}$ & "
                   f"${pct(o['toffoli_paper'], r['toffoli_paper'])}$ & "
                   f"${num(r['depth'])}$ & ${num(o['depth'])}$ & "
                   f"${pct(o['depth'], r['depth'])}$ & ${r['qubits']}$ & ${o['qubits']}$\\\\")
    return "\n".join(out)


def rows_dialog():
    out = []
    for key, p in (("inplace_n3_p7", 7), ("inplace_n4_p11", 11),
                   ("inplace_n5_p31", 31), ("inplace_n6_p61", 61)):
        v = V(key)
        d, e = v["Kaliski division (out-of-place)"], v["EEA dialog + Bezout replay [1128]"]
        out.append(f"${p}$ & ${num(d['toffoli_paper'])}$ & ${d['qubits']}$ & "
                   f"${num(e['toffoli_paper'])}$ & ${e['qubits']}$ & "
                   f"${pct(e['toffoli_paper'], d['toffoli_paper'])}$ & "
                   f"${pct(e['qubits'], d['qubits'])}$\\\\")
    return "\n".join(out)


def rows_padd():
    out = []
    for key, label in (("padd_p=7", "$p=7$"), ("padd_p=11", "$p=11$")):
        v = V(key)
        a, j = v["affine in-place [106] Alg 3"], v["Jacobian out-of-place [106] Alg 4"]
        out.append(f"{label} & affine, in place (Alg.~3) & ${num(a['toffoli_paper'])}$ & "
                   f"${a['qubits']}$ & ${num(a['depth'])}$ & ---\\\\")
        out.append(f" & Jacobian, out of place (Alg.~4) & ${num(j['toffoli_paper'])}$ & "
                   f"${j['qubits']}$ & ${num(j['depth'])}$ & "
                   f"${pct(j['toffoli_paper'], a['toffoli_paper'])}$\\\\")
    w = V("padd_windowed_p7")["windowed w=2 [1128] Alg 1"]
    a7 = V("padd_p=7")["affine in-place [106] Alg 3"]
    out.append("$p=7$ & windowed $w{=}2$, dialog mult.\\ (Alg.~1) & "
               f"${num(w['toffoli_paper'])}$ & ${w['qubits']}$ & ${num(w['depth'])}$ & "
               f"${pct(w['toffoli_paper'], a7['toffoli_paper'])}$\\\\")
    return "\n".join(out)


def rows_tempand():
    out = []
    for n in (5, 8, 12):
        v = V(f"tempand_n{n}")
        allt, tmp = v["all Toffoli (7 T each)"], v["temporary AND (4 T / 0 T)"]
        out.append(f"${n}$ & ${num(tmp['and'])}$ & ${num(tmp['and_dg'])}$ & "
                   f"${num(allt['t'])}$ & ${num(tmp['t'])}$ & "
                   f"${allt['t']/tmp['t']:.2f}\\times$\\\\")
    return "\n".join(out)


def rows_window():
    proj = D["_projections"]["windowing_zigzag"]
    out = []
    for n in (192, 256, 384, 521):
        g = {r["w"]: r for r in proj if r["n"] == n}
        out.append(f"${n}$ & ${num(g[1]['additions'])}$ & ${g[11]['additions']}$ & "
                   f"${g[16]['additions']}$ & ${g[1]['zigzag_registers']}$ & "
                   f"${g[11]['zigzag_registers']}$ & ${g[16]['zigzag_registers']}$ & "
                   f"${num(g[1]['garbage_qubits'])}$ & ${num(g[16]['garbage_qubits'])}$\\\\")
    return "\n".join(out)


def rows_pareto():
    winners = set(PARETO["frontier"])
    return "\n".join(
        f"{r['label']} & ${r['qubits']}$ & ${num(r['toffoli_paper'])}$ & "
        f"${num(r['depth'])}$ & {'yes' if r['label'] in winners else 'no'}\\\\"
        for r in PARETO["rows"]
    )


def rows_full():
    v = V("full_ecdlp")
    ar = v["arithmetic oracle, full control registers"]
    tb = v["table oracle (simulable)"]
    sc = v["table oracle, semiclassical (1 ctrl qubit)"]
    return "\n".join([
        f"real arithmetic, two counting registers & ${ar['qubits']}$ & "
        f"${num(ar['toffoli_paper'])}$ & ${num(ar['gates'])}$ & ${num(ar['depth'])}$\\\\",
        f"permutation oracle, two counting registers & ${tb['qubits']}$ & --- & "
        f"${num(tb['gates'])}$ & ${num(tb['depth'])}$\\\\",
        f"permutation oracle, semiclassical QFT & ${sc['qubits']}$ & --- & "
        f"${num(sc['gates'])}$ & ${num(sc['depth'])}$\\\\"])


def rows_space():
    sp = D["_projections"]["eea_space"]
    out = []
    for n in ("256", "384", "521"):
        s = sp[n]
        out.append(f"${n}$ & ${num(s['record_raw'])}$ & ${num(s['record_compressed'])}$ & "
                   f"${s['record_compressed_per_n']:.3f}n$ & "
                   f"${num(s['with_register_sharing'])}$\\\\")
    return "\n".join(out)


# ---- values quoted in prose, so they too come from the benchmark ----------
def facts():
    f = {}
    gu = next(row for row in GU_AFFINE["rows"] if row["label"] == "divsteps")
    if gu["tvd"] is None:
        raise ValueError("Gu divsteps documentation requires an Aer measurement")
    f["gu_shots"] = num(GU_AFFINE["shots"])
    f["gu_width"] = gu["qubits"]
    f["gu_tvd"] = f"{gu['tvd']:.3f}"
    f["gu_bound"] = f"{GU_AFFINE['tvd_limit']:.3f}"
    p7 = next(row for row in GU_AFFINE["additional_validation"]
              if row["backend"] == "divsteps")
    f["gu_p7_shots"] = num(p7["shots"])
    f["gu_p7_width"] = p7["qubits"]
    f["gu_p7_tvd"] = f"{p7['tvd']:.3f}"
    f["gu_p7_bound"] = f"{p7['tvd_limit']:.3f}"
    p7_decoded = next(row for row in GU_AFFINE["additional_validation"]
                      if row["backend"] == "divsteps-decoded")
    f["gu_p7_decoded_width"] = p7_decoded["qubits"]
    f["gu_p7_decoded_bits"] = p7_decoded["task"].rsplit("=", 1)[-1]
    f["gu_p7_decoded_tvd"] = f"{p7_decoded['tvd']:.3f}"
    f["gu_p7_decoded_bound"] = f"{p7_decoded['tvd_limit']:.3f}"
    packed = next(row for row in GU_AFFINE["additional_validation"]
                  if row["backend"] == "packed-euclid")
    f["gu_p7_packed_width"] = packed["qubits"]
    f["gu_p7_packed_bits"] = packed["task"].rsplit("=", 1)[-1]
    f["gu_p7_packed_tvd"] = f"{packed['tvd']:.3f}"
    f["gu_p7_packed_bound"] = f"{packed['tvd_limit']:.3f}"
    skipped = [row["label"] for row in GU_AFFINE["rows"]
               if row["tvd"] is None]
    skipped_names = " and ".join("\\code{" + label + "}" for label in skipped)
    f["gu_sampling"] = (
        f"On Aer {GU_AFFINE['device']} (limit "
        f"${GU_AFFINE['aer_limit']}$ qubits), the full-history "
        "divsteps circuit's"
    )
    f["gu_unsampled"] = (
        f"The {skipped_names} "
        f"{'row has' if len(skipped) == 1 else 'rows have'} "
        "\\emph{not} been sampled on Aer for this task."
        if skipped else "All affine rows were sampled on Aer for this task."
    )
    f["gu_device_note"] = (
        "The Aer measurements above ran on CPU/MPS; CPU Aer suffices for the "
        "end-to-end toy validation reported here."
        if GU_AFFINE["device"] == "CPU" else
        "The saved benchmark reports an Aer GPU run."
    )
    dv = D["_derivation"]
    f["deriv_offset"] = "yes" if dv["offset_invisible"] else "NO"
    r5 = [e for e in dv["real"] if e["r"] == 5]
    f["p5_m3"] = f"{100*r5[0]['p_usable_correct']:.0f}"
    f["p5_m6"] = f"{100*r5[-1]['p_usable_correct']:.0f}"
    r13 = [e for e in dv["real"] if e["r"] == 13]
    f["p13_m7"] = f"{100*r13[-1]['p_usable_correct']:.0f}"
    f["z13"] = f"{100*r13[-1]['p_a1_zero']:.0f}"
    a32, c32 = V("adder_n32"), V("cadder_n32")
    f["cdkm32"] = a32["CDKM [CDKM04]"]["toffoli_paper"]
    f["gid32"] = a32["Gidney [Gid18]"]["toffoli_paper"]
    f["recon32"] = c32["reconstruct (CDKM)"]["toffoli_paper"]
    f["copy32"] = c32["copy-then-add [106] Fig 4b"]["toffoli_paper"]
    f["cadd_ratio"] = f"{f['recon32']/f['copy32']:.1f}"
    f["copyc32"] = c32["copy-then-add, CDKM inside"]["toffoli_paper"]
    f["cadd_fixed"] = f"{f['recon32']/f['copyc32']:.2f}"
    f["recon32q"] = c32["reconstruct (CDKM)"]["qubits"]
    f["copy32q"] = c32["copy-then-add [106] Fig 4b"]["qubits"]
    d = V("moddbl_q4093")
    f["dbl_exact"] = d["exact (Alg 5)"]["toffoli_paper"]
    f["dbl_pm"] = d["pseudo-Mersenne (Alg 7)"]["toffoli_paper"]
    f["dbl_pm_ratio"] = f"{f['dbl_exact']/f['dbl_pm']:.1f}"
    f["dbl_pm_fail"] = f"{100*d['pseudo-Mersenne (Alg 7)']['failure_rate']:.1f}"
    f["m4"] = f"{100*d['approx msbs=4 (Alg 6)']['failure_rate']:.1f}"
    f["m8"] = f"{100*d['approx msbs=8 (Alg 6)']['failure_rate']:.2f}"
    f["m_ratio"] = f"{d['approx msbs=4 (Alg 6)']['failure_rate']/d['approx msbs=8 (Alg 6)']['failure_rate']:.0f}"
    i = V("inv_n12")
    f["inv_ref"] = num(i["reference Kaliski [HJN+20]"]["toffoli_paper"])
    f["inv_opt"] = num(i["unconditional+postponed [106]"]["toffoli_paper"])
    f["inv_toff_pct"] = pct(i["unconditional+postponed [106]"]["toffoli_paper"],
                            i["reference Kaliski [HJN+20]"]["toffoli_paper"])
    f["inv_depth_pct"] = pct(i["unconditional+postponed [106]"]["depth"],
                             i["reference Kaliski [HJN+20]"]["depth"])
    m = V("mul_n16_w4")
    f["mul_depth_qcsa"] = f"{m['schoolbook double-and-add']['depth']/m['Montgomery+QCSA [106] w=4']['depth']:.1f}"
    e = V("inplace_n6_p61")
    f["dlg_toff_pct"] = pct(e["EEA dialog + Bezout replay [1128]"]["toffoli_paper"],
                            e["Kaliski division (out-of-place)"]["toffoli_paper"])
    f["dlg_q_pct"] = pct(e["EEA dialog + Bezout replay [1128]"]["qubits"],
                         e["Kaliski division (out-of-place)"]["qubits"])
    pp = V("padd_p=11")
    f["padd_proj_pct"] = pct(pp["Jacobian out-of-place [106] Alg 4"]["toffoli_paper"],
                             pp["affine in-place [106] Alg 3"]["toffoli_paper"])
    p7 = V("padd_p=7")
    fw = V("padd_windowed_p7")["windowed w=2 [1128] Alg 1"]
    f["padd_win_pct"] = pct(fw["toffoli_paper"],
                            p7["affine in-place [106] Alg 3"]["toffoli_paper"])
    f["ta_ratio"] = f"{V('tempand_n12')['all Toffoli (7 T each)']['t']/V('tempand_n12')['temporary AND (4 T / 0 T)']['t']:.2f}"
    wz = {r["w"]: r for r in D["_projections"]["windowing_zigzag"] if r["n"] == 256}
    f["w256_none"], f["w256_16"] = wz[1]["additions"], wz[16]["additions"]
    f["w256_ratio"] = f"{wz[1]['additions']/wz[16]['additions']:.0f}"
    f["g256_none"], f["g256_16"] = num(wz[1]["garbage_qubits"]), num(wz[16]["garbage_qubits"])
    f["zz_ratio"] = f"{wz[1]['zigzag_registers']/wz[16]['zigzag_registers']:.0f}"
    f["zz256_none"], f["zz256_16"] = wz[1]["zigzag_registers"], wz[16]["zigzag_registers"]
    fu = V("full_ecdlp")
    f["full_q"] = fu["arithmetic oracle, full control registers"]["qubits"]
    f["full_g"] = num(fu["arithmetic oracle, full control registers"]["gates"])
    f["full_toff"] = num(fu["arithmetic oracle, full control registers"]["toffoli_paper"])
    f["tab_q"] = fu["table oracle (simulable)"]["qubits"]
    f["sc_q"] = fu["table oracle, semiclassical (1 ctrl qubit)"]["qubits"]
    q = D["_projections"]["qubit_totals_1128"]["256"]
    f["q1128_space"], f["q1128_gate"] = q["space_optimized"], q["gate_optimized"]
    f["shor_log2"] = D["_projections"]["shor_toffoli_1128"]["log2"]
    return f


if __name__ == "__main__":
    import sys
    tpl = (ROOT / "bench" / "part7_template.tex").read_text()
    subs = {"DERIVCLEAN": rows_deriv_clean(), "DERIVREAL": rows_deriv_real(),
            "ADDERS": rows_adders(), "MODARITH": rows_modarith(), "MUL": rows_mul(),
            "INV": rows_inv(), "DIALOG": rows_dialog(), "PADD": rows_padd(),
            "TEMPAND": rows_tempand(), "WINDOW": rows_window(),
            "PARETO": rows_pareto(), "GUAFFINE": rows_gu_affine(),
            "FULL": rows_full(),
            "SPACE": rows_space()}
    for k, v in subs.items():
        tpl = tpl.replace(f"%%{k}%%", v)
    for k, v in facts().items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    tpl = expand_listings(tpl)
    assert "%%" not in tpl.replace("%%%", ""), "unsubstituted table placeholder"
    assert "@@" not in tpl, "unsubstituted fact placeholder: " + \
        tpl[tpl.index("@@"):tpl.index("@@") + 40]
    if sys.argv[1:] == ["--write"]:
        doc = ROOT / "shor-complete.tex"
        text = doc.read_text()
        start = text.rindex("\\clearpage", 0, text.index("\\part*{Part VII"))
        end = text.index("\\begin{thebibliography}", start)
        doc.write_text(text[:start] + tpl.rstrip("\n") + "\n\n" + text[end:])
        print(f"Regenerated Part VII in {doc}")
    elif not sys.argv[1:]:
        sys.stdout.write(tpl)
    else:
        sys.exit("usage: ec_make_part7.py [--write]")
