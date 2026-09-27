"""Generate Part VII of shor-complete.tex, with every number pulled from
`ec_ablation.json` and the n = 256 builds (`ec_project_256`, `ec_hier_256`,
`qualtran_compare`, `ec_toffoli_256`, `ec_space_256`).  Run this, paste the output into the document, then run
`ec_check_tex.py` to confirm the document still matches the benchmark.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
D = json.loads((ROOT / "bench" / "ec_ablation.json").read_text())
V = lambda s: D[s]["variants"]

import ec_listings as L

PJ = json.loads((ROOT / "bench" / "ec_project_256.json").read_text())
# built (not projected) at n = 256: bench/ec_hier_256.py, qualtran_compare.py,
# ec_toffoli_256.py, ec_space_256.py
HB = json.loads((ROOT / "bench" / "ec_hier_256.json").read_text())
QC = json.loads((ROOT / "bench" / "qualtran_compare.json").read_text())
TG = json.loads((ROOT / "bench" / "ec_toffoli_256.json").read_text())
SPC = json.loads((ROOT / "bench" / "ec_space_256.json").read_text())


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


REFINE_LABELS = {
    "[1128] Dialog, exact, full width (repo today)":
        "dialog \\cite{ec:schrott26}, exact, full width (before)",
    "[1128] Dialog, fused cmp48, c_pad 2.3, PM":
        "dialog, fused 48-bit compare, width schedule, PM",
    "IonQ CondInv, cmp48, c_pad 2.3, PM, replay=standard":
        "cond.\\ inverted \\cite{ec:ionq26}, dialog replay",
    "IonQ CondInv, cmp48, c_pad 2.3, PM, replay=ci":
        "cond.\\ inverted \\cite{ec:ionq26}, IonQ replay",
    "ECDSA.Fail ping-pong, 704 rounds, PM": "ping-pong \\cite{ec:ecdsafail26}",
    "ECDSA.Fail Jump-2, 261 steps, PM": "Jump-2 \\cite{ec:ecdsafail26}",
}


def rows_refine_gcd():
    out = []
    for name, v in PJ["variants"].items():
        out.append(f"{REFINE_LABELS[name]} & ${v['rounds']}$ & "
                   f"${num(round(v['walk_round_avg']))}$ & ${num(v['replay_step'])}$ & "
                   f"${num(v['in_place_mul'])}$\\\\")
    return "\n".join(out)


def rows_refine_padd():
    V_ = PJ["variants"]
    before = V_["[1128] Dialog, exact, full width (repo today)"]["in_place_mul"]
    best = V_[PJ["point_addition"]["best_gcd"]]["in_place_mul"]
    rows = [
        ("two in-place multiplications", 2 * before, 2 * best),
        ("square-subtract", PJ["square_sub"]["general"], PJ["square_sub"]["pm_fold"]),
        ("table lookups", PJ["lookups_w16"]["ten_recomputed"], PJ["lookups_w16"]["three_mbu"]),
        ("five modular additions", 5 * PJ["modadd"], 5 * PJ["modadd"]),
    ]
    out = [f"{a} & ${num(b)}$ & ${num(c)}$\\\\" for a, b, c in rows]
    pa = PJ["point_addition"]
    out.append("\\midrule")
    out.append(f"total & ${num(pa['today'])}$ & ${num(pa['best'])}$\\\\")
    pub = pa["published"]
    out.append(f"published: \\cite{{ec:schrott26}} / \\cite{{ec:ionq26}} & ${num(pub['1128'])}$ & "
               f"${num(pub['IonQ'])}$\\\\")
    return "\n".join(out)


HIER_LABELS = {
    "[1128] dialog, exact arithmetic (MBU lookups, masks)":
        "dialog \\cite{ec:schrott26}, exact arithmetic, MBU lookups, masked tables",
    "[1128] dialog, fused cmp48, schedule, PM":
        "\\quad + 48-bit comparisons, width schedule, pseudo-Mersenne replay",
    "+ dedicated squarer": "\\quad + dedicated squarer and fold",
    "IonQ: cond.-inverted walk + replay, squarer":
        "cond.\\ inverted walk and replay \\cite{ec:ionq26}, squarer",
    "ECDSA.Fail ping-pong (704 rounds), squarer":
        "ping-pong walk, 704 rounds \\cite{ec:ecdsafail26}, squarer",
    "space: dialog + register sharing + Fig. 1 packing, squarer":
        "dialog, register sharing and Fig.~1 packing, squarer",
}


def rows_hier():
    out = [f"Qualtran's \\texttt{{ECAdd}} \\cite{{ec:litinski23}}, for reference & "
           f"${num(QC['qualtran_litinski_ecadd']['toffoli'])}$ & "
           f"${num(QC['qualtran_litinski_ecadd']['qubits'])}$\\\\"]
    for name, c in HB["addition"].items():
        out.append(f"{HIER_LABELS[name]} & ${num(c['toffoli_paper'])}$ & "
                   f"${num(c['qubits'])}$\\\\")
    return "\n".join(out)


TOFF_LABELS = {
    "IonQ-style, Alg 11 replay cells (before)":
        "cond.\\ inverted walk and replay, Algorithm~11 cells (as above)",
    "+ phase-approximate adder in the replay, 37 careful steps":
        "\\quad + IonQ's adder in the replay, 37 careful steps",
    "+ the same adder in the point addition, approximate negation":
        "\\quad + the same adder in the addition, approximate negation",
    "+ Fig. 1 packing, replay in x's qubits":
        "\\quad + Fig.~1 packing, replay in $x$'s qubits",
}


def rows_toff():
    out = []
    for name, c in TG["rows"].items():
        out.append(f"{TOFF_LABELS[name]} & ${num(c['toffoli'])}$ & ${num(c['expected'])}$ & "
                   f"${num(c['qubits'])}$\\\\")
    pub = TG["published"]["IonQ"]
    out.append("\\midrule")
    out.append(f"published \\cite{{ec:ionq26}} & --- & ${num(pub['toffoli'])}$ & "
               f"${num(pub['qubits'])}$\\\\")
    return "\n".join(out)


def rows_toff_parts():
    before, after = list(TG["components"].values())
    order = (("table lookups and repair", "table lookups and their repair"),
             ("GCD walk and its undoing", "GCD walks and their undoing"),
             ("Bezout replay", "B\\'ezout replays"),
             ("  of which signed additions", "\\quad of which signed additions"),
             ("square-subtract", "square-subtract"),
             ("everything else", "everything else"))
    out = [f"{lab} & ${num(before[k])}$ & ${num(after[k])}$\\\\" for k, lab in order]
    out.append("\\midrule")
    out.append(f"one windowed point addition & ${num(before['total'])}$ & "
               f"${num(after['total'])}$\\\\")
    return "\n".join(out)


SPACE_LABELS = {
    "IonQ-style (cond.-inverted, PM, IonQ replay)":
        "cond.\\ inverted, pseudo-Mersenne, IonQ replay (the start)",
    "+ Fig. 1 packing of the record": "\\quad + Fig.~1 packing of the record",
    "+ Fig. 1 record packing": "\\quad + Fig.~1 packing of the record",
    "+ replay in x's qubits": "\\quad + replay in $x$'s qubits",
    "+ CDKM replay arithmetic (PMSpace)": "\\quad + CDKM replay cells",
    "+ CDKM walk adder": "\\quad + CDKM walk adder",
    "same, dialog replay instead of IonQ's": "\\quad dialog replay instead of IonQ's",
    "[1128] dialog + sharing + Fig. 1, PMSpace":
        "dialog, register sharing, Fig.~1, CDKM replay cells",
    "+ CDKM square-subtract and point-add adders":
        "\\quad + CDKM square-subtract and point-addition adders",
    "+ CDKM dialog walk": "\\quad + CDKM dialog walk",
    "+ lean replay cells (CDKM compare, chunked all-ones, borrowed increment)":
        "\\quad + lean replay cells",
    "+ CDKM squarer": "\\quad + CDKM squarer",
    "+ Gidney where there is headroom (walk 0.94n, squarer 0.78n)":
        "\\quad + Gidney adders where there is headroom",
    "  (side: cond.-inverted walk, same cells -- its record is not shared)":
        "(cond.\\ inverted walk, same cells: its record is not shared)",
}


def rows_space256():
    out = []
    for name, c in SPC.items():
        if name.startswith("_"):
            continue
        out.append(f"{SPACE_LABELS[name]} & ${num(c['qubits'])}$ & ${num(c['toffoli'])}$\\\\")
    return "\n".join(out)


# ---- values quoted in prose, so they too come from the benchmark ----------
def facts():
    f = {}
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
    f["r_lk_old"] = num(PJ["lookups_w16"]["ten_recomputed"])
    f["r_lk_new"] = num(PJ["lookups_w16"]["three_mbu"])
    f["r_sq_old"] = num(PJ["square_sub"]["general"])
    f["r_sq_new"] = num(PJ["square_sub"]["pm_fold"])
    pa = PJ["point_addition"]
    f["r_padd_x"] = f"{pa['today'] / pa['best']:.1f}"
    f["r_full_old"] = num(PJ["full_algorithm"]["today"])
    f["r_full_new"] = num(PJ["full_algorithm"]["best"])
    # --- built at n = 256
    fa = HB["full_algorithm"]
    f["hb_full"], f["hb_full_q"] = num(fa["toffoli_paper"]), num(fa["qubits_semiclassical"])
    f["hb_full_add"] = fa["additions"]
    f["hb_full_cliff"] = num(fa["qualtran"]["clifford"])
    f["hb_full_meas"] = num(fa["qualtran"]["measurement"])
    f["hb_lit"] = num(QC["qualtran_litinski_ecadd"]["toffoli"])
    ionq_row = HB["addition"]["IonQ: cond.-inverted walk + replay, squarer"]
    f["hb_ionq"] = num(ionq_row["toffoli_paper"])
    f["hb_ionq_pct"] = f"{100 * (ionq_row['toffoli_paper'] / QC['published']['IonQ'] - 1):.0f}"
    # --- the Toffoli gap
    cl = TG["cells"]
    f["tg_sa_old"] = cl["signed add, Alg 11 + 0<->q swaps (before)"]["toffoli"]
    f["tg_sa_new"] = cl["signed add, phase-approximate (IonQ Alg 2)"]["toffoli"]
    f["tg_sa_exp"] = f"{cl['signed add, phase-approximate (IonQ Alg 2)']['expected']:.0f}"
    f["tg_half_old"] = cl["halving, 74-bit correction (before)"]["toffoli"]
    f["tg_half_new"] = cl["halving, kappa = 65"]["toffoli"]
    f["tg_kappa"], f["tg_delta"], f["tg_zero"] = TG["kappa"], TG["delta"], TG["zero_steps"]
    rw = list(TG["rows"].values())
    f["tg_before"], f["tg_after"] = num(rw[0]["toffoli"]), num(rw[2]["toffoli"])
    f["tg_after_exp"] = num(rw[2]["expected"])
    f["tg_q_after"], f["tg_q_packed"] = num(rw[2]["qubits"]), num(rw[3]["qubits"])
    pub = TG["published"]["IonQ"]
    f["tg_pub"], f["tg_pub_q"] = num(pub["toffoli"]), num(pub["qubits"])
    f["tg_gap"] = f"{100 * abs(rw[2]['expected'] / pub['toffoli'] - 1):.1f}"
    before, after = list(TG["components"].values())
    f["tg_sa_tot_old"] = num(before["  of which signed additions"])
    f["tg_sa_tot_new"] = num(after["  of which signed additions"])
    fl = TG["full_algorithm"]
    f["tg_full"], f["tg_full_exp"] = num(fl["toffoli"]), num(fl["expected"])
    f["tg_full_add"] = fl["additions"]
    # --- the qubit frontier
    sp = {k: v for k, v in SPC.items() if not k.startswith("_")}
    first, best = sp["IonQ-style (cond.-inverted, PM, IonQ replay)"], \
        sp["+ Gidney where there is headroom (walk 0.94n, squarer 0.78n)"]
    f["sp_start_q"], f["sp_start_t"] = num(first["qubits"]), num(first["toffoli"])
    f["sp_best_q"], f["sp_best_t"] = num(best["qubits"]), num(best["toffoli"])
    f["sp_best_n"] = f"{best['qubits'] / 256:.2f}"
    f["sp_1128"] = num(sp["[1128] dialog + sharing + Fig. 1, PMSpace"]["qubits"])
    f["sp_rec"] = num(SPC["_record_qubits"])
    return f


if __name__ == "__main__":
    import sys
    tpl = (ROOT / "bench" / "part7_template.tex").read_text()
    subs = {"DERIVCLEAN": rows_deriv_clean(), "DERIVREAL": rows_deriv_real(),
            "ADDERS": rows_adders(), "MODARITH": rows_modarith(), "MUL": rows_mul(),
            "INV": rows_inv(), "DIALOG": rows_dialog(), "PADD": rows_padd(),
            "TEMPAND": rows_tempand(), "WINDOW": rows_window(), "FULL": rows_full(),
            "SPACE": rows_space(), "REFINE_GCD": rows_refine_gcd(),
            "REFINE_PADD": rows_refine_padd(), "HIER": rows_hier(), "TOFF": rows_toff(),
            "TOFF_PARTS": rows_toff_parts(), "SPACE256": rows_space256()}
    for k, v in subs.items():
        tpl = tpl.replace(f"%%{k}%%", v)
    for k, v in facts().items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    tpl = expand_listings(tpl)
    assert "%%" not in tpl.replace("%%%", ""), "unsubstituted table placeholder"
    assert "@@" not in tpl, "unsubstituted fact placeholder: " + \
        tpl[tpl.index("@@"):tpl.index("@@") + 40]
    sys.stdout.write(tpl)
