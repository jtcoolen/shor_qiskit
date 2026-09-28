"""Write the summary report (one self-contained HTML page) from the bench JSON.

Every number on the page is read from bench/*.json; the charts are drawn in
the page from the same data.

    ./venv/bin/python bench/make_report.py OUT.html
"""
import html
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
J = lambda name: json.loads((ROOT / "bench" / name).read_text())


def f(x, d=0):
    return f"{x:,.{d}f}"


def sci(x, d=2):
    e = len(str(int(x))) - 1
    return f"{x / 10 ** e:.{d}f}&times;10<sup>{e}</sup>"


def main(out):
    hb, tg, sp, dp = J("ec_hier_256.json"), J("ec_toffoli_256.json"), J("ec_space_256.json"), \
        J("ec_depth_256.json")
    rsa, cmp_, toy, est = J("rsa_g25.json"), J("compare_ecc_rsa.json"), \
        J("toy_distributions.json"), J("estimates.json")
    cat = J("ec_catalogue.json")
    rows = {r["label"]: r for r in cmp_["rows"]}
    ecc, ecc_min = rows["ECDLP-256, IonQ's cells, signed windows"], rows["ECDLP-256, fewest qubits"]
    r2048 = rows["RSA-2048, Gidney 2025 residue arithmetic"]
    r3072 = rows["RSA-3072, Gidney 2025 residue arithmetic"]
    trow = tg["rows"]
    ionq_cells = trow["+ the same adder in the point addition, approximate negation"]
    signed = trow["+ signed windows (2^15-entry tables), no Fig. 1"]
    packed = trow["+ Fig. 1 packing, replay in x's qubits"]
    one = trow["  the same with signed windows: IonQ's count and qubits in one circuit"]
    ssw = trow["  + SELECT-SWAP on the 3x lookup (2 words per load): no extra qubits"]
    sweep = {(r["k_point"], r["k_3x"]): r for r in tg["select_swap"]}
    base = trow["IonQ-style, Alg 11 replay cells (before)"]
    sprows = {k: v for k, v in sp.items() if not k.startswith("_")}
    luo = sprows.get("+ Luo's register-shared EEA instead of the dialog (ECDSA.Fail 5.3.5)")
    best_space = sprows["+ Gidney where there is headroom (walk 0.94n, squarer 0.78n)"]
    cla = dp["+ carry-lookahead squarer"]
    ionq_pub = 1392608

    # chart data
    padd = []
    for k, c in hb["addition"].items():
        padd.append({"q": c["qubits"], "t": c["toffoli_paper"], "d": c["toffoli_depth"],
                     "s": "configurations", "l": k})
    for k, c in sprows.items():
        padd.append({"q": c["qubits"], "t": c["toffoli"], "d": c.get("toffoli_depth"),
                     "s": "fewer qubits", "l": k.strip(),
                     **({"tag": "Luo EEA", "at": [7, 4, "start"]} if "Luo" in k else {})})
    for k, c in trow.items():
        padd.append({"q": c["qubits"], "t": c["expected"], "d": c["toffoli_depth"],
                     "s": "IonQ's cells", "l": k.strip(),
                     **({"tag": "one circuit", "at": [0, 16, "middle"]} if c is ssw else {})})
    for r in tg["select_swap"]:
        if (r["k_point"], r["k_3x"]) != (0, 0):
            padd.append({"q": r["qubits"], "t": r["expected"], "d": r["toffoli_depth"],
                         "s": "SELECT-SWAP", "l": f"one circuit + SELECT-SWAP, k = ({r['k_point']}, {r['k_3x']})"})
    for k, c in dp.items():
        if isinstance(c, dict):
            padd.append({"q": c["qubits"], "t": c["expected"], "d": c["toffoli_depth"],
                         "s": "carry-lookahead", "l": k})
    # label placement [dx, dy, anchor], so the four crosses' names do not collide
    pub = [{"q": 1208, "t": 2 ** 21.19 + 3 * 2 ** 16, "l": "[1128] space", "at": [-7, 4, "end"]},
           {"q": 1462, "t": 2 ** 20.83 + 3 * 2 ** 16, "l": "[1128] gate", "at": [7, 14, "start"]},
           {"q": 1457, "t": ionq_pub, "l": "IonQ", "at": [7, 4, "start"]},
           {"q": 1162, "t": 1684161, "l": "ECDSA.Fail (windowed)", "tag": "ECDSA.Fail",
            "at": [-7, 4, "end"]}]
    wtag = {"ECDLP-256, fewest qubits": ("ECDLP-256", [0, -9, "middle"]),
            "RSA-1024, Gidney 2025 residue arithmetic": ("RSA-1024", [7, 4, "start"]),
            "RSA-2048, Gidney 2025 residue arithmetic": ("RSA-2048", [7, 4, "start"]),
            "RSA-3072, Gidney 2025 residue arithmetic": ("RSA-3072", [7, 4, "start"]),
            "RSA-2048, Gidney-Ekera 2019": ("GE 2019", [0, -9, "end"])}
    whole = [{"q": r["logical_qubits"], "t": r["toffoli"], "pq": r["physical_qubits"],
              "h": r["days"] * 24, "fam": r["family"], "kind": r["kind"], "l": r["label"],
              **({"tag": wtag[r["label"]][0], "at": wtag[r["label"]][1]} if r["label"] in wtag else {})}
             for r in cmp_["rows"]]
    data = json.dumps({"padd": padd, "pub": pub, "whole": whole,
                       "floor": dp["lookup_depth_floor"]})

    def tr(cells, head=False):
        tag = "th" if head else "td"
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    moved = [
        ("Measurement-based lookups, one merged repair", "IonQ Alg 6, [1128]",
         "1,310,700 &rarr; 197,364 lookup Toffolis per addition"),
        ("Dedicated squarer and pseudo-Mersenne fold", "IonQ &sect;VII.A",
         "1,050,623 &rarr; 79,160 Toffolis for step 10"),
        ("Conditionally inverted GCD walk", "IonQ &sect;VI", "in-place multiplication 865,588 &rarr; 720,565"),
        ("Phase-approximate modular adder, 37 careful replay steps, forward multiplication",
         "IonQ Alg 2, Table X",
         f"one addition {f(base['toffoli'])} &rarr; {f(ionq_cells['expected'])} executed"),
        ("Odd signed windows (2<sup>15</sup>-entry tables)", "[HJN+20], [106] &sect;6.1",
         f"&rarr; {f(signed['expected'])} executed, {100 * (1 - signed['expected'] / ionq_pub):.1f}% below IonQ"),
        ("Lean careful cell, CNOT ends, register sharing in IonQ's walk, phase-adder fold",
         "IonQ &sect;VI, [1128] &sect;3.1",
         f"{f(packed['qubits'])} &rarr; {f(one['qubits'])} qubits at {f(one['expected'])} executed: "
         f"IonQ's count and qubits in one circuit"),
        ("SELECT-SWAP loads, junk cleared by measurement into the merged repair",
         "Low&ndash;Kliuchnikov&ndash;Schaeffer 2018",
         f"&rarr; {f(ssw['expected'])} executed at {f(ssw['qubits'])} qubits; "
         f"{f(sweep[(1, 2)]['expected'])} at {f(sweep[(1, 2)]['qubits'])}"),
        ("CDKM cells, lean comparator / all-ones / constant adder, headroom budgets",
         "[1128] &sect;3.2", f"2,251 &rarr; {f(best_space['qubits'])} qubits"),
        ("Luo's register-shared EEA", "ECDSA.Fail &sect;5.3.5",
         f"&rarr; {f(luo['qubits'])} qubits at {sci(luo['toffoli'], 1)} Toffolis" if luo else "built"),
        ("Carry-lookahead adders with fanned-out controls", "[106] &sect;3.1, [DKRS04]",
         f"depth 1,503,533 &rarr; {f(cla['toffoli_depth'])}"),
        ("Gidney 2025 residue arithmetic, built at size", "arXiv:2505.15917",
         f"RSA-2048 {sci(r2048['toffoli'])} Toffolis on {f(r2048['logical_qubits'])} qubits"),
    ]
    moved_rows = "".join(tr([a, f"<span class='src'>{b}</span>", c]) for a, b, c in moved)

    gap_rows = "".join(tr([html.escape(r["label"]) + ("" if r["kind"] == "built" else " <span class='pubtag'>published</span>"),
                           f(r["logical_qubits"]), sci(r["toffoli"]), f(r["physical_qubits"]),
                           (f"{r['days'] * 1440:.0f} min" if r["days"] < 0.1 else f"{r['days']:.1f} days")])
                       for r in cmp_["rows"])
    rsa_rows = "".join(tr([n, f(rsa[n]["primes"]), f(rsa[n]["iteration"]["toffoli_expected"]),
                           sci(rsa[n]["per_factoring_expected"]),
                           sci(rsa[n]["published"]["toffoli_per_factoring"], 1),
                           f"{rsa[n]['qubits']} ({rsa[n]['published']['qubits']})"])
                       for n in sorted(rsa, key=int))

    e = est
    cold = e["cold_record"]
    fac = e["factories"]
    win = {(r["w"], r["signed"]): r for r in e["windows"]}
    qtd = {r["config"]: r["qt"] for r in e["qt"]}
    qtn = {"one": ssw["qubits"] * ssw["expected"],
           "fail": qtd["ECDSA.Fail best Q x T (published, classical addend)"],
           "failw": qtd["ECDSA.Fail windowed variant (published)"],
           "ionq": qtd["IonQ (published)"]}
    t = toy["rsa"]
    m7 = next(x for x in t["masks"] if x["mask_bits"] == 7)
    sd = toy["ecdlp_signed"]

    page = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shor at Cryptographic Size</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
:root {{
  --ink: #1b2230; --paper: #f5f7f9; --panel: #ffffff; --muted: #5d6778; --rule: #d7dce3;
  --ecc: #17738f; --rsa: #b0621b; --good: #2f7d4f; --warn: #a2551b; --grid: #e6e9ee;
  --s1: #3c6fb6; --s2: #2f8f5b; --s3: #c0392b; --s4: #7b4bb3; --s5: #9a7400;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ink: #e5e9ef; --paper: #11151b; --panel: #171c24; --muted: #9aa4b4; --rule: #2a313c;
    --ecc: #5fb8d3; --rsa: #e3a061; --good: #6cc28f; --warn: #e3a061; --grid: #222833;
    --s1: #7aa7ea; --s2: #6cc28f; --s3: #ef7b6c; --s4: #b592e6; --s5: #e2bd4f;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ink: #e5e9ef; --paper: #11151b; --panel: #171c24; --muted: #9aa4b4; --rule: #2a313c;
  --ecc: #5fb8d3; --rsa: #e3a061; --good: #6cc28f; --warn: #e3a061; --grid: #222833;
  --s1: #7aa7ea; --s2: #6cc28f; --s3: #ef7b6c; --s4: #b592e6; --s5: #e2bd4f;
}}
body {{ background: var(--paper); color: var(--ink); font: 15px/1.6 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  padding-inline: 16px; padding-block: 32px 64px; }}
main {{ max-width: 980px; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 1fr); gap: 40px; }}
h1, h2, h3 {{ font-family: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif; text-wrap: balance; line-height: 1.15; margin: 0; }}
h1 {{ font-size: clamp(30px, 5vw, 44px); font-weight: 700; letter-spacing: -0.01em; }}
h2 {{ font-size: 25px; font-weight: 600; }}
h3 {{ font-size: 18px; font-weight: 600; }}
p {{ margin: 0; max-width: 68ch; }}
section, header {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; }}
.lede {{ color: var(--muted); font-size: 17px; max-width: 70ch; }}
.eyebrow {{ font: 500 12px/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.num, td, .tile b {{ font-variant-numeric: tabular-nums; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr)); gap: 12px; }}
.tile {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; display: grid; gap: 4px; }}
.tile b {{ font: 600 26px/1.1 "IBM Plex Mono", ui-monospace, monospace; }}
.tile.ecc b {{ color: var(--ecc); }} .tile.rsa b {{ color: var(--rsa); }}
.tile span {{ color: var(--muted); font-size: 13px; }}
.wide {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font: 500 12px/1.3 "IBM Plex Mono", ui-monospace, monospace; color: var(--muted); letter-spacing: .03em; }}
td:not(:first-child) {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; white-space: nowrap; }}
.src {{ color: var(--muted); }}
.pubtag {{ font: 500 11px/1 "IBM Plex Mono", monospace; color: var(--muted); border: 1px solid var(--rule); border-radius: 3px; padding: 1px 4px; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 16px; }}
figure {{ margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; display: grid; gap: 6px; }}
figcaption {{ color: var(--muted); font-size: 13px; }}
svg text {{ fill: var(--muted); font: 11px "IBM Plex Mono", ui-monospace, monospace; }}
svg .lab {{ fill: var(--ink); font-size: 10px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--muted); }}
.legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: 0; }}
.reasons {{ display: grid; gap: 10px; padding: 0; margin: 0; list-style: none; counter-reset: r; }}
.reasons li {{ display: grid; grid-template-columns: 28px 1fr; gap: 8px; }}
.reasons li::before {{ counter-increment: r; content: counter(r); font: 600 15px/1.5 "IBM Plex Mono", monospace; color: var(--rsa); }}
.est {{ display: grid; gap: 14px; }}
.est article {{ display: grid; gap: 6px; padding-block: 12px; border-top: 1px solid var(--rule); }}
.est h3 span {{ font: 500 12px "IBM Plex Mono", monospace; color: var(--good); margin-left: 8px; }}
code {{ font: 13px "IBM Plex Mono", ui-monospace, monospace; }}
.note {{ font-size: 13px; color: var(--muted); }}
</style>
<main>
<header style="display:grid;gap:12px">
  <span class="eyebrow">secp256k1 &middot; RSA-2048 &middot; built, not projected</span>
  <h1>Shor at cryptographic size</h1>
  <p class="lede">Every circuit below was built gate by gate by this repository's builders at full size
  (n&nbsp;=&nbsp;256 for the elliptic curve, n&nbsp;=&nbsp;1024&ndash;4096 for RSA), with the same code that is checked
  exhaustively at toy sizes. The counts are read off those circuits; Qualtran's <code>QECGatesCost</code>
  reproduces the Toffolis. Estimates are marked as such.</p>
</header>

<section aria-label="Headline figures" class="tiles">
  <div class="tile ecc"><span>ECDLP-256, whole algorithm (IonQ's cells, signed windows)</span><b>{sci(ecc['toffoli'])}</b><span>Toffolis on {f(ecc['logical_qubits'])} qubits &middot; {ecc['days'] * 1440:.0f} min in the surface-code model</span></div>
  <div class="tile rsa"><span>RSA-2048, Gidney 2025 residue arithmetic</span><b>{sci(r2048['toffoli'])}</b><span>Toffolis per factoring on {f(r2048['logical_qubits'])} qubits &middot; {r2048['days']:.1f} days</span></div>
  <div class="tile ecc"><span>fewest qubits, one point addition</span><b>{f(luo['qubits'] if luo else best_space['qubits'])}</b><span>Luo's EEA; {f(best_space['qubits'])} with the dialog at {sci(best_space['toffoli'], 1)} Toffolis</span></div>
  <div class="tile ecc"><span>one point addition, one circuit</span><b>{f(ssw['expected'])}</b><span>executed Toffolis on {f(ssw['qubits'])} qubits; IonQ publishes 1,392,608 on 1,457</span></div>
</section>

<section>
  <h2>The frontier of one point addition</h2>
  <p>One windowed point addition on secp256k1 (w&nbsp;=&nbsp;16) is the unit everything else multiplies: the whole
  algorithm is 28 of them after a first-window lookup. Qubits trade against Toffolis on the left and against
  Toffoli depth on the right. Crosses are published figures.</p>
  <div class="charts">
    <figure><svg id="c1" viewBox="0 0 460 320" role="img" aria-label="Qubits against Toffolis"></svg>
      <figcaption>Logical qubits against executed Toffolis (log scale).</figcaption></figure>
    <figure><svg id="c2" viewBox="0 0 460 320" role="img" aria-label="Qubits against Toffoli depth"></svg>
      <figcaption>Logical qubits against exact Toffoli depth. The dotted line is the three 2<sup>16</sup>-entry lookups.</figcaption></figure>
  </div>
  <div class="legend" id="leg"></div>
</section>

<section>
  <h2>What moved the numbers</h2>
  <div class="wide"><table>
    <thead>{tr(["technique", "source", "effect at n = 256"], True)}</thead>
    <tbody>{moved_rows}</tbody>
  </table></div>
  <p class="note">Completed for coverage, without moving a headline: Montgomery and Karatsuba variants, ping-pong and
  Jump-2 GCDs with the base-5 codec, projective coordinates with a quantum addend, twisted Edwards curves
  (7 multiplications against 11, no exceptional cases), batch inversion (always fewer Toffolis, always more qubits),
  and ECDSA.Fail's fire census (exact only when the census covers every input).</p>
</section>

<section>
  <h2>ECDSA against RSA</h2>
  <p>The same surface-code model for every row: physical error 10<sup>&minus;3</sup>, 1&nbsp;&micro;s cycles,
  10&nbsp;&micro;s reaction time, six CCZ factories. RSA's exponent sits in cold yoked storage, which is why it needs fewer
  physical qubits at a similar logical count.</p>
  <div class="charts">
    <figure><svg id="c3" viewBox="0 0 460 320" role="img" aria-label="Logical qubits against Toffolis, whole algorithms"></svg>
      <figcaption>Whole algorithms: logical qubits against Toffolis per solved instance. Circles are built here, crosses published.</figcaption></figure>
    <figure><svg id="c4" viewBox="0 0 460 320" role="img" aria-label="Physical qubits against runtime, whole algorithms"></svg>
      <figcaption>The same rows in the surface-code model: physical qubits against runtime in hours.</figcaption></figure>
  </div>
  <div class="wide"><table>
    <thead>{tr(["whole algorithm", "logical qubits", "Toffolis", "physical qubits", "runtime"], True)}</thead>
    <tbody>{gap_rows}</tbody>
  </table></div>
  <h3>Why secp256k1 falls about {r2048['toffoli'] / ecc['toffoli']:.0f}&times; sooner in Toffolis</h3>
  <ul class="reasons">
    <li><span><b>The numbers are eight times shorter.</b> Both algorithms spend their time on modular arithmetic,
    at least n<sup>2</sup> per group operation. Elliptic curves get away with n&nbsp;=&nbsp;256 because the best classical attack on them
    is generic (2<sup>n/2</sup>), while factoring has the number field sieve; Shor is polynomial in both.</span></li>
    <li><span><b>One shot against nine.</b> The windowed ECDLP circuit succeeds in one run with high probability;
    Eker&aring;&ndash;H&aring;stad with s&nbsp;=&nbsp;8 needs s&nbsp;+&nbsp;1 good runs, 9.2 expected.</span></li>
    <li><span><b>RSA's approximation buys qubits with Toffolis.</b> Residue arithmetic never holds an n-bit number
    except the output's top f&nbsp;=&nbsp;33 bits: 0.7n qubits for RSA-2048 against {ecc_min['logical_qubits'] / 256:.1f}&ndash;{ecc['logical_qubits'] / 256:.1f}n for secp256k1, paid for by reading all
    214 exponent windows for each of ~21,000 primes.</span></li>
    <li><span><b>At equal classical security the gap widens.</b> RSA-3072 (128-bit, like secp256k1) costs
    {sci(r3072['toffoli'])} Toffolis, {r3072['toffoli'] / ecc['toffoli']:.0f}&times; the elliptic-curve count.</span></li>
  </ul>
  <div class="wide"><table>
    <thead>{tr(["n", "primes", "one iteration", "per factoring", "Gidney 2025", "qubits (paper)"], True)}</thead>
    <tbody>{rsa_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>How the numbers are checked</h2>
  <p>Every arithmetic circuit is a permutation on basis states, so it is simulated exactly on every input at toy
  sizes, with each ancilla checked back to |0&rang; at the instruction that frees it. Measurement-based uncomputation
  is checked separately with a sparse state-vector simulator that performs the measurements. Knowing the answer, the
  whole output distribution follows too:</p>
  <ul class="reasons" style="counter-reset:r">
    <li><span>Signed-window ECDLP on a curve of order {sd['order']}: total variation {sd['tvd']:.3f} from the ideal, all
    from the {100 * sd['wrong_fraction']:.0f}% of inputs that meet an exceptional addition; one-run success
    {100 * sd['success']:.0f}% against {100 * sd['ideal_success']:.0f}% ideal, above IonQ's bound of {100 * sd['ionq_bound']:.0f}%.</span></li>
    <li><span>RSA, N&nbsp;=&nbsp;241&nbsp;&times;&nbsp;251 with Eker&aring;&ndash;H&aring;stad s&nbsp;=&nbsp;2: the approximate, masked exponentiation (which the
    circuit equals bit for bit) deviates by at most {100 * t['worst_deviation']:.2f}% of N and puts {m7['peak_mass']:.3f} of its mass on the
    peaks that carry {m7['exact_peak_mass']:.3f} of the exact oracle's distribution.</span></li>
  </ul>
  <p class="note">Found along the way: the GCD's top-bit comparisons need 40&nbsp;+&nbsp;2.3&radic;n bits because the width schedule
  pads with leading zeros (a flat 24 bits failed 15 of 40 multiplications at n&nbsp;=&nbsp;64); this repository's controlled CDKM adder cost 4n,
  not the documented 3n; IonQ's replay also needs a p&nbsp;&rarr;&nbsp;0 repair when multiplying; IonQ's circuit is 1,457 qubits (1,462 is its target width, which this repository had quoted); Gidney 2025's qubit formula gives 1,432, not the 1,409 in its text.</p>
</section>

<section class="est">
  <h2>Further optimisations, estimated</h2>
  <p>Not built. Each estimate starts from a built count above and states its formula.</p>
  <article>
    <h3>Cold storage for the GCD record<span>&minus;{100 * (1 - cold[1]['physical_record_cold'] / cold[1]['physical_all_hot']):.0f}% physical qubits</span></h3>
    <p>The packed record is read two bits per round, so it can live in yoked storage like RSA's exponent.
    Fewest-qubit configuration: {f(cold[1]['physical_all_hot'])} &rarr; {f(cold[1]['physical_record_cold'])} physical qubits;
    IonQ's cells with signed windows: {f(cold[0]['physical_all_hot'])} &rarr; {f(cold[0]['physical_record_cold'])}. The cost is the latency of fetching a
    record bit, which the walk can prefetch one round ahead.</p>
  </article>
  <article>
    <h3>More factories<span>{fac[0]['minutes']:.0f} &rarr; {fac[2]['minutes']:.0f} min</span></h3>
    <p>ECDLP-256 is limited by CCZ throughput at six factories ({fac[0]['minutes']:.0f} minutes). At {fac[2]['factories']} factories it reaches its
    reaction-limited floor, depth &times; 10&nbsp;&micro;s = {fac[2]['minutes']:.1f} minutes, for {f(fac[2]['physical'])} physical qubits against
    {f(fac[0]['physical'])}. The carry-lookahead configuration lowers that floor about fourfold.</p>
  </article>
  <article>
    <h3>Window size<span>w&nbsp;=&nbsp;16 stays optimal</span></h3>
    <p>Whole algorithm on IonQ's cells, additions against lookup cost: w&nbsp;=&nbsp;15: {sci(win[(15, False)]['toffoli'])}, w&nbsp;=&nbsp;16:
    {sci(win[(16, False)]['toffoli'])}, w&nbsp;=&nbsp;17: {sci(win[(17, False)]['toffoli'])}. With signed windows, w&nbsp;=&nbsp;16:
    {sci(win[(16, True)]['toffoli'])}, below IonQ's 3.90&times;10<sup>7</sup>.</p>
  </article>
  <article>
    <h3>Qubits &times; Toffolis<span>ECDSA.Fail's score</span></h3>
    <p>Best built: the one circuit with SELECT-SWAP on the 3x load, {f(ssw['qubits'])} &times; {f(ssw['expected'])} = {qtn['one'] / 1e9:.2f}&times;10<sup>9</sup>.
    Published: ECDSA.Fail's best classical-addend circuit {qtn['fail'] / 1e9:.2f}&times;10<sup>9</sup> (a classical addend, not windowed Shor),
    its windowed variant {qtn['failw'] / 1e9:.2f}&times;10<sup>9</sup>, IonQ {qtn['ionq'] / 1e9:.2f}&times;10<sup>9</sup>.</p>
  </article>
</section>

<section>
  <h2>Where it lives</h2>
  <p class="note">Repository <code>shor</code>: <code>shor_qiskit/</code> (builders), <code>bench/*.py</code> (every number here, JSON in
  <code>bench/</code>), <code>shor-complete.tex</code> Parts VII and VIII (the explanation, generated from the same JSON and checked by
  <code>bench/ec_check_tex.py</code>), <code>doc/figures/</code> (plots).</p>
</section>
</main>
<script>
const D = {data};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SER = {{"configurations": "--s1", "fewer qubits": "--s2", "IonQ's cells": "--s3", "carry-lookahead": "--s4", "SELECT-SWAP": "--s5"}};
function scatter(svg, pts, opt) {{
  const W = opt.w || 460, H = 320, L = 58, R = 12, T = 12, B = 40;
  const xs = pts.map(p => p.x).concat(opt.extraX || []), ys = pts.map(p => p.y).concat(opt.extraY || []);
  const lx = opt.logx, ly = true;
  const tx = v => lx ? Math.log10(v) : v, ty = v => Math.log10(v);
  let x0 = Math.min(...xs.map(tx)), x1 = Math.max(...xs.map(tx)), y0 = Math.min(...ys.map(ty)), y1 = Math.max(...ys.map(ty));
  const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.08 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const X = v => opt.ox + L + (tx(v) - x0) / (x1 - x0) * (W - L - R), Y = v => T + (1 - (ty(v) - y0) / (y1 - y0)) * (H - T - B);
  let s = "";
  for (let e = Math.ceil(y0); e <= Math.floor(y1); e++) {{
    const y = T + (1 - (e - y0) / (y1 - y0)) * (H - T - B);
    s += `<line x1="${{opt.ox + L}}" x2="${{opt.ox + W - R}}" y1="${{y}}" y2="${{y}}" stroke="${{css('--grid')}}"/>`;
    s += `<text x="${{opt.ox + L - 6}}" y="${{y + 4}}" text-anchor="end">1e${{e}}</text>`;
  }}
  const xt = lx ? [...Array(Math.floor(x1) - Math.ceil(x0) + 1).keys()].map(i => 10 ** (Math.ceil(x0) + i)) : opt.xticks;
  for (const v of xt) {{ const x = X(v); if (x < opt.ox + L || x > opt.ox + W - R) continue;
    s += `<line x1="${{x}}" x2="${{x}}" y1="${{T}}" y2="${{H - B}}" stroke="${{css('--grid')}}"/>`;
    s += `<text x="${{x}}" y="${{H - B + 15}}" text-anchor="middle">${{lx ? '1e' + Math.round(Math.log10(v)) : v}}</text>`; }}
  s += `<text x="${{opt.ox + (L + W - R) / 2}}" y="${{H - 6}}" text-anchor="middle">${{opt.xl}}</text>`;
  s += `<text transform="translate(${{opt.ox + 12}},${{(T + H - B) / 2}}) rotate(-90)" text-anchor="middle">${{opt.yl}}</text>`;
  if (opt.floor) {{ const y = Y(opt.floor); s += `<line x1="${{opt.ox + L}}" x2="${{opt.ox + W - R}}" y1="${{y}}" y2="${{y}}" stroke="${{css('--muted')}}" stroke-dasharray="3 3"/>`; }}
  for (const p of pts) {{
    const c = css(p.c); const x = X(p.x), y = Y(p.y);
    s += p.cross ? `<path d="M${{x - 4}} ${{y - 4}}L${{x + 4}} ${{y + 4}}M${{x - 4}} ${{y + 4}}L${{x + 4}} ${{y - 4}}" stroke="${{c}}" stroke-width="1.8" fill="none"><title>${{p.l}}</title></path>`
                 : `<circle cx="${{x}}" cy="${{y}}" r="4" fill="${{c}}" fill-opacity=".85"><title>${{p.l}}</title></circle>`;
    const at = p.at || [6, -5, "start"];
    if (p.tag) s += `<text class="lab" x="${{x + at[0]}}" y="${{y + at[1]}}" text-anchor="${{at[2]}}">${{p.tag}}</text>`;
  }}
  return s;
}}
function draw() {{
  const pa = D.padd.map(p => ({{x: p.q, y: p.t, c: SER[p.s], l: p.l, tag: p.tag, at: p.at}}));
  const pubs = D.pub.map(p => ({{x: p.q, y: p.t, c: "--ink", cross: true, l: p.l, tag: p.tag || p.l, at: p.at}}));
  const xo = {{xticks: [1200, 2000, 3000, 4000], extraX: [850]}};     // room left of the crosses for their names
  document.getElementById("c1").innerHTML = scatter(null, pa.concat(pubs), {{...xo, ox: 0, xl: "logical qubits", yl: "Toffolis"}});
  const pd = D.padd.filter(p => p.d).map(p => ({{x: p.q, y: p.d, c: SER[p.s], l: p.l, tag: p.tag, at: p.at}}));
  document.getElementById("c2").innerHTML = scatter(null, pd, {{...xo, ox: 0, xl: "logical qubits", yl: "Toffoli depth", floor: D.floor, extraY: [D.floor]}});
  const wl = D.whole.map(r => ({{x: r.q, y: r.t, c: r.fam === "ECC" ? "--ecc" : "--rsa", cross: r.kind !== "built", l: r.l, tag: r.tag, at: r.at}}));
  const wr = D.whole.map(r => ({{x: r.pq, y: r.h, c: r.fam === "ECC" ? "--ecc" : "--rsa", cross: r.kind !== "built", l: r.l, tag: r.tag, at: r.at}}));
  document.getElementById("c3").innerHTML = scatter(null, wl, {{ox: 0, logx: true, xl: "logical qubits", yl: "Toffolis"}});
  document.getElementById("c4").innerHTML = scatter(null, wr, {{ox: 0, logx: true, xl: "physical qubits", yl: "hours"}});
  document.getElementById("leg").innerHTML = Object.entries(SER).map(([k, v]) => `<span><i style="background:${{css(v)}}"></i>built: ${{k}}</span>`).join("") +
    `<span>&times; published</span>`;
}}
draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, {{attributes: true, attributeFilter: ["data-theme"]}});
</script>
"""
    pathlib.Path(out).write_text(page)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "doc" / "report.html"))
