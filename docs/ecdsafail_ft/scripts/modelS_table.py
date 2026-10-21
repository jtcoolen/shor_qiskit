"""Turn data/physical_modelS.json (output of physical_estimates.py) into the LaTeX table of Section 8."""
import json
rows = json.load(open("data/physical_modelS.json"))
short = {
 "ECDSA.fail 8e9c9a2 (26 Jul incumbent) + 16 window qubits, 28 adds + lookups": r"\ecdsafail{} \code{8e9c9a2} (26 Jul, $1{,}151\times1.30$\,M) $+16$ window qubits",
 "ECDSA.fail windowed 8e9c9a2 (Note F; lookups included)": r"\ecdsafail{} windowed \code{8e9c9a2} (Note F, lookups included)",
 "ECDSA.fail a02c716 (7 Oct frontier, Leapfrog) + 16, 28 adds + lookups": r"\ecdsafail{} \code{a02c716} (7 Oct, Leapfrog, $1{,}244\times0.80$\,M) $+16$",
 "ECDSA.fail b6f2b0a low-Q (825) + 16, 28 adds + lookups": r"\ecdsafail{} \code{b6f2b0a} low-$Q$ ($825\times489$\,M) $+16$",
 "ECDSA.fail 10177cd low-Q (792, 7 Oct) + 16, 28 adds + lookups": r"\ecdsafail{} \code{10177cd} low-$Q$ ($792\times657$\,M, 7 Oct) $+16$",
 "Google/Babbush et al. low-qubit variant (whole algorithm)": r"Google, Babbush et al.\ low-qubit variant \cite{babbush26}",
 "Google/Babbush et al. low-gate variant (whole algorithm)": r"Google, Babbush et al.\ low-gate variant \cite{babbush26}",
 "IonQ/Haener et al. (whole algorithm)": r"IonQ, H\"aner et al.\ \cite{haener26}",
 "Schrottenloher space-optimised (whole algorithm)": r"Schrottenloher, space-optimised \cite{schrottenloher26}",
 "Schrottenloher gate-optimised (whole algorithm)": r"Schrottenloher, gate-optimised \cite{schrottenloher26}",
 "Julian: IonQ cells, signed windows, SELECT-SWAP (whole algorithm, executed)": r"\code{shor/}: IonQ cells, signed windows, SELECT-SWAP (executed)",
 "Julian: dialog on the space cells (whole algorithm)": r"\code{shor/}: dialog GCD on the space cells",
 "Julian: three field registers, Luo EEA (whole algorithm)": r"\code{shor/}: three field registers, Luo EEA",
 "RSA-2048, Gidney 2025 (published inputs; 9.2 shots of 12.07 h)": r"RSA-2048, Gidney 2025 inputs (for calibration) \cite{gidney25rsa}",
}
def fmt_t(h):
    return f"{h*60:.0f}\\,min" if h < 1 else (f"{h:.1f}\\,h" if h < 48 else f"{h/24:.1f}\\,d")
with open("data/modelS_table.tex", "w") as f:
    f.write("\\begin{tabular}{@{}p{6.9cm}rrrrrr@{}}\\toprule\n")
    f.write("circuit (whole attack unless noted) & $Q$ & Toffolis & physical & time/run & $p_{\\rm ok}$ & expected\\\\\n\\midrule\n")
    for r in rows:
        lab = short.get(r["label"], r["label"])
        tof = f"${r['toffoli']/1e9:.2f}$\\,B" if r['toffoli'] >= 1e9 else f"${r['toffoli']/1e6:.1f}$\\,M"
        f.write(f"{lab} & {r['logical']:,} & {tof} & {r['physical']:,} & {fmt_t(r['hours'])} & {r['p_ok']:.3f} & {fmt_t(r['days']*24)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print(open("data/modelS_table.tex").read())
