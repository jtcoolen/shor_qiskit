"""Syndrome (lookup-table) decoders for the small codes, emitted as LaTeX tables.

For every single-qubit Pauli error E on every qubit, the syndrome bit s_i is 1 iff E
anticommutes with generator S_i.  A lookup-table decoder inverts this map.  Also
splits data/codes.dat into one file per family for the rate-distance plot.
Outputs: data/syn_steane.tex, data/syn_513.tex, data/syn_shor.tex, data/codes_*.dat
"""
import stim, itertools, collections

def syndrome(err, gens):
    return "".join("1" if not err.commutes(g) else "0" for g in gens)

def P(s): return stim.PauliString(s)

def single_errors(n):
    for q in range(n):
        for p in "XYZ":
            s = ["_"] * n; s[q] = p
            yield f"{p}_{{{q+1}}}", P("".join(s))

# ---------------------------------------------------------------- Steane: Hamming rows
ham = [[1,3,5,7],[2,3,6,7],[4,5,6,7]]           # 1-indexed supports of the three checks
def css(sup, t, n=7):
    return P("".join(t if (i+1) in sup else "_" for i in range(n)))
steane_z = [css(s,"Z") for s in ham]            # detect X errors
steane_x = [css(s,"X") for s in ham]            # detect Z errors
rows = []
for q in range(1,8):
    ex = ["_"]*7; ex[q-1] = "X"; ez = ["_"]*7; ez[q-1] = "Z"
    sz = syndrome(P("".join(ex)), steane_z); sx = syndrome(P("".join(ez)), steane_x)
    assert sz == sx == format(q, "03b")[::-1], (q, sz, sx)
    rows.append((q, sz[::-1]))
with open("data/syn_steane.tex", "w") as f:
    f.write("\\begin{tabular}{@{}c" + "c"*7 + "@{}}\\toprule\n")
    f.write("qubit $j$ & " + " & ".join(str(q) for q,_ in rows) + "\\\\\n")
    f.write("$(s_4,s_2,s_1)$ of $X_j$ (from $Z$-checks) = of $Z_j$ (from $X$-checks) & " + " & ".join(s for _,s in rows) + "\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- [[5,1,3]]
gens513 = [P("XZZX_"), P("_XZZX"), P("X_XZZ"), P("ZX_XZ")]
table = {}
for name, e in single_errors(5):
    s = syndrome(e, gens513)
    assert s not in table, "collision"
    table[s] = name
assert len(table) == 15 and "0000" not in table
with open("data/syn_513.tex", "w") as f:
    keys = sorted(table)
    f.write("\\begin{tabular}{@{}l" + "c"*15 + "@{}}\\toprule\n")
    f.write("$s_1s_2s_3s_4$ & " + " & ".join(keys) + "\\\\\n")
    f.write("error & " + " & ".join("$"+table[k]+"$" for k in keys) + "\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- Shor
shor_z = [P("ZZ_______"),P("_ZZ______"),P("___ZZ____"),P("____ZZ___"),P("______ZZ_"),P("_______ZZ")]
shor_x = [P("XXXXXX___"),P("___XXXXXX")]
with open("data/syn_shor.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lccccccccc@{}}\\toprule\n")
    f.write("qubit $j$ & " + " & ".join(str(q) for q in range(1,10)) + "\\\\\n\\midrule\n")
    xs = []
    zs = []
    for q in range(9):
        e = ["_"]*9; e[q]="X"; xs.append(syndrome(P("".join(e)), shor_z))
        e = ["_"]*9; e[q]="Z"; zs.append(syndrome(P("".join(e)), shor_x))
    f.write("$X_j$ ($Z$-checks, by block) & " + " & ".join(s[:2]+"\\,"+s[2:4]+"\\,"+s[4:] for s in xs) + "\\\\\n")
    f.write("$Z_j$ ($X_{1..6},X_{4..9}$) & " + " & ".join(s for s in zs) + "\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
    # degeneracy: Z_1,Z_2,Z_3 share a syndrome and differ by a stabiliser
    assert zs[0]==zs[1]==zs[2] and zs[3]==zs[4]==zs[5] and zs[6]==zs[7]==zs[8]

# ---------------------------------------------------------------- split codes.dat by family
fam = collections.defaultdict(list)
hdr = None
for line in open("data/codes.dat"):
    if line.startswith("n "): hdr = line; continue
    parts = line.split(None, 6)
    fam[parts[5]].append(line)
for k, lines in fam.items():
    with open(f"data/codes_{k}.dat", "w") as f:
        f.write(hdr); f.writelines(lines)
print("wrote", sorted(fam), {k: len(v) for k, v in fam.items()})
