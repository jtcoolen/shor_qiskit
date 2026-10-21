"""Verify the encoding circuits and syndrome tables quoted in the document with Stim.

Checks, for each code:
  * the encoder maps |0>|0..0> to a +1 eigenstate of every stabiliser and of Z_L,
    and |+>|0..0> to a +1 eigenstate of X_L  (TableauSimulator.peek_observable_expectation)
  * the Heisenberg picture: X on the data qubit -> X_L (mod stabilisers), Z -> Z_L
  * syndrome tables for all single-qubit Paulis
"""
import itertools, stim

def paulistr(s):
    return stim.PauliString(s)

def check_encoder(name, n, data_q, enc, stabs, xl, zl):
    print(f"=== {name} ===")
    # |0_L>
    sim = stim.TableauSimulator()
    sim.do(enc)
    for s in stabs + [zl]:
        e = sim.peek_observable_expectation(paulistr(s))
        assert e == 1, (name, s, e)
    # |+_L>
    sim = stim.TableauSimulator()
    sim.h(data_q)
    sim.do(enc)
    for s in stabs + [xl]:
        e = sim.peek_observable_expectation(paulistr(s))
        assert e == 1, (name, s, e)
    # Heisenberg check: conjugate X_data and Z_data through the encoder
    tab = enc.to_tableau()
    xs = tab.x_output(data_q); zs = tab.z_output(data_q)
    # membership of xs*xl^-1 in the stabiliser group: check it commutes with all logicals and stabilisers
    def in_stab_group(p):
        # p is in the stabiliser group iff it commutes with everything in the normaliser's
        # generators AND is +1 on the code space; test on |0_L>,|+_L> expectation
        for st in (stabs + [xl, zl]):
            if not p.commutes(paulistr(st)):
                return False
        for prep in (lambda s: None, lambda s: s.h(data_q)):
            sim = stim.TableauSimulator(); prep(sim); sim.do(enc)
            if sim.peek_observable_expectation(p) != 1:
                return False
        return True
    assert in_stab_group(xs * paulistr(xl)), ("X_L mismatch", xs)
    assert in_stab_group(zs * paulistr(zl)), ("Z_L mismatch", zs)
    print(f"encoder OK: X_data -> {xs}  (= X_L mod S),  Z_data -> {zs}  (= Z_L mod S)")
    print("gates:", enc.num_operations if hasattr(enc,'num_operations') else len(enc))
    return tab

def syndrome_table(name, n, stabs):
    table = {}
    for q in range(n):
        for P in "XYZ":
            err = ["I"]*n; err[q] = P
            e = paulistr("".join(err))
            syn = tuple(0 if e.commutes(paulistr(s)) else 1 for s in stabs)
            table.setdefault(syn, []).append(f"{P}{q}")
    collisions = {k:v for k,v in table.items() if len(v) > 1}
    print(f"{name}: {len(table)} distinct syndromes for {3*n} single-qubit errors; collisions: {collisions}")
    return table

# ---------------- Steane [[7,1,3]] ----------------
steane_stabs = ["___XXXX", "_XX__XX", "X_X_X_X", "___ZZZZ", "_ZZ__ZZ", "Z_Z_Z_Z"]
steane_enc = stim.Circuit("""
CX 0 1 0 2
H 3 4 5
CX 3 0 3 1 3 6
CX 4 0 4 2 4 6
CX 5 1 5 2 5 6
""")
check_encoder("Steane [[7,1,3]]", 7, 0, steane_enc, steane_stabs, "XXX____", "ZZZ____")
st = syndrome_table("Steane", 7, steane_stabs)
print("Steane X-error syndromes (Z-checks s4 s5 s6):")
for q in range(7):
    err=["_"]*7; err[q]="X"; e=paulistr("".join(err))
    print(" X%d ->"%q, "".join("1" if not e.commutes(paulistr(s)) else "0" for s in steane_stabs[3:]))

# ---------------- Shor [[9,1,3]] ----------------
shor_stabs = ["ZZ_______","_ZZ______","___ZZ____","____ZZ___","______ZZ_","_______ZZ",
              "XXXXXX___","___XXXXXX"]
shor_enc = stim.Circuit("""
CX 0 3 0 6
H 0 3 6
CX 0 1 0 2 3 4 3 5 6 7 6 8
""")
check_encoder("Shor [[9,1,3]]", 9, 0, shor_enc, shor_stabs, "ZZZZZZZZZ", "XXXXXXXXX")
syndrome_table("Shor", 9, shor_stabs)

# ---------------- [[5,1,3]] ----------------
five_stabs = ["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"]
# derive an encoder: tableau whose Z-images are S1..S4 and Z_L ; data qubit = 4
tab = stim.Tableau.from_stabilizers([paulistr(s) for s in five_stabs] + [paulistr("ZZZZZ")],
                                    allow_redundant=False, allow_underconstrained=False)
five_enc = tab.to_circuit(method="elimination")
print("[[5,1,3]] derived encoder:\n", five_enc)
check_encoder("[[5,1,3]]", 5, 4, five_enc, five_stabs, "XXXXX", "ZZZZZ")
tbl = syndrome_table("[[5,1,3]]", 5, five_stabs)
print("[[5,1,3]] syndrome table (s1 s2 s3 s4):")
for syn, errs in sorted(tbl.items()):
    print(" ", "".join(map(str,syn)), errs)

# ---------------- [[15,1,3]] Reed-Muller (punctured RM(1,4)) ----------------
# X stabilisers: rows of the [15,4] simplex-code-like H (weight 8), Z stabilisers: all weight-4 ...
# We only verify transversal-T later in the text; here verify the CSS structure counts.
import numpy as np
def rm_x_stabs():
    rows=[]
    for b in range(4):
        rows.append("".join("X" if ((i>>b)&1) else "_" for i in range(1,16)))
    return rows
def rm_z_stabs():
    # Z stabilisers: products of pairs of the X-type rows (weight 4 patterns) -> generators of RM(2,4)* dual
    rows=[]
    for b1,b2 in itertools.combinations(range(4),2):
        rows.append("".join("Z" if (((i>>b1)&1) and ((i>>b2)&1)) else "_" for i in range(1,16)))
    for b in range(4):
        rows.append("".join("Z" if ((i>>b)&1) else "_" for i in range(1,16)))
    return rows
xs=rm_x_stabs(); zs=rm_z_stabs()
ok = all(paulistr(a).commutes(paulistr(b)) for a in xs for b in zs)
print("[[15,1,3]] RM: X stabs", len(xs), "Z stabs", len(zs), "all commute:", ok)
