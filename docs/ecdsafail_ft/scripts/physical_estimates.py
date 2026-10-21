"""Physical-resource estimates used in the document.

Model S ("superconducting surface code, Gidney-2025 conventions", identical to the model in
Julian's shor_qiskit/physical.py so that the two documents agree):
  p = 1e-3, 1 us cycle, 10 us reaction time, per-patch per-round target 1e-15 -> d = 25,
  hot logical qubit = 2(d+1)^2 = 1352 physical, compute block 7x18 patches = 170,352 physical
  (six CCZ factories + three columns of workspace), one CCZ per 25 us -> time = Toffolis * 25 us.
Also prints the Litinski-style per-gate accounting.
"""
import math, json

class Surface:
    def __init__(s, p=1e-3, cycle_us=1.0, reaction_us=10.0, target=1e-15, a=0.1, p_th=0.0122, factories=6):
        s.p, s.cycle_us, s.reaction_us, s.target, s.a, s.p_th, s.factories = p, cycle_us, reaction_us, target, a, p_th, factories
    def pl(s, d): return s.a * (s.p / s.p_th) ** ((d + 1) / 2)
    def distance(s):
        d = 3
        while s.pl(d) > s.target: d += 2
        return d
    def hot(s, d=None): d = d or s.distance(); return 2 * (d + 1) ** 2
    def compute(s, d=None): return 7 * 18 * s.hot(d)
    def ccz_us(s): return 150.0 / s.factories * s.cycle_us

def estimate(label, logical, toffoli, code=None, shots=1.0, depth=None):
    code = code or Surface()
    d = code.distance(); hot = code.hot(d)
    phys = logical * hot + code.compute(d)
    t_ccz = toffoli * code.ccz_us() * 1e-6
    t_react = (depth or 0) * code.reaction_us * 1e-6
    secs = max(t_ccz, t_react)
    rounds = secs / (code.cycle_us * 1e-6)
    logical_all = logical + 7 * 18
    p_ok = (1 - code.target) ** (logical_all * rounds)
    return dict(label=label, logical=logical, toffoli=toffoli, d=d, hot=hot, physical=phys,
                hours=secs / 3600, p_ok=p_ok, days=secs * shots / 86400 / p_ok)

LOOKUPS = 3 * 2 ** 16   # per windowed addition, IonQ/ECDSA.fail convention
NADD = 28
rows = [
 # label, logical qubits (incl. 16 window qubits where the source excludes them), Toffolis per run
 ("ECDSA.fail 8e9c9a2 (26 Jul incumbent) + 16 window qubits, 28 adds + lookups", 1151 + 16, NADD * (1_299_453 + LOOKUPS)),
 ("ECDSA.fail windowed 8e9c9a2 (Note F; lookups included)", 1162, NADD * 1_683_828),
 ("ECDSA.fail a02c716 (7 Oct frontier, Leapfrog) + 16, 28 adds + lookups", 1244 + 16, NADD * (797_905 + LOOKUPS)),
 ("ECDSA.fail b6f2b0a low-Q (825) + 16, 28 adds + lookups", 825 + 16, NADD * (489_161_900 + LOOKUPS)),
 ("ECDSA.fail 10177cd low-Q (792, 7 Oct) + 16, 28 adds + lookups", 792 + 16, NADD * (656_602_578 + LOOKUPS)),
 ("Google/Babbush et al. low-qubit variant (whole algorithm)", 1200, 90_000_000),
 ("Google/Babbush et al. low-gate variant (whole algorithm)", 1450, 70_000_000),
 ("IonQ/Haener et al. (whole algorithm)", 1457, 39_000_000),
 ("Schrottenloher space-optimised (whole algorithm)", 1208, 72_490_961),
 ("Schrottenloher gate-optimised (whole algorithm)", 1462, 57_698_094),
 ("Julian: IonQ cells, signed windows, SELECT-SWAP (whole algorithm, executed)", 1457, 36_758_053),
 ("Julian: dialog on the space cells (whole algorithm)", 1246, 79_056_082),
 ("Julian: three field registers, Luo EEA (whole algorithm)", 851, 2_385_774_798),
 ("RSA-2048, Gidney 2025 (published inputs; 9.2 shots of 12.07 h)", 1399 + 0, 6_500_000_000),
]
code = Surface()
print(f"Model S: d={code.distance()} hot={code.hot()} compute={code.compute()} CCZ period={code.ccz_us()} us")
out = []
for label, q, t in rows:
    r = estimate(label, q, t, code)
    out.append(r)
    print(f"{label[:60]:60s} Q={q:5d} T={t:.3e} phys={r['physical']:,.0f} hours={r['hours']:.3f} days={r['days']:.3f} p_ok={r['p_ok']:.3f}")
json.dump(out, open("data/physical_modelS.json", "w"), indent=1)

# Litinski-style per-gate accounting at p=1e-3, d=25
d = 25
print("\nPer-gate accounting (surface code, d=25, p=1e-3):")
print(f" logical qubit patch: d^2 data + (d^2-1) ancilla = {2*d*d-1} physical; with routing share 2(d+1)^2 = {2*(d+1)**2}")
print(f" lattice-surgery CNOT/joint measurement: + one ancilla tile {d*d} data (~{2*d*d} physical) for d = {d} rounds = {d*code.cycle_us} us")
print(f" T gate (fast layout): 1 tick = d rounds = {d} us, consumes one |T> from a factory")
fac = {"(15-to-1)_{17,7,7}": (4620, 42.6, 4.5e-8), "(15-to-1)^6_{13,5,5} x (20-to-4)_{23,11,13}": (43300, 130, 1.4e-10),
       "(15-to-1)^6_{13,7,7} x (8-to-CCZ)_{25,15,15}": (47000, 60.0, 5.2e-11)}
for k, (q, c, e) in fac.items():
    print(f"  factory {k}: {q:,} qubits, {c} cycles per state, p_out={e:.1e}")
print(" cultivation (Gidney-Shutty-Jones 2024): ~2e-9 per T at p=1e-3, ~5e4 qubit-rounds per T (Babbush et al.)")
T_total = 4 * NADD * (1_299_453 + LOOKUPS)
print(f" ECDSA.fail 8e9c9a2 full attack: {NADD*(1_299_453+LOOKUPS):,} Toffoli = {T_total:,} T (4T Toffoli) -> at 1e-10/T budget error {T_total*1.4e-10:.3f}")
