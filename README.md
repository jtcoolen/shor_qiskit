# Shor's algorithm in Qiskit — verified implementation and fast variants

Companion code to **`shor-complete.pdf`** (*Reversible modular arithmetic in Shor's
algorithm*). Every listing in Parts IV–V of that document is copied from a file in
here, and every number quoted as "measured" was produced by `bench/`.

The implementation is self-contained: no Qiskit circuit-library arithmetic, and the
QFT is built from scratch. It factors **15, 21 and 33** end to end.

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```python
import sys; sys.path.insert(0, "shor_qiskit")
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from shor_essentials import order_circuit, order_from_counts, find_factor

SIM, SAMPLER = AerSimulator(), Sampler(seed=7)
PM = generate_preset_pass_manager(backend=SIM, optimization_level=1)
run = lambda qc, shots=4096: (
    SAMPLER.run([PM.run(qc)], shots=shots).result()[0].data.out.get_counts())

counts = run(order_circuit(7, 15))            # peaks at 0, 64, 128, 192
print(order_from_counts(counts, 7, 15, t=8))  # -> 4
print(find_factor(15, run))                   # -> 3 or 5
```

### The walkthrough notebook

`notebooks/shor_walkthrough.ipynb` presents all of it in one place, executed, with outputs saved:
the minimal circuits level by level as products of unitaries (QFT → Draper adder → Beauregard
modular adder → multiplier → order finding; then ECDLP), their qubit / Toffoli / T / rotation counts
checked against closed forms, a fault-tolerance section (Clifford+T, Solovay–Kitaev against
Ross–Selinger synthesis, why small phase shifts cost ~100 T), every optimisation with its measured
effect, and Aer runs with each histogram set beside the exact distribution. Re-executes in ~2 min.

```bash
./venv/bin/pip install -r notebooks/requirements.txt
./venv/bin/jupyter lab notebooks/shor_walkthrough.ipynb
```

### The presentation

`presentation/shor_circuits.pdf` is a 60-minute talk in the theme of `doc/references/shor_pres_v4_1.pdf`. It covers:
- the circuits for factoring and ECDLP, built in Qiskit;
- uncomputation, and why garbage kills the interference;
- Clifford+T, Toffoli and Solovay–Kitaev;
- qubit, Toffoli and T counts derived from the construction;
- the optimisations, each with its measured effect;
- why ECDSA falls before RSA at equal classical security.

Its figures and numbers are generated from `shor_qiskit/` with the notebook's seeds. Its code excerpts are cut from the
source and checked, and the usage snippets are executed.

```bash
venv/bin/python presentation/make_figures.py     # figures/ and numbers.json (~1 min)
venv/bin/python presentation/make_listings.py    # listings/, checked against the source
cd presentation && tectonic -X compile shor_circuits.tex
```

Or the whole pipeline as a toy attack — RSA key → factor N → private exponent →
decrypt, and an elliptic-curve key pair → recover the private key — both on the
one-counting-qubit circuits, with every measured histogram printed beside its
exact distribution:

```bash
./venv/bin/python examples/toy_demo.py            # N=15 and a 13-point curve, ~10 s
./venv/bin/python examples/toy_demo.py --N 21     # 1-2 min (dynamic circuits run shot by shot)
./venv/bin/python examples/toy_demo.py --full     # full counting registers, for comparison
```

## Layout

| file | what it is | note § |
|---|---|---|
| `shor_qiskit/shor_essentials.py` | the complete minimal implementation: QFT, Fourier adder, modular adder, multiplier, ladder, classical post-processing | IV |
| `shor_qiskit/rc_adder.py` | ripple-carry (CDKM) arithmetic — `X`/`CNOT`/`Toffoli` only, no rotations | V.18 |
| `shor_qiskit/qrom.py` | unary-iteration lookup (`w` ancillas, not `2^w`), the √L phase fixup, and **temporary ANDs** (4T compute, 0T uncompute) | V.19.5–6, V.20 |
| `shor_qiskit/windowed.py` | Gidney windowing over the multiplicand; quantum-addend modular adder | V.19 |
| `shor_qiskit/nested.py` | windowing over the **exponent** as well (Gidney §3.5) | V.19.7 |
| `shor_qiskit/unlookup.py` | measurement-based uncomputation — hybrid, so it ships as a runner | V.19.6 |
| `shor_qiskit/semiclassical.py` | the **semiclassical inverse QFT** (Griffiths–Niu): one recycled counting qubit, shared by factoring and ECDLP | V.21 |
| `shor_qiskit/onectrl.py` | order finding with **one** counting qubit: 2n+3 instead of 4n+2 | V.21 |
| `shor_qiskit/shor_stats.py` | exact output distributions of both circuits, per-shot success probability, TVD against the shot-noise null | — |
| `shor_qiskit/resources.py` | logical resource counts for any circuit here: qubits, Toffolis, T (plain and temporary-AND), controlled phases, small-angle rotations and their synthesis cost | — |
| `shor_qiskit/coset.py` | Zalka **coset representation**: deletes the modular adder — a plain adder does modular arithmetic | V.22 |
| `tests/` | the verification suite; each file asserts its own results | IV.17 |
| `examples/toy_demo.py` | toy RSA and toy ECDLP broken end to end on Aer | — |
| `bench/` | the cost measurements quoted in the document | — |

Two adders and three multiplier variants are interchangeable: everything above
Level 1 is written once, against the adder interface.

## Running the tests

```bash
./run_tests.sh fast     # unit tests, exhaustive on small moduli   (~3 min)
./run_tests.sh full     # adds end-to-end factoring of 15          (~30 min)
./run_tests.sh all      # adds N=21 and N=33                       (hours, 8 GB peak)
```

`pytest` also works (`pytest.ini` sets the paths). Use `PYTHON=/path/to/python`
to pick an interpreter.

The semiclassical QFT has its own unit test, `test_semiclassical.py` (in `fast`,
~40 s), and so does the resource counter, `test_resources.py` (~4 s: every closed-form count
the notebook quotes, and T pricing cross-checked against Qiskit's own Clifford+T decomposition); the one-control circuits are exercised end to end by `test_1c.py`
(factoring, in `full`) and `test_ec_shor.py` (ECDLP, in `ec`). Any single file runs
on its own:

```bash
PYTHONPATH=shor_qiskit:tests ./venv/bin/python tests/test_semiclassical.py
```

The one-control circuit (`test_1c.py`) defaults to N=15 and 21. Mid-circuit
measurement forces Aer to simulate **shot by shot** rather than evolve one
statevector, so its cost grows steeply despite the low qubit count — N=33 takes
~17 min on 15 qubits. Set `SHOR_1C_BIG=1` to include N=33/35/51/143.

## What is verified

Exhaustively, by exact simulation, with **every ancilla asserted back to |0⟩** —
a circuit can compute the right value and still be broken:

| level | check | result |
|---|---|---|
| 0 | `qft` vs the definition and vs Qiskit's `QFTGate`, n=1..6 | exact |
| 0 | `lookup_ui` every address, m=1..4 | exact `2(L−1)` Toffoli, `m` ancillas |
| 0 | `phase_fixup` = `diag((−1)^F[a])`, every split | exact |
| 1 | `add_const`, `rc_add` exhaustive incl. negative constants | correct, scratch clean |
| 2 | `c_add_mod`, `rc_c_add_mod`, `add_quantum_mod` all (y,X) for N=9,15,21,33 | correct, flag clean |
| 3–4 | `c_ua`, `rc_c_ua`, windowed and nested variants | correct, scratch clean |
| 5 | order finding, N=15 / 21 / 33 | see below |
| 5 | semiclassical inverse QFT vs `qft(t).inverse()`: every Fourier state t=1..6, and arbitrary non-commuting rungs on entangled inputs t=1..5 | exact (branch-enumerated, not sampled) |
| 5 | one-control order finding, N=15 every base | distribution **exactly** the closed form |
| 5 | measured histograms, every end-to-end run | within shot noise of the exact distribution |

End-to-end factoring:

| N | a | r | circuit | qubits | result |
|---|---|---|---|---|---|
| 15 | 7, 2 | 4 | plain, t=2n=8 | 18 | 4 sharp peaks, all shots on peak → 3 × 5 |
| 21 | 2 | 6 | plain, t=2n=10 | 22 | six peaks, smeared (6 ∤ 2ᵗ) → 7 × 3 |
| 33 | 5, 10 | 10, 2 | plain, t=2n=12 | 26 | ten peaks → 11 × 3 |
| 15 | 7, 2 | 4 | **windowed**, t=2n=8 | 24 | → 3 × 5 |
| 21 | 2 | 6 | **windowed**, t=2n=10 | 29 | → 7 × 3 |
| 15 | 7, 2 | 4 | **nested** windowed | 23 | 2 multiplications not 4 → 3 × 5 |
| 15 | 7, 2 | 4 | **one-control** | **11** | 2n+3 qubits → 3 × 5 (6 s) |
| 21 | 2 | 6 | **one-control** | **13** | 2n+3 qubits → 7 × 3 (48 s) |

**Test the distribution, not the factors.** A garbage bug does not give a wrong
answer, it gives a *flat histogram* — and the classical tail verifies its own
candidates, so it can still print the right factors from pure noise.

So the output distribution is computed in closed form (`shor_stats.py`) — for
order finding `P(y) = 4^-t Σ_{x0<r} |Σ_m e^{-2πi m r y/2^t}|²`, for ECDLP the
analogous sum over the lattice `u + kv ≡ z (mod r)` — and checked three ways: it
equals Aer's exact statevector probabilities of the full circuit (to 1e-14), it
equals the branch-enumerated distribution of the one-control circuit (to 1e-13),
and every sampled histogram in the end-to-end tests sits within the total-variation
distance a perfect sampler would reach at that shot count (99.99% quantile,
Monte Carlo). Measured, 2048 shots, one counting qubit:

| N | a | qubits | TVD from exact | noise bound | P(one shot gives r): exact | measured |
|---|---|---|---|---|---|---|
| 15 | 7 | 11 | 0.030 | 0.044 | 0.500 | 0.470 |
| 21 | 2 | 13 | 0.045 | 0.080 | 0.322 | 0.328 |

Flat noise lands at 11× the bound, so the test has teeth; and so does the
semiclassical unit test, which catches each of three deliberate bugs (dropped
phase corrections, reversed rung order, missing reset) at TVD 0.48–0.91.

## What is implemented, from which paper

* Draper, *Addition on a quantum computer* — the Fourier adder.
* Beauregard, *Circuit for Shor's algorithm using 2n+3 qubits* — the seven-block
  modular adder, the in-place multiplier, the control placement.
* Cuccaro–Draper–Kutin–Moulton — the ripple-carry adder.
* Babbush et al. — unary-iteration table lookup.
* Griffiths–Niu semiclassical QFT, via Mosca–Ekert / Parker–Plenio / Beauregard —
  the one-counting-qubit circuit (`onectrl.py`).
* Gidney, *Halving the cost of quantum addition* — temporary ANDs: 4 T to compute,
  0 T to uncompute. Measured 3.50× T-count reduction on every lookup.
* Gidney, *Windowed quantum arithmetic* — windowed modular product addition (§3.3),
  modular multiplication (§3.4), **nested** exponent + multiplication windows (§3.5),
  and measurement-based uncomputation (Fig. 3). §3.2's in-place multiplication is
  not implemented and is not needed: we multiply out-of-place and swap, per Fig. 6.

## Where this sits

This is **2019-era Shor**, complete: everything through Gidney's windowed arithmetic,
plus the semiclassical QFT and temporary ANDs. It is *not* state of the art. The
lineage, and where this stops:

| year | construction | here |
|---|---|---|
| 1996 | Vedral–Barenco–Ekert, 7n+1 qubits | superseded |
| 2003 | Beauregard modular arithmetic, 2n+3 | **built** |
| 2004 | CDKM ripple-carry | **built** |
| 2018 | Gidney temporary AND | **built** |
| 2019 | Gidney windowed arithmetic | **built, in full** |
| 2021 | Gidney–Ekerå: Zalka coset representation | **built** (`coset.py`) |
| 2021 | Gidney–Ekerå: oblivious carry runways | absent — depth only |
| 2017–25 | Ekerå–Håstad short discrete logarithm (1.5n exponent qubits) | **built** (`ekera_hastad.py`) |
| 2025 | Chevignard–Fouque–Schrottenloher: residue number system | absent |
| 2025 | Gidney: approximate residue arithmetic + yoked surface codes | absent |

The coset representation is built and measured (`coset.py`, note §V.22): a value is
stored as the periodic superposition `sum_j |jN + k>` over `cpad` padding qubits, and
a **plain adder then does modular arithmetic**, because the periodicity makes the sum
slide instead of needing a reduction. Measured saving over the seven-block modular
adder: **1.7–2.8×** with the Fourier adder, **3.0–3.9×** with ripple-carry (padding
costs less when the adder is linear rather than quadratic in width).

What remains from Gidney–Ekerå 2021 is **oblivious carry runways**, which attack depth
rather than gate count — the ripple-carry adder is Θ(n)-deep by construction. Beyond
that the lineage leaves this architecture entirely (RNS).

A separate lineage (Regev 2023; Ragavan–Vaikuntanathan 2024) changes the algorithm
rather than the arithmetic, and needs a quantum×quantum multiplier — structurally
different from the quantum×classical-constant multiplier used throughout here.

## Known limits

* Measurement-based uncomputation is **hybrid by necessity** — its fixup table
  depends on the measurement outcome, so it is circuit → measure → classical →
  circuit, and cannot be one static circuit. `run_windowed_acc_mbu` does exactly
  that; on hardware the classical controller sits in the middle.
* The CNOT/depth figures in the windowing sections come from the *static*
  circuits, which uncompute by re-running the lookup. They are a **lower bound**:
  measurement-based uncomputation is cheaper by 2×–26× depending on window size.
* Simulation cost is exponential in qubit count, so this stops at a few bits. On
  real hardware the binding constraint is noise, not qubits.
* The coset representation is **approximate**: deviation ~2^-cpad per addition,
  subadditive over a sequence. Everything else here is exact and asserts exact
  equality; coset tests instead measure an error *rate* against the padding budget.
  `coset.py` supplies the adder and multiplier; **the full coset order-finding
  circuit is not built** — the multiply/swap/uncompute of Level 4 in coset form is
  the remaining piece.

---

# Shor for the elliptic-curve discrete logarithm

Everything above factors integers. This half solves the *elliptic-curve
discrete logarithm* instead, and shares nothing with it but the QFT and the
ripple-carry adder.

Companion implementation of **[eprint 2026/106](https://eprint.iacr.org/2026/106.pdf)**
(Kim, Jang, Wang, Srivastava, Baksi, Song, Seo, Chattopadhyay, *New Quantum
Circuits for ECDLP*) and **[eprint 2026/1128](https://eprint.iacr.org/2026/1128.pdf)**
(Schrottenloher, *Optimized Point Addition Circuits for Elliptic Curve Discrete
Logarithms*), with the [Classiq ECDLP
tutorial](https://docs.classiq.io/explore/algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log)
as the simple reference point.

Given `P` and `Q = [k]P` on a curve over `GF(p)`, recover `k`. Same shape as
order finding — superposition, oracle, QFT, classical post-processing — but the
group is an elliptic curve, so the oracle is point addition, and point addition
needs a **modular inversion**. That inversion is the whole cost, and it is what
both papers attack.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_tests.sh ec
```

```python
import sys; sys.path.insert(0, "shor_qiskit")
import ec_classical as C, ec_shor as S

curve, G = C.CLASSIQ, C.CLASSIQ_G          # y^2 = x^3 + 5x + 4 mod 7, ord(G) = 5
Q = curve.mul(3, G)
k, counts, info = S.solve(curve, G, Q, order=5)
print(k, info["qubits"])                   # -> 3, 12
```

## Reading order

Start at `ec_classical.py`. Every circuit in this half is checked against a
function there, so it doubles as the specification. Then `ec_kaliski.py` (the
reference inversion), then `ec_pointadd.py` (the eleven-step addition), then
whichever optimization interests you.

| file | what it is | source |
|---|---|---|
| **infrastructure** | | |
| `ec_gates.py` | Gidney temporary AND (4 T compute, 0 T uncompute) and the emission context | [Gid18], [JNRV19] |
| `ec_sim.py` | registers, ancilla pooling, exact basis-state simulation, circuit inversion | — |
| `ec_classical.py` | the classical model of every circuit here | — |
| `ec_cost.py` | resource counting; the papers' formulas at cryptographic `n` | — |
| **reference stack** | | |
| `ec_adders.py` | CDKM (2n Toffoli) and Gidney (n Toffoli) adders, comparators, shifts | [CDKM04], [Gid18] |
| `ec_modarith.py` | exact modular add / sub / double / halve / negate, controlled and constant forms | [RNSL17] Alg 5, 8 |
| `ec_mult.py` | schoolbook modular multiplication, squaring, accumulate forms | — |
| `ec_kaliski.py` | Kaliski modular inversion, and division | 106 Alg 2, [HJN+20] |
| `ec_pointadd.py` | in-place controlled affine point addition | 106 Alg 3 |
| `ec_shor.py` | the full ECDLP circuit, both oracles, the semiclassical one-control variant (via `semiclassical.py`), Ekerå post-processing | 106 Fig 1 + App D, 1128 §2, [Eke19] |
| **2026/106** | | |
| `ec_qcsa.py` | quantum carry-save adder: w addends to two in O(log w) depth | 106 §3.2, [KLJ+25] |
| `ec_montgomery.py` | word-level Montgomery (Alg 1b reformulation) + the [HJN+20] QROM baseline | 106 §3.2 |
| `ec_kaliski_opt.py` | unconditional rounds, postponed reduction, control fan-out | 106 §3.3 |
| `ec_proj.py` | Jacobian-affine out-of-place addition (11 multiplications), zig-zag | 106 §4.2, Alg 4, Fig 10 |
| **2026/1128** | | |
| `ec_eea.py` | Euclidean dialog, Bézout replay, in-place multiplication, Fig 1 compression | 1128 §3, Alg 2–4 |
| `ec_approx.py` | approximate and pseudo-Mersenne modular arithmetic | 1128 §4, Alg 6, 7, 9, 10, 11 |
| `ec_window.py` | windowed point addition | 1128 Alg 1 |

## How this is verified

Three layers, because no one of them is sufficient.

**Exact basis-state simulation at full width.** Every circuit here is a
permutation of basis states, so pushing one basis state through it in O(gates)
time is exact, and scales to hundreds of qubits where a statevector dies at
thirty. `ec_sim.simulate` does that. Ancillas are checked back to `|0>` *at the
instruction where they are freed*, not merely at the end — a circuit can compute
the right value and still be broken.

**Statevector cross-checks** (`tests/test_ec_quantum.py`). The fast simulator
only sees basis states, and a bug in it would hide itself from every other test.
So the small circuits are also evolved on genuine superpositions and checked for
exact fidelity *and* for the absence of any leaked amplitude — the statement that
ancillas are unentangled, which basis-state testing cannot see. A stray diagonal
gate would show up here as a phase, and phases are exactly what Shor depends on.

**Property-based tests over random curves** (`tests/test_ec_pbt.py`). Random
prime, random non-singular `(a, b)`, random points and scalars — that is where
curve-shape assumptions hide. The harness (`tests/_pbt.py`) shrinks failures,
staying inside each field's declared domain and re-checking its own result, and
reports a reproduction line. It has been checked against injected bugs: it
isolates the exact offending modulus.

Verified end to end:

| what | result |
|---|---|
| classical models (Kaliski, Montgomery, dialog, Jacobian, zig-zag) | exact; 106's own worked example (n=192, w=11 → 36 additions, m=8) reproduced |
| adders, comparators, modular arithmetic | exhaustive for `p ≤ 61`, randomized to n=128 |
| Kaliski inversion and division | exhaustive; per-round records and counter all return to `|0>` |
| affine point addition | every point pair on two toy curves, both control branches |
| projective point addition | every point pair × every projective representative |
| in-place multiplication (dialog) | every `(x, y)` for `p ≤ 127` |
| **ECDLP end to end** | every discrete log on both toy curves; correct `k` is the **top** candidate every time |
| ECDLP, semiclassical (one control qubit) | same, on 7 qubits instead of 12 (9 instead of 16 at p=11); distribution **exactly** the closed form on p=7 |
| measured histograms, both variants, every k | within shot noise of the exact distribution; P(one shot gives k) 0.644 exact vs 0.645 measured at p=11 |

That last row is the one that matters, and it is stated carefully. A broken
Shor circuit does not return a wrong answer — it returns a *flat histogram*, and
the classical tail then verifies its own candidates and prints the right key
from pure noise. So the test asserts that the correct `k` wins the vote, and a
flat-input control shows a top share of 21% against a uniform 20%: no signal
where there should be none.

## The write-up

`shor-complete.pdf` (Part VII, pp. 54-77) covers this half in the same style as the rest of
the document: what changes when the group is a curve; a full derivation of *why* it works —
the period lattice, its dual, the amplitude computed in closed form, and why the clean case
never actually arises for a prime-order point; why the group law needs a modular inverse and
why that is the whole cost; the eleven-step affine point addition step by step; Kaliski's
algorithm and the three bits per round that cannot be dropped; then each optimization with its
measured effect, an ablation of how they compose, and the verification story.

Every number in Part VII is generated, not typed:

```bash
./venv/bin/python bench/ec_ablation.py    # measure everything -> bench/ec_ablation.json
./venv/bin/python bench/ec_make_part7.py  # render Part VII from the template + that JSON
./venv/bin/python bench/ec_check_tex.py   # assert the .tex still matches the measurements
```

The code listings are generated the same way — `bench/ec_listings.py` pulls each function
out of `shor_qiskit/ec_*.py` by AST at build time, so a listing cannot drift from the
function it claims to show. Where one is shortened the elision is a visible `...`.

`ec_check_tex.py` regenerates the whole part and diffs it against the document, so neither a
stale number nor a stale listing can survive: if the code or the benchmark changes and the
text does not, it fails.

## Measured results

From `bench/ec_resources.py`. These are counted off circuits the suite verifies,
at toy widths — not extrapolations.

**Modular inversion, [HJN+20] against 106 §3.3:**

| n | ref Toffoli | opt Toffoli | saved | ref depth | opt depth | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5242 | 3354 | 36% | 12271 | 7019 | 43% |
| 12 | 11322 | 7530 | 33% | 27239 | 16059 | 41% |
| 16 | 19898 | 13370 | 33% | 48225 | 28775 | 40% |

for 6-12% more qubits (125->132 at n=8, 232->260 at n=16). The paper reports 58–60% T-depth improvement for
division at n=192–521; the direction and rough magnitude agree, against a
different baseline and at toy `n`.

**Point addition, the two constructions** (p=11): affine in-place, 83 qubits /
8632 Toffoli-eq; Jacobian out-of-place, 98 qubits / 5144 Toffoli-eq. The
projective one is much cheaper and needs no inversion — and leaves its 3n-qubit
input behind, which is what the zig-zag schedule exists to manage.

**Division against the Euclidean dialog** — 1128's structural saving:

| n | division qubits / Toffoli | dialog qubits / Toffoli | dialog saves |
|---:|---:|---:|---:|
| 4 | 83 / 3380 | 48 / 1749 | 48% |
| 6 | 115 / 7188 | 66 / 3435 | 52% |

**Temporary ANDs** buy a measured **2.50×** T-count reduction across the
inversion, matching the 7-vs-4-and-0 arithmetic.

**The semiclassical Fourier transform** (Griffiths–Niu) runs the whole thing on
one recycled control qubit: 7 qubits instead of 12 on the p=7 curve, 9 instead
of 16 on p=11. This is the assumption behind both papers reporting only the cost
of the point arithmetic — the two n-bit scalar registers cost one qubit between
them.

**Approximate arithmetic** behaves as the theory says — failure rate tracks
`2^-msbs` (12% → 3% → 0.8% as msbs goes 2 → 4 → 6), and pseudo-Mersenne doubling
costs 5 Toffoli against 17 exact.

## Exceptional cases, stated plainly

The reversible affine addition is correct except on a set of measure `O(1/p)`,
and the circuits do **not** detect it — a check would cost more than it saves.
There are three conditions, not the one usually quoted:

- `x1 == x2` — doubling, adding an inverse, or a point at infinity. This is the
  case both papers name.
- `x3 == x2`, i.e. `P1 = -2·P2` — step 8 divides by `x2 - x3`. Invisible in the
  addition formulas; it only appears once you follow what the *registers* hold.
  1128's in-place-multiplier variant hits the same wall.
- for the windowed circuit adding `O`: `x_R == 0`, because infinity is encoded
  as `(0,0)` and step 6 then divides by `x_R`. Adding the identity looks like it
  could never fail.

`ec_classical.point_add_exceptional` decides the first two; the tests exclude
them explicitly and report how many, rather than quietly passing.

One consequence worth naming: the `q = 0` branch of a controlled addition would
otherwise invert whatever junk its `x` register holds (`x1 + 2x2`, which vanishes
for perfectly ordinary accumulators). `ec_kaliski` fixes that by pushing the
control into the *inversion's setup* — load `u ← p`, `v ← x`, `s ← 1` under the
control, and when it is 0 every round is inert. n ANDs buy a controlled
inversion, which is what makes 106 Fig 8's conditional division blocks
affordable.

## The 2026 optimisations (IonQ, ECDSA.Fail, Litinski, Babbush, the rest of 1128)

Everything above implements [106] and [1128] as the papers print them.  This part adds
the refinements that took the published cost of one 256-bit windowed point addition
from ~2.6M Toffolis ([1128]) to ~1.4M (IonQ, arXiv:2609.05625), plus the GCD variants
of ECDSA.Fail (arXiv:2609.09582), each as a *switch*
next to the construction it refines, never in place of it: every default reproduces
the old builder gate for gate, and `tests/test_ec_regress.py` pins every number in
Part VII.

| module | what it adds | source |
|---|---|---|
| `sparse_sim.py` | sparse state-vector simulator with phases, measurement, feed-forward and supplied outcomes; `assert_coherent` checks a measurement-based uncompute exactly, at any width | — |
| `ec_mbu.py` | measurement-based uncomputation as logical gates (`LookupGate`, `UnlookupGate`, `PhaseFixGate`, `MbuFlagGate`) that the unitary machinery can simulate and invert; `run_live` performs the real X-measurements and repairs | [1128] §2, Litinski Fig 4b, IonQ Alg 6 |
| | ANF lookup (2^w − w − 1 ANDs) and power-product *phaseup* (~2√L ANDs) | Gidney 2025 App A |
| `depth.py` | Toffoli depth, reaction depth, expected (average-executed) Toffolis | — |
| `ec_adders`, `ec_modarith`, `ec_approx` (additions) | conditionally-inverted adder `ci_add` (n−1 ANDs), fused comparator `cgt_fused` (n+1), top-bit comparison, signed modular add, IonQ's complement-trick signed add and approximate negation | IonQ §VI, §X.D |
| `ec_window.windowed_point_add_cfg` | 3 lookups + 1 merged phase repair instead of 10 recomputed lookups; x/y loaded as one word; masked tables (no O entry, no control flag); freed x1/y1; serial loading; pluggable multiplier/squarer/negation | [1128] §2, IonQ Alg 4/6, ECDSA.Fail Era III |
| `ec_gcd.py` | one interface, five GCDs: [1128]'s dialog with fused/top-bit comparisons, width schedule, **register sharing**, the real **5-Toffoli Fig. 1** packing and x-reuse; IonQ's **conditionally-inverted** walk and replay; ECDSA.Fail's **ping-pong** and **Jump-2** walks and its **base-5** transcript codec; exact / approximate / pseudo-Mersenne replay arithmetic | [1128] §3–4, IonQ §VI, ECDSA.Fail §5.3.1–3 |
| `ec_square.py` | dedicated squarer (n(n+3)/2 ANDs) and a pseudo-Mersenne fold for step 10 | IonQ §VII.A, ECDSA.Fail §5.3.4 |
| `ec_shor.ecdlp_windowed` | the full windowed circuit: w recycled control qubits (`semiclassical_iqft_windowed`), first window as a lookup, dropped trailing windows + search post-processing, IonQ's masks; `ecdlp_multikey_1c` reuses one P-half for many keys | [1128] §2, Babbush App A, Litinski §4, IonQ §IV |
| `ekera_hastad.py` | factoring by a short discrete logarithm: 1.5n exponent qubits instead of 2n (s = 1) | Ekerå–Håstad, CFS 2024, Gidney 2025 |
| `physical.py` | surface code + yoked storage + cultivation/CCZ-factory model; reproduces Gidney 2025's 897,864 qubits / 4.96 days from its inputs | Gidney 2025 §3.2 |

**Measured at n = 256** (`bench/ec_project_256.py`: every coefficient is a gate count of a
circuit built at n = 256 with secp256k1's prime and verified exactly at toy sizes):

| | Toffoli-eq |
|---|---:|
| in-place multiplication, [1128] dialog as built before this | 1,955,276 |
| … with fused 48-bit comparisons, width schedule, pseudo-Mersenne replay | 842,330 |
| … IonQ conditionally-inverted walk, IonQ replay | **697,365** |
| … ECDSA.Fail ping-pong (704 rounds) / Jump-2 (261 steps), as composed here | 917,565 / 1,161,099 |
| square-subtract: general multiplier → dedicated squarer + fold | 1,050,623 → **79,160** |
| lookups per addition (w = 16): 10 recomputed → 3 MBU loads + 1 repair | 1,310,700 → **197,364** |
| **one windowed point addition**, before → after | **6,276,990 → 1,676,369** |
| published: [1128] / IonQ | 2,588,963 / 1,392,608 |
| **whole ECDLP-256**: 34 × before → 28 × after | 213M → **47M** |

The "after" column takes the approximate settings (48-bit comparisons, the width
schedule, pseudo-Mersenne arithmetic) at the papers' parameters; their failure rates are
measured at toy sizes (`tests/test_ec_gcd.py`, `test_ec_signed.py`, `test_ec_square.py`),
not at n = 256.  At toy sizes the fixed round budgets distort comparisons between GCDs
(e.g. [1128]'s 1.413n + 2.4√n has a large margin at n = 6), which is why the table is built
at n = 256.

Things the sources get wrong or leave out, found while reproducing them:
- IonQ's replay needs its 0 ↔ p swap *before* the complemented adder; without it every
  e = 1, y = 0 step leaves an ancilla dirty (`ec_approx._swap_zero_q`).
- ECDSA.Fail's ping-pong budget L = 2.75n is not a worst case: exhaustively, n = 16 needs
  3.9n. `PingPong(rounds=...)` exposes it; the tests report the failure rate at 2.75n.
- Gidney 2025's hot-qubit count "3f + 2ℓ + len m = 131" evaluates to 152 (926,256 qubits);
  the published 897,864 uses 131.  Its "9.1 shots" is 9.2.
- At toy N, the Gidney–Ekerå choice A = g^(N+1) puts d outside the short range and never
  factors; Gidney 2025's A = g^(N−1) sometimes does (d is pinned only modulo ord g).

Tests: `test_sparse_sim`, `test_depth`, `test_eh`, `test_physical`, `test_api_surface`
(fast tier); `test_ec_regress`, `test_ec_mbu`, `test_ec_window_cfg`, `test_ec_signed`,
`test_ec_gcd`, `test_ec_square`, `test_ec_windowed` (ec tier).

## What is not implemented

Named rather than glossed:

- **Register sharing** (1128 §3.1) is built in `ec_gcd.Dialog(c_pad=..., share=True)`
  (probabilistic, failure rate measured); `ec_eea` itself remains exact.  The
  **5-Toffoli Fig. 1** packing is `ec_gcd.fig1_compress`; `ec_eea.compress_records`
  keeps its generic 57-Toffoli permutation and is still not wired into `ec_eea`.
- **Gidney's dirty-ancilla constant adder** [Gid25]. Constant addition here uses
  clean ancillas: correct, and more qubits than the paper's accounting.
- **The Ed25519 extension** of 106 §4.4 — the same out-of-place strategy in
  extended Edwards coordinates. The curve model here is short Weierstrass only.
- **QLDPC physical estimates** (Pinnacle, Cain et al.). `physical.py` models the
  surface-code architecture of Gidney 2025 and records the QLDPC figures in
  `physical.REFERENCE`, but does not model them.

And one thing that is implemented but only in one of its two forms: 106 §3.2
distinguishes a *depth-optimized* Montgomery multiplier (fresh output qubits per
two-operand addition) from a *qubit-optimized* one (uncompute and reuse them).
`ec_montgomery` always accumulates in place, i.e. the qubit-optimized shape,
which is why the depth win from the carry-save tree shows up here smaller than
the paper's depth-optimized figures.

And one that is implemented, but not quite as printed: **Algorithm 11** of 1128
(`ec_approx.cmodadd_pm_q`), the pseudo-Mersenne controlled addition that also
handles `x + y = q`. Algorithm 10 cannot, and inside the multiplier that case is
certain, not rare: `r` must end the Bézout replay at 0 and only a swap writes
`r`, so every multiplication with `y ≠ 0` hits `r + s = q` exactly once, at the
replay step mirroring the dialog's first swap — near the *end* of the replay,
the very last step whenever `x` is odd. `ec_eea.bezout_replay` uses one adder for
every iteration, so Algorithm 7 + Algorithm 10 in `inplace_mul` fails on every
such input. As printed, Algorithm 11 clears its flag with an all-zeros test that
also fires for `ctrl = 0, y = 0`, which is the call the replay makes on its opening
iterations; here that test is gated on `ctrl ∧ [y < x]` (and the all-ones test on
`ctrl`). With full-width comparisons Algorithm 7 + Algorithm 11 is exact at
`q = 127` and fails at `q = 61` only where the pseudo-Mersenne shortcut itself
does (sums between `q` and `2^u`), which `tests/test_ec_opt1128.py` asserts
input by input.

The ECDLP circuit with real arithmetic is verified on every basis state — which
for a permutation is the whole truth — but at 69 qubits and ~10^5 gates for a
7-element field it cannot be run in superposition. The end-to-end demonstration
uses a permutation oracle of the same circuit shape. Both halves are tested; the
join between them is an argument, not a simulation, and that is the honest limit
of what runs on a laptop.
