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
  `coset.order_circuit_coset` is the full order-finding circuit in coset form
  (encoding circuit, multiply/swap/uncompute with plain additions);
  `tests/test_coset_order.py` checks its output distribution at N = 15 against the
  exact one, within the bound and falling as the padding grows.

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
| `ec_approx.modadd_pm_phase`, `ec_gcd.PMPhase`, `CondInv(zero_steps=...)` | IonQ's phase-approximate modular adder (carry-out flag, δ-bit phase repair), careful cells only where the replay's structural zero falls, multiplication replayed forwards | IonQ Alg 2, Table X |
| `PMPhase(lean=True)`, `CondInv(share=True, cnot_ends=True)`, `csub_square_pm(arith=...)` | IonQ's cells on 1,457 qubits: the careful cell's scratch freed after its adder, the replay's opening copy and closing clear as CNOTs, register sharing in the conditionally inverted walk (u's high qubits, never x's), the square-subtract's fold on the phase adder | IonQ §VI, [1128] §3.1 |
| `ec_mbu.select_swap_lookup`, `PointAddCfg(select_swap=...)` | SELECT-SWAP loads: a lookup over the high w − k address bits writes 2^k words, k layers of Fredkins move the addressed one into place; the other words are cleared by X-measurement into the addition's merged repair, so the extra qubits live only during the load | Low–Kliuchnikov–Schaeffer 2018 |
| `ec_space.py` | the qubit-lean cells: controlled CDKM at 3n, CDKM comparator (1 ancilla), chunked all-ones test (~2√k ancillas), constant adder whose high part is an increment on *borrowed* qubits; the same answers as `ec_approx`'s cells input for input | [1128] §3.2 / §4, [CDKM04], Gidney 2015 |
| `ec_depth.py` | depth-optimised cells on the carry-lookahead adder: exact modular doubling/halving/controlled addition (`CLAArith`), the dialog walk (`Dialog(walk_cla=True)`), the squarer; shared controls fanned out with CNOTs so n controlled gates take one layer | [106] §3.1, [DKRS04] |
| `ec_cla.py` | Draper–Kutin–Rains–Svore log-depth adder from temporary ANDs, free-uncompute comparator tree, log-depth exact modular add | [106] §3.1, [DKRS04] |
| `ec_edwards.py` | twisted Edwards / Ed25519: complete extended-coordinate out-of-place addition (7 multiplications against Jacobian's 11, no exceptional cases), Niels lookups, maps to Weierstrass | [106] §4.4, [HWCD08] |
| `ec_proj_q.py` | Jacobian addition with a quantum addend (Alg 4 with a lookup), zig-zag chain, projective ECDLP with one Z-inversion | [106] §4.2, Alg 4 |
| `ec_signedwin.py` | odd signed windows: 2^(w−1)-entry tables that never contain O, the sign two merged negations of y; `ec_shor.ecdlp_windowed(signed=True)` | [HJN+20] §5.1, [106] §6.1 |
| `ec_batch.py` | Montgomery's batch inversion: k inverses for 2 inversions + 5(k−1) multiplications, batched affine point additions | Litinski 2023 |
| `ec_luo.py`, `ec_luo_classical.py` | Luo et al.'s register-shared EEA: inversion in exactly 2n + 6⌊log₂n⌋ + 19 qubits (579 at n = 256), reversible division in 4n + 6⌊log₂n⌋ + 19; `PointAddCfg(mul=Luo())` | Luo et al., ECDSA.Fail §5.3.5 |
| `ec_luo3.py` | [Luo26] Sec 5: division and multiplication on three field registers by *venting* y (X-measured, then its phase cancelled by recomputing y around a Z^b), the controlled addition of a classical point (Fig. 14), and the windowed addition on it (`windowed_cfg`: one coordinate loaded at a time); `ec_mbu.vent` / `zfix` | [Luo26] Sec 5–6 |
| `ec_opt.census`, `strip_unfired` | ECDSA.Fail's fire census: drop the Toffolis a sample of inputs never fires; exact when the sample is every input, otherwise measured on fresh inputs | ECDSA.Fail §5.3.7 |
| `hier.exact_depth` | the Toffoli depth of the flat circuit, scheduled while expanding the cached hierarchy (never built); equals `depth.toffoli_depth` | — |
| `rns_classical.py`, `rns.py` | Gidney 2025's approximate residue-number exponentiation: residue system, dlog and transition tables, bit-exact model, [G25] Table 3–5 tallies; the six loops as a circuit, exact against the model at toy N | Gidney 2025 (arXiv:2505.15917) |
| `ec_shor.ecdlp_windowed` | the full windowed circuit: w recycled control qubits (`semiclassical_iqft_windowed`), first window as a lookup, dropped trailing windows + search post-processing, IonQ's masks; `ecdlp_multikey_1c` reuses one P-half for many keys | [1128] §2, Babbush App A, Litinski §4, IonQ §IV |
| `ekera_hastad.py` | factoring by a short discrete logarithm: 1.5n exponent qubits instead of 2n (s = 1) | Ekerå–Håstad, CFS 2024, Gidney 2025 |
| `physical.py` | surface code + yoked storage + cultivation/CCZ-factory model; reproduces Gidney 2025's 897,864 qubits / 4.96 days from its inputs | Gidney 2025 §3.2 |

**Measured at n = 256** (`bench/ec_project_256.py`: every coefficient is a gate count of a
circuit built at n = 256 with secp256k1's prime and verified exactly at toy sizes):

| | Toffoli-eq |
|---|---:|
| in-place multiplication, [1128] dialog as built before this | 1,955,276 |
| … with fused 77-bit comparisons, width schedule, pseudo-Mersenne replay | 865,588 |
| … IonQ conditionally-inverted walk, IonQ replay | **720,565** |
| … ECDSA.Fail ping-pong (704 rounds) / Jump-2 (261 steps), as composed here | 917,565 / 1,161,099 |
| square-subtract: general multiplier → dedicated squarer + fold | 1,050,623 → **79,160** |
| lookups per addition (w = 16): 10 recomputed → 3 MBU loads + 1 repair | 1,310,700 → **197,364** |
| **one windowed point addition**, before → after | **6,276,990 → 1,722,769** |
| published: [1128] / IonQ | 2,588,963 / 1,392,608 |
| **whole ECDLP-256**: 34 × before → 28 × after | 213M → **48M** |

The GCD comparisons use 40 + ⌈2.3√n⌉ = 77 top bits, IonQ's Table X figure: the width schedule
pads u and v with 2.3√n bits of leading zeros, so a k-bit comparison sees only k − 2.3√n
significant bits (an earlier version here used a flat 48 and failed measurably; see below).
The modular arithmetic keeps 48-bit comparisons, on uniformly random field elements.

**Built at n = 256, not projected** (`bench/ec_hier_256.py`, `bench/qualtran_compare.py`).
`hier.py` runs the *unchanged* builders at cryptographic size: `hier.tracing()` wraps a list
of existing builders with a `boundary` decorator for the duration of a `with` block (no
source file changes), and on a `HierMachine` each wrapped call is built once per shape and
reused as one opaque gate.  A complete 256-bit windowed point addition builds in ~20 s in
0.7 GB; `tests/test_hier.py` checks that every field of the count equals the flat build.
`hier.exact_depth(m)` gives the flat circuit's Toffoli depth by scheduling while it expands the
cached hierarchy (the flat circuit is never built; `tests/test_ec_depth.py` checks it against
`depth.toffoli_depth`).  `hier.to_qualtran(m)` exports the tree as Qualtran bloqs
(`pip install qualtran`, optional), and Qualtran's `QECGatesCost` counts exactly the same
Toffolis, plus Cliffords and measurements.  Qualtran has no depth cost key: a call graph
records how often each sub-bloq is called, not on which qubits or in what order.

| one windowed point addition, secp256k1, w = 16 | Toffolis | Toffoli depth | qubits |
|---|---:|---:|---:|
| Qualtran's own `ECAdd` (Litinski 2023), for reference | 8,346,972 | | 1,798 |
| [1128] dialog, exact arithmetic (MBU lookups, masked tables) | 5,163,617 | 4,413,985 | 2,104 |
| … fused 77-bit comparisons, width schedule, pseudo-Mersenne replay | 2,972,327 | 2,409,204 | 2,251 |
| … plus the dedicated squarer | 2,000,864 | 1,503,533 | 2,251 |
| IonQ's conditionally-inverted walk and replay, squarer | **1,711,194** | 1,409,170 | 2,251 |
| ECDSA.Fail ping-pong (704 rounds), squarer | 2,117,534 | 1,932,984 | 2,155 |
| register sharing + Fig. 1 packing (space variant), squarer | 2,006,184 | 1,505,372 | **1,873** |
| published: [1128] 2,588,963 / IonQ 1,392,608 (Toffolis, incl. 3 lookups) | | | 1,192 / 1,457 |

The whole ECDLP-256 circuit (28 windowed additions after a first-window lookup, the last
three windows dropped) builds to 47,978,966 Toffolis at depth 39,494,808 on 2,251 qubits with
recycled controls; Qualtran counts 249,846,912 Cliffords and 29,873,480 measurements.
Part VII of `shor-complete.tex` explains every row of these tables (§39, "At cryptographic
size"), generated from the bench JSON files and checked by `bench/ec_check_tex.py`.

**IonQ's Toffoli count** (`bench/ec_toffoli_256.py`).  Split by component, the IonQ-style
build spends 570k of its 1.71M Toffolis on the replay's signed modular additions: Algorithm 11
between two complements plus the 0 ↔ p swaps, 712 Toffolis a step, of which 256 are the
addition.  IonQ's own adder (their Alg 2, `ec_approx.modadd_pm_phase`) adds with carry-out and
uses the carry as the reduction flag; after the correction the carry equals [y' < x], so it is
cleared by X-measurement and a 32-bit phase comparison that runs half the time: 352 Toffolis worst
case, 336 executed.  It cannot see x + y = p, and the replay produces exactly that (or, dividing,
a 0 entering a complement) once, in its first few steps; `CondInv(zero_steps=37)` keeps the careful
cell there (IonQ's "rounds with x + y = p: 37") and builds multiplication's replay forwards, since
the inverse of "measure, sometimes repair" is a recomputation that always runs.  Signed windows
(`ec_signedwin`, [HJN+20]/[106]) then halve the lookups.

| one windowed point addition, secp256k1 | Toffolis | executed | depth | qubits |
|---|---:|---:|---:|---:|
| IonQ-style, Algorithm 11 replay cells | 1,711,194 | 1,711,194 | 1,409,170 | 2,251 |
| + IonQ's adder in the replay, 37 careful steps | 1,441,968 | 1,430,352 | 1,116,947 | 2,233 |
| + the same adder in the point addition, approximate negation | 1,438,613 | 1,426,965 | 1,114,102 | 2,233 |
| + Fig. 1 packing, replay in x's qubits | 1,443,933 | 1,432,285 | 1,255,253 | 1,845 |
| row 3 + signed windows (2^15-entry tables) | 1,340,563 | 1,328,915 | 1,048,565 | 2,233 |
| row 4 + lean careful cell, CNOT ends, shared walk, phase fold | 1,436,519 | 1,424,871 | 1,296,204 | **1,457** |
| the same with signed windows: **one circuit** | 1,338,469 | 1,326,821 | 1,198,152 | **1,457** |
| + SELECT-SWAP on the 3x lookup | **1,322,341** | **1,310,693** | 1,182,024 | **1,457** |
| published: IonQ | | 1,392,608 | | 1,457 |

On IonQ's cells the whole ECDLP-256 circuit builds to 40,346,698 Toffolis (40,020,554 executed,
2.6% above IonQ's 39.0M) at depth 29,490,972 on 2,233 qubits.  "Executed" counts a
measurement-based repair with the probability it fires (`toffoli_expected` in `ec_cost.count` and
`hier.count`); every other column is worst case.

**IonQ's count and IonQ's qubits in one circuit.**  IonQ reports 39.0M Toffolis at 1,457 logical
qubits.  Starting from row 4 (1,845 qubits), the peak moved four times, and none of the four fixes
adds a Toffoli:

- the careful cell of the first 37 replay steps held its adder's scratch, and a copy register an
  uncontrolled addition never touches, until the end: freed after the addition, the same
  703 Toffolis run on 259 scratch qubits instead of 647 (`PMPhase(lean=True)`);
- the replay's opening s = y (adding r into 0) and closing clear (subtracting r from r) were exact
  modular additions, 1,023 Toffolis on 516 scratch qubits; both are copies, so n CNOTs
  (`CondInv(cnot_ends=True)`);
- the square-subtract's fold ran 8 exact generic subtractions beside the 2n + 1-qubit square;
  on the phase adder each is 352 Toffolis on 258 scratch qubits (`csub_square_pm(arith=...)`);
- the walk ended holding u at full width next to the whole record: register sharing
  (`CondInv(share=True)`) frees u's high qubits (never x's, so the replay still runs in x) as
  the schedule narrows, and the record grows into them.

The peak is then the replay: record 669 + multiplicand and accumulator 512 + window 16 + one
cell's scratch = **1,457**.  With signed windows one addition is 1,326,821 executed Toffolis
(4.7% below IonQ's 1,392,608 at the same width), and the whole ECDLP-256 circuit is
**37,535,781 Toffolis (37,209,637 executed) on 1,457 qubits**, against IonQ's 39.0M
on 1,457.  The price is depth, 33,606,905 against 29,490,972, most of it from row 4's packing
and reuse of x's qubits.  `tests/test_ec_gcd.py`, `test_ec_square.py` and `test_ec_space.py`
check that the lean cell, the CNOT ends and the shared walk change no output on any input tried,
and that the fold adds only the ~f/q inputs a pseudo-Mersenne subtraction gets wrong.

**SELECT-SWAP lookups** (`ec_mbu.select_swap_lookup`, `PointAddCfg(select_swap=(k_point, k_3x))`,
Low–Kliuchnikov–Schaeffer).  A lookup over the high w − k address bits writes 2^k words and k
layers of Fredkins move the addressed one into place: 2^(w−k) − 2 ANDs + (2^k − 1)·b Fredkins.
The other words hold a known function of the address, so they are X-measured at once and join
the addition's merged repair: no extra repair, and the extra qubits live only during the load.
At an (x, y) load 1,054 qubits are live, so even k = 1 there costs qubits (1,565); at the 3x
load only 798 are, and k = 1 fits under the 1,457-qubit peak for free.  The signed one-circuit
addition, k swap bits for (point loads, 3x load):

| k | qubits | Toffolis | executed | depth |
|---|---:|---:|---:|---:|
| (0, 0) | 1,457 | 1,338,469 | 1,326,821 | 1,198,152 |
| (0, 1) | 1,457 | 1,322,341 | 1,310,693 | 1,182,024 |
| (1, 1) | 1,565 | 1,290,597 | 1,278,949 | 1,150,276 |
| (1, 2) | 1,565 | 1,282,917 | 1,271,269 | 1,142,347 |
| (2, 2) | 2,588 | 1,268,581 | 1,256,933 | 1,126,989 |
| (3, 3) | 4,635 | 1,261,413 | 1,249,765 | 1,117,009 |

The whole ECDLP-256 circuit with the free setting, (0, 1): **37,084,197 Toffolis
(36,758,053 executed) on 1,457 qubits**, depth 33,155,321;
5.9% below IonQ per addition at the same width.  `tests/test_ec_window_cfg.py` checks
every setting on every input at toy size, the cost against the formula exactly, and the
measurements performed literally.

**Fewer qubits** (`bench/ec_space_256.py`, same builders at n = 256, w = 16; every row adds
one option to the row above it).  The peak of a point addition is the Bézout replay:
the packed GCD record, the two n-bit registers, and whatever scratch the arithmetic
draws.  So the work is: take the scratch out of every cell that runs at that moment, then
check where the peak went (the square-subtract, then the dialog walk, then the replay's
constant adder, then the squarer):

| one windowed point addition, secp256k1 | qubits | Toffolis | depth |
|---|---:|---:|---:|
| IonQ-style (cond.-inverted walk, PM arithmetic, IonQ replay) | 2,251 | 1,711,194 | 1,409,170 |
| + Fig. 1 packing of the record | 2,119 | 1,716,514 | 1,505,184 |
| + replay in x's qubits (the walk leaves them empty) | 1,863 | 1,716,514 | 1,505,403 |
| + CDKM replay arithmetic (`PMSpace`) | 1,713 | 2,042,914 | 1,835,003 |
| [1128] dialog + register sharing + Fig. 1, `PMSpace` | 1,557 | 2,333,400 | 1,916,396 |
| + CDKM square-subtract and point-addition adders | 1,389 | 2,335,990 | 1,917,407 |
| + CDKM dialog walk (`Dialog(walk_space=True)`) | 1,309 | 2,577,278 | 2,159,123 |
| + lean replay cells (`PMSpace(lean=True)`) | 1,299 | 2,868,531 | 2,387,717 |
| + CDKM squarer | **1,246** | 2,935,857 | 2,455,813 |
| + Gidney adders where there is headroom (walk ≤ 0.94n, squarer ≤ 0.78n ancillas) | **1,246** | 2,853,603 | 2,372,502 |
| + SELECT-SWAP on the 3x lookup (`select_swap=(0, 1)`: its junk fits under the peak) | **1,246** | **2,821,091** | 2,339,990 |
| + Luo's register-shared EEA instead of the dialog (`PointAddCfg(mul=Luo())`) | 1,107 | 86,033,905 | 66,428,000 |
| three field registers ([Luo26] Sec 5, `ec_luo3.windowed_cfg`, signed windows) | **851** | 85,204,396 | 65,520,277 |
| published: [1128] space-optimised, secp256k1 (+16 window qubits) | 1,208 | 2,390,000 | |
| published: IonQ | 1,457 | 1,392,608 | |

1,246 is 4.87n.  It breaks down as:

- the packed record, 669 qubits (x's own 256 among them);
- y and the replay accumulator, 512;
- the window address, 16;
- 38 qubits of scratch, 33 of them the register that holds f = 2^32 + 977 for the
  constant adder;
- a few flags.

[1128] reports 4.355n + O(√n) = 1,192 (+16).  The lean cells answer exactly what `ec_approx`'s
do, input by input and failures included (`tests/test_ec_space.py`).  The subtractions they run
backwards fail when the accumulator is below f on entry: probability ~f/q, ~2^−224 for secp256k1.

The Luo row trades 30× the Toffolis for 139 qubits: its division shares the two remainders' and the
cofactors' lanes, so the record disappears, but every quotient bit costs a shifted comparison.

**Three field registers** (`ec_luo3`, [Luo26] Sec 5).  `Luo.div` still holds four field registers at
its peak (x in the bank, y, the second bank, the quotient).  Luo et al. *vent* y: once z = y/x is in
the second bank, y is a function of the other registers, so it is X-measured and its register serves,
clean, as the bank of the backward EEA; the phase (−1)^(b·y) is cancelled afterwards by recomputing
y = xz around a Z^b (`ec_mbu.vent` / `zfix`; `run_live` performs both, and the tests check the phase
cancels, and that without the Z^b it does not).  The multiplications take p as a classical constant:
[Luo26] uses Gidney's constant-workspace adder for that, not built here; the exact generic cells
serve at toy size (exact on every input, exactly 3n + 6⌊log₂n⌋ + 19 qubits) and the lean
pseudo-Mersenne cells at n = 256, which fit in the qubits the EEA leaves idle.  At n = 256: the
division on 835 qubits (the formula, exactly), the controlled addition of a classical point
([Luo26] Fig. 14) on 836, the signed windowed addition on **851** (the point loaded one
coordinate at a time: [Luo26]'s five lookups), and the whole ECDLP-256 circuit:
**2,385,774,798 Toffolis on 851 qubits** (Luo et al. report 2^30.88 ≈ 1.98e9).

**Depth** (`bench/ec_depth_256.py`).  Every configuration above ripples, and the GCD rounds
are sequential, so depth runs at 0.8–0.95 of the Toffoli count.  [106]'s depth-optimised
choice is the carry-lookahead adder (`ec_cla`, [DKRS04]: ~4 log₂ n deep, ~7n Toffolis),
put where the depth is (`ec_depth`: the replay, the walk, the squarer), with every shared
control fanned out by CNOTs so that n controlled gates take one layer:

| one windowed point addition | qubits | Toffolis | depth |
|---|---:|---:|---:|
| dialog, PM replay (ripple adders) | 2,251 | 2,000,864 | 1,503,533 |
| + carry-lookahead replay | 2,601 | 6,769,556 | 936,530 |
| + carry-lookahead walk | 2,601 | 8,655,996 | 456,379 |
| + carry-lookahead squarer | 2,601 | 9,184,472 | **394,281** |
| + register sharing, Fig. 1 | 2,223 | 9,189,792 | 394,917 |

The three lookups (unary iteration, 2^16 steps each) are a floor of 196,608.  Plots of both
frontiers, qubits against Toffolis and against depth, are `doc/figures/ecdlp_padd_*.pdf`.

**RSA, built at size** (`rns_classical.py`, `rns.py`, `bench/rsa_g25_2048.py`).  Gidney 2025's
approximate residue-number exponentiation, loop for loop: per prime, dlogs of the window
multipliers are looked up and added (merged with the previous prime's uncompute), reduced mod
p − 1 by long division, exponentiated by windowed multiplications (old results measured out),
and the truncated CRT contribution subtracted into an f-bit masked accumulator.  At toy N the
circuit's output equals a bit-exact integer model on every exponent (`tests/test_rns.py`).  At
size, one prime's iteration is built for three primes of a real residue system (identical
counts: no cost depends on the tables) and multiplied by |P|:

| n | primes | one iteration | per shot (executed) | per factoring | Gidney 2025 | qubits (paper) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 6,242 | 20,109 | 1.26e8 | 1.18e9 | 1.1e9 | 800 (742) |
| 1536 | 11,956 | 30,446 | 3.64e8 | 3.39e9 | 3.1e9 | 1,138 (1,074) |
| 2048 | 21,320 | 35,449 | 7.56e8 | **6.95e9** | 6.5e9 | 1,467 (1,399) |
| 3072 | 47,814 | 44,858 | 2.14e9 | 1.97e10 | 1.9e10 | 2,115 (2,043) |
| 4096 | 74,245 | 62,774 | 4.66e9 | 4.29e10 | 4e10 | 2,766 (2,692) |

**ECDSA against RSA** (`bench/compare_ecc_rsa.py`, one surface-code model for every row:
p = 1e-3, 1 µs cycles, 10 µs reaction, six CCZ factories; plots in `doc/figures/`):

| whole algorithm | logical qubits | Toffolis | physical qubits | runtime |
|---|---:|---:|---:|---:|
| ECDLP-256, IonQ's cells, signed windows (built) | 1,457 | 3.68e7 | 2,140,216 | 15 min |
| ECDLP-256, dialog on the space cells (built) | 1,246 | 7.91e7 | 1,854,944 | 33 min |
| ECDLP-256, three field registers (built) | 851 | 2.39e9 | 1,320,904 | 0.73 days |
| RSA-2048, Gidney 2025 (built) | 1,467 | 6.95e9 | 973,576 | 2.1 days |
| RSA-3072, Gidney 2025 (built) | 2,115 | 1.97e10 | 1,259,592 | 6.4 days |
| RSA-2048, Gidney 2025 (published) | 1,399 | 6.5e9 | 881,640 | 1.9 days |
| ECDLP-256, IonQ (published) | 1,457 | 3.9e7 | 2,140,216 | 16 min |

At comparable logical qubit counts secp256k1 costs ~189× fewer Toffolis than RSA-2048 and
~537× fewer than RSA-3072, its classical-security equal.  The numbers are eight times shorter
(arithmetic costs ≥ n² per group operation); ECDLP needs one shot where Ekerå–Håstad with s = 8
needs 9.2; and Gidney's residue arithmetic buys RSA its low qubit count (0.7n, the exponent in
cold storage) with Toffolis.  Part VIII of `shor-complete.tex` sets this out.

**Output distributions, checked against the known answer** (`bench/toy_distributions.py`).
Every circuit here is a permutation on basis states, so its exact frequency distribution
follows from what it computes on each input.  The signed-window ECDLP circuit on a curve of
odd prime order matches the ideal up to its exceptional inputs, above IonQ's bound; the
approximate, masked RSA exponentiation at N = 241 × 251 (s = 2, a residue system without N's
factors) puts the same mass on the exact oracle's peaks.

Things the sources get wrong or leave out, found while reproducing them:
- The GCD's top-bit comparisons act on registers padded by the width schedule, so they need
  k + 2.3√n bits for k significant ones (IonQ Table X's 40 + 2.3√n = 77 at n = 256).  A flat
  24 bits at n = 64 fails 15 of 40 multiplications; this package's n = 256 builds used a flat
  48 until the n = 64 simulation caught it (every number above now uses 77).
- IonQ's replay needs its 0 ↔ p swap *before* the complemented adder; without it every
  e = 1, y = 0 step leaves an ancilla dirty (`ec_approx._swap_zero_q`).  IonQ's text names
  only that swap; run in the multiplication direction, the same replay step is a complemented
  subtraction whose result is 0, which comes out as p and needs the p → 0 repair as well
  (`ec_approx._fix_zero_q`).  The careful cell here does both.
- ECDSA.Fail's ping-pong budget L = 2.75n is not a worst case: exhaustively, n = 16 needs
  3.9n. `PingPong(rounds=...)` exposes it; the tests report the failure rate at 2.75n.
- Gidney 2025's hot-qubit count "3f + 2ℓ + len m = 131" evaluates to 152 (926,256 qubits);
  the published 897,864 uses 131.  Its "9.1 shots" is 9.2.
- At toy N, the Gidney–Ekerå choice A = g^(N+1) puts d outside the short range and never
  factors; Gidney 2025's A = g^(N−1) sometimes does (d is pinned only modulo ord g).
- And two of ours.  This package quoted IonQ's circuit at 1,462 qubits; that is the width IonQ
  targeted ([1128]'s, window qubits included).  IonQ's own figure is "39 million Toffoli gates
  at 1457 logical qubits" (arXiv:2609.05625, Sec. I).
- `ec_adders.cdkm_add` documented a controlled CDKM as 3n Toffolis, but it
  puts the control on both MAJ and UMA, which is 4n.  The textbook 3n (control on the UMA
  only) is `ec_space.cdkm_cadd`; it takes ~200k Toffolis off every dialog replay at n = 256.

Tests: `test_sparse_sim`, `test_depth`, `test_eh`, `test_physical`, `test_api_surface`,
`test_g25_arith`, `test_coset_order`, `test_rns` (fast tier); `test_ec_regress`, `test_ec_mbu`,
`test_ec_window_cfg`, `test_ec_signed`, `test_ec_gcd`, `test_ec_square`, `test_ec_windowed`,
`test_ec_padd_mont`, `test_ec_opt`, `test_hier`, `test_ec_space`, `test_ec_cla`,
`test_ec_edwards`, `test_ec_proj_q`, `test_ec_depth`, `test_ec_signedwin`, `test_ec_batch`,
`test_ec_luo`, `test_ec_luo3` (ec tier).

## What is not implemented

Named rather than glossed:

- **Register sharing** (1128 §3.1) is built in `ec_gcd.Dialog(c_pad=..., share=True)`
  (probabilistic, failure rate measured); `ec_eea` itself remains exact.  The
  **5-Toffoli Fig. 1** packing is `ec_gcd.fig1_compress`; `ec_eea.compress_records`
  keeps its generic 57-Toffoli permutation and is still not wired into `ec_eea`.
- **Gidney's constant-workspace classical–quantum adder** (arXiv:2507.23079), which [Luo26]
  uses for its multipliers' modular reductions with any p.  `ec_luo3` builds [Luo26]'s
  three-register division exactly for any p with the generic cells (more scratch), and at
  n = 256 with the lean pseudo-Mersenne cells, which fit in the qubits the EEA leaves idle.
- **IonQ's Karatsuba square-subtract** (their Algs 7–11, fused slice additions and phase
  comparators).  The square-subtract here is the dedicated squarer and a fold (79,160);
  a Karatsuba over this package's general cells was measured and costs more.
- **Gidney's dirty-ancilla constant adder** [Gid25], in full.  `ec_space`'s lean constant
  adder borrows dirty qubits only for the increment above f's top bit; the bits of f
  itself still sit in bitlen(f) clean qubits (33 for secp256k1, the largest scratch left
  at the space variant's peak).
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
