# CFS ePrint 2024/222 low-bit RNS toy (`shor_qiskit/rns_cfs.py`)

Source: Chevignard, Fouque, Schrottenloher, *Reducing the Number of Qubits in
Quantum Factoring*, IACR ePrint 2024/222 (full 44-page version, HAL
hal-05436363).  Sections used: 4.1–4.4 (Eqs. 12–27, Lemmas 5–6, Prop. 1,
Heuristic 3), Algorithm 1, Theorem 2, Lemma 7, Sec. 6.1, Algorithm 2 and
Theorem 3.  This is the **low-bit** multi-product
`x -> [[prod A_i^{x_i}]_N]_{2^r}`.  It is separate from the Gidney
(arXiv:2505.15917) high-bit toy in `rns_gidney.py`.

## Implemented

* **Algorithm 1, steps 1–23**, as one coherent reversible Qiskit circuit
  (`cfs_multiproduct`).  It uses the Eq. 27 cofactors `B_p, C_{p,i}, D_{0,i}, D_{p,i}`
  and the paper's register widths: qM has `u + len(qM) + 1` bits, QN has `u' + r`
  bits and Res has `r` bits.  `u` comes from Lemma 5 and `u'` from Heuristic 3,
  with tunable `nu`.  The circuit contains no measurements.  The two shifts
  that discard bits (`qM >> u`, `QN >> u'`) are done by addressing the high
  slice of each register.  The low garbage bits are then erased by
  **Theorem 2's recompute-and-subtract pass**, giving 4|P| multi-product
  circuits in total, the count stated in Sec. 6.1.
* **Lemma 7 residues.** The circuit computes `S = sum x_i alpha_i`
  (discrete logs) with controlled constant additions, then reads
  `g_p^S mod p` from a table indexed by S, never by X.  This is the
  single-window case of the Sec. 6.1 lookup.
* **Register-exact classical mirror** (`CfsConfig.algorithm1`) and
  exhaustive failure labelling (`failures`):
  * `lemma5`: qM is wrong.
  * `heuristic3`: QN is wrong.
  * `internal=True` also flags inputs where an internal value is wrong but
    the r output bits agree by chance.
* **Toy Algorithm 2** (`CfsFactoringToy`, `cfs_factoring_circuit`) for
  N = 15, G = 2:
  * Steps 1–14 on 6 input qubits (a: n/2 + ℓ = 4, b: ℓ = 2), r = 2.
  * The β-compressed bit goes into one qubit and Res is cleared by
    Algorithm 1's inverse.
  * An inverse QFT is applied to a and b, then j and k are measured.

## Verified evidence

Evidence comes from `tests/test_rns_cfs.py` and `bench/rns_cfs_toy.json`.

* **Circuit equals reference.** A bit-sliced exhaustive simulation over all
  64 inputs shows Res = `algorithm1(x)`, x preserved and every scratch
  register back at 0.  This holds for the exact config (primes 7, 11, 13, 19;
  56 qubits; 0 failures).  It also holds for the failing config (primes 3, 5,
  7, 11; 48 qubits):
  * output failures: 12 `heuristic3` and 4 `lemma5`;
  * 28 internal failures in total.
* **Coherence.** H · U · Z · U⁻¹ · H on Aer MPS: scratch is |0⟩ in every
  shot, and P(x = 0) matches the Walsh–Hadamard prediction.
* **Algorithm 2 on Aer MPS** (2000 shots, about 5 s, 57 qubits):
  * For β ∈ {1, 2, 3} the total-variation distance to the exact ideal
    distribution is ≤ 0.03.
  * P(0) is 0.625 for β = 1, 2 and 0.5 for β = 3.  Theorem 3 bounds it by
    1/2 + ε + 1/8 ≈ 0.635.
  * Every measured non-zero shot is a good pair for d = 1, and every other
    candidate scores strictly less.  Post-processing recovers d = 1 and the
    factors (3, 5).  Caveat: at N = 15, d = 1 is also the only candidate in
    range that factors N, so the factoring check alone is not
    discriminating; the test asserts the good-pair scores separately.
* **Order finding, N = 21.** Shor order finding of 2 mod 21 through the same
  compressed oracle (m = 5, primes 5, 7, 11, 13, 29, ν = 2, 61 qubits, about
  23 s):
  * total-variation distance to ideal: 0.031;
  * recovered order: 6;
  * per-shot success: 0.24.

## Deviations (labelled; this is not a full paper implementation)

* **Toy RNS (Remark 1).** M > ∏A_i, so every X is representable, instead
  of the Lemma 4 Hamming-weight bound.  `t_N = ceil(log2(N ∏A_i))`.
* **Toy validity.** Configs with toy `t_N < u'` are rejected, because the
  truncation in Eq. 27 is meaningless there.
* **Primes.** Primes dividing some A_i are rejected rather than set aside
  as in Lemma 7.  p = 2 is excluded.
* **Sums.** A sequential Lemma 7 sum is used, not the Lemma 8/9
  multi-bit-sum trees.
* **Lookup.** A single-window table, not the three-window Sec. 6.1 lookup.
* **Adders.** CDKM ripple-carry adders and direct multi-controlled-X lookups,
  so gate counts are not Table 3/5 counts.
* **Algorithm 2 post-processing.** The Ekerå–Håstad lattice step is
  replaced by enumerating the toy-range short log d, ranked by the
  good-pair test and then checked by factoring.
* **Algorithm 2 setup.** β is fixed per circuit.  The Step 4 error target
  is not enforced; exactness is checked exhaustively instead.
* **N = 21.** This path is plain Shor order finding (`cfs_order_circuit`),
  not Algorithm 2, Seifert, or Ekerå–Håstad.  Postprocessing tries the
  continued-fraction convergents and reduces each to the minimal order.

## Not implemented

Real parameter scales and the qubit/gate counts of Tables 3–5; Sec. 5.2
general-N (Seifert); Ekerå–Håstad lattice reduction for s > 1;
windowed or unary-iteration lookups; Lemma 8/9 depth-optimised sums.
