"""Shor's algorithm in Qiskit: verified implementations of factoring and ECDLP.

Factoring (order finding)
-------------------------
shor_essentials : the minimal complete implementation (Part IV of the note)
rc_adder        : ripple-carry (CDKM) arithmetic, X/CNOT/Toffoli only
qrom            : unary-iteration table lookup and the sqrt(L) phase fixup
windowed        : Gidney windowing over the multiplicand
nested          : windowing over the exponent as well (Gidney Sec 3.5)
unlookup        : measurement-based uncomputation (hybrid: needs a runner)
onectrl         : order finding with one counting qubit (2n+3)
coset           : Zalka coset representation

Shared by both halves
---------------------
semiclassical   : the semiclassical inverse QFT (one recycled counting qubit)
shor_stats      : exact output distributions, and testing samples against them
resources       : logical counts: qubits, Toffoli, T, rotations (and synthesis cost)

Elliptic-curve discrete logarithm
---------------------------------
Infrastructure
  ec_gates      : Gidney temporary AND, and the emission context
  ec_sim        : registers, ancilla pooling, exact basis-state simulation
  ec_classical  : the classical reference model for every circuit below
  ec_cost       : resource counting, and the papers' projections

Reference stack (Roetteler et al. Asiacrypt'17; Haener et al. PQCrypto'20)
  ec_adders     : CDKM and Gidney adders, comparators, shifts
  ec_modarith   : exact modular arithmetic over GF(p)
  ec_mult       : schoolbook modular multiplication
  ec_kaliski    : Kaliski modular inversion and division
  ec_pointadd   : in-place controlled affine point addition
  ec_shor       : the full ECDLP circuit and its post-processing

Optimizations of eprint 2026/106 (Kim et al.)
  ec_qcsa       : quantum carry-save adder
  ec_montgomery : word-level Montgomery multiplication (Alg. 1b, and the
                  [HJN+20] lookup baseline)
  ec_kaliski_opt: unconditional rounds, postponed reduction, low-depth rounds
  ec_proj       : Jacobian-affine out-of-place point addition, zig-zag schedule

Optimizations of eprint 2026/1128 (Schrottenloher)
  ec_eea        : Euclidean dialog, Bezout replay, in-place multiplication
  ec_approx     : approximate and pseudo-Mersenne modular arithmetic
  ec_window     : windowed point addition
"""
