"""Toy transcription of Chevignard-Fouque-Schrottenloher, "Reducing the Number
of Qubits in Quantum Factoring", IACR ePrint 2024/222 (HAL hal-05436363).

Verified source: the 44-page PDF, Secs. 4.1-4.4 (Eqs. 12-27, Lemmas 5-6,
Prop. 1, Heuristic 3), Algorithm 1, Theorem 2, Lemma 7, Sec. 6.1 and
Algorithm 2.  This is the *low-bit* RNS multi-product

    (x_0..x_{m-1}, 0) -> (x, [[A_X]_N]_{2^r}),   A_X = prod_i A_i^{x_i},

and is distinct from the Gidney arXiv:2505.15917 high-bit toy in
`rns_gidney.py`.

What is transcribed (all coherent, no mid-circuit measurement):

* Algorithm 1, steps 1-23, with the Eq. 27 cofactors B_p, C_{p,i}, D_{0,i},
  D_{p,i} on registers of the paper's widths: qM on u+len(qM)+1 bits, QN on
  u'+r bits, Res on r bits, u from Lemma 5 and u' from Heuristic 3.
* The per-prime residue [A_X]_p by Lemma 7: S = sum_i x_i alpha_i (discrete
  logs to a generator g_p, controlled constant additions) followed by g_p^S
  mod p read from a table indexed by S.  This is the single-window case of
  the Sec. 6.1 lookup (the paper splits S into three 7-bit windows); the
  table has |S| = O(m p) entries and is never indexed by X.
* Theorem 2's cleanup: after Res is final, qM and QN are erased by
  recomputing every residue and subtracting, so the circuit uses
  4 |P| multi-product (lookup) circuits: compute + uncompute per prime,
  twice (Sec. 6.1).

Irreversible steps as printed in Algorithm 1 and how they are made
reversible here (no measurement is needed anywhere):

* `x := x0 + 2 x1 + ...` and `qM := b0 + 2 b1 + ...` are relabellings.
* `qM <- (qM >> u) + 1` and `QN <- QN >> u'` discard low bits; they are
  implemented by addressing the high slice of the same register (the low
  bits stay as garbage) and are undone by Theorem 2's recomputation pass.

Deviations, all labelled:

* Toy RNS (Remark 1): primes are chosen so that M exceeds the product of all
  A_i, i.e. every X is representable, instead of the Lemma 4 Hamming-weight
  bound M >= 2^{n(m/2 + 2 m^{2/3})}; t_N is the smallest t with
  2^t >= N * prod(A_i).  Lemma 5 / Heuristic 3 failures are still possible
  and are reported by `failures()`.
* Lemma 7 "sets aside" A_i that vanish mod p; here such primes are rejected.
* Lemma 7's sequential sum is used, not the Lemma 8/9 multi-bit-sum trees.
* Additions are CDKM ripple-carry (`rc_adder.py`) with a shared clean
  ancilla register; the lookup is a direct open/closed-controlled-X table,
  not the [1] unary-iteration lookup.  Gate counts are therefore not the
  paper's Table 3/5 counts.
* Algorithm 2 toy: s, mu, r are laptop sized; the classical Ekera-Hastad
  lattice post-processing is replaced by enumerating the short logarithm d
  (dimension-2 lattice, toy range) ranked by the EH good-pair criterion and
  checked by factoring.  The beta-compressed bit is computed into one qubit
  and Res is erased by running Algorithm 1's circuit in reverse (so one
  Step-8 run costs 8 |P| multi-product circuits; the paper does not spell
  out how Res is cleared).  beta is fixed per circuit; Step 7's random beta
  is a per-run parameter.
* `cfs_order_circuit` / `order_postprocess` are plain Shor order finding on
  the compressed oracle (for odd N that are not of the Algorithm 2 shape);
  this is NOT Algorithm 2 / Seifert / Ekera-Hastad.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, prod

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from rc_adder import rc_add_const
from shor_essentials import qft


def clog2(x):
    """ceil(log2(x)) for integers x >= 1."""
    return (x - 1).bit_length()


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def primitive_root(p):
    qs = [q for q in range(2, p) if (p - 1) % q == 0 and is_prime(q)]
    return next(g for g in range(2, p)
                if all(pow(g, (p - 1) // q, p) != 1 for q in qs))


def discrete_log(g, a, p):
    return next(d for d in range(p - 1) if pow(g, d, p) == a % p)


@dataclass(frozen=True)
class CfsConfig:
    """Parameters of Algorithm 1 for multipliers A_0..A_{m-1} mod N."""
    modulus: int
    multipliers: tuple
    primes: tuple
    out_bits: int
    nu: int = 1
    u_override: int | None = None
    generators: tuple = field(init=False)
    alphas: tuple = field(init=False)

    def __post_init__(self):
        N, A, P = self.modulus, tuple(self.multipliers), tuple(self.primes)
        object.__setattr__(self, "multipliers", A)
        object.__setattr__(self, "primes", P)
        if N < 3 or N % 2 == 0:
            raise ValueError("N must be odd and >= 3")
        if not A or any(not 1 <= a < N or gcd(a, N) != 1 for a in A):
            raise ValueError("Lemma 6 needs units A_i in [1, N)")
        if (not P or len(set(P)) != len(P)
                or not all(is_prime(p) and p > 2 for p in P)):
            raise ValueError("RNS needs distinct odd primes (Lemma 5: p != 2)")
        if any(a % p == 0 for a in A for p in P):
            raise ValueError("toy omits Lemma 7's zero-residue set-aside")
        if self.out_bits < 1 or self.nu < 0:
            raise ValueError("requires r >= 1 and nu >= 0")
        if prod(P) <= prod(A):
            raise ValueError("Remark 1 toy RNS: M must exceed prod(A_i)")
        if self.u_override is not None and self.u_override < 1:
            raise ValueError("u must be positive")
        if self.t_N < self.u_prime:
            raise ValueError("toy t_N < u': Eq. 27 truncation to u' bits "
                             "is meaningless; use fewer primes or larger A_i")
        gens = tuple(primitive_root(p) for p in P)
        object.__setattr__(self, "generators", gens)
        object.__setattr__(self, "alphas", tuple(
            tuple(discrete_log(g, a, p) for a in A) for g, p in zip(gens, P)))

    # --- Sec. 4.1-4.3 parameters -------------------------------------------
    @property
    def m(self):
        return len(self.multipliers)

    @property
    def n(self):
        return clog2(self.modulus)

    @property
    def M(self):
        return prod(self.primes)

    def Mp(self, p):
        return self.M // p

    def wp(self, p):
        return pow(self.Mp(p), -1, p)

    @property
    def u(self):
        """Lemma 5: u = ceil(log2 sum p^2)."""
        if self.u_override is not None:
            return self.u_override
        return clog2(sum(p * p for p in self.primes))

    @property
    def q_bits(self):
        """len(qM): q_M = floor(sum [A_X]_p w_p / p) < sum w_p."""
        return sum(self.wp(p) for p in self.primes).bit_length()

    @property
    def qm_width(self):
        return self.u + self.q_bits + 1

    @property
    def t_N(self):
        """Prop. 1 condition 2^tN >= A_X N for every X (toy worst case)."""
        return clog2(prod(self.multipliers) * self.modulus)

    @property
    def u_prime(self):
        """Heuristic 3: ceil(log2(sum ceil(log2 p) + len qM)) + nu."""
        return clog2(sum(clog2(p) for p in self.primes) + self.q_bits) + self.nu

    @property
    def qn_width(self):
        return self.u_prime + self.out_bits

    def s_max(self, j):
        return sum(self.alphas[j])

    @property
    def s_width(self):
        return max(1, max(self.s_max(j) for j in range(len(self.primes))
                          ).bit_length())

    @property
    def residue_width(self):
        return max(p.bit_length() for p in self.primes)

    # --- Eq. 27 cofactors --------------------------------------------------
    def B(self, p):
        return (self.Mp(p) * self.wp(p)) % (1 << self.out_bits)

    def C(self, p, i):
        return ((1 << i) * self.wp(p) * ((1 << self.u) // p)
                ) % (1 << self.qm_width)

    def D0(self, i):
        t, up = self.t_N, self.u_prime
        val = (1 << i) * ((1 << (t + self.out_bits))
                          - self.M * ((1 << t) // self.modulus))
        return (val >> (t - up)) % (1 << self.qn_width)

    def Dp(self, p, i):
        t, up = self.t_N, self.u_prime
        val = (1 << i) * self.Mp(p) * self.wp(p) * ((1 << t) // self.modulus)
        return (val >> (t - up)) % (1 << self.qn_width)

    def table(self, j):
        """Sec. 6.1 lookup [g_p^S]_p for S = 0..s_max (single window)."""
        p, g = self.primes[j], self.generators[j]
        return [pow(g, s, p) for s in range(self.s_max(j) + 1)]

    # --- exact quantities --------------------------------------------------
    def bits(self, x):
        return [(x >> i) & 1 for i in range(self.m)]

    def multi_product(self, x):
        return prod(a for a, b in zip(self.multipliers, self.bits(x)) if b)

    def dlog_sum(self, j, x):
        return sum(a for a, b in zip(self.alphas[j], self.bits(x)) if b)

    def residue(self, j, x):
        """Lemma 7: g_p^{sum x_i alpha_i} mod p == [A_X]_p."""
        return self.table(j)[self.dlog_sum(j, x)]

    def exact_output(self, x):
        return (self.multi_product(x) % self.modulus) % (1 << self.out_bits)

    def exact_qM(self, x):
        """Eq. 13."""
        return sum(self.residue(j, x) * self.Mp(p) * self.wp(p)
                   for j, p in enumerate(self.primes)) // self.M

    def exact_QN(self, x):
        """Eq. 13 / Prop. 1 (A_X < M so [A'_X]_M = A_X)."""
        return self.multi_product(x) // self.modulus

    # --- Algorithm 1, register-exact classical mirror ------------------------
    def algorithm1(self, x):
        r, qw, nw, u, up = (self.out_bits, self.qm_width, self.qn_width,
                            self.u, self.u_prime)
        qM = QN = res = 0
        for j, p in enumerate(self.primes):
            v = self.residue(j, x)
            for i in range(p.bit_length()):
                if v >> i & 1:
                    qM = (qM + self.C(p, i)) % (1 << qw)
                    QN = (QN + self.Dp(p, i)) % (1 << nw)
            res = (res + (v % (1 << r)) * self.B(p)) % (1 << r)
        q_hi = ((qM >> u) + 1) % (1 << (qw - u))
        res = (res - q_hi * self.M) % (1 << r)
        for i in range(qw - u):
            if q_hi >> i & 1:
                QN = (QN + self.D0(i)) % (1 << nw)
        qn_hi = (QN >> up) % (1 << r)
        res = (res - qn_hi * self.modulus) % (1 << r)
        return dict(res=res, qM=q_hi, QN=qn_hi)

    def failures(self, internal=False):
        """Every X (exhaustive, toy only) where Algorithm 1's output is wrong
        (or, with internal=True, where qM or [QN]_{2^r} is wrong even if the
        r output bits happen to agree), labelled by the approximation
        responsible: Lemma 5 (qM) or Heuristic 3 (QN)."""
        bad = {}
        for x in range(1 << self.m):
            got = self.algorithm1(x)
            qm_ok = got["qM"] == self.exact_qM(x)
            qn_ok = got["QN"] == self.exact_QN(x) % (1 << self.out_bits)
            wrong = got["res"] != self.exact_output(x)
            if wrong or (internal and not (qm_ok and qn_ok)):
                bad[x] = "lemma5" if not qm_ok else "heuristic3"
        return bad


# --- quantum circuit ---------------------------------------------------------
def _lookup(qc, addr, values, target):
    """target ^= values[addr]; one (open/closed)-controlled X per set bit."""
    for s, v in enumerate(values):
        for b, q in enumerate(target):
            if v >> b & 1:
                qc.mcx(list(addr), q, ctrl_state=s)


@dataclass
class CfsTally:
    residue_computations: int = 0
    controlled_additions: int = 0
    constant_additions: int = 0
    lookups: int = 0
    lookup_entries: int = 0


def cfs_multiproduct(conf):
    """Algorithm 1 as a clean reversible circuit.

    |x>|0...> -> |x>|Res = Alg1(x)>|0...>.  Returns (circuit, regs, tally)
    where regs = dict(x, s, residue, qm, qn, res, anc)."""
    if not isinstance(conf, CfsConfig):
        raise TypeError("conf must be a CfsConfig")
    x = QuantumRegister(conf.m, "x")
    s = QuantumRegister(conf.s_width, "dlogsum")
    v = QuantumRegister(conf.residue_width, "residue")
    qm = QuantumRegister(conf.qm_width, "qM")
    qn = QuantumRegister(conf.qn_width, "QN")
    res = QuantumRegister(conf.out_bits, "Res")
    anc = QuantumRegister(max(conf.qm_width, conf.qn_width, conf.s_width,
                              conf.out_bits) + 1, "anc")
    qc = QuantumCircuit(x, s, v, qm, qn, res, anc)
    tally = CfsTally()
    r, u, up = conf.out_bits, conf.u, conf.u_prime

    def cadd(value, target, ctrls=()):
        target = list(target)
        if value % (1 << len(target)) == 0:
            return
        rc_add_const(qc, value, target, list(anc[:len(target) + 1]),
                     list(ctrls))
        if ctrls:
            tally.controlled_additions += 1
        else:
            tally.constant_additions += 1

    def residue(j, sign):
        """Lemma 7 + Sec. 6.1: sign=+1 computes [A_X]_p, -1 erases it."""
        p = conf.primes[j]
        word = list(v[:p.bit_length()])
        tab = conf.table(j)
        if sign < 0:
            _lookup(qc, s, tab, word)
        for i, alpha in enumerate(conf.alphas[j]):
            cadd(sign * alpha, s, [x[i]])
        if sign > 0:
            _lookup(qc, s, tab, word)
            tally.residue_computations += 1
        tally.lookups += 1
        tally.lookup_entries += len(tab)
        return word

    def accumulate(j, word, sign, with_res):
        p = conf.primes[j]
        for i in range(p.bit_length()):
            cadd(sign * conf.C(p, i), qm, [word[i]])
            cadd(sign * conf.Dp(p, i), qn, [word[i]])
            if with_res and i < r:
                cadd((conf.B(p) << i), res, [word[i]])

    q_hi = list(qm[u:])
    qn_hi = list(qn[up:up + r])
    for j in range(len(conf.primes)):                  # steps 4-14
        word = residue(j, +1)
        accumulate(j, word, +1, True)
        residue(j, -1)
    cadd(1, q_hi)                                      # step 15
    for i in range(min(r, len(q_hi))):                 # step 16
        cadd(-(conf.M << i), res, [q_hi[i]])
    for i in range(len(q_hi)):                         # steps 17-20
        cadd(conf.D0(i), qn, [q_hi[i]])
    for i in range(r):                                 # steps 21-22
        cadd(-(conf.modulus << i), res, [qn_hi[i]])
    # Theorem 2 cleanup: erase QN and qM by recomputation.
    for i in range(len(q_hi)):
        cadd(-conf.D0(i), qn, [q_hi[i]])
    cadd(-1, q_hi)
    for j in range(len(conf.primes)):
        word = residue(j, +1)
        accumulate(j, word, -1, False)
        residue(j, -1)
    regs = dict(x=x, s=s, residue=v, qm=qm, qn=qn, res=res, anc=anc)
    return qc, regs, tally


# --- bit-sliced classical simulator (exact check on all basis inputs) --------
def simulate_basis(qc, inputs):
    """Run a circuit of X/CX/CCX/MCX gates on many basis states at once.

    `inputs` maps qubit -> list of bits (one per lane).  Returns a dict
    qubit -> int whose bit l is the qubit's value in lane l."""
    lanes = len(next(iter(inputs.values())))
    full = (1 << lanes) - 1
    state = {q: 0 for q in qc.qubits}
    for q, col in inputs.items():
        state[q] = sum(b << l for l, b in enumerate(col))
    for inst in qc.data:
        op, qargs = inst.operation, inst.qubits
        if op.name == "barrier":
            continue
        if op.name == "x":
            state[qargs[0]] ^= full
            continue
        if not (getattr(op, "num_ctrl_qubits", 0)
                and getattr(op, "base_gate", None) is not None
                and op.base_gate.name == "x"):
            raise ValueError(f"non-classical gate {op.name}")
        ctrl_state = op.ctrl_state
        mask = full
        for k, c in enumerate(qargs[:-1]):
            mask &= state[c] if ctrl_state >> k & 1 else ~state[c] & full
        state[qargs[-1]] ^= mask
    return state


def register_values(state, reg, lanes):
    return [sum(((state[q] >> l) & 1) << b for b, q in enumerate(reg))
            for l in range(lanes)]


# --- Algorithm 2 (compressed Ekera-Hastad) toy -------------------------------
@dataclass(frozen=True)
class CfsFactoringToy:
    """Algorithm 2 for an RSA-like N = P Q with P, Q of n/2 bits."""
    modulus: int
    base: int
    primes: tuple
    out_bits: int
    beta: int
    s: int = 1
    nu: int = 1

    def __post_init__(self):
        if self.n % 2:
            raise ValueError("Algorithm 2 toy needs even n = ceil(log2 N)")
        if not 1 <= self.beta < (1 << self.out_bits):
            raise ValueError("beta must be a non-zero r-bit mask")
        if gcd(self.base, self.modulus) != 1:
            raise ValueError("G must be invertible mod N")

    @property
    def n(self):
        return clog2(self.modulus)

    @property
    def half(self):
        return self.n // 2

    @property
    def ell(self):
        return -(-self.n // (2 * self.s))

    @property
    def a_bits(self):
        return self.half + self.ell

    @property
    def b_bits(self):
        return self.ell

    @property
    def A(self):
        """Step 2: A = G^{(N-1)/2 - 2^{n/2-1}}."""
        N = self.modulus
        return pow(self.base, (N - 1) // 2 - (1 << (self.half - 1)), N)

    def multipliers(self):
        """Step 5: G^{2^i} (i < n/2 + ell), then A^{-2^j} (j < ell)."""
        N, G, A = self.modulus, self.base, self.A
        return ([pow(G, 1 << i, N) for i in range(self.a_bits)]
                + [pow(A, -(1 << j), N) for j in range(self.b_bits)])

    def config(self):
        return CfsConfig(self.modulus, self.multipliers(), self.primes,
                         self.out_bits, self.nu)

    def compressed(self, x, approximate=True):
        """Step 8: beta . ([A_X]_N mod 2^r)."""
        conf = self.config()
        val = conf.algorithm1(x)["res"] if approximate else conf.exact_output(x)
        return bin(self.beta & val).count("1") & 1


def cfs_factoring_circuit(toy):
    """Step 8 quantum subroutine: H, compressed multi-product into one
    qubit, Algorithm 1 inverse (clears Res), inverse QFT on a and b."""
    oracle, regs, tally = cfs_multiproduct(toy.config())
    flag = QuantumRegister(1, "flag")
    ca = ClassicalRegister(toy.a_bits, "j")
    cb = ClassicalRegister(toy.b_bits, "k")
    qc = QuantumCircuit(*oracle.qregs, flag, ca, cb)
    x = regs["x"]
    a, b = list(x[:toy.a_bits]), list(x[toy.a_bits:])
    qc.h(x)
    qc.compose(oracle, inplace=True)
    for i, q in enumerate(regs["res"]):
        if toy.beta >> i & 1:
            qc.cx(q, flag[0])
    qc.compose(oracle.inverse(), inplace=True)
    qc.append(qft(len(a)).inverse(), a)
    qc.append(qft(len(b)).inverse(), b)
    qc.measure(a, ca)
    qc.measure(b, cb)
    return qc, regs, tally


def ideal_distribution(toy, approximate=True):
    """Exact P(j, k) of Step 9 from the classical compressed function."""
    A, B = 1 << toy.a_bits, 1 << toy.b_bits
    f = np.array([[toy.compressed(a | (b << toy.a_bits), approximate)
                   for a in range(A)] for b in range(B)]).T
    probs = np.zeros((A, B))
    for value in (0, 1):
        amp = np.fft.fft2((f == value).astype(float)) / (A * B)
        probs += np.abs(amp) ** 2
    return probs


def split_counts(counts):
    """Qiskit key 'k j' -> {(j, k): count}."""
    out = {}
    for key, c in counts.items():
        k, j = key.split()
        out[(int(j, 2), int(k, 2))] = out.get((int(j, 2), int(k, 2)), 0) + c
    return out


def eh_score(toy, outcomes, d):
    """Shots among the non-zero outcomes (j, k) that are Ekera-Hastad good
    pairs for candidate d: |{d j + 2^m k}_{2^{l+m}}| <= max(1, 2^{m-2})."""
    m, L = toy.half, toy.a_bits
    bound = max(1, 1 << (m - 2)) if m >= 2 else 1
    tot = 0
    for (j, k), c in outcomes.items():
        if (j, k) == (0, 0):
            continue
        e = (d * j + (k << m)) % (1 << L)
        tot += c * (min(e, (1 << L) - e) <= bound)
    return tot


def eh_postprocess(toy, outcomes):
    """Step 12-14 toy: rank short logs d by `eh_score` over the non-zero
    outcomes, then deduce P + Q = 2 (d + 2^{n/2-1} + 1) and P, Q.  Candidates
    with no good pair are never tried.  Returns (P, Q, d) or None."""
    def score(d):
        return eh_score(toy, outcomes, d)

    N = toy.modulus
    for d in sorted(range(-1, 1 << toy.half), key=score, reverse=True):
        if score(d) == 0:
            break
        s = 2 * (d + (1 << (toy.half - 1)) + 1)
        disc = s * s - 4 * N
        if disc < 0:
            continue
        root = int(round(disc ** 0.5))
        if root * root == disc and (s - root) % 2 == 0:
            P, Q = (s - root) // 2, (s + root) // 2
            if P > 1 and P * Q == N:
                return P, Q, d
    return None


# --- Shor-style order finding with the same oracle (Sec. 5.2 general N) ------
def cfs_order_circuit(conf, beta):
    """Toy order finding of A_i = G^{2^i}: deviation from Algorithm 2 (plain
    Shor counting register instead of Seifert / Ekera-Hastad)."""
    oracle, regs, tally = cfs_multiproduct(conf)
    flag = QuantumRegister(1, "flag")
    out = ClassicalRegister(conf.m, "frequency")
    qc = QuantumCircuit(*oracle.qregs, flag, out)
    qc.h(regs["x"])
    qc.compose(oracle, inplace=True)
    for i, q in enumerate(regs["res"]):
        if beta >> i & 1:
            qc.cx(q, flag[0])
    qc.compose(oracle.inverse(), inplace=True)
    qc.append(qft(conf.m).inverse(), regs["x"])
    qc.measure(regs["x"], out)
    return qc, regs, tally


def order_postprocess(counts, base, modulus, m):
    """Try every continued-fraction convergent denominator q < N of each
    frequent outcome y / 2^m; keep the first with base^q = 1 mod N and reduce
    it to the minimal such divisor.  With only m < 2 log N counting bits the
    best convergent can be wrong or a multiple of the order (e.g. 9/32 ~ 5/18
    -> 18 for ord_21(2) = 6), so earlier convergents are tried too."""
    for key in sorted(counts, key=counts.get, reverse=True):
        y = int(key.replace(" ", ""), 2)
        if y == 0:
            continue
        frac, dens = Fraction(y, 1 << m), []
        for bound in range(1, modulus):
            q = frac.limit_denominator(bound).denominator
            if q not in dens:
                dens.append(q)
        for r in dens:
            if pow(base, r, modulus) != 1:
                continue
            p = 2
            while p <= r:
                while r % p == 0 and pow(base, r // p, modulus) == 1:
                    r //= p
                p += 1
            return r
    return 0


def ideal_order_distribution(conf, beta, approximate=True):
    """Exact P(y) of `cfs_order_circuit` from the classical compressed bit."""
    size = 1 << conf.m
    f = np.array([bin(beta & (conf.algorithm1(x)["res"] if approximate
                              else conf.exact_output(x))).count("1") & 1
                  for x in range(size)])
    probs = np.zeros(size)
    for value in (0, 1):
        probs += np.abs(np.fft.fft((f == value).astype(float)) / size) ** 2
    return probs
