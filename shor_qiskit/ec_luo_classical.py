"""Classical model of Luo et al.'s register-shared extended Euclidean algorithm.

[Luo26] H. Luo, Z. Yang, J. Luo, Z. Wang, Y. Su, X. Sun, L. Li, T. Li,
        "Quantum Algorithm for Elliptic Curve Discrete Logarithms with
        Space-Efficient Point Addition", arXiv:2607.13816v3, Sec 3 and App A.
[EF26]  ECDSA.Fail contributors, arXiv:2609.09582, Sec 5.3.5 ("Register-Shared
        EEA and Lane-Lifetime Refinements"), which runs this EEA as the
        inversion of its 851-qubit point addition.
[PZ03]  J. Proos, C. Zalka, "Shor's discrete logarithm quantum algorithm for
        elliptic curves", QIC 3(4), 2003 -- the four-phase framework.

This module is the specification `ec_luo` is transcribed from: the exact
bit-level content of the two shared banks after every microstep, so the
circuit can be checked against it one step at a time, not just end to end.

The Euclidean algorithm on (p, x)
---------------------------------
r_0 = p, r_1 = x, r_{i+1} = r_{i-1} - q_i r_i with q_i = floor(r_{i-1}/r_i),
and the cofactors t_0 = 0, t_1 = 1, t_{i+1} = t_{i-1} + q_i t_i.  They satisfy

    r_{i-1} t_i + r_i t_{i-1} = p                                   ([Luo26] 3.2)

-- [EF26] eq (38) writes the same thing as rho c + rho' c' = p, with
(rho, rho', c, c') = (r, r', t, t') below.  When r_m = 1, t_{m-1} is -+x^{-1}:
the sign is the parity of the number of iterations, recorded in `Iter`.

Since r shrinks while t grows, (r_{i-1}, t_i) share one (n+3)-qubit bank and
(r_i, t_{i-1}) the other ([PZ03]'s idea; [Luo26] makes the boundaries exact):

    W1 = t | 0 | q | r        t little-endian from the left, then one zero,
                              then the partial quotient q (big-endian), then
                              r big-endian against the right end
    W2 = rot_ls(t' | r')      t' little-endian from the left, r' big-endian
                              against the right end, the whole bank rotated
                              left by the cursor ls

Rotating W2 left by ls lines r' up as 2^ls r' under r, and t' as 2^-ls t'
under t -- so the shifted operands never exist as separate registers.  The
field boundaries are four small length registers lt, lq, lr (= l_r'), ls.

The four phases ([PZ03], [Luo26] Algorithm 2)
---------------------------------------------
Each quotient q_i of b_i + 1 bits costs 4(b_i + 1) steps:

  1  rotate W2 left, ls += 1, until 2^ls r' > r      (finds b_i + 1)
  2  rotate right, ls -= 1; one quotient bit per step: [r >= 2^ls r'], and
     r -= 2^ls r' when it is 1; the bit is parked in q (lq += 1)
  3  take the bit back out of q (lq -= 1); t' += 2^ls t when it is 1;
     rotate left, ls += 1
  4  rotate right, ls -= 1 -- phase 1's bit-length information is
     uncomputed by the mirror comparison [t' >= 2^ls t]
  then swap W1 <-> W2, recompute lt, lr from the contents, Iter ^= 1.

The phase bits (P1, P2) and the one-bit `Sign` are driven by exactly the
phase-update rules of [Luo26] Algorithm 2 (see `step`).  Phase 2 is written as
"compare, then subtract under the result"; [Luo26] writes it as "subtract, add
back if negative" -- the two leave the same state, and the comparison is the
form the circuit uses.  Algorithm 3's merges are cost optimisations of the same
state sequence.

`table4()` is [Luo26] Table 4 (p = 37, x = 13) and `test_ec_luo` checks this
model reproduces all 37 rows bit for bit.

Padding
-------
The circuit runs a fixed Nmax = 4 ceil(c n) steps (`step_bound`); inputs that
finish early must idle reversibly.  [Luo26] keeps running phase 1: W2 keeps
rotating and ls keeps counting (Table 4 rows 33-36).  That moves t' to an
input-dependent place, and ls (floor(log2 n) + 1 bits) can wrap at n a power
of two.  `padding="block"` (what `ec_luo` builds) instead leaves the banks
alone and counts idle *blocks of four* steps in ls, incremented at steps
T = 1 mod 4: every iteration ends at T = 0 mod 4, so ls >= 1 exactly when
the run has finished at least one block ago, the swap test (lq = ls = 0)
never fires while idle, and the counter needs only ceil(c n) - n + 1 values.
t' stays at W2's left end, where the division reads it.

Sign convention.  [Luo26] Algorithm 1 runs on p - x when x > p/2 (so that
q_1 >= 2 and t_1 < t_2) and records that in Iter.  At the end t' = -x^{-1}
when Iter = 0 and +x^{-1} when Iter = 1 ([Luo26] Alg. 1: "if Iter = 0 then
t' <- p - t'"; [EF26] (38): c' = (-1)^(pi+1) x^{-1}).
"""

import math

# --- [Luo26] App A constants -------------------------------------------------
RHO = 2 + math.sqrt(3)                  # spectral radius of M(1) M(2)
LAM = RHO ** (1 / 3)                    # growth per unit of quotient weight
C = 1 / math.log2(LAM)                  # = 3 / log2(2 + sqrt 3) = 1.5791...
ETA = (4 * math.sqrt(3) - 3) / 13 * LAM ** 2
DELTA = -math.log(ETA, LAM)             # = 0.726208...


def bitlen(v):
    return int(v).bit_length()


def step_bound(n):
    """[Luo26] App A.1: every x needs at most 4 ceil(c n) steps (and >= 4n)."""
    return 4 * math.ceil(C * n)


def meta_bits(n):
    """Width of each length register: floor(log2 n) + 1 ([Luo26] Sec 3.2)."""
    return int(math.floor(math.log2(n))) + 1


def quotients(p, x):
    """The quotient sequence of the EEA on (p, min(x, p - x))."""
    a, b = p, (x if 2 * x < p else p - x)
    qs = []
    while b:
        qs.append(a // b)
        a, b = b, a % b
    return qs


def steps_needed(p, x):
    """N(x) = 4 sum_i (floor(log2 q_i) + 1)  ([Luo26] eq (9))."""
    return 4 * sum(bitlen(q) for q in quotients(p, x))


# --- the Luo qubit formula --------------------------------------------------
def luo_qubits(n):
    """[Luo26] Sec 6.1: modular inversion in 2n + 6 floor(log2 n) + 19 qubits
    (2n + 4 floor(log2 n) + 15 for Figs 5-6, 2 floor(log2 n) + 4 for the
    unary iteration)."""
    return 2 * n + 6 * int(math.floor(math.log2(n))) + 19


# [Luo26] Table 6: compiled Qiskit counts, in millions --
# n: (forward inversion Toffoli, CNOT, point addition Toffoli, CNOT)
LUO_TABLE6 = {
    64: (1.20, 1.85, 5.37, 8.72),
    128: (4.24, 6.60, 19.20, 31.72),
    160: (6.33, 9.95, 28.80, 48.13),
    192: (8.87, 14.01, 40.51, 68.03),
    224: (11.86, 18.77, 54.26, 92.21),
    256: (15.35, 24.30, 70.29, 118.50),
    384: (33.11, 52.91, 152.50, 260.44),
    512: (58.15, 92.94, 268.26, 456.81),
}


def luo_pointadd_qubits(n):
    """[Luo26] Sec 6.1: affine point addition in 3n + 6 floor(log2 n) + 19."""
    return 3 * n + 6 * int(math.floor(math.log2(n))) + 19


# =============================================================================
# The state
# =============================================================================
class State:
    """Bit-level state of the two banks and the metadata.

    W1, W2 are lists of n + 3 bits; index j - 1 holds [Luo26]'s position j
    (u_j, v_j), counted from the left.  W2 is stored *as rotated*.
    """

    def __init__(self, n):
        self.n = n
        self.N = n + 3
        self.W1 = [0] * self.N
        self.W2 = [0] * self.N
        self.lt = self.lq = self.lr = self.ls = 0
        self.P1 = self.P2 = self.Sign = self.Iter = 0

    def copy(self):
        s = State(self.n)
        s.__dict__.update({k: (list(v) if isinstance(v, list) else v)
                           for k, v in self.__dict__.items()})
        return s

    # -- field access (positions are 1-based, inclusive) ---------------------
    def be(self, W, L, R):
        """Big-endian value of positions L..R (R least significant)."""
        v = 0
        for j in range(L, R + 1):
            v = 2 * v + W[j - 1]
        return v

    def set_be(self, W, L, R, v):
        assert 0 <= v < (1 << (R - L + 1)), (v, L, R)
        for j in range(R, L - 1, -1):
            W[j - 1] = v & 1
            v >>= 1

    def le(self, W, a, b):
        """Little-endian value of positions a..b (a least significant)."""
        return sum(W[j - 1] << (j - a) for j in range(a, b + 1))

    def set_le(self, W, a, b, v):
        assert 0 <= v < (1 << (b - a + 1)), (v, a, b)
        for j in range(a, b + 1):
            W[j - 1] = (v >> (j - a)) & 1

    def rotl(self):
        self.W2 = self.W2[1:] + self.W2[:1]

    def rotr(self):
        self.W2 = self.W2[-1:] + self.W2[:-1]

    def unrotated_w2(self, shift=None):
        k = (self.ls if shift is None else shift) % self.N
        return self.W2[-k:] + self.W2[:-k] if k else list(self.W2)

    # -- decoded fields (Table 4's columns) ----------------------------------
    def fields(self, rotation=None):
        """(t, q, r, t', r') as [Luo26] Table 4 prints them."""
        N, lt, lq, lr = self.N, self.lt, self.lq, self.lr
        t = self.le(self.W1, 1, lt)
        q = self.be(self.W1, lt + 2, lt + lq + 1) << self.ls if lq else 0
        r = self.be(self.W1, lt + lq + 2, N)
        U2 = self.unrotated_w2(rotation)
        tp = self.le(U2, 1, N - lr)
        rp = self.be(U2, N - lr + 1, N) if lr else 0
        return t, q, r, tp, rp

    def row(self):
        t, q, r, tp, rp = self.fields()
        return (t, q, r, tp, rp, self.lt, self.lq, self.lr, self.ls,
                self.P1, self.P2, self.Iter, self.Sign)

    def bank_strings(self):
        """W1 as t|q|r and W2 as t'(h)|r'|t'(l), Table 4's separators."""
        s1 = "".join(map(str, self.W1))
        s2 = "".join(map(str, self.W2))
        a, b = self.lt + 1, self.lt + 1 + self.lq
        w1 = s1[:a] + "|" + s1[a:b] + "|" + s1[b:]
        c = self.N - self.lr - self.ls
        w2 = s2[:c] + "|" + s2[c:c + self.lr] + "|" + s2[c + self.lr:]
        return w1, w2


# =============================================================================
# Initialisation ([Luo26] Algorithm 1)
# =============================================================================
def init(p, x):
    """W1 = 1 | 0 | p, W2 = 0 | x, with the x > p/2 reflection."""
    n = bitlen(p)
    s = State(n)
    assert 0 < x < p
    s.Iter = int(2 * x > p)
    xx = p - x if s.Iter else x
    s.W1[0] = 1                                   # t = 1
    s.set_be(s.W1, 3, s.N, p)                     # r = p (n + 1 bits, top 0)
    s.set_be(s.W2, 4, s.N, xx)                    # r' = x (n bits)
    s.lt, s.lr = 1, bitlen(xx)
    return s


# =============================================================================
# One microstep
# =============================================================================
class LayoutError(AssertionError):
    pass


def _need(cond, msg):
    if not cond:
        raise LayoutError(msg)


def step(s, T, padding="block", trace=None):
    """Apply microstep T (1-based) in place.  `trace`, if a dict, receives the
    location registers each operation used (for the active-window checks)."""
    N, n = s.N, s.n
    done = s.lr == 0
    tr = trace if trace is not None else {}

    if done:
        if padding == "luo":                      # Table 4 rows 33-36
            s.rotl()
            s.ls += 1
        elif padding == "block":
            if T % 4 == 1:
                s.ls += 1                          # count idle blocks
        else:
            raise ValueError(padding)
    else:
        ph = (s.P1, s.P2)
        if ph == (0, 0):                                   # --- phase 1
            s.rotl()
            s.ls += 1
            L, R = s.lt + s.lq + 2, N - s.ls
            tr["r"] = (L, R)
            _need(L <= R, "phase 1: empty r window")
            a, b = s.be(s.W1, L, R), s.be(s.W2, L, R)
            _check_r_window(s, L, R)
            s.Sign ^= int(a < b)
        elif ph == (0, 1):                                 # --- phase 2
            s.rotr()
            s.ls -= 1
            L, R = s.lt + s.lq + 2, N - s.ls
            tr["r"] = (L, R)
            _check_r_window(s, L, R)
            a, b = s.be(s.W1, L, R), s.be(s.W2, L, R)
            s.Sign ^= int(a < b)
            s.Sign ^= 1                                    # = [r >= 2^ls r']
            if s.Sign:
                s.set_be(s.W1, L, R, a - b)
            _need(s.W1[L - 1] == 0, "phase 2: quotient lane not free")
            s.lq += 1
            J = s.lt + s.lq + 1
            tr["swap"] = J
            s.Sign, s.W1[J - 1] = s.W1[J - 1], s.Sign
            _need(s.Sign == 0, "phase 2: Sign not returned clean")
        elif ph == (1, 0):                                 # --- phase 3
            J = s.lt + s.lq + 1
            tr["swap"] = J
            s.Sign, s.W1[J - 1] = s.W1[J - 1], s.Sign
            s.lq -= 1
            B = s.lt + 1
            tr["t"] = B
            _check_t_window(s, B)
            a, b = s.le(s.W1, 1, B), s.le(s.W2, 1, B)
            if s.Sign:
                _need(a + b < (1 << B), "phase 3: t' overflows its window")
                s.set_le(s.W2, 1, B, a + b)
                b = a + b
            s.Sign ^= int(b >= a)
            _need(s.Sign == 0, "phase 3: Sign not returned clean")
            s.rotl()
            s.ls += 1
        else:                                              # --- phase 4
            B = N - s.lr - s.ls
            tr["t"] = B
            _check_t_window(s, B)
            a, b = s.le(s.W1, 1, B), s.le(s.W2, 1, B)
            s.Sign ^= int(b >= a)
            s.rotr()
            s.ls -= 1

    # --- phase update ([Luo26] Alg 2, last three blocks) ----------------------
    if s.lq == 0 and s.lr > 0:
        s.P2 ^= s.Sign ^ s.P1
        s.Sign ^= s.P2
    if s.ls == 0:
        s.P1 ^= 1
        s.P2 ^= 1
    if s.lq == 0 and s.ls == 0:
        _need(padding != "block" or T % 4 == 0, "iteration ends off the 4-grid")
        _need(not done, "swap fired while idle")
        swap_banks(s, tr)


def _check_r_window(s, L, R):
    """The r / 2^ls r' window must hold all of both operands: W1 has only r's
    top bits left of R... and W2 has only zeros and r' in [L, R]."""
    N = s.N
    U2 = s.unrotated_w2()
    rp = s.be(U2, N - s.lr + 1, N) if s.lr else 0
    _need(s.be(s.W2, L, R) == rp, f"r' does not fit the window [{L}, {R}]")
    # nothing of t' may sit inside the window
    _need(s.le(s.W2, 1, L - 1) == s.le(U2, 1, N - s.lr) >> s.ls
          if L - 1 >= 1 else True, "t'(h) reaches into the r window")


def _check_t_window(s, B):
    """t (with its appended zero) and t'(h) must be exactly the window 1..B."""
    _need(s.le(s.W1, 1, B) == s.le(s.W1, 1, s.lt), "W1 window holds more than t")
    U2 = s.unrotated_w2()
    tp = s.le(U2, 1, s.N - s.lr)
    _need(s.le(s.W2, 1, B) == tp >> s.ls, "W2 window is not t' >> ls")


def swap_banks(s, tr=None):
    """End of an iteration: W1 <-> W2, then lt = len(t), lr = len(r')."""
    N, lr_old, lt_old = s.N, s.lr, s.lt
    s.W1, s.W2 = s.W2, s.W1
    t = s.le(s.W1, 1, N - lr_old)
    lt = bitlen(t)
    rp = s.be(s.W2, lt + 2, N)
    lr = bitlen(rp)
    if tr is not None:
        tr["len"] = (lt_old, lt, lr_old, lr)
    _need(s.le(s.W2, 1, lt_old) == s.le(s.W2, 1, lt + 1), "old t spills")
    s.lt, s.lr = lt, lr
    s.Iter ^= 1


# =============================================================================
# Whole runs
# =============================================================================
def run(p, x, steps=None, padding="block", history=False):
    """Run the full schedule.  Returns the final state (and the per-step
    states when history=True)."""
    s = init(p, x)
    T = steps if steps is not None else step_bound(s.n)
    hist = [s.copy()] if history else None
    for i in range(1, T + 1):
        step(s, i, padding)
        if history:
            hist.append(s.copy())
    return (s, hist) if history else s


def inverse_from(s, p):
    """x^{-1} from a finished run: t' (at W2's left end) with the Iter sign."""
    tp = s.le(s.W2, 1, s.N - s.lr)
    return tp % p if s.Iter else (-tp) % p


def final_ok(s, p):
    """The normalised endpoint: r = 1, r' = 0, q = 0, t = p, lr = lq = 0,
    Sign = 0, phases (0, 0), W2 unrotated with t' < p at its left end."""
    n, N = s.n, s.N
    if (s.lr, s.lq, s.P1, s.P2, s.Sign, s.lt) != (0, 0, 0, 0, 0, n):
        return False
    if s.le(s.W1, 1, n) != p or s.W1[n] != 0 or s.be(s.W1, n + 2, N) != 1:
        return False
    tp = s.le(s.W2, 1, N)
    return tp < p


def check_prime(p, padding="block"):
    """Every x in [1, p): normalised endpoint, x^{-1} recovered, within the
    step bound.  Returns (max steps used, step bound, max pad blocks)."""
    n = bitlen(p)
    Nmax = step_bound(n)
    worst, blocks = 0, 0
    for x in range(1, p):
        N_x = steps_needed(p, x)
        assert 4 * n <= N_x <= Nmax, (p, x, N_x, Nmax)
        s = run(p, x, Nmax, padding)
        assert final_ok(s, p), (p, x, s.row())
        assert inverse_from(s, p) * x % p == 1, (p, x)
        worst = max(worst, N_x)
        blocks = max(blocks, s.ls)
    return worst, Nmax, blocks


def pad_blocks_bound(n):
    """Idle blocks at most ceil(c n) - n (the step count is in [4n, Nmax]),
    which fits the floor(log2 n) + 1 bits of ls."""
    return math.ceil(C * n) - n


# =============================================================================
# [Luo26] Table 4
# =============================================================================
# step: (W1, W2, t, q, r, t', r', lt, lq, lr', ls, P1, P2, Iter, Sign)
TABLE4 = {
    0: ("10||0100101", "00000|1101|", 1, 0, 37, 0, 13, 1, 0, 4, 0, 0, 0, 0, 0),
    1: ("10||0100101", "0000|1101|0", 1, 0, 37, 0, 13, 1, 0, 4, 1, 0, 0, 0, 0),
    2: ("10||0100101", "000|1101|00", 1, 0, 37, 0, 13, 1, 0, 4, 2, 0, 1, 0, 0),
    3: ("10|1|001011", "0000|1101|0", 1, 2, 11, 0, 13, 1, 1, 4, 1, 0, 1, 0, 0),
    4: ("10|10|01011", "00000|1101|", 1, 2, 11, 0, 13, 1, 2, 4, 0, 1, 0, 0, 0),
    5: ("10|1|001011", "0000|1101|0", 1, 2, 11, 0, 13, 1, 1, 4, 1, 1, 0, 0, 0),
    6: ("10||0001011", "000|1101|01", 1, 0, 11, 2, 13, 1, 0, 4, 2, 1, 1, 0, 1),
    7: ("10||0001011", "1000|1101|0", 1, 0, 11, 2, 13, 1, 0, 4, 1, 1, 1, 0, 0),
    8: ("010||001101", "10000|1011|", 2, 0, 13, 1, 11, 2, 0, 4, 0, 0, 0, 1, 0),
    9: ("010||001101", "0000|1011|1", 2, 0, 13, 1, 11, 2, 0, 4, 1, 0, 1, 1, 0),
    10: ("010|1|00010", "10000|1011|", 2, 1, 2, 1, 11, 2, 1, 4, 0, 1, 0, 1, 0),
    11: ("010||000010", "1000|1011|1", 2, 0, 2, 3, 11, 2, 0, 4, 1, 1, 1, 1, 1),
    12: ("110||001011", "0100000|10|", 3, 0, 11, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0),
    13: ("110||001011", "100000|10|0", 3, 0, 11, 2, 2, 2, 0, 2, 1, 0, 0, 0, 0),
    14: ("110||001011", "00000|10|01", 3, 0, 11, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0),
    15: ("110||001011", "0000|10|010", 3, 0, 11, 2, 2, 2, 0, 2, 3, 0, 1, 0, 0),
    16: ("110|1|00011", "00000|10|01", 3, 4, 3, 2, 2, 2, 1, 2, 2, 0, 1, 0, 0),
    17: ("110|10|0011", "100000|10|0", 3, 4, 3, 2, 2, 2, 2, 2, 1, 0, 1, 0, 0),
    18: ("110|101|001", "0100000|10|", 3, 5, 1, 2, 2, 2, 3, 2, 0, 1, 0, 0, 0),
    19: ("110|10|0001", "010000|10|1", 3, 4, 1, 5, 2, 2, 2, 2, 1, 1, 0, 0, 0),
    20: ("110|1|00001", "10000|10|10", 3, 4, 1, 5, 2, 2, 1, 2, 2, 1, 0, 0, 0),
    21: ("110||000001", "0100|10|100", 3, 0, 1, 17, 2, 2, 0, 2, 3, 1, 1, 0, 1),
    22: ("110||000001", "00100|10|10", 3, 0, 1, 17, 2, 2, 0, 2, 2, 1, 1, 0, 0),
    23: ("110||000001", "000100|10|1", 3, 0, 1, 17, 2, 2, 0, 2, 1, 1, 1, 0, 0),
    24: ("100010||010", "11000000|1|", 17, 0, 2, 3, 1, 5, 0, 1, 0, 0, 0, 1, 0),
    25: ("100010||010", "1000000|1|1", 17, 0, 2, 3, 1, 5, 0, 1, 1, 0, 0, 1, 0),
    26: ("100010||010", "000000|1|11", 17, 0, 2, 3, 1, 5, 0, 1, 2, 0, 1, 1, 0),
    27: ("100010|1|00", "1000000|1|1", 17, 2, 0, 3, 1, 5, 1, 1, 1, 0, 1, 1, 0),
    28: ("100010|10|0", "11000000|1|", 17, 2, 0, 3, 1, 5, 2, 1, 0, 1, 0, 1, 0),
    29: ("100010|1|00", "1000000|1|1", 17, 2, 0, 3, 1, 5, 1, 1, 1, 1, 0, 1, 0),
    30: ("100010||000", "100100|1|10", 17, 0, 0, 37, 1, 5, 0, 1, 2, 1, 1, 1, 1),
    31: ("100010||000", "0100100|1|1", 17, 0, 0, 37, 1, 5, 0, 1, 1, 1, 1, 1, 0),
    32: (None, None, 37, 0, 1, 17, 0, 6, 0, 0, 0, 0, 0, 0, 0),
    33: (None, None, 37, 0, 1, 17, 0, 6, 0, 0, 1, 0, 0, 0, 0),
    34: (None, None, 37, 0, 1, 17, 0, 6, 0, 0, 2, 0, 0, 0, 0),
    35: (None, None, 37, 0, 1, 17, 0, 6, 0, 0, 3, 0, 0, 0, 0),
    36: (None, None, 37, 0, 1, 17, 0, 6, 0, 0, 4, 0, 0, 0, 0),
}


def table4_mismatches():
    """Rows of [Luo26] Table 4 this model does not reproduce (should be [])."""
    s = init(37, 13)
    bad = []
    for T in range(0, 37):
        if T:
            step(s, T, padding="luo")
        want = TABLE4[T]
        row = s.fields() + (s.lt, s.lq, s.lr, s.ls, s.P1, s.P2,
                                        s.Iter, s.Sign)
        if tuple(row) != tuple(want[2:]):
            bad.append((T, "fields", row, want[2:]))
        if want[0] is not None:
            w1, w2 = s.bank_strings()
            if (w1, w2) != (want[0], want[1]):
                bad.append((T, "banks", (w1, w2), want[:2]))
    return bad


# =============================================================================
# Step-dependent active windows ([Luo26] Sec 4.2, App A.2)
# =============================================================================
def luo_windows(n, T):
    """[Luo26]'s bounds on the positions each location-controlled block of
    step T can touch, over every input.  (lo, hi), 1-based positions:

      r      the r / 2^ls r' window [L, R], L = lt + lq + 2, R = n + 3 - ls
      swap   J = lt + lq + 1
      t      the right boundary of the t / t' window (lt + 1 or n+3-lr-ls)
      lt     the length-update scan for lt: [lt, n + 3 - lr]   (4 | T only)
      lr     the length-update scan for lr: [lt* + 2, n + 3]  (4 | T only)
    """
    c4 = 4 * C
    k1 = max(math.ceil((T - n - 1 - 4 * DELTA) / (c4 - 1)), 1) + 2
    k2 = max(math.ceil((T - 3 * (n + 1) - 4 * DELTA) / (c4 - 3)), 1) + 1
    K2 = min(T // 2 + 2, n + 2)
    K3 = min(T // 4 + 2, n + 1)
    k4 = max(math.ceil((T - 4 * (n + 1) - 4 * DELTA) / (c4 - 4)), 1)
    K4 = min(T // 4 + 3, n + 2)
    k5 = max(math.ceil((T - 4 * DELTA) / c4), 1)
    return {"r": (k1, n + 3), "swap": (k2, K2), "t": (1, K3),
            "lt": (k4, K4), "lr": (k5, n + 3)}


def observed_windows(p, xs=None):
    """The exact ranges the model uses at each step, over `xs` (default all)."""
    n = bitlen(p)
    Nmax = step_bound(n)
    obs = {}
    for x in (xs if xs is not None else range(1, p)):
        s = init(p, x)
        for T in range(1, Nmax + 1):
            tr = {}
            step(s, T, "block", tr)
            o = obs.setdefault(T, {})
            for key, val in tr.items():
                if key == "len":
                    lt_old, lt, lr_old, lr = val
                    vals = {"lt": (lt_old, n + 3 - lr_old), "lr": (lt + 2, n + 3)}
                elif key == "r":
                    vals = {"r": val}
                else:
                    vals = {key: (val, val)}
                for k2, (a, b) in vals.items():
                    lo, hi = o.get(k2, (a, b))
                    o[k2] = (min(lo, a), max(hi, b))
    return obs


def window_violations(p, xs=None):
    """Steps where the observed range leaves [Luo26]'s analytic window."""
    n = bitlen(p)
    bad = []
    for T, o in observed_windows(p, xs).items():
        w = luo_windows(n, T)
        for k, (lo, hi) in o.items():
            if lo < w[k][0] or hi > w[k][1]:
                bad.append((T, k, (lo, hi), w[k]))
    return bad
