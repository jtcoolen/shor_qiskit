"""Gidney 2025's approximate residue-number modular exponentiation: the classical half.

[G25] (C. Gidney, "How to factor 2048 bit RSA integers with less than a million
noisy qubits", 2025) computes V ~ g^e mod N without ever holding an n-bit
number in superposition.  With the exponent's bits taken w1 at a time, g^e is
the *exact integer* product of W1 classical multipliers M[j][k_j] = g^(k 2^(j w1))
mod N.  That product is below N^W1, so it is determined by its residues modulo a
set P of small primes whose product L exceeds N^W1.  For each prime p:

  loop1   S_p = sum_j dlog_{g_p}(M[j][k_j])            additions of looked-up dlogs
  loop2   S_p mod (p - 1)                              binary long division
  loop3   V_p = g_p^(S_p mod (p-1)) mod p              windowed multiplications mod p
  loop4   acc += truncate(V_p u_p mod L mod N)         u_p the CRT basis element
  unloop3, unloop2, and loop1 for the next prime (merged: it adds the
  *difference* of the two primes' dlog tables)

Only the loop4 accumulator is n-sized, and it is truncated to its top f bits
(modulo trunc = N >> t, t = n - f); every other register is ~l = log log N.  The
accumulator starts at a random mask, which is what makes the truncation harmless
for period finding ([G25] Sec 2.2).

This module holds everything classical: the residue system (primes of bit
length l that divide no multiplier, with L mod N small), generators and
discrete-log tables, every lookup table the circuit reads, a bit-exact integer
model of the circuit's output (`reference`), and the symbolic cost tallies of
[G25] Tables 3-5 (`tallies`).  `rns.py` builds the circuit and is checked
against `reference` exhaustively at toy N.
"""

import math
import random
from dataclasses import dataclass, field


# =============================================================================
# Parameters
# =============================================================================
@dataclass(frozen=True)
class Params:
    """[G25] Table 2.  m exponent qubits (Ekera-Hastad: n/2 + n/s), l-bit
    primes, windows w1 (loop1), w3 (loop3/unloop3), w4 (loop4), f kept bits."""
    n: int
    s: int
    ell: int
    w1: int
    w3: int
    w4: int
    f: int
    m: int = None

    def __post_init__(self):
        if self.m is None:
            object.__setattr__(self, "m", math.ceil(self.n / 2 + self.n / self.s))

    @property
    def len_m(self):
        return self.m.bit_length()

    @property
    def W1(self):
        return -(-self.m // self.w1)

    @property
    def W3(self):
        return -(-self.ell // self.w3)

    @property
    def W4(self):
        return -(-self.ell // self.w4)

    @property
    def num_primes(self):
        """|P| ~ n m / (l w1) ([G25] Table 2): L must exceed N^W1."""
        return math.ceil(self.n * self.W1 / (self.ell - 0.5))


# [G25] Table 5: n -> (s, l, w1, w3, w4, f, m, P_deviant, E(shots), Toffolis, qubits)
TABLE5 = {
    1024: (8, 18, 6, 3, 6, 28, 640, 0.0287, 9.4, 1.1e9, 742),
    1536: (8, 21, 6, 3, 5, 31, 960, 0.0183, 9.3, 3.1e9, 1074),
    2048: (8, 21, 6, 3, 5, 33, 1280, 0.0125, 9.2, 6.5e9, 1399),
    3072: (8, 21, 6, 3, 5, 35, 1920, 0.0091, 9.2, 1.9e10, 2043),
    4096: (8, 24, 6, 3, 5, 36, 2560, 0.0080, 9.2, 4.0e10, 2692),
    6144: (8, 24, 6, 3, 5, 39, 3840, 0.0042, 9.1, 1.2e11, 3978),
    8192: (8, 24, 6, 3, 5, 40, 5120, 0.0040, 9.1, 2.7e11, 5261),
}


def params_table5(n):
    s, ell, w1, w3, w4, f, m, *_ = TABLE5[n]
    return Params(n=n, s=s, ell=ell, w1=w1, w3=w3, w4=w4, f=f, m=m)


# =============================================================================
# Small-number helpers
# =============================================================================
def is_prime(p):
    if p < 2:
        return False
    for d in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if p % d == 0:
            return p == d
    d, r = p - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, p)
        if x in (1, p - 1):
            continue
        for _ in range(r - 1):
            x = x * x % p
            if x == p - 1:
                break
        else:
            return False
    return True


def primitive_root(p):
    fs, q, d = [], p - 1, 2
    while d * d <= q:
        if q % d == 0:
            fs.append(d)
            while q % d == 0:
                q //= d
        d += 1
    if q > 1:
        fs.append(q)
    for g in range(2, p):
        if all(pow(g, (p - 1) // f_, p) != 1 for f_ in fs):
            return g
    raise ValueError(p)


def primes_of_length(ell):
    """Every prime of bit length ell (a sieve)."""
    hi = 1 << ell
    sieve = bytearray([1]) * hi
    sieve[0:2] = b"\x00\x00"
    for d in range(2, int(hi ** 0.5) + 1):
        if sieve[d]:
            sieve[d * d::d] = bytearray(len(range(d * d, hi, d)))
    return [p for p in range(1 << (ell - 1), hi) if sieve[p]]


def residue_system_size(N, par, rng):
    """(|P|, L) for a random residue system of l-bit primes with L >= N^W1,
    the exclusion rule aside: what a shot at this size iterates over."""
    pool = primes_of_length(par.ell)
    rng.shuffle(pool)
    need, L, k = N ** par.W1, 1, 0
    while L < need:
        L *= pool[k]
        k += 1
    return k, L, pool[:k]


def dlog_table(p, g):
    """{x: dlog_g(x) mod p} for every nonzero x: one pass over g's powers."""
    t, x = {}, 1
    for e in range(p - 1):
        t[x] = e
        x = x * g % p
    return t


def bits_of(v, lo, width):
    return (v >> lo) & ((1 << width) - 1)


def window_width(total, j, w):
    return max(0, min(w, total - j * w))


# =============================================================================
# The residue system and every table the circuit reads
# =============================================================================
@dataclass
class Config:
    """One shot's precomputation ([G25]'s ExecutionConfig).

    lookup1[i][j][k]    difference of prime i's and prime i-1's dlog of M[j][k]
                        (i = |P| returns the accumulator to 0), mod 2^D
    lookup3c[i][a]      g_p^a mod p for the first two loop3 windows (2 w3 bits)
    lookup3a[i][j][k]   (l0 2^(k w3) X_j(l1)) mod p, address l0 + (l1 << a0):
                        loop3's windowed multiply by X_j(l1) = g_p^(l1 2^(j w3))
                        (a0, a1 the actual widths of result window k and
                        exponent window j)
    lookup3b[i][j][k]   the same with X_j(l1)^-1: unloop3's un-multiply
    lookup4[i][j][v]    -(truncated contribution of V_p's window j = v) mod trunc
    """
    N: int
    g: int
    par: Params
    primes: list
    gens: list
    D: int
    t: int
    trunc: int
    mask_bits: int
    L: int
    lookup1: list = field(repr=False)
    lookup3c: list = field(repr=False)
    lookup3a: list = field(repr=False)
    lookup3b: list = field(repr=False)
    lookup4: list = field(repr=False)
    dlogs: list = field(repr=False)


def multipliers(N, g, par, bases=None):
    """M[j][k] = the product selected by window j holding k: g^(k 2^(j w1))
    mod N, or, with `bases` (one per exponent bit), the product of the bases
    of the bits set in k -- Ekera-Hastad's two registers, x on powers of g and
    y on powers of A^-1, are one exponent register this way."""
    out = []
    for j in range(par.W1):
        w = min(par.w1, par.m - j * par.w1)
        if bases is None:
            base = pow(g, 1 << (j * par.w1), N)
            out.append([pow(base, k, N) for k in range(1 << w)])
            continue
        bs = bases[j * par.w1:j * par.w1 + w]
        row = []
        for k in range(1 << w):
            v = 1
            for i, b in enumerate(bs):
                if k >> i & 1:
                    v = v * b % N
            row.append(v)
        out.append(row)
    return out


def choose_primes(N, par, M, rng, pool_bits=None, tries=20000, need_small_offset=True):
    """Primes of bit length l (or `pool_bits` .. l), dividing no multiplier, with
    L >= N^W1 and (if asked) L mod N < N >> f -- the offset that each of the
    |P| W4 truncated terms would otherwise carry ([G25] Fig 3's assertion).
    Random subsets are tried until one satisfies it."""
    lo, hi = 1 << ((pool_bits or par.ell) - 1), 1 << par.ell
    pool = [p for p in range(lo | 1, hi, 2) if is_prime(p) and N % p
            and all(x % p for row in M for x in row)]     # (N % p: at toy size a prime
                                                          # could be a factor of N)
    need = N ** par.W1
    rng.shuffle(pool)
    for _ in range(tries):
        rng.shuffle(pool)
        sel, L = [], 1
        extra = rng.randrange(3)                 # vary the size, not only the members
        for p in pool:
            if L >= need:
                if extra == 0:
                    break
                extra -= 1
            sel.append(p)
            L *= p
        if L < need:
            raise ValueError("not enough primes: raise l")
        if not need_small_offset or L % N < (N >> par.f) or (N >> par.f) == 0 and L % N == 0:
            return sorted(sel), L
    raise ValueError("no residue system with a small L mod N found")


def make_config(N, g, par, rng=None, mask_bits=None, primes=None, need_small_offset=True,
                L=None, bases=None, pool_bits=None):
    """`primes` fixes the residue system (then `L` may give the product of a
    larger system of which these are a part: tables for a few primes of a
    cryptographic-size system, for building one prime's iteration)."""
    rng = rng or random.Random(0)
    n = N.bit_length()
    assert par.n == n, (par.n, n)
    M = multipliers(N, g, par, bases)
    if primes is None:
        primes, L = choose_primes(N, par, M, rng, pool_bits=pool_bits,
                                  need_small_offset=need_small_offset)
    else:
        assert all(x % p for p in primes for row in M for x in row), "a prime divides M"
        L = L or math.prod(primes)
    gens = [primitive_root(p) for p in primes]
    D = par.ell + par.len_m
    t = n - par.f
    trunc = N >> t
    mask_bits = par.f - 1 if mask_bits is None else mask_bits
    dl = []
    for p, gp in zip(primes, gens):
        tab = dlog_table(p, gp)
        dl.append([[tab[x % p] for x in row] for row in M])
    zero = [[0] * len(row) for row in M]
    lookup1 = []
    for i in range(len(primes) + 1):
        cur = dl[i] if i < len(primes) else zero
        prev = dl[i - 1] if i > 0 else zero
        lookup1.append([[(cur[j][k] - prev[j][k]) % (1 << D) for k in range(len(M[j]))]
                        for j in range(len(M))])
    w3 = par.w3
    l3c, l3a, l3b, l4 = [], [], [], []
    for p, gp in zip(primes, gens):
        l3c.append([pow(gp, a, p) for a in range(1 << min(2 * w3, par.ell))])
        ta, tb = {}, {}
        for j in range(2, par.W3):
            a1 = window_width(par.ell, j, w3)
            for k in range(par.W3):
                a0 = window_width(par.ell, k, w3)
                rowa, rowb = [], []
                for addr in range(1 << (a0 + a1)):
                    l0, l1 = addr & ((1 << a0) - 1), addr >> a0
                    X = pow(gp, l1 << (j * w3), p)
                    rowa.append((l0 << (k * w3)) * X % p)
                    rowb.append((l0 << (k * w3)) * pow(X, -1, p) % p)
                ta[j, k], tb[j, k] = rowa, rowb
        l3a.append(ta)
        l3b.append(tb)
        u = (L // p) * pow(L // p, -1, p)
        rows = []
        for j in range(par.W4):
            rows.append([(-((((u * v) << (j * par.w4)) % L % N) >> t)) % trunc
                         for v in range(1 << par.w4)])
        l4.append(rows)
    conf = Config(N=N, g=g, par=par, primes=primes, gens=gens, D=D, t=t, trunc=trunc,
                  mask_bits=mask_bits, L=L, lookup1=lookup1, lookup3c=l3c, lookup3a=l3a,
                  lookup3b=l3b, lookup4=l4, dlogs=dl)
    conf.multipliers = M
    return conf


# =============================================================================
# Bit-exact model of the circuit
# =============================================================================
def loop2_model(v, modulus, D, c):
    """[G25] loop2 on a D-bit register: returns (register, remainder) with the
    quotient flags left in the high bits, exactly as the circuit leaves them."""
    reg = v
    n = D
    while n > c:
        n -= 1
        thr = modulus << (n - c)
        lowmask = (1 << (n + 1)) - 1
        low = (reg & lowmask) - thr
        low %= 1 << (n + 1)
        reg = (reg & ~lowmask) | low
        if (reg >> n) & 1:
            reg = (reg & ~((1 << n) - 1)) | (((reg & ((1 << n) - 1)) + thr) % (1 << n))
    return reg, reg & ((1 << c) - 1)


def residues(conf, e):
    """(S_p, S_p mod (p-1), V_p) for every prime, as the circuit computes them."""
    par, out = conf.par, []
    for i, p in enumerate(conf.primes):
        S = sum(conf.dlogs[i][j][bits_of(e, j * par.w1, par.w1)] for j in range(par.W1))
        assert S < 1 << conf.D
        _, Sc = loop2_model(S, p - 1, conf.D, par.ell)
        assert Sc == S % (p - 1)
        out.append((S, Sc, pow(conf.gens[i], Sc, p)))
    return out


def reference(conf, e, mask):
    """The output register's final value: (mask + sum of truncated
    contributions) mod trunc, accumulated by subtracting the flipped tables."""
    par, acc = conf.par, mask
    assert 0 <= mask < conf.trunc
    for i, (_, _, V) in enumerate(residues(conf, e)):
        for j in range(par.W4):
            acc = (acc - conf.lookup4[i][j][bits_of(V, j * par.w4, par.w4)]) % conf.trunc
    return acc


def exact_value(conf, e):
    """What the approximation approximates: the product of the selected
    multipliers mod N (g^e mod N for the plain bases)."""
    par, v = conf.par, 1
    M = conf.multipliers
    for j in range(par.W1):
        v = v * M[j][bits_of(e, j * par.w1, par.w1)] % conf.N
    return v


def deviation(conf, e, mask=0):
    """|approximation - g^e mod N| / N (modular distance)."""
    approx = (reference(conf, e, mask) - mask) % conf.trunc << conf.t
    err = (approx - exact_value(conf, e)) % conf.N
    return min(err, conf.N - err) / conf.N


def deviation_bound(conf):
    """[G25] Fig 3: at most ~3 |P| l / 2^f (truncation, L mod N offsets, and the
    floor in trunc = N >> t)."""
    return 3 * len(conf.primes) * conf.par.ell / 2 ** conf.par.f


# =============================================================================
# [G25] Tables 3 and 4: symbolic tallies
# =============================================================================
def lookup_toffolis(a):
    return (1 << a) - a - 1


def phaseup_toffolis(a):
    return 2 * math.ceil(2 ** (a / 2))


def tallies(par, primes=None):
    """Toffolis per shot and peak qubits from [G25] Tables 3-4: additions of an
    r-qubit register cost r - 1, lookups 2^a - a - 1, phaseups ~2 sqrt(2^a).
    Returns a dict with the per-subroutine breakdown."""
    P = primes if primes is not None else par.num_primes
    ell, lm, f = par.ell, par.len_m, par.f
    D = ell + lm
    rows = {
        "loop1": ((P + 1) * par.W1, D, par.w1, 1, 1, 0),
        "loop2": (P * lm, D, 0, 2, 0, 0),
        "loop3 (startup)": (P, ell, 2 * par.w3, 0, 1, 0),
        "loop3 (body)": (P * (par.W3 - 2) * par.W3, ell, 2 * par.w3, 2, 1, 0),
        "loop4": (P * par.W4, f, par.w4, 2.5, 1.5, 1),
        "unloop3 (body)": (P * (par.W3 - 2) * 2 * par.W3, ell, 2 * par.w3, 2.5, 1.5, 1),
        "unloop3 (cleanup)": (P, ell, 2 * par.w3, 0, 0, 1),
        "unloop2": (P * lm, D, 0, 2, 0, 0),
    }
    out, total = {}, 0.0
    for k, (it, size, addr, add, lk, ph) in rows.items():
        c = it * (add * (size - 1) + lk * lookup_toffolis(addr) + ph * phaseup_toffolis(addr))
        out[k] = c
        total += c
    qubits = {
        "loop1": par.m + f + 3 * ell + 3 * lm,
        "loop3": par.m + f + 4 * ell + lm,
        "loop4": par.m + 3 * f + 2 * ell + lm,
    }
    return {"per_shot": total, "breakdown": out, "qubits": max(qubits.values()),
            "qubits_by_phase": qubits, "primes": P}


def per_factoring(par):
    """[G25] Table 5's Toffolis column: per shot x E(shots)."""
    shots = TABLE5[par.n][8]
    return tallies(par)["per_shot"] * shots


# =============================================================================
# The output distribution of a shot, exactly (toy N)
# =============================================================================
def eh_bases(N, g, A, lx, ly):
    """Per-bit bases of Ekera-Hastad's exponent register (x, y): g^(2^i) for
    the lx bits of x, A^(-2^i) for the ly bits of y."""
    Ainv = pow(A, -1, N)
    return [pow(g, 1 << i, N) for i in range(lx)] + [pow(Ainv, 1 << i, N) for i in range(ly)]


def eh_distribution(conf, lx, ly, approx=True, mask_bits=None):
    """P[j, k] of the two-register frequency measurement, exactly.

    approx=False: the textbook oracle f(x, y) = g^x A^-y mod N.
    approx=True:  what the circuit does -- the output register starts in a
    uniform superposition of masks s < 2^mask_bits and ends at
    (s + V(x, y)) mod trunc; after it is measured at o, the (x, y) that
    interfere are those with o - V(x, y) mod trunc a valid mask.  Summed over o
    ([G25] Sec 2.2)."""
    import numpy as np
    qa, qb = 1 << lx, 1 << ly
    V = np.zeros((qa, qb), dtype=np.int64)
    for x in range(qa):
        for y in range(qb):
            e = x | (y << lx)
            V[x, y] = exact_value(conf, e) if not approx else reference(conf, e, 0)
    groups = []
    if not approx:
        for val in np.unique(V):
            groups.append((V == val).astype(float))
        norm = 1.0
    else:
        mb = conf.mask_bits if mask_bits is None else mask_bits
        M = 1 << mb
        for o in range(conf.trunc):
            groups.append((((o - V) % conf.trunc) < M).astype(float))
        norm = 1.0 / M
    P = np.zeros((qa, qb))
    for G in groups:
        P += np.abs(np.fft.fft2(G)) ** 2
    return P * norm / (qa * qb) ** 2
