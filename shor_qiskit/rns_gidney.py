"""Toy Qiskit transcription of Gidney arXiv:2505.15917 App. A.1 `approx_modexp`.

Verified source: the arXiv:2505.15917 PDF, Secs. 2.1-2.4 (Eqs. 1-53),
Fig. 3 and App. A.1 (`approx_modexp`, `loop1`, `loop2`, `loop3`, `loop4`,
`unloop3`, `unloop2`).  The Zenodo upload [Gid25] that defines
`ExecutionConfig` and the lookup tables was not available, so every table
here is rebuilt only from the equations and docstrings:

* loop1: `Q_dlog += lookup1[i, j][Q_k]` holds Eq. 52 differences of the
  discrete logs (Eq. 48) of window products `g ** (x << j*w1) mod N`; index
  `len(P)` uncomputes (difference to zero).
* loop2/unloop2: the printed binary-long-division code, gate for gate.
* loop3: only the "skip first 2 iterations by direct lookup" startup is
  built, i.e. `2 * window3 >= ell`, so `lookup3c[i][l] = pow(g_i, l, p_i)`
  follows from the loop3 docstring.  The windowed multiply body
  (`lookup3a`/`lookup3b`) is not reconstructed.
* loop4: window4 = 1, so a window holds one residue bit and its table entry
  is the Eq. 21 constant `C[j, k]`.  The fifth optimisation of Sec. 2.4
  turns the addition into subtraction of `(-C) mod (N >> t)`; the wrap qubit
  is removed by an X-basis measurement and, on outcome 1, the printed phase
  comparison `Q_acc[:-1] >= table2[Q_k]` with `table2 = trunc - table`.
* unloop3 cleanup: the lookup output is deleted by X-basis measurement and a
  phase correction `parity(mx & lookup3c[l])` on the address.

Deviations, all labelled: Shor-style `g**e` period finding (as Sec. 2
itself describes), not Ekera-Hastad `g**x h**y`; arithmetic is ancilla-free
Fourier addition with address-controlled phases instead of materialised
unary-iteration lookups, so there are no lookup "vents" to clear; widths
(`ell`, `f`, windows) are laptop toys far outside the paper's grid.  The
paper's measured-and-phased qubits are reset here with a conditional X.
This is not the CFS ePrint 2024/222 least-significant-bit algorithm.
"""

from dataclasses import dataclass, field
from math import gcd, prod

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from shor_essentials import add_const, c_add_const, qft


def _len(a):
    """Table 1: len a = ceil(log2(max(1, a)))."""
    return (max(1, a) - 1).bit_length()


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _primitive_root(p):
    factors = {d for d in range(2, p) if (p - 1) % d == 0 and _is_prime(d)}
    return next(g for g in range(2, p)
                if all(pow(g, (p - 1) // q, p) != 1 for q in factors))


def modular_deviation(a, b, modulus):
    """Eq. 14."""
    diff = (a - b) % modulus
    return min(diff, modulus - diff) / modulus


@dataclass(frozen=True)
class ApproxModexpConfig:
    modulus: int
    base: int
    exponent_bits: int
    primes: tuple
    kept_bits: int
    window1: int = 1
    window3: int = 3
    mask_bits: int = 0
    generators: tuple = field(init=False)

    def __post_init__(self):
        n, P = self.modulus, tuple(self.primes)
        object.__setattr__(self, "primes", P)
        if n < 3 or gcd(self.base % n, n) != 1 or self.exponent_bits < 1:
            raise ValueError("requires N >= 3, unit base, exponent_bits >= 1")
        if (not P or len(set(P)) != len(P) or not all(map(_is_prime, P))
                or len({p.bit_length() for p in P}) != 1 or min(P) < 3):
            raise ValueError("Eq. 2: distinct odd primes of one bit length")
        if not 1 <= self.kept_bits <= n.bit_length() or self.window1 < 1:
            raise ValueError("requires 1 <= f <= len N and window1 >= 1")
        if prod(P) < n ** self.num_windows1:
            raise ValueError("Eq. 8 (windowed): prod(P) >= N ** W1")
        if prod(P) % n >= n >> self.kept_bits:
            raise ValueError("Fig. 3 assert: L % N < N >> f")
        if any(self.window_multiplier(j, x) % p == 0 for p in P
               for j in range(self.num_windows1)
               for x in range(1 << self.window_width(j))):
            raise ValueError("Eq. 49: a multiplier vanishes modulo a prime")
        if 2 * self.window3 < self.ell:
            raise ValueError("only the loop3 startup lookup is implemented: "
                             "requires 2 * window3 >= ell")
        if self.mask_bits < 0 or 1 << self.mask_bits > self.truncated_modulus:
            raise ValueError("mask range must fit the truncated modulus")
        for p in P:
            if any(loop2_reference(d, p - 1, self.ell, self.len_dlog)
                   % (1 << self.ell) != d % (p - 1)
                   for d in range(self.num_windows1 * (p - 2) + 1)):
                raise ValueError("loop2 long division must reduce every "
                                 "reachable dlog sum modulo p - 1")
        object.__setattr__(self, "generators",
                           tuple(_primitive_root(p) for p in P))

    # --- Table 2 parameters -------------------------------------------------
    @property
    def ell(self):
        return self.primes[0].bit_length()

    @property
    def crt_modulus(self):
        return prod(self.primes)

    @property
    def shift(self):
        return self.modulus.bit_length() - self.kept_bits

    @property
    def truncated_modulus(self):
        return self.modulus >> self.shift

    @property
    def num_windows1(self):
        return -(-self.exponent_bits // self.window1)

    @property
    def len_dlog(self):
        return self.ell + _len(self.exponent_bits)

    def window_width(self, j):
        return min(self.window1, self.exponent_bits - j * self.window1)

    def window_multiplier(self, j, x):
        return pow(self.base, x << (j * self.window1), self.modulus)

    # --- classical tables ---------------------------------------------------
    def dlog(self, i, j, x):
        """Eq. 48 for the window product: g_i ** D == multiplier (mod p_i)."""
        p, g = self.primes[i], self.generators[i]
        target = self.window_multiplier(j, x) % p
        return next(d for d in range(p - 1) if pow(g, d, p) == target)

    def lookup1(self, i, j, x):
        """Eq. 52 difference D[i] - D[i-1]; D[-1] = D[len(P)] = 0."""
        now = self.dlog(i, j, x) if i < len(self.primes) else 0
        before = self.dlog(i - 1, j, x) if i else 0
        return (now - before) % (1 << self.len_dlog)

    def lookup3c(self, i):
        p, g = self.primes[i], self.generators[i]
        return [pow(g, l, p) for l in range(1 << self.ell)]

    def coefficient(self, i):
        """Eq. 10."""
        L, p = self.crt_modulus, self.primes[i]
        return (L // p) * pow(L // p, -1, p)

    def contribution(self, i, k):
        """Eq. 21: C[i, k]."""
        L = self.crt_modulus
        return (((self.coefficient(i) << k) % L % self.modulus) >> self.shift
                ) % self.truncated_modulus

    # --- classical mirrors --------------------------------------------------
    def windows(self, e):
        return [e >> (j * self.window1) & ((1 << self.window_width(j)) - 1)
                for j in range(self.num_windows1)]

    def dlog_sum(self, i, e):
        """Eq. 50 with windows: S_p as the integer the register holds."""
        return sum(self.dlog(i, j, x) for j, x in enumerate(self.windows(e)))

    def residue(self, i, e):
        """Eq. 51: V_p = g_p ** (S_p mod (p - 1)) mod p."""
        p = self.primes[i]
        return pow(self.generators[i], self.dlog_sum(i, e) % (p - 1), p)

    def product(self, e):
        """The never-reduced integer the residues represent (< L by Eq. 8)."""
        return prod(self.window_multiplier(j, x)
                    for j, x in enumerate(self.windows(e)))

    def approximate(self, e):
        """Eq. 20: the truncated accumulator, before the mask is added."""
        return sum(self.contribution(i, k)
                   for i in range(len(self.primes))
                   for k in range(self.ell)
                   if self.residue(i, e) >> k & 1) % self.truncated_modulus

    def deviation(self, e):
        return modular_deviation(pow(self.base, e, self.modulus),
                                 self.approximate(e) << self.shift,
                                 self.modulus)


def fig3_reference(g, Q_e, N, P, f):
    """Gidney Fig. 3, transcribed (unwindowed, no uncomputation)."""
    L = prod(P)
    ell = max(p.bit_length() for p in P)
    m = Q_e.bit_length()
    t = N.bit_length() - f
    assert L >= N ** m and (L % N) < (N >> f)
    Q_total = 0
    for p in P:
        Q_residue = 1
        for k in range(m):
            if Q_e & (1 << k):
                Q_residue = Q_residue * (pow(g, 1 << k, N) % p) % p
        u = (L // p) * pow(L // p, -1, p)
        for k in range(ell):
            if Q_residue & (1 << k):
                Q_total = (Q_total + ((((u << k) % L % N) >> t) % (N >> t))
                           ) % (N >> t)
    return Q_total << t


def loop2_reference(value, modulus, compressed_len, length):
    """App. A.1 loop2 on a classical integer; returns the new register.

    `modulus` is the long-division divisor (p - 1 for a dlog register).  The
    low `compressed_len` bits hold the remainder when the input is small
    enough; the higher bits keep the (complemented) quotient bits so that
    `unloop2_reference` can invert the step."""
    mask = (1 << length) - 1
    n = length
    while n > compressed_len:
        n -= 1
        threshold = modulus << (n - compressed_len)
        low = (value & ((1 << (n + 1)) - 1)) - threshold
        value = (value & ~((1 << (n + 1)) - 1)) | (low % (1 << (n + 1)))
        if value >> n & 1:
            low = ((value & ((1 << n) - 1)) + threshold) % (1 << n)
            value = (value & ~((1 << n) - 1)) | low
    return value & mask


def unloop2_reference(value, modulus, compressed_len, length):
    n = compressed_len
    while n < length:
        threshold = modulus << (n - compressed_len)
        if value >> n & 1:
            low = ((value & ((1 << n) - 1)) - threshold) % (1 << n)
            value = (value & ~((1 << n) - 1)) | low
        low = ((value & ((1 << (n + 1)) - 1)) + threshold) % (1 << (n + 1))
        value = (value & ~((1 << (n + 1)) - 1)) | low
        n += 1
    return value


# --- quantum building blocks ------------------------------------------------
def _flip_zeros(qc, qubits, value):
    for b, q in enumerate(qubits):
        if not value >> b & 1:
            qc.x(q)


def _add_addressed(qc, addr, table, target):
    """target += table[addr] (mod 2**len(target)): the paper's `Q += T[Q_k]`
    as address-controlled Fourier phases, never materialising T[Q_k]."""
    n = len(target)
    qc.append(qft(n), target)
    for x, value in enumerate(table):
        value %= 1 << n
        if not value:
            continue
        _flip_zeros(qc, addr, x)
        for i in range(n):
            qc.mcp(2 * np.pi * value * 2.0 ** (i - n), list(addr), target[i])
        _flip_zeros(qc, addr, x)
    qc.append(qft(n).inverse(), target)


def _xor_lookup(qc, addr, table, target):
    """target ^= table[addr], one open/closed-controlled X per set bit."""
    for x, value in enumerate(table):
        for b, q in enumerate(target):
            if value >> b & 1:
                qc.mcx(list(addr), q, ctrl_state=x)


def _phase_lookup(qc, addr, flags):
    """Multiply |a> by (-1) ** flags[a] (a phaseup)."""
    for x, flag in enumerate(flags):
        if flag:
            _flip_zeros(qc, addr, x)
            if len(addr) == 1:
                qc.z(addr[0])
            else:
                qc.mcp(np.pi, list(addr[:-1]), addr[-1])
            _flip_zeros(qc, addr, x)


@dataclass
class PaperTally:
    """Sec. 3.1 accounting as in Table 3: subtractions and comparisons count
    as additions; a GHZ lookup is not a lookup (loop2/unloop2 steps count 2
    additions, a loop4 window 1 plus a comparison that fires on X outcome 1).
    """
    additions: int = 0
    conditional_additions: int = 0
    lookups: int = 0
    phaseups: int = 0
    x_measurements: int = 0


def approx_modexp(conf):
    """|e>|0> -> |e>|mask + approx(e) mod (N >> t)>, all workspace cleared.

    Returns `(circuit, registers, tally)`.  The circuit contains mid-circuit
    X-basis measurements with classically conditioned phase fixes."""
    if not isinstance(conf, ApproxModexpConfig):
        raise TypeError("conf must be an ApproxModexpConfig")
    f, ell = conf.kept_bits, conf.ell
    trunc = conf.truncated_modulus
    exponent = QuantumRegister(conf.exponent_bits, "exponent")
    result = QuantumRegister(f + 1, "result")
    dlog = QuantumRegister(conf.len_dlog, "dlog")
    residue = QuantumRegister(ell, "residue")
    wrap_bit = ClassicalRegister(1, "wrap_mx")
    residue_bits = ClassicalRegister(ell, "residue_mx")
    qc = QuantumCircuit(exponent, result, dlog, residue, wrap_bit,
                        residue_bits)
    tally = PaperTally()
    acc, wrap = list(result), result[f]
    for q in result[:conf.mask_bits]:
        qc.h(q)

    def loop1(i):
        for j in range(conf.num_windows1):
            addr = exponent[j * conf.window1:][:conf.window1]
            table = [conf.lookup1(i, j, x) for x in range(1 << len(addr))]
            _add_addressed(qc, addr, table, list(dlog))
            tally.additions += 1
            tally.lookups += 1

    for i, p in enumerate(conf.primes):
        loop1(i)
        n = conf.len_dlog                                    # loop2
        while n > ell:
            n -= 1
            threshold = (p - 1) << (n - ell)
            add_const(qc, -threshold, list(dlog[:n + 1]))
            c_add_const(qc, [dlog[n]], threshold, list(dlog[:n]))
            tally.additions += 2
        address = list(dlog[:ell])
        table3 = conf.lookup3c(i)                            # loop3 startup
        _xor_lookup(qc, address, table3, list(residue))
        tally.lookups += 1
        for k in range(ell):                                 # loop4
            sub = -conf.contribution(i, k) % trunc
            if not sub:
                continue
            c_add_const(qc, [residue[k]], -sub, acc)
            c_add_const(qc, [wrap], trunc, acc[:f])
            tally.additions += 1
            qc.h(wrap)
            qc.measure(wrap, wrap_bit[0])
            tally.x_measurements += 1
            with qc.if_test((wrap_bit[0], 1)):
                qc.h(wrap)                                   # |1> -> |->
                bump = (1 << f) - (trunc - sub)
                c_add_const(qc, [residue[k]], bump, acc)     # carry kicks phase
                c_add_const(qc, [residue[k]], -bump, acc[:f])
                qc.h(wrap)
                qc.x(wrap)
            tally.conditional_additions += 1
        qc.h(residue)                                        # unloop3 cleanup
        qc.measure(residue, residue_bits)
        tally.x_measurements += ell
        for b in range(ell):
            with qc.if_test((residue_bits[b], 1)):
                _phase_lookup(qc, address, [v >> b & 1 for v in table3])
                qc.x(residue[b])
        tally.phaseups += 1
        n = ell                                              # unloop2
        while n < conf.len_dlog:
            threshold = (p - 1) << (n - ell)
            c_add_const(qc, [dlog[n]], -threshold, list(dlog[:n]))
            add_const(qc, threshold, list(dlog[:n + 1]))
            tally.additions += 2
            n += 1
    loop1(len(conf.primes))
    return qc, {"exponent": exponent, "result": result, "dlog": dlog,
                "residue": residue}, tally


def approx_period_circuit(conf):
    """Masked approximate period finding (Sec. 2.2) with Shor-style input."""
    oracle, regs, tally = approx_modexp(conf)
    exponent, result = regs["exponent"], regs["result"]
    approx = ClassicalRegister(conf.kept_bits, "approx")
    frequency = ClassicalRegister(conf.exponent_bits, "frequency")
    qc = QuantumCircuit(*oracle.qregs, *oracle.cregs, approx, frequency)
    qc.h(exponent)
    qc.compose(oracle, inplace=True)
    qc.measure(result[:conf.kept_bits], approx)
    qc.append(qft(conf.exponent_bits).inverse(), exponent)
    qc.measure(exponent, frequency)
    return qc, regs, tally


def ideal_joint_distribution(conf):
    """Exact P(approx=v, frequency=y) of the noiseless pre-measurement state
    Eq. 26 (with `approximate` as f~), by direct Fourier analysis."""
    m, trunc = conf.exponent_bits, conf.truncated_modulus
    size = 1 << m
    mask = 1 << conf.mask_bits
    out = {}
    for v in range(trunc):
        amp = np.array([sum((s + conf.approximate(e)) % trunc == v
                            for s in range(mask)) for e in range(size)],
                       dtype=float) / np.sqrt(size * mask)
        freq = np.fft.fft(amp) / np.sqrt(size)
        for y, a in enumerate(freq):
            if abs(a) ** 2 > 1e-12:
                out[(v, y)] = abs(a) ** 2
    return out


def split_counts(counts, circuit):
    """Aer keys list registers last-added-first; return (fields, count)."""
    names = [creg.name for creg in reversed(circuit.cregs)]
    return [(dict(zip(names, (int(part, 2) for part in key.split()))), value)
            for key, value in counts.items()]
