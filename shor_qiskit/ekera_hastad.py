"""Factoring by a short discrete logarithm (Ekera-Hastad), at toy size.

Textbook Shor finds the order r of g, with a 2n-qubit exponent.  Ekera and
Hastad instead extract a *short* logarithm directly: for N = pq and g of order
dividing phi(N)/2,

    g^((N-1)/2)  =  g^((p+q-2)/2),

so A = g^((N-1)/2 - 2^(n/2 - 1)) = g^D with D = (p-1)/2 + (q-1)/2 - 2^(n/2-1)
-- CFS 2024's centred choice -- which is at most ~2^(n/2) in size.  Knowing D
gives p + q, and with N that gives p and q.  A short logarithm of m bits needs
an exponent of only m + l (for g) plus l (for A) qubits, l = ceil(m / s): with
s = 1, 1.5n qubits instead of 2n, one run; with larger s, n/2 + n/s qubits per
run and s + 1 runs.  That is the formulation of Gidney-Ekera 2019, CFS 2024
and Gidney 2025, and what makes residue arithmetic worthwhile at all.

The circuit computes g^x A^(-y) = g^(x - D y) and Fourier transforms x and y
separately -- the same shape as the ECDLP circuit, with z = x - D y.  Its
exact output is `shor_stats.ecdlp_probs2(r, -D mod r, lx, ly)`.

Post-processing (`eh_postprocess`) never uses r, which the factoring setting
does not know.  A pair (j, k) is good when {d j + 2^m k} mod 2^(m + l),
centred, is small; the short d that makes it smallest is the candidate.
At toy size d is found by enumeration over 0 < d <= 2^m, shortest first among
equal scores; at scale it is the closest-vector problem of Ekera-Hastad's
lattice, which enumeration stands in for here.

Variants: "cfs" (the centred A above); "ge" (Gidney-Ekera: A = g^(N+1), d = p + q)
and "g25" (Gidney 2025: A = g^(N-1)... d = p + q - 2).  At toy N the last two
make d exceed ord(g), so the quantum part only pins d modulo r and the
post-processing succeeds only when the tie-break happens to land on it; the
tests measure that and show the centred D doing at least as well.
"""

import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from rc_adder import rc_c_ua
from semiclassical import semiclassical_iqft
from shor_essentials import qft


class EHInstance:
    def __init__(self, N, g, variant="cfs", s=1):
        self.N, self.g, self.variant, self.s = N, g, variant, s
        self.n = N.bit_length()
        self.m = math.ceil(self.n / 2)
        self.l = math.ceil(self.m / s)
        self.lx, self.ly = self.m + self.l, self.l
        if variant == "cfs":
            self.e = (N - 1) // 2 - (1 << (self.n // 2 - 1))
        elif variant == "ge":
            self.e = N + 1
        elif variant == "g25":
            self.e = N - 1
        else:
            raise ValueError(variant)
        self.A = pow(g, self.e, N)
        self.Ainv = pow(self.A, -1, N)

    def true_d(self, p, q):
        """The logarithm the variant targets (not reduced mod ord g)."""
        return {"cfs": (p - 1) // 2 + (q - 1) // 2 - (1 << (self.n // 2 - 1)),
                "ge": p + q, "g25": p + q - 2}[self.variant]


def _registers(inst):
    n = inst.n
    tgt, anc = QuantumRegister(n, "tgt"), QuantumRegister(n, "y")
    sf, rca = QuantumRegister(2, "sf"), QuantumRegister(n + 2, "rca")
    return tgt, anc, sf, rca


def _rung(inst, base, k, tgt, anc, sf, rca):
    """Controlled multiplication by base^(2^k)."""
    mult = pow(base, 1 << k, inst.N)
    return lambda qc, c: rc_c_ua(qc, c, mult, list(tgt), list(anc), sf[0], sf[1],
                                 inst.N, list(rca))


def eh_circuit(inst):
    """Two exponent registers (lx for g, ly for A^-1), each Fourier
    transformed and measured into ox, oy."""
    tgt, anc, sf, rca = _registers(inst)
    xr, yr = QuantumRegister(inst.lx, "x"), QuantumRegister(inst.ly, "yx")
    ox, oy = ClassicalRegister(inst.lx, "ox"), ClassicalRegister(inst.ly, "oy")
    qc = QuantumCircuit(xr, yr, tgt, anc, sf, rca, ox, oy)
    qc.x(tgt[0])
    qc.h(xr)
    qc.h(yr)
    for k in range(inst.lx):
        _rung(inst, inst.g, k, tgt, anc, sf, rca)(qc, xr[k])
    for k in range(inst.ly):
        _rung(inst, inst.Ainv, k, tgt, anc, sf, rca)(qc, yr[k])
    qc.append(qft(inst.lx).inverse(), list(xr))
    qc.append(qft(inst.ly).inverse(), list(yr))
    qc.measure(xr, ox)
    qc.measure(yr, oy)
    return qc


def eh_circuit_1c(inst):
    """The same on ONE recycled control qubit (semiclassical, twice)."""
    tgt, anc, sf, rca = _registers(inst)
    ctr = QuantumRegister(1, "ctr")
    ox, oy = ClassicalRegister(inst.lx, "ox"), ClassicalRegister(inst.ly, "oy")
    qc = QuantumCircuit(ctr, tgt, anc, sf, rca, ox, oy)
    qc.x(tgt[0])
    semiclassical_iqft(qc, ctr[0], ox,
                       [_rung(inst, inst.g, k, tgt, anc, sf, rca) for k in range(inst.lx)])
    semiclassical_iqft(qc, ctr[0], oy,
                       [_rung(inst, inst.Ainv, k, tgt, anc, sf, rca) for k in range(inst.ly)])
    return qc


def _centred(v, mod):
    v %= mod
    return v - mod if v >= mod // 2 else v


def score(inst, d, j, k):
    """|{d j + 2^m k}_(2^(m+l))|: small for a good pair and the right d."""
    return abs(_centred(d * j + (k << inst.m), 1 << (inst.m + inst.l)))


def eh_postprocess(samples, inst, span=None):
    """Rank candidate short logarithms d by how well they explain the pairs.

    `samples`: [(j, k)] (or {(j, k): shots}).  With s = 1 one good pair
    suffices; with larger s, s + 1 pairs are combined.  Returns [(d, total
    score)] best first.  ord(g) is not used.
    """
    if isinstance(samples, dict):
        items = list(samples.items())
    else:
        items = [(p, 1) for p in samples]
    span = span or (1 << inst.m)
    tot = {}
    for (j, k), c in items:
        for d in range(1, span + 1):                  # 0 < d <= 2^m, as in [EH17]
            tot[d] = tot.get(d, 0) + c * score(inst, d, j, k)
    # one pair fixes d only modulo 2^(m+l) / gcd(j, 2^(m+l)); among the
    # candidates that explain it equally well, the *shortest* is the lattice
    # answer -- hence the tie-break on d
    return sorted(tot.items(), key=lambda kv: (kv[1], kv[0]))


def factors_from_d(inst, d):
    """p, q from the short logarithm, or None."""
    N, n = inst.N, inst.n
    ssum = {"cfs": 2 * (d + 1 + (1 << (n // 2 - 1))), "ge": d, "g25": d + 2}[inst.variant]
    disc = ssum * ssum - 4 * N
    if disc < 0:
        return None
    rt = math.isqrt(disc)
    if rt * rt != disc:
        return None
    p, q = (ssum - rt) // 2, (ssum + rt) // 2
    return (p, q) if p * q == N and 1 < p < N else None
