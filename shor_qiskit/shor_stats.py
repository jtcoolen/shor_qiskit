"""Exact measurement statistics of Shor's circuits, and how to test samples.

A broken Shor circuit does not give a wrong answer; it gives a flat or
distorted histogram, and the classical tail can still verify its way to the
right factors from noise.  So the output distribution itself is what to check.
Both distributions here are computed in closed form from the circuit's
defining state -- no simulation -- which makes them an independent reference
for the full-register and the semiclassical circuits alike.

Order finding.  After the ladder the state is 2^(-t/2) sum_x |x>|A^x mod N>.
The target takes r distinct values, one per residue x0 = x mod r, so

    P(y) = 4^-t  sum_{x0 < r} | sum_{m < M(x0)} exp(-2 pi i m r y / 2^t) |^2,
    M(x0) = ceil((2^t - x0) / r).

ECDLP.  After the oracle the state is 2^-bits sum_{u,v} |u>|v>|S + [u + k v]P>,
and the point register depends on (u, v) only through z = u + k v mod r, so

    P(j1, j2) = q^-4 sum_{z < r} | sum_{u + k v = z} exp(-2 pi i (u j1 + v j2) / q) |^2,
    q = 2^bits.

Samples are compared by total variation distance against the shot-noise null:
the TVD that a *perfect* sampler of the exact distribution would exceed only
with small probability at the same shot count.
"""

from fractions import Fraction

import numpy as np

import ec_classical as C
from shor_essentials import order_from_counts


def multiplicative_order(A, N):
    r, x = 1, A % N
    while x != 1:
        x, r = x * A % N, r + 1
    return r


# =============================================================================
# Exact distributions
# =============================================================================
def order_finding_probs(A, N, t):
    """P(y) for y in [0, 2^t): the exact output of order_circuit(A, N, t)."""
    r, q = multiplicative_order(A, N), 1 << t
    y = np.arange(q, dtype=np.int64)
    probs = np.zeros(q)
    for x0 in range(r):
        m = np.arange((q - x0 + r - 1) // r, dtype=np.int64)
        ph = (np.outer(y, m) * r) % q                    # exact integer phase
        probs += np.abs(np.exp(-2j * np.pi * ph / q).sum(axis=1)) ** 2
    return probs / q**2


def ecdlp_probs(order, k, bits):
    """P[j1, j2]: the exact output of the ECDLP circuit for Q = [k]P."""
    q = 1 << bits
    u = np.arange(q)
    F = np.exp(-2j * np.pi * np.outer(u, u) / q)         # F[u, j]
    z = (u[:, None] + k * u[None, :]) % order            # z[u, v]
    probs = np.zeros((q, q))
    for zz in range(order):
        amp = F.T @ (z == zz).astype(float) @ F          # [j1, j2]
        probs += np.abs(amp) ** 2
    return probs / q**4


# =============================================================================
# Per-shot success: does this one outcome, alone, give the answer?
# =============================================================================
def order_success_mask(A, N, t):
    """mask[y] is True when y alone yields the order via continued fractions."""
    r = multiplicative_order(A, N)
    return np.array([order_from_counts({format(y, f"0{t}b"): 1}, A, N, t) == r
                     for y in range(1 << t)])


def ecdlp_success_mask(order, k, bits):
    """mask[j1, j2] is True when (j1, j2) alone yields k (no search)."""
    q = 1 << bits
    mask = np.zeros((q, q), dtype=bool)
    for j1 in range(q):
        for j2 in range(q):
            c = C.ecdlp_postprocess({(j1, j2): 1}, order, bits, search=0)
            mask[j1, j2] = bool(c) and c[0][0] == k
    return mask


# =============================================================================
# Comparing samples to the exact distribution
# =============================================================================
def empirical(counts, shape):
    """counts: {index: shots}, index an int or a tuple -> normalised array."""
    emp = np.zeros(shape)
    for key, c in counts.items():
        emp[key] += c
    return emp / emp.sum()


def tvd(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def tvd_null(probs, shots, quantile=0.9999, trials=20000, seed=0):
    """The TVD a perfect sampler of `probs` exceeds with prob. 1 - quantile.

    Monte Carlo over multinomial draws, so it makes no large-count assumption
    and stays honest on sparse, many-outcome histograms.  The quantile is
    strict because a suite makes dozens of these comparisons."""
    flat = np.asarray(probs, dtype=float).ravel()
    flat = flat / flat.sum()
    rng = np.random.default_rng(seed)
    d = np.concatenate([
        0.5 * np.abs(rng.multinomial(shots, flat, size=n) / shots - flat).sum(axis=1)
        for n in [1000] * (trials // 1000)])
    return float(np.quantile(d, quantile))


def order_counts(counts):
    """Qiskit bit-string counts of the `out` register -> {y: shots}."""
    return {int(b, 2): c for b, c in counts.items()}


def nearest_fraction(y, t, N):
    return Fraction(y, 2**t).limit_denominator(N - 1)


def ecdlp_probs2(order, k, bits_k, bits_l):
    """P[j1, j2] for registers of different lengths (u: bits_k, v: bits_l).

    The shape of the circuit once trailing windows of the second register are
    dropped ([Litinski23], Babbush et al., IonQ): fewer qubits, a coarser j2.
    Reduces to `ecdlp_probs` when the lengths agree."""
    qa, qb = 1 << bits_k, 1 << bits_l
    u, v = np.arange(qa), np.arange(qb)
    Fa = np.exp(-2j * np.pi * np.outer(u, u) / qa)
    Fb = np.exp(-2j * np.pi * np.outer(v, v) / qb)
    z = (u[:, None] + k * v[None, :]) % order
    probs = np.zeros((qa, qb))
    for zz in range(order):
        amp = Fa.T @ (z == zz).astype(float) @ Fb
        probs += np.abs(amp) ** 2
    return probs / (qa * qb) ** 2


def ecdlp_probs_oracle(f, bits_k, bits_l, phase=None):
    """P[j1, j2] for the circuit that *actually* computes f(u, v).

    `f[u][v]` is whatever the oracle leaves in the accumulator on basis input
    (u, v) -- correct, or garbage on an exceptional input -- as any hashable
    value; `phase[u][v]` its phase (default +1).  The output is exact:
    group the (u, v) by the value they reach, Fourier transform each group,
    sum the squared amplitudes.  This is how an imperfect oracle's success
    probability is computed without trusting any bound."""
    qa, qb = 1 << bits_k, 1 << bits_l
    u, v = np.arange(qa), np.arange(qb)
    Fa = np.exp(-2j * np.pi * np.outer(u, u) / qa)
    Fb = np.exp(-2j * np.pi * np.outer(v, v) / qb)
    groups = {}
    for a in range(qa):
        for b in range(qb):
            groups.setdefault(f[a][b], []).append((a, b))
    probs = np.zeros((qa, qb))
    for members in groups.values():
        M = np.zeros((qa, qb), dtype=complex)
        for a, b in members:
            M[a, b] = 1 if phase is None else phase[a][b]
        amp = Fa.T @ M @ Fb
        probs += np.abs(amp) ** 2
    return probs / (qa * qb) ** 2


def success_probability(P, order, k, bits_k, bits_l, curve=None, Pt=None, Q=None):
    """Probability that one run's (j1, j2) yields k: equal registers use
    `ecdlp_postprocess` (no search), unequal ones the verified search."""
    s = 0.0
    for j1 in range(P.shape[0]):
        for j2 in range(P.shape[1]):
            if P[j1, j2] < 1e-15:
                continue
            if bits_k == bits_l:
                c = C.ecdlp_postprocess({(j1, j2): 1}, order, bits_k, search=0)
            else:
                c = C.ecdlp_postprocess_short({(j1, j2): 1}, order, bits_k, bits_l,
                                              curve, Pt, Q)
            if c and c[0][0] == k:
                s += P[j1, j2]
    return s
