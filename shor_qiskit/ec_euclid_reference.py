"""Exact classical oracles for the packed Euclidean walk of Gu et al. SI II.2.

Binary long division specifies the arithmetic.  ``luo_*`` is a bit-position
model of Luo et al. arXiv:2607.13816 Alg. 1-3 (two opposed-end words, length
and shift counters); it reproduces the paper's Table 4 and is the oracle for
the reversible circuit in ``ec_packed_euclid``.
"""

from math import gcd
from dataclasses import dataclass


@dataclass(frozen=True)
class PackedSnapshot:
    """Two opposed-end words and the live boundaries of an EEA state."""

    work1: int
    work2: int
    t_bits: int
    q_bits: int
    r_bits: int
    t_prev_bits: int
    r_next_bits: int
    width: int


def pack_state(t, quotient, r_prev, t_prev, r, width):
    """Store (t, quotient, r_prev) and (t_prev, r) from opposite ends.

    A spare bit after t separates the growing coefficient from the quotient.
    This models the live boundaries of Luo et al.'s Work1/Work2; it is a
    classical layout specification, not a quantum permutation or gate count.
    """
    values = (t, quotient, r_prev, t_prev, r)
    if width < 1 or any(value < 0 for value in values):
        raise ValueError("packed EEA values must be nonnegative")
    bt, bq, br, btp, brn = (value.bit_length() for value in values)
    if bt + 1 + bq + br > width or btp + brn > width:
        raise ValueError("packed EEA values overlap at this width")
    first = t | (quotient << (bt + 1)) | (r_prev << (width - br))
    second = t_prev | (r << (width - brn))
    return PackedSnapshot(first, second, bt, bq, br, btp, brn, width)


def unpack_state(snapshot):
    """Read back the five integers using the recorded coherent boundaries."""
    s = snapshot
    low = lambda bits: (1 << bits) - 1
    return (s.work1 & low(s.t_bits),
            (s.work1 >> (s.t_bits + 1)) & low(s.q_bits),
            s.work1 >> (s.width - s.r_bits),
            s.work2 & low(s.t_prev_bits),
            s.work2 >> (s.width - s.r_next_bits))


def binary_divmod(numerator, denominator):
    """Binary long division by compare/subtract at each scanned bit."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("expected a nonnegative numerator and positive divisor")
    remainder = quotient = 0
    for bit in range(numerator.bit_length() - 1, -1, -1):
        remainder = 2 * remainder + ((numerator >> bit) & 1)
        if remainder >= denominator:
            remainder -= denominator
            quotient |= 1 << bit
    return quotient, remainder


def inverse(w, p):
    """Compute the signed-coefficient inverse from Eq. (60), without truncation."""
    if p < 3 or p % 2 == 0 or not 0 < w < p or gcd(w, p) != 1:
        raise ValueError("expected 0 < w < odd p with gcd(w, p) = 1")
    prev_r, r = p, w
    prev_t, t = 0, 1
    parity = 0
    while r != 1:
        quotient, next_r = binary_divmod(prev_r, r)
        if next_r == 0:
            raise AssertionError("unit input reached a zero remainder")
        prev_r, r = r, next_r
        prev_t, t = t, prev_t + quotient * t
        parity ^= 1
    return (-t if parity else t) % p


def packed_trace(w, p):
    """Return exact toy packed-boundary snapshots before/after quotient scans."""
    if p < 3 or p % 2 == 0 or not 0 < w < p or gcd(w, p) != 1:
        raise ValueError("expected a unit w modulo odd p")
    width = p.bit_length() + 3
    prev_r, r, prev_t, t = p, w, 0, 1
    trace = [pack_state(t, 0, prev_r, prev_t, r, width)]
    while r != 1:
        q, next_r = binary_divmod(prev_r, r)
        if next_r == 0:
            raise AssertionError("unit input reached zero remainder")
        trace.append(pack_state(t, q, next_r, prev_t, r, width))
        prev_r, r, prev_t, t = r, next_r, t, prev_t + q * t
        if prev_r * t + r * prev_t != p:
            raise AssertionError("packed Euclidean determinant changed")
        trace.append(pack_state(t, 0, prev_r, prev_t, r, width))
    return trace


# --- Luo et al. Algorithms 1 and 3 at physical-position level ---------------
#
# Positions are one-based from the left, as in Luo Table 4.  ``work1`` holds
# t little-endian from position 1, a zero separator, the quotient window and
# r big-endian ending at position W=n+3.  ``work2`` holds t' little-endian
# and r' big-endian, rotated left by ``ls``.  Every update below acts on
# positions selected by the coherent counters only, so it is the reference
# for the gate-level step in ``ec_packed_euclid``.


def luo_step_bound(n):
    """Algorithm 1's fixed schedule 4 ceil(c n), c = 3 / log2(2 + sqrt 3)."""
    from math import ceil, log2, sqrt
    return 4 * ceil(3 / log2(2 + sqrt(3)) * n - 1e-12)


@dataclass
class LuoState:
    work1: list
    work2: list
    lt: int
    lq: int
    lrp: int
    ls: int
    phase1: int
    phase2: int
    iter: int
    sign: int

    def key(self):
        return (tuple(self.work1), tuple(self.work2), self.lt, self.lq,
                self.lrp, self.ls, self.phase1, self.phase2, self.iter,
                self.sign)


def luo_init(x, p):
    """Algorithm 1 preparation: Work1=(1,p), Work2=(0,x or p-x)."""
    n = p.bit_length()
    width = n + 3
    if p < 3 or p % 2 == 0 or not 0 <= x < p:
        raise ValueError("expected 0 <= x < odd p")
    flip = int(x > p // 2)
    xr = p - x if flip else x
    w1 = [0] * width
    w2 = [0] * width
    w1[0] = 1
    for k in range(n):
        w1[width - 1 - k] = (p >> k) & 1
        w2[width - 1 - k] = (xr >> k) & 1
    return LuoState(w1, w2, 1, 0, xr.bit_length(), 0, 0, 0, flip, 0)


def _rotate_left(word, k=1):
    k %= len(word)
    return word[k:] + word[:k]


def _interval_addsub(target, addend, lo, hi, subtract):
    """Big-endian (target[lo..hi]) +/- addend; return the carry/borrow bit."""
    if lo > hi + 1:
        raise AssertionError("location-controlled interval is inverted")
    if lo > hi:
        return 0
    size = hi - lo + 1
    a = int("".join(map(str, target[lo - 1:hi])), 2)
    b = int("".join(map(str, addend[lo - 1:hi])), 2)
    value = a - b if subtract else a + b
    out = int(value < 0 or value >= 1 << size)
    value %= 1 << size
    target[lo - 1:hi] = [(value >> (size - 1 - k)) & 1 for k in range(size)]
    return out


def _prefix_addsub(target, addend, hi, subtract):
    """Little-endian (target[1..hi]) +/- addend; return carry/borrow."""
    a = sum(target[k] << k for k in range(hi))
    b = sum(addend[k] << k for k in range(hi))
    value = a - b if subtract else a + b
    out = int(value < 0 or value >= 1 << hi)
    value %= 1 << hi
    target[:hi] = [(value >> k) & 1 for k in range(hi)]
    return out


def _highest_one(word, bound):
    return max((j for j in range(1, bound + 1) if word[j - 1]), default=0)


def _right_length(word, start):
    first = next((j for j in range(start, len(word) + 1) if word[j - 1]), None)
    return 0 if first is None else len(word) + 1 - first


def luo_step(s):
    """One Algorithm-3 step (Luo's optimized ordering), in place.

    After termination (``lrp = 0``) only the shift counter advances: Luo's
    reference circuit rotates Work2 during this padding instead.  Advancing
    ``ls`` alone keeps the terminal state distinct from the step that
    produced it, so the fixed schedule stays a permutation without moving t'.
    """
    width = len(s.work1)
    live = s.lrp != 0
    if not s.phase1:
        if live:
            s.work2 = _rotate_left(s.work2, 1)
        s.ls += 1
        if s.phase2 and live:
            s.work2 = _rotate_left(s.work2, -2)
            s.ls -= 2
    lo, hi = s.lt + s.lq + 2, width - s.ls
    if not s.phase1 and live:
        s.sign ^= _interval_addsub(s.work1, s.work2, lo, hi, True)
    if not s.phase1 and s.phase2:
        s.sign ^= 1
    if not s.phase1 and not (s.phase2 and s.sign) and live:
        _interval_addsub(s.work1, s.work2, lo, hi, False)
    if not s.phase1 and s.phase2:
        s.lq += 1
    if s.phase1 ^ s.phase2:
        j = s.lt + s.lq + 1
        s.sign, s.work1[j - 1] = s.work1[j - 1], s.sign
    if s.phase1 and not s.phase2:
        s.lq -= 1
    # Phase 3 adds 2^ls t on the left-most lt+1 lanes.  Phase 4 compares on
    # the left-most n+3-lr'-ls lanes (Luo Sec. 3.2); there Work1 holds only t
    # and zeros, while lt+1+Phase2 lanes can miss high bits of t'.
    hi = width - s.lrp - s.ls if s.phase2 else s.lt + 1
    if s.phase1 and not (not s.phase2 and s.sign):
        _prefix_addsub(s.work2, s.work1, hi, True)
    if s.phase1:
        s.sign ^= 1
        s.sign ^= _prefix_addsub(s.work2, s.work1, hi, False)
        s.work2 = _rotate_left(s.work2, 1)
        s.ls += 1
        if s.phase2:
            s.work2 = _rotate_left(s.work2, -2)
            s.ls -= 2
    if s.lq == 0 and s.lrp != 0:
        s.phase2 ^= s.sign ^ s.phase1
        s.sign ^= s.phase2
    if s.ls == 0:
        s.phase1 ^= 1
        s.phase2 ^= 1
    if s.lq == 0 and s.ls == 0:
        s.work1, s.work2 = s.work2, s.work1
        bound = width - s.lrp
        s.lt ^= _highest_one(s.work2, bound) ^ _highest_one(s.work1, bound)
        start = s.lt + 2
        s.lrp ^= (_right_length(s.work1, start)
                  ^ _right_length(s.work2, start))
        s.iter ^= 1
    if min(s.lt, s.lq, s.lrp, s.ls) < 0 or s.lt + s.lq + 1 > width:
        raise AssertionError("Luo counters left their physical range")
    return s


def luo_decode(s):
    """Read (t, q, r, t', r') from the packed words and counters."""
    width = len(s.work1)
    t = sum(s.work1[k] << k for k in range(s.lt))
    q = sum(s.work1[s.lt + 1 + k] << (s.ls + s.lq - 1 - k) for k in range(s.lq))
    r = int("".join(map(str, s.work1[s.lt + s.lq + 1:])) or "0", 2)
    w2 = s.work2 if s.lrp == 0 else _rotate_left(s.work2, -s.ls)
    tp = sum(w2[k] << k for k in range(width - s.lrp))
    rp = int("".join(map(str, w2[width - s.lrp:])) or "0", 2)
    return t, q, r, tp, rp


def luo_run(x, p, steps=None):
    """Algorithm 1: prepare, run the fixed step schedule, return the trace."""
    s = luo_init(x, p)
    steps = luo_step_bound(p.bit_length()) if steps is None else steps
    trace = [LuoState(**{**s.__dict__, "work1": list(s.work1),
                         "work2": list(s.work2)})]
    for _ in range(steps):
        luo_step(s)
        trace.append(LuoState(**{**s.__dict__, "work1": list(s.work1),
                                 "work2": list(s.work2)}))
    return trace


def luo_output(s, p):
    """Algorithm 1 read-out: x^-1 = t' if Iter else p - t' (0 maps to 0)."""
    tp = sum(s.work2[k] << k for k in range(p.bit_length()))
    return tp if s.iter or tp == 0 else p - tp
