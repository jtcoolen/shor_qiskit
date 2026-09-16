"""Classical reference models for every quantum circuit in this package.

Each function here is the *specification* of a circuit elsewhere: the tests
build the circuit, push basis states through `ec_sim.simulate`, and compare
against the function of the same name here.  Where a paper gives pseudocode,
this file transcribes it literally -- including the parts that look redundant
classically but exist to make the quantum version reversible.

Sources
    [RNSL17] Roetteler, Naehrig, Svore, Lauter, Asiacrypt 2017
    [HJN+20] Haener, Jaques, Naehrig, Roetteler, Soeken, PQCrypto 2020
    [106]    Kim et al., eprint 2026/106
    [1128]   Schrottenloher, eprint 2026/1128
    [Eke19]  Ekera, Revisiting Shor's algorithm for discrete logarithms
"""

from math import gcd


# =============================================================================
# Elliptic curves over GF(p), short Weierstrass form
# =============================================================================
class Point:
    """An affine point, or the point at infinity (`inf=True`)."""

    __slots__ = ("x", "y", "inf")

    def __init__(self, x=0, y=0, inf=False):
        self.x, self.y, self.inf = x, y, inf

    def __eq__(self, o):
        if self.inf or o.inf:
            return self.inf and o.inf
        return self.x == o.x and self.y == o.y

    def __hash__(self):
        return hash((self.inf, self.x, self.y))

    def __repr__(self):
        return "O" if self.inf else f"({self.x},{self.y})"


O = Point(inf=True)


class Curve:
    """y^2 = x^3 + a x + b over GF(p), p prime > 3."""

    def __init__(self, p, a, b, name=""):
        assert (4 * a**3 + 27 * b**2) % p != 0, "singular curve"
        self.p, self.a, self.b, self.name = p, a % p, b % p, name

    def on_curve(self, P):
        if P.inf:
            return True
        return (P.y * P.y - P.x**3 - self.a * P.x - self.b) % self.p == 0

    def neg(self, P):
        return O if P.inf else Point(P.x, (-P.y) % self.p)

    def add(self, P, Q):
        p = self.p
        if P.inf:
            return Point(Q.x, Q.y, Q.inf)
        if Q.inf:
            return Point(P.x, P.y, P.inf)
        if P.x == Q.x:
            if (P.y + Q.y) % p == 0:
                return O
            lam = (3 * P.x * P.x + self.a) * pow(2 * P.y, -1, p) % p   # doubling
        else:
            lam = (Q.y - P.y) * pow(Q.x - P.x, -1, p) % p
        x3 = (lam * lam - P.x - Q.x) % p
        return Point(x3, (lam * (P.x - x3) - P.y) % p)

    def mul(self, k, P):
        R, Q, k = O, Point(P.x, P.y, P.inf), k % self.order_hint() if False else k
        while k > 0:
            if k & 1:
                R = self.add(R, Q)
            Q = self.add(Q, Q)
            k >>= 1
        return R

    def order_hint(self):
        return None

    def points(self):
        """Every point on the curve. Only for toy p -- O(p^2) work."""
        pts = [O]
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            for y in range(self.p):
                if y * y % self.p == rhs:
                    pts.append(Point(x, y))
        return pts

    def point_order(self, P):
        n, R = 1, Point(P.x, P.y, P.inf)
        while not R.inf:
            R, n = self.add(R, P), n + 1
        return n


# --- toy instances used by the tests ----------------------------------------
# The Classiq tutorial's curve: y^2 = x^3 + 5x + 4 mod 7, G = (0,5) of order 5.
CLASSIQ = Curve(7, 5, 4, "classiq-p7")
CLASSIQ_G = Point(0, 5)
CLASSIQ_Q = Point(0, 2)          # = [4]G, so the discrete log is 4

TOY11 = Curve(11, 1, 6, "toy-p11")          # (2,7) has order 13
TOY11_G = Point(2, 7)


# =============================================================================
# In-place affine point addition, the reversible form  ([RNSL17] Alg 1, [106] Alg 3)
# =============================================================================
def point_add_ctrl_ref(curve, ctrl, x1, y1, x2, y2):
    """The exact map the in-place circuit implements.

    (x1,y1) is the quantum accumulator, (x2,y2) a *classical* constant point.
    When ctrl=0 the accumulator is untouched; when ctrl=1 it becomes the sum.
    Both papers assume the exceptional cases (x1==x2, either point at infinity)
    never arise -- for a random accumulator they have probability O(1/p), and
    the algorithm tolerates that failure rate.
    """
    if not ctrl:
        return x1, y1
    p = curve.p
    assert x1 != x2, "exceptional case: circuit is only correct for x1 != x2"
    lam = (y1 - y2) * pow(x1 - x2, -1, p) % p
    x3 = (lam * lam - x1 - 2 * x2) % p
    y3 = (lam * (x2 - x3) - y2) % p
    return x3, y3


def point_add_exceptional(curve, P1, P2):
    """Is (P1, P2) outside the domain the reversible addition circuit covers?

    There are two divisions in the circuit and each one excludes a case:

      x1 == x2   step 3 divides by x1 - x2.  This is the case both papers name:
                 doubling, adding an inverse, or a point at infinity.
      x3 == x2   step 8 divides by x2 - x3, where x3 is the answer.  x3 == x2
                 means P1 + P2 = -P2, i.e. P1 = -2 P2.  This one is easy to
                 miss -- it is invisible in the formulas and only appears once
                 you follow what the *registers* hold -- but it is inherent to
                 the construction, not to this implementation: [1128]'s variant
                 reaches the same place, where its in-place multiplier would be
                 asked to multiply by zero.

    Both are O(1/p) for a random accumulator, which is what makes the circuit
    usable in Shor's algorithm.  On a toy curve they are a large fraction of the
    inputs, so the tests exclude them explicitly and report how many.
    """
    if P1.inf or P2.inf:
        return True
    if P1.x == P2.x:
        return True
    P3 = curve.add(P1, P2)
    return P3.inf or P3.x == P2.x


def point_add_steps(curve, ctrl, x1, y1, x2, y2):
    """[106] Algorithm 3, step by step, as the circuit runs it.

    Returned as a dict so a test can check each register at each barrier, which
    is how circuit bugs get localised: a wrong final answer says nothing about
    *which* of the eleven steps went wrong.
    """
    p, q = curve.p, ctrl
    s = {}
    x = (x1 - x2) % p;                      s["1_sub_x"] = x
    y = (y1 - q * y2) % p;                  s["2_csub_y"] = y
    lam = (q * y * pow(x, -1, p)) % p;      s["3_cdiv"] = lam
    y = y ^ (x * lam % p);                  s["4_mul_xor"] = y      # y becomes 0
    x = (x + 3 * x2) % p;                   s["5_add_3x2"] = x
    x = (x - lam * lam) % p;                s["6_sub_lamsq"] = x
    y = (y + x * lam) % p;                  s["7_mul_acc"] = y
    lam = lam ^ (q * y * pow(x, -1, p) % p if x % p else 0)
    s["8_cdiv_clear"] = lam                                         # lam becomes 0
    x = (x - (q * x2 + (1 - q) * 2 * x2)) % p; s["9_csub"] = x
    y = (y - q * y2) % p;                   s["10_csub_y"] = y
    x = (-x) % p if q else x;               s["11_cneg"] = x
    s["out"] = (x, y)
    return s


# =============================================================================
# Kaliski's algorithm  ([106] Algorithm 2, the [HJN+20] swap-based formulation)
# =============================================================================
def kaliski_round(u, v, r, s, p, reduce_mod=True, conditional=False):
    """One round of [106] Algorithm 2. Returns (u, v, r, s, bswap, active).

    `active` is the round's contribution to the counter k.

    `conditional` is the difference between the two variants, and it is the
    whole of [106] Sec 3.3.  With conditional=True ([HJN+20]) a round after
    v has hit 0 does nothing at all, so r stops at -x^{-1} 2^k and a counter is
    needed to undo the 2^k.  With conditional=False ([106]) the round still
    runs -- everything is inert except the doubling of r -- so r keeps doubling
    to a fixed -x^{-1} 2^{rounds}, independent of k, and the counter and its
    correction disappear.
    """
    active = v != 0
    if conditional and not active:
        return u, v, r, s, False, False
    bswap = False
    if (u % 2 == 0 and v % 2 == 1) or (u % 2 == 1 and v % 2 == 1 and u > v):
        u, v, r, s, bswap = v, u, s, r, True
    if u % 2 == 1 and v % 2 == 1:
        v -= u
        s = r + s
    v //= 2
    r = 2 * r
    if reduce_mod:
        r %= p
        s %= p
    if bswap:
        u, v, r, s = v, u, s, r
    return u, v, r, s, bswap, active


def kaliski(x, p, rounds=None, reduce_mod=True, conditional=False):
    """Run Kaliski to completion (or for a fixed number of rounds).

    Returns (r, k, u, v, s) where k counts the *active* rounds.

    The invariant, verified in the tests: on termination
        r == (-x^{-1} * 2^k) mod p        and   u == 1,  s == p.
    So r is a *pseudo*-inverse: it is off by a power of two and a sign, both of
    which the callers fix up.  With `rounds` fixed at 2n the extra inert rounds
    just keep doubling r, giving r == (-x^{-1} * 2^(2n)) mod p regardless of k --
    which is [106]'s reason for dropping the v!=0 test entirely.
    """
    u, v, r, s, k = p, x % p, 0, 1, 0
    i = 0
    while (v != 0 if rounds is None else i < rounds):
        u, v, r, s, _, active = kaliski_round(
            u, v, r, s, p, reduce_mod, conditional)
        k += int(active)
        i += 1
    return r, k, u, v, s


def mod_inverse_kaliski(x, p, n=None):
    """x^{-1} mod p via Kaliski, with the [HJN+20] correction applied.

    Kaliski leaves -x^{-1} 2^k; the circuit corrects by negating and halving
    k times (equivalently doubling n-k times when working in Montgomery form).
    """
    r, k, u, v, s = kaliski(x, p)
    assert u == 1 and v == 0
    inv = (-r) % p                          # kill the sign
    inv = inv * pow(2, -k, p) % p           # kill the 2^k
    assert inv * x % p == 1
    return inv


def mod_inverse_kaliski_unconditional(x, p, n):
    """[106] Sec 3.3: 2n unconditional rounds, no counter, no v!=0 test.

    The result is -x^{-1} 2^{2n} mod p.  If x arrived in Montgomery form
    (x = X 2^n) this is -X^{-1} 2^n, i.e. the Montgomery form of -X^{-1}:
    the correction that [HJN+20] pays a counter and 2n-k doublings for is
    already done, for free, by the rounds that would otherwise be skipped.
    """
    r, k, u, v, s = kaliski(x, p, rounds=2 * n)
    assert u == 1 and v == 0, "2n rounds did not suffice"
    return r, k


# =============================================================================
# Montgomery arithmetic  ([106] Algorithm 1)
# =============================================================================
def mont_mul_wordlevel(a, b, p, w, n):
    """[106] Algorithm 1(b): word-level Montgomery with the precomputed d.

    Algorithm 1(a) computes M = c_w p'_w mod 2^w then c += M p.  Substituting M
    gives c += c_w (p'_w mod 2^w) p, and the bracket depends only on the
    modulus -- so define d = (p'_w mod 2^w) p once, classically, and the
    reduction becomes a second `c += c_w * d`, structurally identical to the
    multiplication step.  That is what lets one carry-save tree serve both.
    """
    assert n % w == 0, "n must be a whole number of words"
    s = n // w
    mask = (1 << w) - 1
    pw_prime = (-pow(p, -1, 1 << w)) % (1 << w)      # p' = -p^{-1} mod 2^w
    d = (pw_prime & mask) * p
    c = 0
    for i in range(s):
        ai = (a >> (i * w)) & mask
        c += ai * b
        c += (c & mask) * d
        assert c % (1 << w) == 0, "Montgomery reduction did not clear the low word"
        c >>= w
    return c % p if c < 2 * p else c % p


def mont_d_constant(p, w):
    """The classically precomputed d(x) = (p'_w mod 2^w) p of [106] Alg 1(b)."""
    pw_prime = (-pow(p, -1, 1 << w)) % (1 << w)
    return pw_prime * p


def to_mont(x, p, n):
    return x * pow(2, n, p) % p


def from_mont(x, p, n):
    return x * pow(2, -n, p) % p


# =============================================================================
# [1128] Sec 3: the Euclidean algorithm as a bit-vector, and Bezout replay
# =============================================================================
def eea_dialog(u, v, iters):
    """[1128] Algorithm 2. Returns (bits, u, v) where bits is [(b0, b0b1), ...].

    This is the *first* key idea: run the binary GCD but do not maintain the
    Bezout coefficients.  Just record, per iteration, the two decision bits.
    That record is enough to replay the coefficient updates later, and it is
    far smaller than carrying (r,s) along -- 1.5 bits per iteration on average
    instead of 2n qubits.
    """
    assert u % 2 == 1, "u must be odd (it is the prime q)"
    bits = []
    for _ in range(iters):
        b0 = v & 1
        b1 = int(u > v)
        b0b1 = b0 & b1
        bits.append((b0, b0b1))
        if b0b1:
            u, v = v, u
        if b0:
            v -= u
        v >>= 1
    return bits, u, v


def eea_iterations(n, c_iter=2.4):
    """[1128]: 1.413n + c_iter*sqrt(n) iterations, c_iter ~ 2.4 (four sigma).

    Each iteration removes one bit for certain and a second with probability
    1/2, so (uv) shrinks by 3/8 per step and 2n / log2(8/3) ~ 1.413n steps
    suffice on average, with standard deviation ~0.6 sqrt(n).
    """
    return int(1.413 * n + c_iter * n**0.5) + 1


def bezout_replay(bits, r, s, q):
    """[1128] Algorithm 3, reading the dialog in reverse.

    The updates to (r,s) are linear and controlled only by (b0, b0&b1), so they
    can be replayed on *any* starting pair.  Started at (y, 0) this yields
    (0, y*x mod q) -- inversion and in-place multiplication in one pass, which
    is the second key idea and the reason no separate multiplier is needed.
    """
    for b0, b0b1 in reversed(bits):
        s = 2 * s % q
        if b0:
            s = (s + r) % q
        if b0b1:
            r, s = s, r
    return r, s


def bezout_replay_inv(bits, r, s, q):
    """The inverse of `bezout_replay`, step for step -- i.e. the circuit run
    backwards.  Started at (0, z) it returns (z * x^{-1} mod q, 0).

    This is [1128]'s remark that "the inverse circuit will multiply by x^{-1}
    instead", and it is why one circuit covers both in-place multiplications in
    the point addition.  Note the direction matters: Algorithm 3 replayed in the
    dialog's *original* order is a different linear map, not this one.
    """
    for b0, b0b1 in bits:
        if b0b1:
            r, s = s, r
        if b0:
            s = (s - r) % q
        s = s * pow(2, -1, q) % q
    return r, s


def bezout_inverse_classical(bits, q):
    """x^{-1} mod q read straight off the dialog, tracking one coefficient.

    Kept as an independent cross-check on `bezout_replay`: it maintains
    u == ru*x, v == rv*x (mod q) alongside the dialog and ends with ru = x^{-1}.
    The circuit does not use this form -- it needs modular halving, whereas the
    reversed replay needs only doubling and addition.
    """
    ru, rv, half = 0, 1, pow(2, -1, q)
    for b0, b0b1 in bits:
        if b0b1:
            ru, rv = rv, ru
        if b0:
            rv = (rv - ru) % q
        rv = rv * half % q
    return ru


def inplace_mul_ref(x, y, q, iters=None):
    """[1128] Algorithm 4: |x,y> -> |x, y*x mod q>, via dialog then replay."""
    n = q.bit_length()
    iters = iters or eea_iterations(n)
    bits, u, v = eea_dialog(q, x % q, iters)
    assert v == 0 and u == 1, "dialog did not converge -- increase iters"
    r, s = bezout_replay(bits, y % q, 0, q)
    assert r == 0, "replay left r != 0"
    return s


def inplace_div_ref(x, z, q, iters=None):
    """|x,z> -> |x, z * x^{-1} mod q>: `inplace_mul_ref` run backwards."""
    n = q.bit_length()
    iters = iters or eea_iterations(n)
    bits, u, v = eea_dialog(q, x % q, iters)
    assert v == 0 and u == 1, "dialog did not converge -- increase iters"
    r, s = bezout_replay_inv(bits, 0, z % q, q)
    assert s == 0, "inverse replay left s != 0"
    return r


def compress_triple(pairs):
    """[1128] Fig. 1: three (b0, b0&b1) pairs -> 5 bits.

    Each pair is one of (0,0), (1,0), (1,1) -- never (0,1), since b0&b1 can
    only be set when b0 is.  So three pairs carry 3^3 = 27 < 32 states and fit
    in 5 bits instead of 6, releasing one qubit per three iterations.  Over
    1.413n iterations that is the difference between 2.83n and 2.355n garbage
    qubits.
    """
    assert len(pairs) == 3
    v = 0
    for i, (b0, b0b1) in enumerate(pairs):
        assert (b0, b0b1) in ((0, 0), (1, 0), (1, 1)), "impossible pair"
        digit = 0 if b0 == 0 else (1 if b0b1 == 0 else 2)
        v += digit * 3**i
    return v


def decompress_triple(v):
    out = []
    for _ in range(3):
        d, v = v % 3, v // 3
        out.append((0, 0) if d == 0 else ((1, 0) if d == 1 else (1, 1)))
    return out


# =============================================================================
# [106] Sec 4.2: out-of-place point addition in Jacobian-affine coordinates
# =============================================================================
def jacobian_add_ref(curve, X1, Y1, Z1, x2, y2):
    """[106] Algorithm 4. (X1:Y1:Z1) + (x2,y2,1) -> (X3:Y3:Z3), 11 multiplies.

    Mixed Jacobian-affine: the constant point has Z2=1, which kills every term
    involving Z2 in the general addition law.  No inversion is needed, which is
    the whole point -- but the representation is not unique, so the input
    survives as garbage.  That is what the zig-zag schedule and windowing exist
    to pay for.
    """
    p = curve.p
    Z1_2 = Z1 * Z1 % p                       # 1
    U2 = x2 * Z1_2 % p                       # 2
    Z1_3 = Z1_2 * Z1 % p                     # 3
    H = (U2 - X1) % p                        # 4
    H2 = H * H % p                           # 5
    H3 = H2 * H % p                          # 6
    S2 = y2 * Z1_3 % p                       # 7
    R = (S2 - Y1) % p                        # 8
    R2 = R * R % p                           # 9
    V = X1 * H2 % p                          # 10
    twoV = 2 * V % p                         # 11 (an addition, not a multiply)
    T0 = (R2 - H3) % p                       # 12
    X3 = (T0 - twoV) % p                     # 13
    T1 = (V - X3) % p                        # 14
    T2 = R * T1 % p                          # 15
    T3 = Y1 * H3 % p                         # 16
    Y3 = (T2 - T3) % p                       # 17
    Z3 = Z1 * H % p                          # 18
    return X3, Y3, Z3


def jacobian_to_affine(curve, X, Y, Z):
    p = curve.p
    if Z % p == 0:
        return O
    zi = pow(Z, -1, p)
    return Point(X * zi * zi % p, Y * zi * zi * zi % p)


def affine_to_jacobian(P):
    return (P.x, P.y, 1)


# =============================================================================
# Windowing and the zig-zag schedule
# =============================================================================
def window_table(curve, base, w, offset=None):
    """The 2^w points {offset + [i]base}.

    [106] Sec 4.2 and [1128] Sec 2 both add a fixed offset T so that i=0 does
    not produce the point at infinity, which the addition formulas cannot
    represent.  Pass offset=None to keep the plain multiples (and accept O at
    index 0, which the addition circuit then has to special-case).
    """
    pts, acc = [], (O if offset is None else offset)
    for _ in range(1 << w):
        pts.append(Point(acc.x, acc.y, acc.inf))
        acc = curve.add(acc, base)
    return pts


def n_point_additions(n, w):
    """[106] Sec 4.2: 2*ceil((n+1)/w) windowed additions instead of 2n+2."""
    return 2 * -(-(n + 1) // w)


def zigzag_registers(n_adds):
    """[106] Fig. 10: smallest m with m(m+1)/2 >= n_adds.

    Point additions in projective coordinates leave their input behind as
    garbage.  Arranging the additions as m + (m-1) + ... + 1 lets each output
    register be cleared by the reverse of the addition that made it, so m
    registers serve n_adds additions instead of n_adds registers.
    """
    m = 1
    while m * (m + 1) // 2 < n_adds:
        m += 1
    return m


def zigzag_schedule(n_adds, m=None):
    """The tape of [106] Fig. 10: runs of m, m-1, ..., 1 forward additions.

    The projective addition is out-of-place and preserves its input, so:
      * computing state j+1 needs state j live and one free register;
      * un-computing state j (the addition run backwards) needs state j-1 live,
        and frees j's register.
    State 0 is the input; it is never freed and does not count towards m.

    The schedule: fill all m registers with a run of m additions, then walk
    back un-computing all but the last state of the run -- that is legal
    because each state's predecessor is still live, and it returns m-1
    registers.  Repeat with runs of m-1, m-2, ..., 1.  Total forward additions
    m + (m-1) + ... + 1 = m(m+1)/2, which is the paper's 10 = 4+3+2+1.

    Returns (tape, m, residual):
      tape     [("add"|"unadd", state index)] in execution order
      residual the intermediate states still live at the end -- exactly the
               last state of each run.  There are m of them, which is the
               paper's "garbage qubits are required for only m of them".

    NOTE.  `residual` is garbage, not zero.  For Shor those registers must not
    stay entangled with the control register, so `ec_proj` uses this tape for
    its register *budget* and then unwinds the whole thing (compute, copy the
    final affine point out, uncompute).  `ec_cost` reports the paper's mG
    figure, which is the budget.
    """
    if m is None:
        m = zigzag_registers(n_adds)
    tape, live, cur, done = [], {0}, 0, 0
    run = m
    while done < n_adds and run > 0:
        this = min(run, n_adds - done)
        start_state = cur
        for _ in range(this):                       # forward run
            cur += 1
            tape.append(("add", cur))
            live.add(cur)
            done += 1
        for j in range(cur - 1, start_state, -1):   # walk back, keep the last
            tape.append(("unadd", j))
            live.discard(j)
        run -= 1
    assert done == n_adds, f"zigzag with m={m} only reached {done}/{n_adds}"
    return tape, m, live - {0, cur}


# =============================================================================
# [Eke19] classical post-processing for the elliptic-curve discrete log
# =============================================================================
def ecdlp_postprocess(counts, order, bits, search=1):
    """Recover k from measured (j1, j2) pairs of the two ECDLP registers.

    `counts` maps (j1, j2) -> number of shots.  Returns [(k, weight)] sorted by
    weight, heaviest first.

    Where the relation comes from.  The circuit computes S + [u]P + [v]Q with
    Q = [k]P, so the third register depends on (u, v) only through u + k v mod r.
    The level sets of that are translates of the lattice generated by (-k, 1)
    and (r, 0); the QFT concentrates on the dual, so a measured (j1, j2) obeys

        j2 * r / q  ==  k * (j1 * r / q)   (mod r),      q = 2^bits

    i.e. with a_i = round(j_i r / q),   a2 == k a1 (mod r),  so k = a2 / a1.
    a1 must be invertible mod r, which discards j1 = 0 automatically.

    The rounding is exact only when q is a multiple of r.  Otherwise the peaks
    smear by less than half a step, and [Eke19] shows a *small* search around the
    rounded value restores the success rate.  `search` is that radius, and it
    must stay small: on a toy curve a radius comparable to r enumerates the whole
    group and turns any measurement, including pure noise, into a "recovery" of
    every k.  Off-by-one candidates are down-weighted so an exact hit always
    outvotes a searched one.
    """
    if isinstance(counts, dict):
        items = list(counts.items())
    else:
        items = [(pair, 1) for pair in counts]

    votes, q = {}, 1 << bits
    for (j1, j2), c in items:
        for da in range(-search, search + 1):
            for db in range(-search, search + 1):
                a1 = (round(j1 * order / q) + da) % order
                a2 = (round(j2 * order / q) + db) % order
                if gcd(a1, order) != 1:
                    continue
                k = a2 * pow(a1, -1, order) % order
                w = c / (1 + abs(da) + abs(db)) ** 2
                votes[k] = votes.get(k, 0.0) + w
    return sorted(votes.items(), key=lambda kv: -kv[1])


def verify_dlog(curve, P, Q, k):
    return curve.mul(k, P) == Q
