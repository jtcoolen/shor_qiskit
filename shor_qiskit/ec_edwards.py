"""Twisted Edwards curves and out-of-place extended-coordinate point addition
-- [106] Sec 4.4 ("Extension to Ed25519"), Fig. 11, Table 4.

The curve
---------
    E_{a,d} :  a x^2 + y^2 = 1 + d x^2 y^2   over GF(p),

identity (0, 1), negation (x, y) -> (-x, y).  Ed25519 [BDL+12] is a = -1,
d = -121665/121666 over p = 2^255 - 19, cofactor 8.  The unified law

    (x1, y1) + (x2, y2) = ( (x1 y2 + y1 x2) / (1 + d x1 x2 y1 y2),
                            (y1 y2 - a x1 x2) / (1 - d x1 x2 y1 y2) )

is *complete* when a is a square and d is not [BL07, BBJ+08 Thm 3.3]: the
denominators never vanish, so one formula covers the identity, doubling and
P + (-P).  That is the structural difference from `ec_proj`, whose Jacobian
formulas break at H = 0.

Extended coordinates [HWCD08]
-----------------------------
(X : Y : Z : T) with x = X/Z, y = Y/Z, T = XY/Z.  For a = -1, "add-2008-hwcd-3":

    A = (Y1 - X1)(Y2 - X2)   B = (Y1 + X1)(Y2 + X2)   C = T1 * 2d * T2
    D = 2 Z1 Z2              E = B - A    F = D - C    G = D + C    H = B + A
    X3 = E F     Y3 = G H     Z3 = F G     T3 = E H

8 multiplications and one by the constant 2d.  [106] Sec 4.4 uses the mixed
form (classical addend, Z2 = 1, T2 = x2 y2): the addend enters only through
(y2 - x2, y2 + x2, 2d x2 y2) -- its affine Niels form -- and D = 2 Z1 is an
addition.  7 multiplications, against 11 for Weierstrass Alg. 4.

The circuit ([106] Fig. 11)
---------------------------
     1  S  = Y1 - X1          6  D  = 2 Z1  (2 Z1 Z2 for a quantum addend)
     2  Y1 = Y1 + X1          7  E  = B - A
     3  A  = S  (y2 - x2)     8  A  = A + B   (= H, in place)
     4  B  = Y1 (y2 + x2)     9  F  = D - C
     5  C  = T1 (2d x2 y2)   10  C  = C + D   (= G, in place)
    11-14  X3 = E F,  Y3 = G H,  Z3 = F G,  T3 = E H

The figure leaves S, Y1 + X1 and A..F behind; here steps 1-10 are recorded and
unwound once the outputs exist (their inputs are all still live), so as in
`ec_proj` the only garbage is the 4n-qubit input point.  Uncompute costs the
three products A, B, C -- 7 + 3 multiplications per addition, against 11 + 10
for `ec_proj.jacobian_add`, and 7n intermediates against 15n.

Variants: the addend classical (`edwards_add_const`, the analogue of
`jacobian_add`), looked up ([106]: "we load (y2 - x2), (y2 + x2), and 2dx2y2
from the quantum lookup table" -- `edwards_add_niels`, `edwards_add_window`),
controlled (`edwards_add_const_ctrl`), or a second quantum point in extended
coordinates (`edwards_add`, full hwcd-3).  Completeness pays twice in Shor: a
window's i = 0 entry is the identity, Niels (1, 1, 0), with no offset and no
special case, and a controlled addition is just an addition of either P or O.

The circuits hardcode a = -1 (H = B + A).  An a = 1 curve with p = 1 mod 4 is
the a = -1 curve with d -> -d under x -> sqrt(-1) x; `to_a_minus_1` does that.

Weierstrass
-----------
E_{a,d} is birational to the Montgomery curve B v^2 = u^3 + A u^2 + u with
A = 2(a + d)/(a - d), B = 4/(a - d) [BBJ+08 Thm 3.2], u = (1 + y)/(1 - y),
v = u/x; and that to a short Weierstrass curve by u = B t - A/3, v = B w.
When d/a is not a square, E(F_p) has no points at infinity and the map is a
group isomorphism onto the Weierstrass group: (0, 1) -> O, (0, -1) -> the
point of order 2.  [106] uses this ([MS16]) to run its in-place affine
addition on an Edwards instance; here it lets an Edwards ECDLP be checked
against the Weierstrass machinery in `ec_classical`.

Sources
    [106]    Kim et al., eprint 2026/106, Sec 4.4
    [HWCD08] Hisil, Wong, Carter, Dawson, Twisted Edwards curves revisited,
             Asiacrypt 2008
    [BBJ+08] Bernstein, Birkner, Joye, Lange, Peters, Twisted Edwards curves,
             Africacrypt 2008
    [BL07]   Bernstein, Lange, Faster addition and doubling on elliptic
             curves, Asiacrypt 2007
    [BDL+12] Bernstein, Duif, Lange, Schwabe, Yang, High-speed high-security
             signatures, JCEN 2012 (Ed25519); parameters as in RFC 8032
    [MS16]   Moody, Shumow, Analogues of Velu's formulas for isogenies on
             alternate models of elliptic curves, Math. Comp. 2016
"""

import ec_modarith as MA
import ec_mult as MU
from ec_classical import O as W_INF, Curve, Point
from qrom import lookup_ui


# =============================================================================
# Field helpers
# =============================================================================
def _is_square(v, p):
    v %= p
    return v == 0 or pow(v, (p - 1) // 2, p) == 1


def _sqrt_mod(v, p):
    """A square root of v mod p (Tonelli-Shanks), or None."""
    v %= p
    if v == 0:
        return 0
    if not _is_square(v, p):
        return None
    q, s = p - 1, 0
    while q % 2 == 0:
        q, s = q // 2, s + 1
    z = 2
    while _is_square(z, p):
        z += 1
    mm, c, t, r = s, pow(z, q, p), pow(v, q, p), pow(v, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2, i = t2 * t2 % p, i + 1
        b = pow(c, 1 << (mm - i - 1), p)
        mm, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


# =============================================================================
# The curve, in affine and extended coordinates
# =============================================================================
class TwistedEdwards:
    """a x^2 + y^2 = 1 + d x^2 y^2 over GF(p), p an odd prime > 3.

    Points are `ec_classical.Point`s; `inf` is never set -- the identity is the
    affine point (0, 1).
    """

    def __init__(self, p, a, d, name=""):
        a, d = a % p, d % p
        assert a and d and a != d, "degenerate twisted Edwards curve"
        self.p, self.a, self.d, self.name = p, a, d, name

    @property
    def identity(self):
        return Point(0, 1)

    def complete(self):
        """a square, d non-square: the unified law has no exceptions."""
        return _is_square(self.a, self.p) and not _is_square(self.d, self.p)

    def is_on_curve(self, P):
        p, x, y = self.p, P.x, P.y
        return not P.inf and (self.a * x * x + y * y - 1 - self.d * x * x * y * y) % p == 0

    def neg(self, P):
        return Point((-P.x) % self.p, P.y)

    def add(self, P, Q):
        """The unified affine law.  Raises only if the curve is not complete."""
        p = self.p
        t = self.d * P.x * Q.x * P.y * Q.y % p
        dx, dy = (1 + t) % p, (1 - t) % p
        if dx == 0 or dy == 0:
            raise ZeroDivisionError("exceptional pair: this curve is not complete")
        x3 = (P.x * Q.y + P.y * Q.x) * pow(dx, -1, p) % p
        y3 = (P.y * Q.y - self.a * P.x * Q.x) * pow(dy, -1, p) % p
        return Point(x3, y3)

    def mul(self, k, P):
        """[k]P by double-and-add.  k < 0 negates.  No special cases needed."""
        if k < 0:
            k, P = -k, self.neg(P)
        R, Q = self.identity, Point(P.x, P.y)
        while k:
            if k & 1:
                R = self.add(R, Q)
            Q = self.add(Q, Q)
            k >>= 1
        return R

    def points(self):
        """Every point. Toy p only -- O(p^2) work."""
        return [Point(x, y) for x in range(self.p) for y in range(self.p)
                if self.is_on_curve(Point(x, y))]

    def point_order(self, P):
        n, R = 1, Point(P.x, P.y)
        while R != self.identity:
            R, n = self.add(R, P), n + 1
        return n

    def recover_x(self, y, odd=0):
        """The x with (x, y) on the curve and x = odd (mod 2), or None."""
        p = self.p
        den = (self.d * y * y - self.a) % p
        if den == 0:
            return None
        x = _sqrt_mod((y * y - 1) * pow(den, -1, p), p)
        if x is None:
            return None
        return x if x % 2 == odd or x == 0 else p - x

    # --- extended coordinates [HWCD08] ---------------------------------------
    def to_extended(self, P, lam=1):
        """(x, y) -> (lam x : lam y : lam : lam x y), any lam != 0."""
        p = self.p
        assert lam % p
        return (lam * P.x % p, lam * P.y % p, lam % p, lam * P.x * P.y % p)

    def from_extended(self, X, Y, Z, T):
        p = self.p
        assert Z % p, "Z = 0 is not a point"
        zi = pow(Z, -1, p)
        return Point(X * zi % p, Y * zi % p)

    def on_curve_extended(self, X, Y, Z, T):
        """(aX^2 + Y^2) Z^2 = Z^4 + d X^2 Y^2 and XY = ZT, Z != 0."""
        p, a, d = self.p, self.a, self.d
        return (Z % p != 0
                and (X * Y - Z * T) % p == 0
                and ((a * X * X + Y * Y) * Z * Z - Z**4 - d * X * X * Y * Y) % p == 0)

    def add_extended(self, P1, P2):
        """The unified law on (X, Y, Z, T) tuples: hwcd-3 if a = -1, else hwcd."""
        if self.a == self.p - 1:
            return ext_add_ref(self, P1, P2)
        return ext_add_general_ref(self, P1, P2)

    def niels(self, P):
        """(y - x, y + x, 2d x y): the three values [106] looks up per window."""
        return niels_const(P.x, P.y, self.p, self.d)

    # --- a = 1 -> a = -1 ----------------------------------------------------
    def to_a_minus_1(self):
        """(curve with a' = -a, d' = -d, and x -> sqrt(-1) x) -- needs p = 1 mod 4.

        a x^2 + y^2 = 1 + d x^2 y^2 with x = i x' is -a x'^2 + y^2 = 1 - d x'^2
        y^2.  The circuits hardcode a = -1, so this is how an a = 1 curve is run.
        """
        p = self.p
        i = _sqrt_mod(-1, p)
        assert i is not None, "needs p = 1 mod 4"
        E2 = TwistedEdwards(p, -self.a, -self.d, self.name + "-twisted")
        ii = pow(i, -1, p)
        return E2, (lambda P: Point(P.x * ii % p, P.y)), (lambda P: Point(P.x * i % p, P.y))

    # --- birational maps: Montgomery and short Weierstrass [BBJ+08] ---------
    def montgomery(self):
        """(A, B) of B v^2 = u^3 + A u^2 + u."""
        p, a, d = self.p, self.a, self.d
        inv = pow(a - d, -1, p)
        return 2 * (a + d) * inv % p, 4 * inv % p

    def to_montgomery(self, P):
        p = self.p
        if P == self.identity:
            return W_INF
        if P.x == 0:                        # (0, -1), order 2
            return Point(0, 0)
        u = (1 + P.y) * pow(1 - P.y, -1, p) % p
        return Point(u, u * pow(P.x, -1, p) % p)

    def from_montgomery(self, M):
        p = self.p
        if M.inf:
            return self.identity
        if M.y == 0:
            assert M.x == 0, "a point at infinity of E: d/a is a square"
            return Point(0, p - 1)
        return Point(M.x * pow(M.y, -1, p) % p,
                     (M.x - 1) * pow(M.x + 1, -1, p) % p)

    def weierstrass(self):
        """The short Weierstrass curve t^3 + a' t + b' birational to this one."""
        p = self.p
        A, B = self.montgomery()
        aw = (3 - A * A) * pow(3 * B * B, -1, p) % p
        bw = (2 * A**3 - 9 * A) * pow(27 * B**3, -1, p) % p
        return Curve(p, aw, bw, (self.name or "edwards") + "-weierstrass")

    def to_weierstrass(self, P):
        M = self.to_montgomery(P)
        if M.inf:
            return W_INF
        p = self.p
        A, B = self.montgomery()
        Bi = pow(B, -1, p)
        return Point((M.x + A * pow(3, -1, p)) * Bi % p, M.y * Bi % p)

    def from_weierstrass(self, W):
        if W.inf:
            return self.identity
        p = self.p
        A, B = self.montgomery()
        return self.from_montgomery(Point((B * W.x - A * pow(3, -1, p)) % p, B * W.y % p))

    def __repr__(self):
        return f"TwistedEdwards({self.name}: p={self.p}, a={self.a}, d={self.d})"


def niels_const(x, y, p, d):
    """Affine Niels form of (x, y): (y - x, y + x, 2 d x y) mod p."""
    return (y - x) % p, (y + x) % p, 2 * d * x * y % p


NIELS_IDENTITY = (1, 1, 0)        # niels_const(0, 1, ...) for every p, d


def niels_window(curve, base, w):
    """[i] base in Niels form, i = 0 .. 2^w - 1.

    Entry 0 is the identity, (1, 1, 0) -- no offset needed, unlike
    `ec_classical.window_table` for Weierstrass, where [0] base = O has no
    affine encoding and every other entry must dodge the exceptional cases.
    """
    out, acc = [], curve.identity
    for _ in range(1 << w):
        out.append(curve.niels(acc))
        acc = curve.add(acc, base)
    return out


# --- toy instances ------------------------------------------------------------
# All have a square and d non-square (complete).  p = 1 mod 4 throughout, so
# a = -1 is a square; the a = 1 curves exercise the general law.  Orders:
# 20 = 4*5, 24 = 8*3, 28 = 4*7, 40 = 8*5, 44 = 4*11, 44 = 4*11.
ED13 = TwistedEdwards(13, -1, 6, "ed-p13")         # n = 4 bits, the quantum test curve
ED17 = TwistedEdwards(17, -1, 7, "ed-p17")         # n = 5, cofactor 8 like Ed25519
ED_TOY = [
    ED13,
    ED17,
    TwistedEdwards(29, -1, 2, "ed-p29"),
    TwistedEdwards(29, 1, 2, "ed-p29-a1"),
    TwistedEdwards(37, -1, 14, "ed-p37"),
    TwistedEdwards(37, 1, 15, "ed-p37-a1"),
]

# --- Ed25519 [BDL+12], RFC 8032 Sec 5.1 -----------------------------------------
ED25519_P = 2**255 - 19
ED25519 = TwistedEdwards(ED25519_P, -1, -121665 * pow(121666, -1, ED25519_P), "ed25519")
ED25519_L = 2**252 + 27742317777372353535851937790883648493     # prime subgroup order
ED25519_H = 8                                                   # cofactor
ED25519_B = Point(
    15112221349535400772501151409588531511454012693041857206046113283949847762202,
    46316835694926478169428394003475163141307993866256225615783033603165251855960)


# =============================================================================
# Reference formulas -- the specifications of the circuits below
# =============================================================================
def ext_add_ref(curve, P1, P2):
    """add-2008-hwcd-3 [HWCD08], a = -1: 8 multiplications + 1 by k = 2d.

    P1, P2, and the result are (X, Y, Z, T) tuples.  The spec of `edwards_add`.
    """
    p = curve.p
    assert curve.a == p - 1, "hwcd-3 is the a = -1 formula"
    X1, Y1, Z1, T1 = P1
    X2, Y2, Z2, T2 = P2
    A = (Y1 - X1) * (Y2 - X2) % p
    B = (Y1 + X1) * (Y2 + X2) % p
    C = T1 * 2 * curve.d * T2 % p
    D = Z1 * 2 * Z2 % p
    E, F, G, H = (B - A) % p, (D - C) % p, (D + C) % p, (B + A) % p
    return E * F % p, G * H % p, F * G % p, E * H % p


def ext_madd_ref(curve, P1, x2, y2):
    """[106] Sec 4.4 mixed addition, (X1:Y1:Z1:T1) + (x2, y2): 7 multiplications.

    The spec of `edwards_add_const`; with the Niels triple looked up, of
    `edwards_add_niels`.
    """
    return ext_add_niels_ref(curve, P1, niels_const(x2, y2, curve.p, curve.d))


def ext_add_niels_ref(curve, P1, N):
    p = curve.p
    X1, Y1, Z1, T1 = P1
    ymx, ypx, kt = N
    A = (Y1 - X1) * ymx % p                 # 3
    B = (Y1 + X1) * ypx % p                 # 4
    C = T1 * kt % p                         # 5
    D = 2 * Z1 % p                          # 6  an addition, not a multiply
    E, F, G, H = (B - A) % p, (D - C) % p, (D + C) % p, (B + A) % p
    return E * F % p, G * H % p, F * G % p, E * H % p


def ext_add_general_ref(curve, P1, P2):
    """add-2008-hwcd [HWCD08 Sec 3.1], any a: 9 multiplications + 2 constants.

    E = (X1 + Y1)(X2 + Y2) - A - B and H = B - aA; used for the a = 1 curves.
    """
    p, a, d = curve.p, curve.a, curve.d
    X1, Y1, Z1, T1 = P1
    X2, Y2, Z2, T2 = P2
    A, B = X1 * X2 % p, Y1 * Y2 % p
    C, D = d * T1 * T2 % p, Z1 * Z2 % p
    E = ((X1 + Y1) * (X2 + Y2) - A - B) % p
    F, G, H = (D - C) % p, (D + C) % p, (B - a * A) % p
    return E * F % p, G * H % p, F * G % p, E * H % p


# Multiplications per out-of-place addition: (forward, uncompute), each split
# as general multiplications + multiplications by a classical constant.
OPS = {
    "jacobian_add (Weierstrass, [106] Alg. 4)": ((9, 2), (8, 2)),
    "edwards_add_const ([106] Sec 4.4, classical addend)": ((4, 3), (0, 3)),
    "edwards_add_niels ([106] Sec 4.4, looked-up addend)": ((7, 0), (3, 0)),
    "edwards_add (hwcd-3, quantum addend)": ((8, 1), (4, 1)),
}


# =============================================================================
# Quantum circuits
# =============================================================================
def _copy_into(m, src, dst):
    for a, b in zip(src, dst):
        m.ctx.cx(a, b)


def _sub_into(m, a, b, out, p):
    """out (|0>) <- a - b mod p."""
    _copy_into(m, a, out)
    MA.modsub(m, b, out, p)


def _mul(m, x, k, out, p):
    """out (|0>) <- x k mod p.  k a register, or a classical constant."""
    if isinstance(k, int):
        MU.modmul_const(m, x, k, out, p)
    else:
        MU.modmul(m, x, k, out, p)


def _front(m, P1, N, S, A, B, C, D, E, F, p):
    """Steps 1-10 of [106] Fig. 11: everything before the four products.

    N = (y2 - x2, y2 + x2, 2d x2 y2, Z2): registers or classical ints; Z2 is
    None for an affine addend (then D = 2 Z1 costs an addition, not a multiply).
    """
    X1, Y1, Z1, T1 = P1
    ymx, ypx, kt, z2 = N
    _sub_into(m, Y1, X1, S, p)              # 1  S  = Y1 - X1
    MA.modadd(m, X1, Y1, p)                 # 2  Y1 = Y1 + X1
    _mul(m, S, ymx, A, p)                   # 3  A  = (Y1 - X1)(y2 - x2)
    _mul(m, Y1, ypx, B, p)                  # 4  B  = (Y1 + X1)(y2 + x2)
    _mul(m, T1, kt, C, p)                   # 5  C  = T1 * 2d x2 y2
    if z2 is None:                          # 6  D  = 2 Z1
        _copy_into(m, Z1, D)
    else:                                   #    D  = 2 Z1 Z2
        MU.modmul(m, Z1, z2, D, p)
    MA.moddbl(m, D, p)
    _sub_into(m, B, A, E, p)                # 7  E  = B - A
    MA.modadd(m, B, A, p)                   # 8  A  = A + B  = H
    _sub_into(m, D, C, F, p)                # 9  F  = D - C
    MA.modadd(m, D, C, p)                   # 10 C  = C + D  = G


def _core(m, P1, N, P3, p):
    """(X3:Y3:Z3:T3) <- P1 + N, every intermediate returned to |0>."""
    n = len(P1[0])
    X3, Y3, Z3, T3 = P3
    S, A, B, C, D, E, F = (m.anc(n, nm) for nm in "SABCDEF")
    _, front = m.step(_front, m, P1, N, S, A, B, C, D, E, F, p)
    H, G = A, C
    MU.modmul(m, E, F, X3, p)               # 11 X3 = E F
    MU.modmul(m, G, H, Y3, p)               # 12 Y3 = G H
    MU.modmul(m, F, G, Z3, p)               # 13 Z3 = F G
    MU.modmul(m, E, H, T3, p)               # 14 T3 = E H
    m.undo(front)                           # A, B, C cost a multiply each
    m.free(F, E, D, C, B, A, S)


def edwards_add_const(m, P1, x2, y2, P3, p, d):
    """P3 <- P1 + (x2, y2) for a classical affine addend.  [106] Sec 4.4.

    P1 = (X1, Y1, Z1, T1) survives unchanged -- the garbage, as in
    `ec_proj.jacobian_add`.  P3 must be |0>.  a = -1.  Complete: (x2, y2) may
    be the identity, equal to P1, or its negation.
    """
    _core(m, P1, niels_const(x2, y2, p, d) + (None,), P3, p)


def edwards_add_const_inv(m, P1, x2, y2, P3, p, d):
    """The PA-dagger of [106] Fig. 10: clears P3 given P1."""
    m.emit_inverse(edwards_add_const, m, P1, x2, y2, P3, p, d)


def edwards_add_niels(m, P1, N2, P3, p):
    """P3 <- P1 + P2 with P2 given by registers N2 = (y2 - x2, y2 + x2, 2d x2 y2).

    The windowed form of [106] Sec 4.4: seven general multiplications.
    """
    ymx, ypx, kt = N2
    _core(m, P1, (ymx, ypx, kt, None), P3, p)


def _load_ctrl(m, ctrl, regs, v1, v0):
    """regs ^= (ctrl ? v1 : v0), bitwise.  X and CNOT only; self-inverse."""
    for r, a, b in zip(regs, v1, v0):
        for i, q in enumerate(r):
            if (b >> i) & 1:
                m.ctx.x(q)
            if ((a ^ b) >> i) & 1:
                m.ctx.cx(ctrl, q)


def edwards_add_const_ctrl(m, ctrl, P1, x2, y2, P3, p, d):
    """P3 <- P1 + [ctrl] (x2, y2).

    Completeness makes the control free: load Niels(P2) if ctrl else Niels(O)
    = (1, 1, 0) -- CNOTs, no Toffoli -- and add unconditionally.  The price is
    that A, B, C become general multiplications (the window-of-one case of
    `edwards_add_window`).
    """
    n = len(P1[0])
    N = [m.anc(n, nm) for nm in ("ymx", "ypx", "kt")]
    v1 = niels_const(x2, y2, p, d)
    _load_ctrl(m, ctrl, N, v1, NIELS_IDENTITY)
    edwards_add_niels(m, P1, N, P3, p)
    _load_ctrl(m, ctrl, N, v1, NIELS_IDENTITY)
    m.free(*N)


def edwards_add_window(m, addr, P1, table, P3, p):
    """P3 <- P1 + [addr] base, `table` = niels_window(curve, base, len(addr)).

    One unary-iteration lookup loads all three Niels values (3n bits per
    entry), another unloads them.  Entry 0 is the identity: no special case.
    """
    n, w = len(P1[0]), len(addr)
    assert len(table) == 1 << w
    N = [m.anc(n, nm) for nm in ("ymx", "ypx", "kt")]
    words = [a | b << n | c << 2 * n for a, b, c in table]
    one, lk = m.anc(1, "one"), m.anc(w, "lk")
    m.ctx.x(one[0])
    lookup_ui(m.ctx, one[0], addr, N[0] + N[1] + N[2], words, lk)
    edwards_add_niels(m, P1, N, P3, p)
    lookup_ui(m.ctx, one[0], addr, N[0] + N[1] + N[2], words, lk)
    m.ctx.x(one[0])
    m.free(lk, one, *N)


def edwards_add(m, P1, P2, P3, p, d):
    """P3 <- P1 + P2, both quantum, extended coordinates.  add-2008-hwcd-3.

    P2 is put into projective Niels form in place (Y2 - X2 into scratch,
    Y2 += X2, 2d T2 into scratch), the Fig. 11 core runs with D = 2 Z1 Z2, and
    the preparation is unwound.  P1 and P2 survive; P3 must be |0>.  P1 and P2
    must be distinct registers (their values may be equal: doubling is fine).
    """
    X2, Y2, Z2, T2 = P2
    n = len(X2)
    S2, K2 = m.anc(n, "S2"), m.anc(n, "K2")

    def prep():
        _sub_into(m, Y2, X2, S2, p)         # Y2 - X2
        MA.modadd(m, X2, Y2, p)             # Y2 + X2
        MU.modmul_const(m, T2, 2 * d % p, K2, p)     # 2d T2

    _, body = m.step(prep)
    _core(m, P1, (S2, Y2, K2, Z2), P3, p)
    m.undo(body)
    m.free(K2, S2)


def edwards_add_inv(m, P1, P2, P3, p, d):
    m.emit_inverse(edwards_add, m, P1, P2, P3, p, d)
