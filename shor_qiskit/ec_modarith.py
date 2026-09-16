"""Exact modular arithmetic over GF(p).

These are the [RNSL17] circuits, in the form [1128] Sec 4 quotes them as
Algorithms 5 and 8 before optimising them.  Everything here is *exact*: it is
correct for every input, and the tests check that exhaustively.  The
approximate versions -- MSB-only comparisons, pseudo-Mersenne reduction -- live
in `ec_approx`, and are checked against these.

Every routine takes a `Machine` and draws its scratch from that machine's pool,
so ancillas are shared across the whole point addition rather than allocated per
operation.  That pooling is what keeps the qubit count near the papers'.

Register convention: little-endian, and a doubling is a *relabelling* (see
`ec_adders.shift_up`), so `moddbl` returns the register its result lives in.
"""

import ec_adders as A
from ec_sim import Reg


# --- zero test ---------------------------------------------------------------
def is_zero(m, x, out):
    """out ^= [x == 0].  n-1 ANDs; x restored."""
    ctx, n = m.ctx, len(x)
    if n == 1:
        ctx.x(x[0])
        ctx.cx(x[0], out)
        ctx.x(x[0])
        return
    for q in x:
        ctx.x(q)
    anc = m.anc(n - 1, "iz")
    ctx.and_(x[0], x[1], anc[0])
    for i in range(2, n):
        ctx.and_(anc[i - 2], x[i], anc[i - 1])
    ctx.cx(anc[n - 2], out)
    for i in range(n - 1, 1, -1):
        ctx.and_dg(anc[i - 2], x[i], anc[i - 1])
    ctx.and_dg(x[0], x[1], anc[0])
    for q in x:
        ctx.x(q)
    m.free(anc)


def is_nonzero(m, x, out):
    is_zero(m, x, out)
    m.ctx.x(out)


# --- modular addition of a quantum operand -----------------------------------
def cmodadd(m, ctrl, x, y, p):
    """y <- (y + x) mod p when ctrl (or unconditionally if ctrl is None).

    [1128] Algorithm 8, the exact circuit of [RNSL17]:

        cadd(ctrl, x+anc_x, y+anc_y);  sub(p, y+anc_y)
        cadd(anc_y, p, y);  clt(ctrl, y, x, anc_y);  x(anc_y)

    The elegance is in the last two lines.  After subtracting p, anc_y is the
    borrow, i.e. [x+y < p], i.e. "no reduction happened".  No reduction means
    the answer is x+y, which is >= x; a reduction means the answer is x+y-p,
    which is < x because y < p.  So [answer < x] is exactly the complement of
    anc_y, and comparing the answer against x clears the flag -- no extra
    ancilla, no second copy of the inputs.
    """
    ctx, n = m.ctx, len(y)
    assert len(x) == n
    anc_x, anc_y = m.anc(1, "ax"), m.anc(1, "ay")
    xe, ye = x + anc_x, y + anc_y
    cp, a = m.anc(n + 1, "cp"), m.anc(n + 1, "a")

    if ctrl is None:
        A.add(ctx, xe, ye, a)
    else:
        A.cadd(ctx, ctrl, xe, ye, cp, a)
    A.sub_const(ctx, ye, p, cp, a)
    A.cadd_const(ctx, anc_y[0], y, p, cp[:n], a)
    if ctrl is None:
        A.lt_uint(ctx, y, x, anc_y[0], a)
    else:
        A.clt_uint(ctx, ctrl, y, x, anc_y[0], a, cp)
    ctx.x(anc_y[0])
    m.free(cp, a, anc_x, anc_y)


def modadd(m, x, y, p):
    cmodadd(m, None, x, y, p)


def modsub(m, x, y, p):
    """y <- (y - x) mod p: `modadd` run backwards."""
    m.emit_inverse(modadd, m, x, y, p)


def cmodsub(m, ctrl, x, y, p):
    m.emit_inverse(cmodadd, m, ctrl, x, y, p)


# --- modular addition of a classical constant --------------------------------
def cmodadd_const(m, ctrl, y, k, p):
    """y <- (y + k) mod p, for classical k in [0, p)."""
    ctx, n = m.ctx, len(y)
    k %= p
    hi = m.anc(1, "hi")
    ye = y + hi
    cp, a = m.anc(n + 1, "cp"), m.anc(n + 1, "a")
    if ctrl is None:
        A.add_const(ctx, ye, k, cp, a)
    else:
        A.cadd_const(ctx, ctrl, ye, k, cp, a)
    A.sub_const(ctx, ye, p, cp, a)
    A.cadd_const(ctx, hi[0], y, p, cp[:n], a)
    # hi == [y+k < p] == [answer >= k]; clear it by testing the answer against k
    t = m.anc(1, "t")
    A.lt_const(ctx, y, k, t[0], cp[:n], a)
    if ctrl is None:
        ctx.cx(t[0], hi[0])
    else:
        ctx.ccx(ctrl, t[0], hi[0])
    A.lt_const(ctx, y, k, t[0], cp[:n], a)
    ctx.x(hi[0])
    m.free(t, cp, a, hi)


def modadd_const(m, y, k, p):
    cmodadd_const(m, None, y, k, p)


def modsub_const(m, y, k, p):
    m.emit_inverse(modadd_const, m, y, k, p)


def cmodsub_const(m, ctrl, y, k, p):
    m.emit_inverse(cmodadd_const, m, ctrl, y, k, p)


# --- modular doubling --------------------------------------------------------
def _moddbl_body(m, xe, p):
    """Reduce an (n+1)-bit register holding 2v to v's width, mod p.

    [1128] Algorithm 5, the exact circuit of [RNSL17].  Clearing the reduction
    flag costs one CNOT and no ancilla: 2v is even and p is odd, so the answer
    is even exactly when no reduction happened.
    """
    ctx = m.ctx
    n = len(xe) - 1
    a2 = m.anc(1, "a2")
    cp, a = m.anc(n + 2, "cp"), m.anc(n + 2, "a")
    A.sub_const(ctx, xe + a2, p, cp, a)
    A.cadd_const(ctx, a2[0], xe, p, cp[: n + 1], a)
    ctx.x(a2[0])
    ctx.cx(xe[0], a2[0])
    m.free(cp, a, a2)


def moddbl(m, x, p):
    """x <- 2x mod p, in place: the register handle stays valid.

    The shift itself is free -- it renames wires, which is how [1128] writes it
    (`shift_right(x_reg + anc1, 1)`).  What is not free is putting the answer
    back where the caller left it: the reduced value lands one position low, so
    n SWAPs rotate it home and hand the borrowed qubit back.

    Those SWAPs are pure CNOT -- zero Toffoli, zero T -- so they do not move the
    metric either paper reports, and in exchange every routine composes and
    inverts without the caller tracking a rotating layout.  `moddbl_rotate`
    below is the free version, used where the layout is under local control.
    """
    ctx, n = m.ctx, len(x)
    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])                 # n+1 bits, holds 2x
    _moddbl_body(m, xe, p)
    # value bits now sit at xe[0..n-1]; xe[n] (the old MSB) is clean
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
    m.free(z)


def moddbl_rotate(m, x, p):
    """`moddbl` without the SWAPs: returns the register the answer moved to.

    Zero gates for the shift, but the caller must adopt the returned handle.
    Used by the Bezout replay of [1128] Alg. 3, which doubles once per iteration
    on a register it owns outright, so the rotation costs nothing and is safe.
    """
    n = len(x)
    z = m.anc(1, "z")
    xe = A.shift_up(x, z[0])
    _moddbl_body(m, xe, p)
    out = Reg(list(xe[:n]), getattr(x, "name", "x"))
    m.free(Reg([xe[n]]))
    return out


def cmoddbl(m, ctrl, x, p):
    """x <- 2x mod p when ctrl, x unchanged otherwise.  In place.

    The trick is that `shift_up` already makes the register *read* as 2x for
    free, so the control is applied the other way round: when ctrl is 0 the
    contents are walked back down one position, which restores the reading x.
    n CSWAPs, then the ordinary reduction.

    Clearing the reduction flag needs one more control than `moddbl`: the
    "answer is even iff no reduction" argument only holds for the doubled
    branch, so the CNOT from the LSB becomes a Toffoli with ctrl.
    """
    ctx, n = m.ctx, len(x)
    z = m.anc(1, "z")
    xe = Reg([z[0]] + list(x), "xe")          # reads as 2x
    ctx.x(ctrl)
    for i in range(n):                        # ctrl = 0: walk back down to x
        ctx.cswap(ctrl, xe[i], xe[i + 1])
    ctx.x(ctrl)
    a2 = m.anc(1, "a2")
    cp, sc = m.anc(n + 2, "cp"), m.anc(n + 2, "sc")
    A.sub_const(ctx, xe + a2, p, cp, sc)
    A.cadd_const(ctx, a2[0], xe, p, cp[: n + 1], sc)
    ctx.x(a2[0])
    ctx.ccx(ctrl, xe[0], a2[0])
    m.free(cp, sc, a2)
    for i in range(n - 1, -1, -1):            # rotate the answer home
        ctx.swap(xe[i], xe[i + 1])
    m.free(z)


def cmodhalf(m, ctrl, x, p):
    """x <- x/2 mod p when ctrl: `cmoddbl` run backwards."""
    m.emit_inverse(cmoddbl, m, ctrl, x, p)


def modhalf(m, x, p):
    """x <- x/2 mod p, in place: `moddbl` run backwards."""
    m.emit_inverse(moddbl, m, x, p)


# --- modular negation --------------------------------------------------------
def cmodneg(m, ctrl, x, p):
    """x <- (-x) mod p when ctrl.

    p - x is a complement plus a constant, so the whole thing is a controlled
    complement and a controlled constant addition.  The only subtlety is x = 0,
    where p - x would give p rather than 0; a zero test gates the operation, and
    the *same* test clears the flag afterwards because negation preserves
    "is zero".
    """
    ctx, n = m.ctx, len(x)
    nz = m.anc(1, "nz")
    is_nonzero(m, x, nz[0])
    g = nz
    if ctrl is not None:
        g = m.anc(1, "g")
        ctx.and_(ctrl, nz[0], g[0])
    for q in x:
        ctx.cx(g[0], q)
    cp, a = m.anc(n, "cp"), m.anc(n, "a")
    A.cadd_const(ctx, g[0], x, p + 1, cp, a)
    m.free(cp, a)
    if ctrl is not None:
        ctx.and_dg(ctrl, nz[0], g[0])
        m.free(g)
    is_nonzero(m, x, nz[0])
    m.free(nz)


def modneg(m, x, p):
    cmodneg(m, None, x, p)
