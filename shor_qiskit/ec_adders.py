"""Unsigned adders, comparators and shifts -- the bottom of the stack.

Both papers pick an adder per call site, from the same menu, on the basis of
how many ancillas happen to be free at that point:

    CDKM  [CDKM04]   addition 2n Toffoli, controlled addition 3n, ~1 ancilla
    Gidney [Gid18]   addition  n Toffoli, controlled addition 2n,  n ancillas

[1128] Sec 4 says it plainly: "we use strategically components based on the
latter when ancillas are available ... and components based on the former when
this is not the case".  So every routine here takes an explicit ancilla budget
and `add()` dispatches on it.

Two structural notes.

* Controlled addition is *never* built by promoting the adder's CNOTs to
  Toffolis.  [106] Fig. 4(b) copies the operand into clean ancillas under the
  control and then adds unconditionally; that is 2n rather than 3n, and the
  copies parallelise.  `cadd` does exactly that.
* Comparison is the forward half of the Gidney carry chain, read off, then
  reversed.  Because the AND uncompute is free, an n-bit comparison costs n
  Toffoli-equivalents, not 2n.
"""

from ec_sim import Reg


# --- helpers -----------------------------------------------------------------
def _bits(k, n):
    k %= 1 << n
    return [(k >> i) & 1 for i in range(n)]


def encode_const(ctx, reg, k, ctrl=None):
    """Write the classical constant k into `reg` (which must be |0>).

    With `ctrl`, writes k when ctrl=1 and 0 otherwise -- the copy step of
    [106] Fig. 4(b).  Self-inverse, so the same call undoes it.
    """
    for i, b in enumerate(_bits(k, len(reg))):
        if b:
            ctx.cx(ctrl, reg[i]) if ctrl is not None else ctx.x(reg[i])


# --- CDKM ripple-carry: cheap in space, 2n Toffoli ---------------------------
# Reused verbatim from `rc_adder`, this package's existing CDKM implementation
# (already exhaustively verified by tests/test_rc.py).  It emits only mcx/cx/ccx,
# which `Ctx` exposes, so a Ctx drops straight in where it expects a circuit.
from rc_adder import rc_add as _rc_add


def cdkm_add(ctx, x, y, carry, ctrls=()):
    """|x>|y> -> |x>|(y+x) mod 2^n>.  One clean ancilla `carry`, returned clean.

    2n Toffoli; with controls, 3n.  [CDKM04].
    """
    assert len(x) == len(y), "operands must be the same width"
    _rc_add(ctx, x, y, carry, ctrls)


# --- Gidney: n Toffoli, n-1 ancillas -----------------------------------------
def gidney_add(ctx, x, y, anc):
    """|x>|y> -> |x>|(y+x) mod 2^n>.  n-1 clean ancillas, returned clean.

    The carry recurrence c_{i+1} = MAJ(x_i, y_i, c_i) is rewritten using
    MAJ(x,y,c) = c XOR ((x XOR c) AND (y XOR c)), so each carry is one AND on a
    *clean* target -- which is the whole point: 4 T to compute, 0 to uncompute.
    """
    n = len(y)
    assert len(x) == n, "operands must be the same width"
    if n == 0:
        return
    a = anc[: n - 1]
    assert len(a) == n - 1, f"gidney_add needs {n-1} ancillas, got {len(anc)}"
    for i in range(n - 1):
        if i:
            ctx.cx(a[i - 1], x[i])
            ctx.cx(a[i - 1], y[i])
        ctx.and_(x[i], y[i], a[i])
        if i:
            ctx.cx(a[i - 1], a[i])
    ctx.cx(x[n - 1], y[n - 1])
    if n > 1:
        ctx.cx(a[n - 2], y[n - 1])
    for i in range(n - 2, -1, -1):
        if i:
            ctx.cx(a[i - 1], a[i])
        ctx.and_dg(x[i], y[i], a[i])
        if i:
            ctx.cx(a[i - 1], x[i])
        ctx.cx(x[i], y[i])


def add(ctx, x, y, anc):
    """y += x mod 2^n, picking the adder the ancilla budget allows."""
    n = len(y)
    if len(anc) >= n - 1 and n > 1:
        gidney_add(ctx, x, y, anc)
    else:
        assert len(anc) >= 1, "need at least one ancilla for CDKM"
        cdkm_add(ctx, x, y, anc[0])


def sub(ctx, x, y, anc):
    """y -= x mod 2^n.  y - x == NOT(NOT(y) + x), so one adder covers both."""
    for q in y:
        ctx.x(q)
    add(ctx, x, y, anc)
    for q in y:
        ctx.x(q)


def cadd(ctx, ctrl, x, y, copy, anc):
    """y += x if ctrl, via [106] Fig. 4(b): copy under control, then add.

    `copy`: len(x) clean ancillas.  2n Toffoli (n for the copy, n for the add)
    against 3n for promoting the adder's CNOTs, and the copies all commute so
    they parallelise -- which is why the paper does it this way.
    """
    assert len(copy) == len(x)
    for i, q in enumerate(x):
        ctx.and_(ctrl, q, copy[i])
    add(ctx, copy, y, anc)
    for i, q in enumerate(x):
        ctx.and_dg(ctrl, q, copy[i])


def csub(ctx, ctrl, x, y, copy, anc):
    for q in y:
        ctx.x(q)
    cadd(ctx, ctrl, x, y, copy, anc)
    for q in y:
        ctx.x(q)


# --- constant operands -------------------------------------------------------
def add_const(ctx, y, k, creg, anc):
    """y += k mod 2^n for a classical k.  `creg`: n clean ancillas."""
    encode_const(ctx, creg, k)
    add(ctx, creg, y, anc)
    encode_const(ctx, creg, k)


def sub_const(ctx, y, k, creg, anc):
    add_const(ctx, y, -k, creg, anc)


def cadd_const(ctx, ctrl, y, k, creg, anc):
    """y += k if ctrl.  The copy of [106] Fig. 4(b) is just CNOTs from ctrl."""
    encode_const(ctx, creg, k, ctrl)
    add(ctx, creg, y, anc)
    encode_const(ctx, creg, k, ctrl)


def csub_const(ctx, ctrl, y, k, creg, anc):
    cadd_const(ctx, ctrl, y, -k, creg, anc)


# --- comparison: the carry chain, read, then reversed ------------------------
def carry_out(ctx, x, y, out, anc):
    """out ^= carry-out of (x + y).  x, y restored; n clean ancillas.

    n AND gates to compute the chain and n free uncomputes to undo it, so this
    costs n Toffoli-equivalents -- not the 2n a subtract-and-restore would.
    """
    n = len(x)
    assert len(y) == n and len(anc) >= n
    a = anc[:n]
    for i in range(n):
        if i:
            ctx.cx(a[i - 1], x[i])
            ctx.cx(a[i - 1], y[i])
        ctx.and_(x[i], y[i], a[i])
        if i:
            ctx.cx(a[i - 1], a[i])
    ctx.cx(a[n - 1], out)
    for i in range(n - 1, -1, -1):
        if i:
            ctx.cx(a[i - 1], a[i])
        ctx.and_dg(x[i], y[i], a[i])
        if i:
            ctx.cx(a[i - 1], x[i])
            ctx.cx(a[i - 1], y[i])


def gt_uint(ctx, x, y, out, anc):
    """out ^= [x > y].  x + NOT(y) carries out exactly when x > y."""
    for q in y:
        ctx.x(q)
    carry_out(ctx, x, y, out, anc)
    for q in y:
        ctx.x(q)


def lt_uint(ctx, x, y, out, anc):
    """out ^= [x < y]."""
    gt_uint(ctx, y, x, out, anc)


def geq_const(ctx, x, k, out, creg, anc):
    """out ^= [x >= k] for classical k.  `creg`: len(x) clean ancillas.

    The boundaries are not decoration: k = 0 makes the predicate constantly
    true, and encoding k-1 = -1 would wrap to all-ones and silently answer
    "false".  Both ends are decided classically, for free.
    """
    n = len(x)
    if k <= 0:
        ctx.x(out)                      # x >= 0 always
        return
    if k >= 1 << n:
        return                          # x >= 2^n never
    encode_const(ctx, creg, k - 1)
    gt_uint(ctx, x, creg, out, anc)
    encode_const(ctx, creg, k - 1)


def lt_const(ctx, x, k, out, creg, anc):
    """out ^= [x < k]."""
    geq_const(ctx, x, k, out, creg, anc)
    ctx.x(out)


def clt_uint(ctx, ctrl, x, y, out, anc, copy):
    """out ^= ctrl AND [x < y].  Compares into scratch, then ANDs in."""
    t = copy[:1]
    lt_uint(ctx, x, y, t[0], anc)
    ctx.ccx(ctrl, t[0], out)
    lt_uint(ctx, x, y, t[0], anc)


# --- shifts: relabelling, not gates ------------------------------------------
def shift_up(reg, zero):
    """2*reg, as a register: a fresh |0> becomes the new LSB.

    Costs nothing.  A doubling is a renaming of wires, and both papers treat it
    that way -- [1128] Alg. 5-7 open with `shift_right(x_reg + anc1, 1)`, which
    in their little-endian list convention is exactly this.
    """
    return Reg([zero] + list(reg), getattr(reg, "name", "") + "<<1")


def shift_down(reg):
    """reg // 2, as a register.  Returns (value register, freed LSB qubit)."""
    return Reg(list(reg[1:]), getattr(reg, "name", "") + ">>1"), reg[0]
