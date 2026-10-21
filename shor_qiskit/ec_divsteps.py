"""Toy-width reversible divsteps inversion using Gu et al. Eq. (6).

This initial circuit keeps exact per-round branch history. It does not yet
implement the paper's Appendix A history-free decoder or packed wire layout.
"""

import ec_adders as A
import ec_modarith as MA
from ec_divsteps_reference import walk


def _signed_neg(m, ctrl, reg):
    for bit in reg:
        m.ctx.cx(ctrl, bit)
    _add_const_small(m, ctrl, reg, 1)


def _add_signed(m, ctrl, src, dst):
    carry = m.anc(1, "carry")
    A.cdkm_add(m.ctx, src, dst, carry[0], ctrls=(ctrl,))
    m.free(carry)


def _add_const_small(m, ctrl, reg, value, carry=None):
    cp = m.anc(len(reg), "const")
    owned = carry is None
    if owned:
        carry = m.anc(1, "carry")[0]
    A.encode_const(m.ctx, cp, value, ctrl)
    A.cdkm_add(m.ctx, cp, reg, carry)
    A.encode_const(m.ctx, cp, value, ctrl)
    m.free(cp)
    if owned:
        m.free([carry])


def _cmoddbl_small(m, ctrl, x, p, borrowed=None):
    """Same exact modular double as ec_modarith.cmoddbl, using CDKM scratch."""
    ctx, n = m.ctx, len(x)
    z = m.anc(1, "z") if borrowed is None else borrowed[:1]
    xe = [z[0]] + list(x)
    ctx.x(ctrl)
    for i in range(n):
        ctx.cswap(ctrl, xe[i], xe[i + 1])
    ctx.x(ctrl)
    hi = m.anc(1, "hi") if borrowed is None else borrowed[1:2]
    carry = None if borrowed is None else borrowed[2]
    _add_const_small(m, None, xe + list(hi), -p, carry)
    _add_const_small(m, hi[0], xe, p, carry)
    ctx.x(hi[0])
    ctx.ccx(ctrl, xe[0], hi[0])
    if borrowed is None:
        m.free(hi)
    for i in range(n - 1, -1, -1):
        ctx.swap(xe[i], xe[i + 1])
    if borrowed is None:
        m.free(z)


def _cmodneg_small(m, ctrl, x, p, carry=None):
    ctx = m.ctx
    nz = m.anc(1, "nz")
    MA.is_nonzero(m, x, nz[0])
    gate = m.anc(1, "neg")
    ctx.and_(ctrl, nz[0], gate[0])
    for bit in x:
        ctx.cx(gate[0], bit)
    _add_const_small(m, gate[0], x, p + 1, carry)
    ctx.and_dg(ctrl, nz[0], gate[0])
    m.free(gate)
    MA.is_nonzero(m, x, nz[0])
    m.free(nz)


def _cmodadd_small(m, ctrl, x, y, p):
    """Exact modular add with a low-space CDKM carry instead of copied operands."""
    ctx, n = m.ctx, len(y)
    ax, ay = m.anc(1, "ax"), m.anc(1, "ay")
    carry = m.anc(1, "carry")
    A.cdkm_add(ctx, x + ax, y + ay, carry[0], ctrls=(ctrl,))
    m.free(carry)
    _add_const_small(m, None, y + ay, -p)
    _add_const_small(m, ay[0], y, p)
    scratch, test = m.anc(n, "compare"), m.anc(1, "test")
    A.clt_uint(ctx, ctrl, y, x, ay[0], scratch, test)
    ctx.x(ay[0])
    m.free(scratch, test, ax, ay)


def _round(m, z, f, g, fv, gv, p, record):
    active, swap, odd = (r[0] for r in record)
    ctx = m.ctx
    MA.is_nonzero(m, g, active)
    ctx.cx(g[0], odd)
    ctx.x(z[-1])
    ctx.and_(odd, z[-1], swap)
    ctx.x(z[-1])

    for left, right in ((f, g), (fv, gv)):
        for a, b in zip(left, right):
            ctx.cswap(swap, a, b)

    _signed_neg(m, swap, g)
    _add_signed(m, odd, f, g)
    _cmodneg_small(m, swap, gv, p)
    _cmodadd_small(m, odd, fv, gv, p)
    _cmoddbl_small(m, active, fv, p)

    _signed_neg(m, swap, z)
    ctx.x(swap)
    increment = m.anc(1, "increment")
    ctx.and_(active, swap, increment[0])
    _add_const_small(m, increment[0], z, 1)
    ctx.and_dg(active, swap, increment[0])
    ctx.x(swap)
    m.free(increment)

    for i in range(len(g) - 1):
        ctx.swap(g[i], g[i + 1])
    ctx.cx(g[-2], g[-1])


def mod_inv(m, x, out, p, ctrl=None):
    """XOR the exact inverse of nonzero x into out; clean every ancilla.

    Limited to toy primes: the fixed schedule is exhaustively determined at
    construction time, not inferred from a 256-bit paper failure budget.
    """
    n = len(x)
    if (n != p.bit_length() or p < 3 or p > 127 or
            any(p % d == 0 for d in range(2, int(p ** 0.5) + 1))):
        raise ValueError("divsteps circuit requires a toy prime <= 127")
    rounds = max(walk(w, p)[1] for w in range(1, p))
    width = n + 1
    z_width = (rounds + 1).bit_length() + 1
    z = m.anc(z_width, "z")
    f, g = m.anc(width, "f"), m.anc(width, "g")
    fv, gv = m.anc(n, "fv"), m.anc(n, "gv")
    A.encode_const(m.ctx, f, p, ctrl)
    for src, dst in zip(x, g):
        if ctrl is None:
            m.ctx.cx(src, dst)
        else:
            m.ctx.ccx(ctrl, src, dst)
    if ctrl is None:
        m.ctx.x(gv[0])
    else:
        m.ctx.cx(ctrl, gv[0])

    start = m.begin()
    records = []
    for _ in range(rounds):
        record = tuple(m.anc(1, label) for label in ("active", "swap", "odd"))
        _round(m, z, f, g, fv, gv, p, record)
        records.append(record)
    body = m.since(start)

    inverse = m.anc(n, "inverse")
    for a, b in zip(fv, inverse):
        m.ctx.cx(a, b)
    correction = m.begin()
    _cmodneg_small(m, f[-1], inverse, p, g[0])
    # The terminated remainder is zero, so its wires can carry clean scratch.
    for record in reversed(records):
        m.emit_inverse(_cmoddbl_small, m, record[0][0], inverse, p,
                       g[:3])
    corrections = m.since(correction)
    for a, b in zip(inverse, out):
        m.ctx.cx(a, b)
    m.undo(corrections)
    for a, b in zip(fv, inverse):
        m.ctx.cx(a, b)
    m.free(inverse)
    m.undo(body)

    A.encode_const(m.ctx, f, p, ctrl)
    for src, dst in zip(x, g):
        if ctrl is None:
            m.ctx.cx(src, dst)
        else:
            m.ctx.ccx(ctrl, src, dst)
    if ctrl is None:
        m.ctx.x(gv[0])
    else:
        m.ctx.cx(ctrl, gv[0])
    m.free(z, f, g, fv, gv, *(bit for record in records for bit in record))
