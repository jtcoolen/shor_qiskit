"""Arithmetic Appendix A divsteps branch decoder for signed coefficients.

Unlike the toy-state lookup, this circuit uses the two's-complement integer
coefficients of Gu et al. Eq. (6), not residues modulo p. For an ambiguous
step it evaluates Eq. (A11) with a reversible bitwise inverse modulo 2**j.
Circuit generation depends on register widths, never on enumerated inputs.
"""

import ec_adders as A
import ec_modarith as MA


def _low_product(m, left, right, width):
    """Compute a truncated integer product, including carries, into scratch."""
    product = m.anc(width, "product")
    mark = m.begin()
    for shift, control in enumerate(right[:width]):
        carry = m.anc(1, "product_carry")
        A.cdkm_add(m.ctx, left[:width - shift], product[shift:],
                   carry[0], ctrls=(control,))
        m.free(carry)
    return product, m.since(mark)


def _inverse_bit(m, q, inverse, k):
    product, body = _low_product(m, q, inverse[:k], k + 1)
    m.ctx.cx(product[k], inverse[k])
    m.undo(body)
    m.free(product)


def _window(m, fv, gv, depth, target, enabled):
    """Toggle the ambiguous decision at valuation depth using Eq. (A11)."""
    ctx = m.ctx
    q = m.anc(depth, "odd_part")
    for i in range(depth):
        source = fv[depth + i] if depth + i < len(fv) else fv[-1]
        ctx.cx(source, q[i])
    inverse = m.anc(depth, "odd_inverse")
    ctx.x(inverse[0])
    for k in range(1, depth):
        _inverse_bit(m, q, inverse, k)
    product, body = _low_product(m, inverse, gv, depth)
    ctx.ccx(enabled, product[-1], target)
    m.undo(body)
    m.free(product)
    for k in range(depth - 1, 0, -1):
        _inverse_bit(m, q, inverse, k)
    ctx.x(inverse[0])
    m.free(inverse)
    for i in range(depth):
        source = fv[depth + i] if depth + i < len(fv) else fv[-1]
        ctx.cx(source, q[i])
    m.free(q)


def toggle(m, z, g, fv, gv, active, swap, odd, max_depth=None):
    """XOR a round's (swap, odd) decisions from its updated signed state.

    The active bit distinguishes a frozen round from the terminating one.
    Inputs must be reachable states of the signed Eq. (6) recurrence, with
    coefficients fitting the supplied two's-complement registers.
    """
    if (len(z) < 2 or len(fv) < 4 or len(gv) != len(fv)
            or not g or len(set(z + g + fv + gv + [active, swap, odd]))
            != len(z) + len(g) + len(fv) + len(gv) + 3):
        raise ValueError("decoder needs disjoint signed divsteps registers")
    if max_depth is None:
        max_depth = len(fv) - 1
    if not 0 <= max_depth < len(fv):
        raise ValueError("depth bound must fit the signed coefficient word")
    ctx = m.ctx
    g_nonzero = m.anc(1, "g_nonzero")
    MA.is_nonzero(m, g, g_nonzero[0])
    z_nonzero = m.anc(1, "z_nonzero")
    MA.is_nonzero(m, z, z_nonzero[0])
    nonpositive = m.anc(1, "nonpositive")
    ctx.x(z[-1])
    ctx.and_(z_nonzero[0], z[-1], nonpositive[0])
    ctx.x(z[-1])
    ctx.x(nonpositive[0])

    ctx.ccx(active, fv[1], swap)
    ctx.ccx(active, fv[1], odd)
    ctx.x(g_nonzero[0])
    ctx.x(fv[1])
    ctx.mcx((active, g_nonzero[0], nonpositive[0], fv[1]), odd)
    ctx.x(fv[1])
    ctx.x(g_nonzero[0])

    for depth in range(2, max_depth + 1):
        for wire in fv[:depth]:
            ctx.x(wire)
        selected = m.anc(1, "depth")
        ctx.mcx((active, g_nonzero[0], nonpositive[0],
                 *fv[:depth + 1]), selected[0])
        _window(m, fv, gv, depth, odd, selected[0])
        ctx.mcx((active, g_nonzero[0], nonpositive[0],
                 *fv[:depth + 1]), selected[0])
        m.free(selected)
        for wire in fv[:depth]:
            ctx.x(wire)

    ctx.x(nonpositive[0])
    ctx.x(z[-1])
    ctx.and_dg(z_nonzero[0], z[-1], nonpositive[0])
    ctx.x(z[-1])
    m.free(nonpositive)
    MA.is_nonzero(m, z, z_nonzero[0])
    MA.is_nonzero(m, g, g_nonzero[0])
    m.free(z_nonzero, g_nonzero)
