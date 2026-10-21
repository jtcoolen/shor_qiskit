"""Toy-width reversible divsteps inversion using Gu et al. Eq. (6).

The arithmetic history mode uses signed coefficients and Appendix A's
bit-window decoder. The original modes retain modular coefficients and
either explicit history or a finite-toy-state lookup. None uses the
paper's packed wire layout.
"""

import ec_adders as A
import ec_modarith as MA
from ec_divsteps_reference import State, step, walk
import ec_divsteps_codec as codec
import ec_divsteps_arith_codec as arithmetic_codec


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


def _signed_width(p, *, include_noncanonical=False, max_rounds=None):
    """Bound signed intermediates over units or the bounded input domain."""
    largest = 1
    stop = 1 << p.bit_length() if include_noncanonical else p
    for w in range(1, stop):
        state = State(0, p, w, 0, 1)
        count = 0
        while state.g:
            if max_rounds is not None and count >= max_rounds:
                break
            fv, gv = state.fv, state.gv
            if state.g & 1 and state.z >= 0:
                fv, gv = gv, -fv
            largest = max(largest, abs(fv), abs(gv),
                          abs(gv + (state.g & 1) * fv), abs(2 * fv))
            state, _ = step(state)
            count += 1
    return largest.bit_length() + 2


def _round_depths(p, rounds, *, include_noncanonical=False):
    """Per-round valuation windows needed by the arithmetic decoder.

    Entry r is the largest valuation v_2(f_v') over ambiguous nonswap states
    after round r, enumerated over the same toy input domain as
    `_signed_width`; 1 means that round never needs an Eq. (A11) window.
    """
    depths = [1] * rounds
    stop = 1 << p.bit_length() if include_noncanonical else p
    for w in range(1, stop):
        state = State(0, p, w, 0, 1)
        for index in range(rounds):
            if not state.g:
                break
            state, _ = step(state)
            if state.fv % 4 != 2 and state.z <= 0 and state.g:
                valuation = (abs(state.fv) & -abs(state.fv)).bit_length() - 1
                depths[index] = max(depths[index], valuation)
    return depths


def _signed_double(m, ctrl, x):
    """Reversible signed double when the sign has one bit of headroom."""
    z = m.anc(1, "top")
    word = z + x
    for i in range(len(x), 0, -1):
        m.ctx.cswap(ctrl, word[i], word[i - 1])
    m.ctx.ccx(ctrl, x[-1], z[0])
    m.free(z)


def _round_signed(m, z, f, g, fv, gv, record):
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
    _signed_neg(m, swap, gv)
    _add_signed(m, odd, fv, gv)
    _signed_double(m, active, fv)
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


def _copy_signed_residue(m, signed, residue, p):
    """Add signed two's-complement coefficients into a reduced residue."""
    for i, control in enumerate(signed):
        weight = (-(1 << i) if i == len(signed) - 1 else 1 << i) % p
        MA.cmodadd_const(m, control, residue, weight, p)


def mod_inv(m, x, out, p, ctrl=None, history="full", *,
            round_limit=None, failure=None):
    """XOR the exact inverse of nonzero x into out; clean every ancilla.

    Limited to toy primes: the fixed schedule is exhaustively determined at
    construction time, not inferred from a 256-bit paper failure budget.
    A shorter arithmetic schedule requires a separate clean failure output:
    unsuccessful branches produce out=0, failure=1 rather than a false inverse.
    """
    n = len(x)
    if (n != p.bit_length() or p < 3 or p > 127 or
            any(p % d == 0 for d in range(2, int(p ** 0.5) + 1))):
        raise ValueError("divsteps circuit requires a toy prime <= 127")
    if history not in ("full", "decoded", "arithmetic"):
        raise ValueError(f"unknown divsteps history mode {history!r}")
    rounds = max(walk(w, p)[1] for w in range(1, p))
    if round_limit is not None:
        if (history != "arithmetic" or type(round_limit) is not int
                or not 1 <= round_limit <= rounds
                or failure is None or failure in x or failure in out
                or failure == ctrl):
            raise ValueError("bounded schedule requires arithmetic decoding, "
                             "a clean distinct failure wire and 1..exact rounds")
        rounds = round_limit
    width = n + 1
    z_width = (rounds + 1).bit_length() + 1
    z = m.anc(z_width, "z")
    f, g = m.anc(width, "f"), m.anc(width, "g")
    coeff_width = (_signed_width(p, include_noncanonical=round_limit is not None,
                                 max_rounds=rounds)
                   if history == "arithmetic" else n)
    depths = (_round_depths(p, rounds,
                            include_noncanonical=round_limit is not None)
              if history == "arithmetic" else None)
    fv, gv = m.anc(coeff_width, "fv"), m.anc(coeff_width, "gv")
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

    shared = (m.anc(1, "swap"), m.anc(1, "odd")) if history != "full" else None
    layout = (z, f, g, fv, gv)
    tables = codec.tables(p, tuple(map(len, layout)) + (1,)) if history == "decoded" else None
    start = m.begin()
    records = []
    bodies = []
    for index in range(rounds):
        active = m.anc(1, "active")
        record = (active, *shared) if shared else (
            active, m.anc(1, "swap"), m.anc(1, "odd"))
        mark = m.begin()
        if history == "arithmetic":
            _round_signed(m, z, f, g, fv, gv, record)
        else:
            _round(m, z, f, g, fv, gv, p, record)
        if shared:
            bodies.append(m.since(mark))
            if history == "arithmetic":
                arithmetic_codec.toggle(
                    m, z, g, fv, gv, active[0], record[1][0], record[2][0],
                    max_depth=depths[index])
            else:
                codec.toggle(m, (*layout, active),
                             (record[1][0], record[2][0]), tables[index])
        records.append(record)
    body = m.since(start) if not shared else None

    inverse = m.anc(n, "inverse")
    if history == "arithmetic":
        _copy_signed_residue(m, fv, inverse, p)
    else:
        for a, b in zip(fv, inverse):
            m.ctx.cx(a, b)
    correction = m.begin()
    _cmodneg_small(m, f[-1], inverse, p,
                   None if round_limit is not None else g[0])
    # The terminated remainder is zero, so its wires can carry clean scratch.
    for record in reversed(records):
        m.emit_inverse(_cmoddbl_small, m, record[0][0], inverse, p,
                       None if round_limit is not None else g[:3])
    corrections = m.since(correction)
    success = None
    if round_limit is not None:
        g_nonzero = m.anc(1, "unfinished")
        MA.is_nonzero(m, g, g_nonzero[0])
        geq, nz = m.anc(1, "out_of_range"), m.anc(1, "input_nonzero")
        cp, sc = m.anc(n, "constant"), m.anc(n, "compare")
        A.geq_const(m.ctx, x, p, geq[0], cp, sc)
        MA.is_nonzero(m, x, nz[0])
        m.ctx.x(nz[0])
        invalid = m.anc(1, "invalid")
        m.ctx.cx(geq[0], invalid[0])
        m.ctx.cx(nz[0], invalid[0])
        active_invalid = m.anc(1, "active_invalid")
        if ctrl is None:
            m.ctx.cx(invalid[0], active_invalid[0])
        else:
            m.ctx.and_(ctrl, invalid[0], active_invalid[0])
        m.ctx.cx(g_nonzero[0], failure)
        m.ctx.cx(active_invalid[0], failure)
        m.ctx.ccx(g_nonzero[0], active_invalid[0], failure)
        if ctrl is None:
            m.ctx.cx(invalid[0], active_invalid[0])
        else:
            m.ctx.and_dg(ctrl, invalid[0], active_invalid[0])
        m.free(active_invalid)
        m.ctx.cx(nz[0], invalid[0])
        m.ctx.cx(geq[0], invalid[0])
        m.free(invalid)
        m.ctx.x(nz[0])
        MA.is_nonzero(m, x, nz[0])
        A.geq_const(m.ctx, x, p, geq[0], cp, sc)
        m.free(nz, geq, cp, sc)
        MA.is_nonzero(m, g, g_nonzero[0])
        m.free(g_nonzero)
        success = m.anc(1, "success")
        m.ctx.x(success[0])
        m.ctx.cx(failure, success[0])
    for a, b in zip(inverse, out):
        if success is None:
            m.ctx.cx(a, b)
        else:
            m.ctx.ccx(success[0], a, b)
    if success is not None:
        m.ctx.cx(failure, success[0])
        m.ctx.x(success[0])
        m.free(success)
    m.undo(corrections)
    if history == "arithmetic":
        m.emit_inverse(_copy_signed_residue, m, fv, inverse, p)
    else:
        for a, b in zip(fv, inverse):
            m.ctx.cx(a, b)
    m.free(inverse)
    if shared:
        for index in range(rounds - 1, -1, -1):
            record = records[index]
            if history == "arithmetic":
                arithmetic_codec.toggle(
                    m, z, g, fv, gv, record[0][0], shared[0][0], shared[1][0],
                    max_depth=depths[index])
            else:
                codec.toggle(m, (*layout, record[0]),
                             (shared[0][0], shared[1][0]), tables[index])
            m.undo(bodies[index])
    else:
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
    if shared:
        m.free(z, f, g, fv, gv, shared[0], shared[1],
               *(record[0] for record in records))
    else:
        m.free(z, f, g, fv, gv,
               *(bit for record in records for bit in record))
