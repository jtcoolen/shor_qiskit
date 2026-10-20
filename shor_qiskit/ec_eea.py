"""In-place modular multiplication via a Euclidean dialog -- [1128] Sec 3.

The expensive operation in a point addition is |x, y> -> |x, x y mod q>.
Everyone before [1128] built it out of a modular inversion plus an out-of-place
multiplication, ran the inversion forwards to get x^{-1}, and ran it backwards
to erase the garbage.  [1128], following [14], does something better.

The two ideas
-------------
1. Split the extended Euclidean algorithm in half.  Run the binary GCD on
   (u, v) alone and do *not* maintain the Bezout coefficients; just record the
   two decision bits per iteration.  That record -- the "dialog" -- is 1.5 bits
   per iteration on average, against the 2n qubits carrying (r, s) would cost.

2. Replay the record onto any pair you like.  The updates to (r, s) are linear
   and depend only on those two bits, so starting the replay at (y, 0) instead
   of (1, 0) ends at (0, y x mod q).  The inversion and the multiplication come
   out of one pass -- the multiplication is *free*.

And because the replay is a circuit, running it backwards multiplies by x^{-1}
instead, which is the other in-place multiplication the point addition needs.
One circuit, both steps.

Direction matters
-----------------
The record is replayed in *reverse* order.  Read forwards, the same update rules
compute a different linear map -- not the inverse.  `ec_classical` has both, and
the tests pin down which is which.

What is and is not implemented here
-----------------------------------
Implemented: the dialog (Alg. 2), the Bezout replay (Alg. 3), and in-place
multiplication and division (Alg. 4).  The streaming five-Toffoli compression
(Fig. 1) is implemented and tested but opt-in (`compress=True`): the measured
multiplier keeps the raw record.

Opt-in: the register *sharing* of Sec 3.1, where u and v shrink and freed
qubits absorb the record. This is probabilistic (c_pad ~ 2.3), not a drop-in
replacement for the exact default. Basis-state simulation checks every
shrinking boundary and raises on overflow rather than truncating. The circuit
uses the existing fixed five-bit/three-round encoding, so it does not claim the
paper's tighter 2.12n asymptotic record bound. `inplace_mul` remains exact and
uses its original layout.

The compressed record is emitted as each group of three rounds completes.
Its clean sixth wire is returned to the ancilla pool and reused by later
dialog/replay workspace; each packed group is decoded only while its bits are
used.
"""

import ec_adders as A
import ec_modarith as MA
from ec_classical import eea_iterations, eea_shrink_schedule
from ec_sim import Reg


# --- Algorithm 2: the dialog ------------------------------------------------
def dialog_round(m, u, v, rec):
    """One iteration of [1128] Algorithm 2.  `rec` = (b0, b0b1), kept.

    b0 is v's parity and b0b1 is "swap"; both have to be recorded because v is
    halved and u and v are swapped, so neither is recoverable afterwards.
    """
    ctx, n = m.ctx, len(u)
    b0, b0b1 = rec
    ctx.cx(v[0], b0[0])                              # b0 = v mod 2

    t, sc = m.anc(1, "t"), m.anc(n, "sc")
    A.gt_uint(ctx, u, v, t[0], sc)                   # t = [u > v]
    ctx.and_(b0[0], t[0], b0b1[0])                   # b0b1 = b0 AND b1
    A.gt_uint(ctx, u, v, t[0], sc)
    m.free(t, sc)

    for i in range(n):
        ctx.cswap(b0b1[0], u[i], v[i])               # if b0b1: swap u, v

    cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
    A.csub(ctx, b0[0], u, v, cp, sc)                 # if b0: v -= u
    m.free(cp, sc)

    for i in range(n - 1):                           # v /= 2 (v is even now)
        ctx.swap(v[i], v[i + 1])


def dialog(m, x, q, iters=None, ctrl=None, compress=False,
           register_sharing=False, c_pad=2.3, c_iter=2.4):
    """Consume x, produce the dialog. Returns (records, u, v).

    On termination u = 1 and v = 0, so x's own register comes back clean -- the
    value has been traded for the record. That is [1128] Alg. 4 step 1.
    ``register_sharing=True`` opts into Sec. 3.1's probabilistic layout; in that
    case u/v are only the final active slices and simulator cleanliness checks
    detect scheduled-width overflow.
    """
    ctx, n = m.ctx, len(x)
    iters = iters or eea_iterations(n, c_iter)
    assert ctrl is None, (
        "a controlled dialog is not implemented -- gate the caller instead. "
        "Unlike Kaliski, loading v <- 0 does not make the rounds inert here: "
        "the dialog would record a different bit string and the replay would "
        "no longer invert it.")
    if register_sharing:
        return dialog_register_sharing(
            m, x, q, iters=iters, compress=compress, c_pad=c_pad,
            c_iter=c_iter)

    hi = m.anc(1, "vhi")
    u = m.anc(n + 1, "u")
    v = Reg(list(x) + list(hi), "v")
    A.encode_const(ctx, u, q)                        # u <- q (odd)

    recs = []
    spares = []
    for _ in range(iters):
        rec = (
            m.anc(1, "b0", exclude=spares if compress else ()),
            m.anc(1, "b1", exclude=spares if compress else ()),
        )
        dialog_round(m, u, v, rec)
        recs.append(rec)
        if compress and len(recs) % 3 == 0:
            group = _record_wires(recs[-3:])
            compress_triple_circuit(m, group)
            spare = Reg([group[5]])
            m.free(spare)
            spares.append(spare)
    return recs, u, v


def dialog_register_sharing(m, x, q, iters=None, compress=True, c_pad=2.3,
                            c_iter=2.4):
    """Build [1128] Sec. 3.1's probabilistic shrinking-width EEA dialog.

    Wires above each scheduled u/v width are returned to the clean-ancilla
    pool and can hold dialog bits. ``Machine.free`` installs a simulator check
    at each boundary, so a basis-state overflow raises instead of being
    silently truncated. Returned u/v handles cover only the final active
    slices; upper wires may now hold dialog data.

    ``inplace_mul(..., register_sharing=True)`` uses this layout end to end.
    A quantum device cannot expose a non-destructive overflow flag for this
    probabilistic schedule; callers must validate the arithmetic result.
    """
    recs, u, v, widths = _build_dialog_register_sharing(
        m, x, q, iters, compress, c_pad, c_iter)
    return recs, u[:widths[-1]], v[:widths[-1]]


def _build_dialog_register_sharing(m, x, q, iters, compress, c_pad, c_iter):
    n = len(x)
    iters = iters or eea_iterations(n, c_iter)
    widths = shrink_schedule(n, c_pad, iters=iters, c_iter=c_iter)
    width_max = widths[0]
    if width_max < n:
        raise ValueError("register-sharing schedule is narrower than the input")

    u = m.anc(width_max, "u")
    vhi = m.anc(width_max - n, "vhi")
    v = Reg(list(x) + list(vhi), "v")
    A.encode_const(m.ctx, u, q)

    recs = []
    spares = []
    previous_width = width_max
    for width in widths:
        if width < previous_width:
            m.free(u[width:previous_width], v[width:previous_width])
        rec = (
            m.anc(1, "b0", exclude=spares if compress else ()),
            m.anc(1, "b1", exclude=spares if compress else ()),
        )
        dialog_round(m, u[:width], v[:width], rec)
        recs.append(rec)
        if compress and len(recs) % 3 == 0:
            group = _record_wires(recs[-3:])
            compress_triple_circuit(m, group)
            spare = Reg([group[5]])
            m.free(spare)
            spares.append(spare)
        previous_width = width
    return recs, u, v, widths


def dialog_undo(m, x, q, recs, u, v, iters=None, compressed=False):
    """Rebuild x: Algorithm 2 run backwards ([1128] Alg. 4 step 3)."""
    n = len(x)
    iters = iters or eea_iterations(n)
    assert len(recs) == iters
    if compressed:
        full = len(recs) - len(recs) % 3
        spares = [Reg([_record_wires(recs[i:i + 3])[5]])
                  for i in range(0, full, 3)]
        m.reserve(*spares)
        for i in reversed(range(full, len(recs))):
            m.emit_inverse(dialog_round, m, u, v, recs[i])
        for start in reversed(range(0, full, 3)):
            group_recs = recs[start:start + 3]
            m.emit_inverse(compress_triple_circuit, m, _record_wires(group_recs))
            for rec in reversed(group_recs):
                m.emit_inverse(dialog_round, m, u, v, rec)
    else:
        for rec in reversed(recs):
            m.emit_inverse(dialog_round, m, u, v, rec)
    m.emit_inverse(A.encode_const, m.ctx, u, q)


def clear_dialog_end(m, u, v):
    """After a dialog, u = 1 and v = 0: return both registers to |0>."""
    m.ctx.x(u[0])
    m.free(u)
    m.free(Reg([v[-1]]))              # the borrowed high bit of v


# --- Algorithm 3: the Bezout replay -----------------------------------------
def bezout_replay(m, r, s, recs, q, dbl=None, cadd=None, compressed=False):
    """(r, s) updated by [1128] Algorithm 3, reading the record in reverse.

    Started at (y, 0) this ends at (0, y x mod q).  `dbl` and `cadd` default to
    the exact modular doubling and controlled modular addition; `ec_approx`
    supplies the cheaper approximate ones.  The same `cadd` runs on every
    iteration, and once per multiplication it is handed r + s = q exactly (see
    `ec_approx`), so a pseudo-Mersenne `cadd` must be Algorithm 11, not 10.

    Returns the register that holds s on exit.  With the default doubling it
    is *not* the handle passed in: the replay owns s outright, so it uses
    `moddbl_rotate`, which lets the doubled value live one wire lower each
    round instead of paying n SWAPs to move it home.  Callers must adopt the
    returned handle.  A caller-supplied `dbl` is applied in place as before.
    """
    if dbl is None:
        state = {"s": s}

        def do_dbl():
            state["s"] = MA.moddbl_rotate(m, state["s"], q)
    else:
        state = {"s": s}

        def do_dbl():
            dbl(m, state["s"])
    cadd = cadd or (lambda mm, c, a, b: MA.cmodadd(mm, c, a, b, q))

    def replay_round(b0, b0b1):
        do_dbl()                                         # s <- 2s mod q
        s = state["s"]
        cadd(m, b0[0], r, s)                             # if b0: s += r mod q
        for i in range(len(r)):
            m.ctx.cswap(b0b1[0], r[i], s[i])             # if b0b1: swap r, s

    if compressed:
        full = len(recs) - len(recs) % 3
        for i in reversed(range(full, len(recs))):
            replay_round(*recs[i])
        for start in reversed(range(0, full, 3)):
            group_recs = recs[start:start + 3]
            group = _record_wires(group_recs)
            m.emit_inverse(compress_triple_circuit, m, group)
            for rec in reversed(group_recs):
                replay_round(*rec)
            compress_triple_circuit(m, group)
    else:
        for rec in reversed(recs):
            replay_round(*rec)
    return state["s"]


# --- Algorithm 4: in-place modular multiplication ---------------------------
def inplace_mul(m, x, y, q, iters=None, dbl=None, cadd=None,
                register_sharing=False, c_pad=2.3, c_iter=2.4, compress=False):
    """|x, y> -> |x, y*x mod q>.  [1128] Algorithm 4.

    Three phases: x becomes a dialog, the dialog is replayed onto (y, 0), the
    dialog becomes x again.  The answer lands in the replay's scratch half, so n
    SWAPs put it back in y's own qubits and the caller's handle stays valid --
    pure CNOT, no Toffoli.

    The rebuild also consumes the record: running a dialog round backwards
    un-copies b0 and un-ANDs b0b1, so every record qubit returns to |0> and the
    whole multiplication is garbage-free.

    ``register_sharing=True`` opts into the probabilistic Sec. 3.1 layout.
    Overflow is checked by the basis-state simulator at register-shrink
    boundaries; the default remains the exact fixed-width implementation.

    ``compress=True`` streams Fig. 1 packing through all three phases.  It is
    off by default: it costs 5 Toffolis per three rounds, each way, and at the
    widths simulated here the freed wire does not lower the peak qubit count
    (39 qubits either way at p = 7), so the measured multiplier is unpacked.
    """
    n = len(x)
    iters = iters or eea_iterations(n, c_iter)
    if register_sharing:
        return _inplace_mul_register_sharing(
            m, x, y, q, iters, dbl, cadd, c_pad, c_iter)
    if not compress:
        return _inplace_mul_plain(m, x, y, q, iters, dbl, cadd)

    recs, u, v = dialog(m, x, q, iters, compress=True)   # 1: x -> dialog
    clear_dialog_end(m, u, v)

    full = len(recs) - len(recs) % 3
    spares = [Reg([_record_wires(recs[i:i + 3])[5]])
              for i in range(0, full, 3)]
    m.reserve(*spares)
    s = m.anc(n, "bz")                                   # 2: replay onto (y, 0)
    s = bezout_replay(m, y, s, recs, q, dbl, cadd, compressed=True)
    for a, b in zip(y, s):                               # r is 0, s is the answer
        m.ctx.swap(a, b)
    m.free(s)

    hi = m.anc(1, "vhi")                                 # 3: dialog -> x
    u2 = m.anc(n + 1, "u")
    v2 = Reg(list(x) + list(hi), "v")
    m.ctx.x(u2[0])                                       # the dialog ends at u = 1
    for i in reversed(range(full, len(recs))):
        m.emit_inverse(dialog_round, m, u2, v2, recs[i])
        m.free(*recs[i])
    for start in reversed(range(0, full, 3)):
        group_recs = recs[start:start + 3]
        group = _record_wires(group_recs)
        m.emit_inverse(compress_triple_circuit, m, group)
        for rec in reversed(group_recs):
            m.emit_inverse(dialog_round, m, u2, v2, rec)
        m.free(Reg(group))
    A.encode_const(m.ctx, u2, q)                         # and starts at u = q
    m.free(u2, hi)


def _inplace_mul_plain(m, x, y, q, iters, dbl, cadd):
    """[1128] Algorithm 4 with the record kept as raw (b0, b0b1) pairs."""
    n = len(x)
    recs, u, v = dialog(m, x, q, iters)                  # 1: x -> dialog
    clear_dialog_end(m, u, v)

    s = m.anc(n, "bz")                                   # 2: replay onto (y, 0)
    s = bezout_replay(m, y, s, recs, q, dbl, cadd)
    for a, b in zip(y, s):                               # r is 0, s is the answer
        m.ctx.swap(a, b)
    m.free(s)

    hi = m.anc(1, "vhi")                                 # 3: dialog -> x
    u2 = m.anc(n + 1, "u")
    v2 = Reg(list(x) + list(hi), "v")
    m.ctx.x(u2[0])                                       # the dialog ends at u = 1
    for rec in reversed(recs):
        m.emit_inverse(dialog_round, m, u2, v2, rec)
        m.free(*rec)
    A.encode_const(m.ctx, u2, q)                         # and starts at u = q
    m.free(u2, hi)


def _inplace_mul_register_sharing(m, x, y, q, iters, dbl, cadd, c_pad, c_iter):
    """Run shared EEA, replay it, then invert the captured dialog gates.

    Shrinking-register wires can successively hold register bits, records, and
    replay workspace. After uncomputation, consolidate those clean ancillas
    back into the pool state that existed on entry.
    """
    start = m.begin()
    initial_qubits = set(m.qc.qubits)
    initial_pool = list(m._pool)
    live_before = m._live

    recs, u, v, _ = _build_dialog_register_sharing(
        m, x, q, iters, True, c_pad, c_iter)
    dialog_body = m.since(start)

    n = len(x)
    full = len(recs) - len(recs) % 3
    spares = [Reg([_record_wires(recs[i:i + 3])[5]])
              for i in range(0, full, 3)]
    m.reserve(*spares)
    s = m.anc(n, "bz")
    s = bezout_replay(m, y, s, recs, q, dbl, cadd, compressed=True)
    for a, b in zip(y, s):
        m.ctx.swap(a, b)
    m.free(s)

    m.undo(dialog_body)

    local = {qbit for ci in m.qc.data[start:] for qbit in ci.qubits}
    local.update(qbit for qbit in m.qc.qubits if qbit not in initial_qubits)
    inputs = set(x) | set(y)
    local_qubits = [
        qbit for qbit in m.qc.qubits if qbit in local and qbit not in inputs
    ]

    local_set = set(local_qubits)
    pooled_local = [qbit for qbit in m._pool if qbit in local_set]
    if pooled_local:
        m.reserve(Reg(pooled_local))
    if local_qubits:
        m.free(Reg(local_qubits))

    new_qubits = [qbit for qbit in m.qc.qubits if qbit not in initial_qubits]
    m._pool = initial_pool + new_qubits
    m._live = live_before


def inplace_div(m, x, y, q, iters=None, dbl=None, cadd=None,
                register_sharing=False, c_pad=2.3, c_iter=2.4, compress=False):
    """|x, y> -> |x, y*x^{-1} mod q>: `inplace_mul` run backwards."""
    m.emit_inverse(
        inplace_mul, m, x, y, q, iters, dbl, cadd,
        register_sharing=register_sharing, c_pad=c_pad, c_iter=c_iter,
        compress=compress)


# --- Fig. 1: compressing three (b0, b0&b1) pairs into five bits -------------
def _compress_bits(bits):
    q = list(bits)

    def cx(control, target):
        q[target] ^= q[control]

    def ccx(control_a, control_b, target):
        q[target] ^= q[control_a] & q[control_b]

    cx(1, 0); cx(3, 2); cx(5, 4); cx(0, 2); cx(5, 3)
    q[2] ^= 1
    q[4] ^= 1
    ccx(1, 3, 5)
    cx(1, 4)
    ccx(3, 4, 5)
    ccx(4, 5, 1)
    ccx(2, 5, 0)
    ccx(0, 1, 5)
    return q


def compress_triple_circuit(m, qs):
    """Apply [1128] Fig. 1 in place; five Toffolis and a clean sixth wire."""
    assert len(qs) == 6
    ctx = m.ctx
    ctx.cx(qs[1], qs[0])
    ctx.cx(qs[3], qs[2])
    ctx.cx(qs[5], qs[4])
    ctx.cx(qs[0], qs[2])
    ctx.cx(qs[5], qs[3])
    ctx.x(qs[2])
    ctx.x(qs[4])
    ctx.ccx(qs[1], qs[3], qs[5])
    ctx.cx(qs[1], qs[4])
    ctx.ccx(qs[3], qs[4], qs[5])
    ctx.ccx(qs[4], qs[5], qs[1])
    ctx.ccx(qs[2], qs[5], qs[0])
    ctx.ccx(qs[0], qs[1], qs[5])


def _record_wires(recs):
    return [q for rec in recs for reg in rec for q in reg]


def compression_permutation():
    """The 6-bit permutation of [1128] Fig. 1, as an explicit map.

    Only 27 of the 64 input patterns can occur -- (b0, b0&b1) is never (0, 1) --
    so three pairs fit in 5 bits with one to spare.  Over 1.413n iterations that
    is 2.355n record qubits instead of 2.83n, and unlike the shift-based
    encoding it leaves the record a *fixed* size, which is what lets the rest of
    the circuit be laid out statically.
    """
    return {
        code: sum(bit << i for i, bit in enumerate(
            _compress_bits([(code >> j) & 1 for j in range(6)])))
        for code in range(64)
    }


def compress_records(m, recs, free=False):
    """Compress complete record triples; optionally pool each clean spare."""
    freed = []
    for i in range(0, len(recs) - 2, 3):
        qs = _record_wires(recs[i:i + 3])
        compress_triple_circuit(m, qs)
        freed.append(qs[5])
        if free:
            m.free(Reg([qs[5]]))
    return freed


# --- space accounting -------------------------------------------------------
def dialog_width(n, c_iter=2.4):
    """Uncompressed record baseline: two qubits per iteration."""
    return 2 * eea_iterations(n, c_iter)


def dialog_width_compressed(n, c_iter=2.4):
    """Packed record used by `inplace_mul`: five bits per full triple."""
    it = eea_iterations(n, c_iter)
    return 2 * it - it // 3


def shrink_schedule(n, c_pad=2.3, iters=None, c_iter=2.4):
    """[1128] Sec 3.1: the width u and v would occupy at each iteration.

    n - 0.5 log2(8/3) i + c_pad sqrt(n). Used only by the opt-in
    ``dialog_register_sharing`` layout; the default circuit remains exact.
    """
    return eea_shrink_schedule(n, c_pad, iters, c_iter)
