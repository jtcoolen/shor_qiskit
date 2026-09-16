"""Word-level Montgomery multiplication -- [106] Sec 3.2, and its baseline.

Two multipliers, because the paper's contribution is only visible against what
it replaces:

`mont_mul_lookup`  [HJN+20].  Per word, one quantum lookup indexed by the low
    word of the accumulator loads the whole multiple M*p, and a single addition
    applies it.  The w controlled additions of the bit-level scheme collapse to
    one -- at the price of building a 2^w-entry table.  Reuses this package's
    existing unary-iteration QROM.

`mont_mul_qcsa`    [106].  Algorithm 1(a) is rewritten as Algorithm 1(b):

        M = c_w p'_w mod 2^w ;  c += M p        becomes    c += c_w d,
        d = (p'_w mod 2^w) p  precomputed classically

    so the reduction step has the *same shape* as the multiplication step --
    "load w shifted copies of a constant under w control bits, then sum them" --
    and one carry-save tree serves both.  No lookup table at all.

    The reformulation is not free, and the paper does not dwell on it.  M p and
    c_w d differ by 2^w * floor(c_w p'_w / 2^w) * p, and with both c_w and p'_w
    below 2^w that reaches about 2^{2w} p -- so the reduction over-adds by up to
    2^{2w} p, and the accumulator settles at n + 2w bits rather than the n + 2 of
    textbook Montgomery.  A final reduction mops that up.  `_acc_width` and the
    tests measure the width rather than trusting it.

Garbage.  Both multipliers keep the pre-reduction low word of each round.  They
have to: the reduction drives those bits to zero, which is what makes the shift
legal, and having driven them to zero it can no longer tell you what they were.
That is n qubits over the whole multiplication -- and it costs nothing extra in
space, because the shift frees exactly w qubits per round for them to live in.
It is uncomputed when the caller runs the multiplier backwards, which is the
Mul / Mul-dagger pairing of [106] Fig. 7.
"""

import ec_adders as A
import ec_qcsa as Q
from ec_classical import mont_d_constant
from ec_sim import Reg


def _acc_width(n, w):
    """[106] Alg. 1(b) settles the accumulator at n + 2w bits; +2 for headroom."""
    return n + 2 * w + 2


def _reduce_below_p(m, c, p, extra):
    """c (< p * 2^extra) -> c mod p, by binary conditional subtraction.

    The comparison flags cannot be cleared -- c after the step does not say
    whether the step fired -- so they are returned as garbage, which is fine
    here: a multiplier is always used compute / use / uncompute.
    """
    ctx = m.ctx
    W = len(c)
    flags = []
    cp, sc = m.anc(W, "rcp"), m.anc(W, "rsc")
    for j in reversed(range(extra)):
        t = m.anc(1, "rf")
        A.geq_const(ctx, c, p << j, t[0], cp, sc)
        A.csub_const(ctx, t[0], c, p << j, cp, sc)
        flags.append(t)
    m.free(cp, sc)
    return flags


def _load_shifted(m, ctrl_bits, const_or_reg, W, classical):
    """The "Set input" phase of [106] Fig. 5.

    w registers, register j holding the operand shifted up by j when control
    bit j is set.  For a classical operand that is CNOTs from the control; for a
    quantum one it is ANDs onto a clean target.  Self-inverse in the classical
    case, and undone by AND-dagger in the quantum case.
    """
    ops = []
    for j, cbit in enumerate(ctrl_bits):
        r = m.anc(W, "csain")
        if classical:
            k = const_or_reg
            for b in range(k.bit_length()):
                if (k >> b) & 1 and j + b < W:
                    m.ctx.cx(cbit, r[j + b])
        else:
            for b, q in enumerate(const_or_reg):
                if j + b < W:
                    m.ctx.and_(cbit, q, r[j + b])
        ops.append(r)
    return ops


def _unload_shifted(m, ctrl_bits, const_or_reg, ops, W, classical):
    for j, cbit in enumerate(ctrl_bits):
        r = ops[j]
        if classical:
            k = const_or_reg
            for b in range(k.bit_length()):
                if (k >> b) & 1 and j + b < W:
                    m.ctx.cx(cbit, r[j + b])
        else:
            for b, q in enumerate(const_or_reg):
                if j + b < W:
                    m.ctx.and_dg(cbit, q, r[j + b])
        m.free(r)


def mont_mul_qcsa(m, a, b, out, p, w):
    """out (|0>) <- a*b*2^-n mod p.  [106] Sec 3.2.  Returns the garbage.

    a and b are quantum and preserved.  n must be a whole number of w-bit words.
    """
    n = len(b)
    assert len(a) == n and n % w == 0, "n must be a whole number of words"
    s, W = n // w, _acc_width(n, w)
    d = mont_d_constant(p, w)

    c = m.anc(W, "mc")
    garbage = []
    for i in range(s):
        awrd = Reg(list(a[i * w:(i + 1) * w]), "aw")

        ops = _load_shifted(m, awrd, b, W, classical=False)     # c += a_i * b
        Q.qcsa_acc(m, ops, c)
        _unload_shifted(m, awrd, b, ops, W, classical=False)

        cw = m.anc(w, "cw")                                     # keep the low word
        for j in range(w):
            m.ctx.cx(c[j], cw[j])
        garbage.append(cw)

        ops = _load_shifted(m, cw, d, W, classical=True)        # c += c_w * d
        Q.qcsa_acc(m, ops, c)
        _unload_shifted(m, cw, d, ops, W, classical=True)

        # the low word is now zero, so the shift is a rotation: those w qubits
        # come back at the top, clean, and cost nothing
        c = Reg(list(c[w:]) + list(c[:w]), "mc")

    garbage += _reduce_below_p(m, c, p, W - p.bit_length())
    for i in range(len(out)):
        m.ctx.cx(c[i], out[i])
    garbage.append(c)
    return garbage


def mont_mul_lookup(m, a, b, out, p, w):
    """out (|0>) <- a*b*2^-n mod p.  [HJN+20]: a QROM lookup per word.

    The table has 2^w entries, entry i being ((i p'_w) mod 2^w) p -- the exact
    multiple of p that clears the low word.  Being exact is what keeps the
    accumulator at n + 2 bits instead of n + 2w.
    """
    from qrom import lookup_ui

    n = len(b)
    assert len(a) == n and n % w == 0
    s = n // w
    pw_prime = (-pow(p, -1, 1 << w)) % (1 << w)
    table = [((i * pw_prime) % (1 << w)) * p for i in range(1 << w)]
    tw = max(t.bit_length() for t in table)
    W = n + w + 2

    c = m.anc(W, "mc")
    one = m.anc(1, "one")
    m.ctx.x(one[0])
    garbage = []
    for i in range(s):
        awrd = Reg(list(a[i * w:(i + 1) * w]), "aw")
        ops = _load_shifted(m, awrd, b, W, classical=False)     # c += a_i * b
        Q.qcsa_acc(m, ops, c)
        _unload_shifted(m, awrd, b, ops, W, classical=False)

        cw = m.anc(w, "cw")
        for j in range(w):
            m.ctx.cx(c[j], cw[j])
        garbage.append(cw)

        mp = m.anc(tw, "Mp")                                    # lookup M*p
        anc = m.anc(max(w, 1), "lk")
        lookup_ui(m.ctx, one[0], cw, mp, table, anc)
        m.free(anc)
        adder = m.anc(W, "ad")
        A.add(m.ctx, Reg(list(mp) + list(m.anc(W - tw, "z"))), c, adder)
        m.free(adder)
        garbage.append(mp)

        c = Reg(list(c[w:]) + list(c[:w]), "mc")

    garbage += _reduce_below_p(m, c, p, W - p.bit_length())
    for i in range(len(out)):
        m.ctx.cx(c[i], out[i])
    m.ctx.x(one[0])
    m.free(one)
    garbage.append(c)
    return garbage


def mont_mul_clean(m, a, b, out, p, w, kind="qcsa"):
    """out (|0>) <- a*b*2^-n mod p, with every scratch qubit returned to |0>.

    Compute, copy the answer out, uncompute -- the shape of [106] Fig. 7.  Two
    multiplications for one clean result, which is the standard price and the
    reason the papers count multiplications rather than gates.
    """
    fn = mont_mul_qcsa if kind == "qcsa" else mont_mul_lookup
    n = len(b)
    tmp = m.anc(n, "mt")
    start = len(m.qc.data)
    garbage = fn(m, a, b, tmp, p, w)
    body = list(m.qc.data[start:])
    for i in range(len(out)):
        m.ctx.cx(tmp[i], out[i])
    for ci in reversed(body):
        m.qc.append(ci.operation.inverse(), ci.qubits, ci.clbits)
    m.free(tmp, *garbage)
