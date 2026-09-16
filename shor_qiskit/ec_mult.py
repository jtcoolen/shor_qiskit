"""Modular multiplication: the schoolbook double-and-add reference.

This is the baseline both papers optimise away from -- [106] replaces it with
word-level Montgomery over a carry-save tree (`ec_montgomery`), [1128] replaces
it with a Euclidean dialog and Bezout replay (`ec_eea`).  Keeping the plain
version around is what makes those two testable: they must agree with this on
every input.

Cost here is n modular doublings plus n controlled modular additions, i.e.
O(n^2) Toffoli, with no Montgomery representation and no approximation.
"""

import ec_modarith as MA


def copy_reg(ctx, src, dst):
    """dst ^= src, bitwise.  Free (CNOTs only)."""
    assert len(src) == len(dst)
    for a, b in zip(src, dst):
        ctx.cx(a, b)


def modmul(m, x, y, z, p):
    """z (which must be |0>) <- x*y mod p.  x and y are preserved.

    Double-and-add over the bits of x, most significant first: the accumulator
    doubles and conditionally takes y, so y itself never moves.  That is why
    this direction is chosen over doubling y -- it saves undoing n doublings.
    """
    n = len(x)
    assert len(z) == len(y) == n
    for i in reversed(range(n)):
        MA.moddbl(m, z, p)
        MA.cmodadd(m, x[i], y, z, p)


def modmul_const(m, x, k, z, p):
    """z (|0>) <- k*x mod p for a classical k.  x preserved.

    Cheaper than a full multiplication: the accumulator doubles once per bit of
    k, but takes x only on k's set bits, so it is bitlen(k) doublings and
    popcount(k) additions instead of n of each.  Used for the two steps of the
    projective addition where the addend point is a compile-time constant.
    """
    k %= p
    if k == 0:
        return
    for i in reversed(range(k.bit_length())):
        MA.moddbl(m, z, p)
        if (k >> i) & 1:
            MA.modadd(m, x, z, p)


def cmodmul(m, ctrl, x, y, z, p):
    """z (|0>) <- x*y mod p if ctrl, else stays 0."""
    n = len(x)
    anc = m.anc(1, "cm")
    for i in reversed(range(n)):
        MA.moddbl(m, z, p)
        m.ctx.and_(ctrl, x[i], anc[0])
        MA.cmodadd(m, anc[0], y, z, p)
        m.ctx.and_dg(ctrl, x[i], anc[0])
    m.free(anc)


def modsqr(m, x, z, p):
    """z (|0>) <- x^2 mod p.  Needs n scratch qubits for the second operand."""
    n = len(x)
    t = m.anc(n, "sq")
    copy_reg(m.ctx, x, t)
    modmul(m, t, x, z, p)
    copy_reg(m.ctx, x, t)
    m.free(t)


# --- accumulate forms: compute the product, use it, uncompute it -------------
def _with_product(m, x, y, p, use):
    """Compute x*y into scratch, hand it to `use`, then uncompute it.

    Two multiplications per accumulate.  That is the standard price -- [106]
    Fig. 7 shows exactly this shape (Mul then Mul-dagger around the payload) --
    and it is what makes the accumulate forms reversible with no garbage.
    """
    n = len(x)
    t = m.anc(n, "prod")
    modmul(m, x, y, t, p)
    use(t)
    m.emit_inverse(modmul, m, x, y, t, p)
    m.free(t)


def modmul_add(m, x, y, acc, p):
    """acc <- (acc + x*y) mod p.  [106] Alg. 3 step 7 ("Mul+")."""
    _with_product(m, x, y, p, lambda t: MA.modadd(m, t, acc, p))


def modmul_sub(m, x, y, acc, p):
    """acc <- (acc - x*y) mod p.  [106] Alg. 3 step 6 ("Mul-")."""
    _with_product(m, x, y, p, lambda t: MA.modsub(m, t, acc, p))


def modmul_xor(m, x, y, acc, p):
    """acc ^= (x*y mod p), bitwise.  [106] Alg. 3 step 4 ("Mul-xor").

    Used only where the accumulator is *known* to already equal the product, so
    the XOR zeroes it.  Bitwise XOR is CNOTs -- far cheaper than a modular
    subtraction -- and the caller's algebra guarantees the result.
    """
    _with_product(m, x, y, p, lambda t: copy_reg(m.ctx, t, acc))


def modsqr_sub(m, x, acc, p):
    """acc <- (acc - x^2) mod p.  [106] Alg. 3 step 6 with lambda twice."""
    n = len(x)
    t = m.anc(n, "sq")
    copy_reg(m.ctx, x, t)
    modmul_sub(m, t, x, acc, p)
    copy_reg(m.ctx, x, t)
    m.free(t)
