"""Out-of-place point addition in Jacobian-affine coordinates -- [106] Sec 4.2.

Algorithm 4, eleven multiplications and no inversion.  Mixed coordinates: the
accumulator is (X1 : Y1 : Z1), the addend is a classical (or looked-up) affine
point with Z2 = 1, and every term involving Z2 in the general Jacobian addition
law drops out.  [106] claims the fewest multiplications of any quantum
projective addition -- 11 against the previous 12.

The trade
---------
No inversion is a big win: `ec_kaliski` is the most expensive thing in the
affine circuit by a wide margin.  The cost is that projective coordinates are
not a unique representation, so (P + Q) - Q comes back as a *different triple*
for the same point and the input cannot be erased.  Every addition therefore
leaves its input behind.

What can and cannot be cleaned
------------------------------
The fifteen intermediates -- Z1^2, U2, Z1^3, H, H^2, H^3, S2, R, R^2, V, 2V,
T0, T1, T2, T3 -- all *can* be cleaned, and this module cleans them.  Each is
uncomputed by reversing the step that made it, and every such step's inputs are
still live at that point.  The order is not simply "backwards": X3 is an output
and must survive, so T0 is unwound via step 12 (from R^2 and H^3) rather than
via step 13.  `Machine.undo` exists for exactly this.

So the garbage is 3n qubits per addition -- the input point, and nothing else.
That is the G of [106] Sec 4.2, and the two mitigations in that section are
about reducing how many G's are live at once:

  windowing   2*ceil((n+1)/w) additions instead of 2n+2, so 2*ceil((n+1)/w) G
  zig-zag     m registers serve them all, so m*G  (see `ec_classical.zigzag_*`)

Correctness caveat: like the affine circuit, this one assumes the addition is
generic.  H = 0 (equal x-coordinates) makes the formulas degenerate.
"""

import ec_modarith as MA
import ec_mult as MU
from ec_classical import zigzag_schedule


def _copy(m, src):
    """A fresh register holding a copy of `src` -- squaring needs two operands."""
    t = m.anc(len(src), "cp")
    for a, b in zip(src, t):
        m.ctx.cx(a, b)
    return t


def jacobian_add(m, X1, Y1, Z1, x2, y2, X3, Y3, Z3, p):
    """(X3:Y3:Z3) <- (X1:Y1:Z1) + (x2, y2).  [106] Algorithm 4.

    X3, Y3, Z3 must be |0>.  X1, Y1, Z1 survive unchanged -- they are the
    garbage.  Every other scratch register is returned to |0>.
    """
    n = len(X1)
    A = m.anc

    def mul(a, b, out):
        MU.modmul(m, a, b, out, p)

    def sqr(a, out):
        c = _copy(m, a)
        MU.modmul(m, c, a, out, p)
        for q, r in zip(a, c):
            m.ctx.cx(q, r)
        m.free(c)

    def sub_into(a, b, out):
        """out (|0>) <- a - b mod p."""
        for q, r in zip(a, out):
            m.ctx.cx(q, r)
        MA.modsub(m, b, out, p)

    Z1sq = A(n, "Z1sq"); _, b1 = m.step(sqr, Z1, Z1sq)                    # 1
    U2 = A(n, "U2");     _, b2 = m.step(MU.modmul_const, m, Z1sq, x2, U2, p)   # 2
    Z1cu = A(n, "Z1cu"); _, b3 = m.step(mul, Z1sq, Z1, Z1cu)              # 3
    H = A(n, "H");       _, b4 = m.step(sub_into, U2, X1, H)              # 4
    Hsq = A(n, "Hsq");   _, b5 = m.step(sqr, H, Hsq)                      # 5
    Hcu = A(n, "Hcu");   _, b6 = m.step(mul, Hsq, H, Hcu)                 # 6
    S2 = A(n, "S2");     _, b7 = m.step(MU.modmul_const, m, Z1cu, y2, S2, p)   # 7
    R = A(n, "R");       _, b8 = m.step(sub_into, S2, Y1, R)              # 8
    Rsq = A(n, "Rsq");   _, b9 = m.step(sqr, R, Rsq)                      # 9
    V = A(n, "V");       _, b10 = m.step(mul, X1, Hsq, V)                 # 10

    twoV = A(n, "2V")
    def mk2V():
        for q, r in zip(V, twoV):
            m.ctx.cx(q, r)
        MA.modadd(m, V, twoV, p)
    _, b11 = m.step(mk2V)                                                 # 11

    T0 = A(n, "T0");     _, b12 = m.step(sub_into, Rsq, Hcu, T0)          # 12
    _, b13 = m.step(sub_into, T0, twoV, X3)                        # 13  OUTPUT
    T1 = A(n, "T1");     _, b14 = m.step(sub_into, V, X3, T1)             # 14
    T2 = A(n, "T2");     _, b15 = m.step(mul, R, T1, T2)                  # 15
    T3 = A(n, "T3");     _, b16 = m.step(mul, Y1, Hcu, T3)                # 16
    _, b17 = m.step(sub_into, T2, T3, Y3)                          # 17  OUTPUT
    _, b18 = m.step(mul, Z1, H, Z3)                                # 18  OUTPUT

    # unwind everything but the three outputs.  Note T0 goes back via step 12,
    # not step 13 -- step 13 wrote X3, which stays.
    for body in (b16, b15, b14, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1):
        m.undo(body)
    m.free(T3, T2, T1, T0, twoV, V, Rsq, R, S2, Hcu, Hsq, H, Z1cu, U2, Z1sq)


def jacobian_add_inv(m, X1, Y1, Z1, x2, y2, X3, Y3, Z3, p):
    """The reverse addition: clears (X3, Y3, Z3) given its input point.

    This is the PA-dagger of [106] Fig. 10 -- the operation the zig-zag schedule
    uses to hand a register back.
    """
    m.emit_inverse(jacobian_add, m, X1, Y1, Z1, x2, y2, X3, Y3, Z3, p)


# --- the zig-zag schedule, applied ------------------------------------------
def zigzag_chain(m, start, points, p, n, m_regs=None):
    """Run a chain of point additions under the schedule of [106] Fig. 10.

    `start` is (X0, Y0, Z0); `points` is the list of classical addends.  Returns
    (final register triple, list of registers still live).

    Registers are recycled by the schedule, so the peak is m triples rather than
    one per addition -- at the cost of re-running additions backwards.  The
    residual live set is the m intermediate points the schedule cannot clear;
    `ecdlp_projective` deals with them.
    """
    tape, m_regs, residual = zigzag_schedule(len(points), m_regs)
    pool, live = [], {0: start}
    for op, j in tape:
        if op == "add":
            if pool:
                trip = pool.pop()
            else:
                trip = (m.anc(n, "Xp"), m.anc(n, "Yp"), m.anc(n, "Zp"))
            src = live[j - 1]
            x2, y2 = points[j - 1]
            jacobian_add(m, *src, x2, y2, *trip, p)
            live[j] = trip
        else:
            src = live[j - 1]
            x2, y2 = points[j - 1]
            jacobian_add_inv(m, *src, x2, y2, *live[j], p)
            pool.append(live.pop(j))
    final = max(live)
    return live[final], [live[k] for k in live if k not in (0, final)], m_regs
