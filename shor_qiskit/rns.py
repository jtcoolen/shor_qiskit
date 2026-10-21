"""Gidney 2025's approximate residue-number modular exponentiation, as a circuit.

[G25] Appendix A.1, transcribed loop for loop onto this package's `Machine`
(see `rns_classical` for the algorithm and every table).  One shot computes

    result <- mask + sum_{p in P} truncate(V_p u_p mod L mod N)   (mod trunc)
          ~  mask + (g^e mod N) >> t

with the exponent e (m qubits) and the f + 1 qubit result register the only
large registers; everything per prime lives in ~l = 21 qubits.

The building blocks and what they cost (as [G25] Table 3 counts them):

  lookup            `ec_mbu.lookup(anf=True)`: 2^a - a - 1 ANDs; unloaded by
                    X-measurement, the repair either merged (loop1's, one
                    phaseup per exponent window at the very end) or done at
                    once (~2 sqrt(2^a) ANDs)
  addition          Gidney's adder, r - 1 ANDs on r qubits
  sub mod q         [G25]'s subtract-and-underflow: acc -= T with an extra top
                    qubit, add q back on the borrow, and clear the borrow by
                    X-measurement -- on outcome 1 (half the time) a comparison
                    phase against a looked-up threshold (lookup + addition)
  loop2             binary long division: subtract a constant, add it back on
                    the sign, per step
  loop3             windowed multiplications mod p; each old result is
                    *measured out* rather than uncomputed, its phase repaired
                    later by unloop3, which recomputes it anyway

Measured-out registers are `ClearGate`s: the price is the measurement (and the
probability-weighted repair), the unitary definition recomputes the value so
that the basis-state simulator can check everything clears.  One simplification
against [G25]: loop3's lookup repairs are done in place rather than merged
into unloop3's (a phaseup of 2 w3 address bits, ~2% of a prime's cost).
"""

from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister

import ec_adders as A
import ec_mbu as MB
import rns_classical as RC
from ec_gates import Ctx
from ec_sim import Machine, Reg


# =============================================================================
# Clearing a register by measurement
# =============================================================================
class ClearGate(Gate):
    """Clear `out` (the last nout qubits), which holds a function of the data
    qubits, by X-measurement.  `recompute(qc, qs)` XORs the function back in:
    it is the unitary definition (for the basis-state simulator) and the
    inverse.  ec_cost: `fix` Toffolis firing with probability `p_fire`."""

    def __init__(self, nq, nout, recompute, fix, p_fire, rc_cost, label="rns_clear"):
        super().__init__("rns_clear", nq, [], label=label)
        self.recompute, self.rc_cost = recompute, rc_cost
        self.ec_cost = {"toffoli": fix, "measure": nout, "p_fire": p_fire}

    def _define(self):
        qc = QuantumCircuit(QuantumRegister(self.num_qubits, "q"))
        self.recompute(qc, list(qc.qubits))
        self.definition = qc

    def inverse(self, annotated=False):
        return _Recompute(self)


class _Recompute(Gate):
    def __init__(self, fwd):
        super().__init__("rns_recompute", fwd.num_qubits, [])
        self.fwd = fwd
        self.ec_cost = {"toffoli": fwd.rc_cost, "measure": 0}

    def _define(self):
        self.definition = self.fwd.definition

    def inverse(self, annotated=False):
        return self.fwd


def clear(m, data, out, recompute, fix=0, p_fire=0.5, rc_cost=0, nanc=0, label="rns_clear"):
    anc = m.anc(nanc, "clr") if nanc else []
    g = ClearGate(len(data) + len(out) + nanc, len(out), recompute, fix, p_fire, rc_cost, label)
    m.qc.append(g, list(data) + list(out) + list(anc))
    if nanc:
        m.free(anc)


def _table_xor(qc, addr, out, table, anc):
    """out ^= table[addr], a plain unary-iteration lookup on the given
    ancillas (only inside definitions: this is what a clearing recomputes)."""
    from ec_mbu import LookupGate
    k = MB.walk_ancillas(len(addr), False)
    g = LookupGate(len(addr), len(out), table, False, k)
    qc.append(g, list(addr) + list(out) + list(anc[:k]))


def _lk_anc(a):
    return MB.walk_ancillas(a, False)


# =============================================================================
# Subtract-and-underflow modulo q, borrow cleared by measurement
# =============================================================================
def sub_wrap(m, acc, addr, table, q, fix=True):
    """acc (value qubits + one top qubit, value < q) <- acc - table[addr] mod q,
    with 0 <= table[a] <= q.  [G25] Sec 2.4 / A.1:

        acc -= T;  if borrow: acc += q;  measure the borrow out.

    After the correction the borrow equals [acc >= q - T], so its phase repair
    (outcome 1, half the time) is a lookup of q - T and a comparison.  With
    fix=False the repair is deferred (loop3: [G25] merges it into unloop3's)."""
    ctx = m.ctx
    val, top = Reg(acc[:-1]), acc[-1]
    r = len(val)
    T = m.anc(r, "T")
    MB.lookup(m, addr, T, table, anf=True)
    z = m.anc(1, "z")
    sc = m.anc(r, "sb")
    A.sub(ctx, Reg(list(T) + list(z)), acc, sc)                 # borrow -> top
    m.free(sc, z)
    cp, sc = m.anc(r, "cp"), m.anc(r, "sc")
    A.cadd_const(ctx, top, val, q, cp, sc)                       # + q on the borrow
    m.free(cp, sc)
    MB.unlookup(m, addr, T, table)
    m.free(T)
    thr = [q - v for v in table]
    a = len(addr)
    k = _lk_anc(a)

    def recompute(qc, qs):
        vv, aa, fl, anc = qs[:r], qs[r:r + a], qs[r + a], qs[r + a + 1:]
        tmp, sc_, lk = anc[:r], anc[r:2 * r], anc[2 * r:]
        _table_xor(qc, aa, tmp, thr, lk)
        A.lt_uint(Ctx(qc, "and"), vv, tmp, fl, sc_)             # fl ^= [acc < q - T]
        qc.x(fl)                                                  # fl ^= [acc >= q - T]
        _table_xor(qc, aa, tmp, thr, lk)

    cost = RC.lookup_toffolis(a) + r - 1 if fix else 0
    clear(m, list(val) + list(addr), [top], recompute, fix=cost, p_fire=0.5,
          rc_cost=RC.lookup_toffolis(a) + r - 1, nanc=2 * r + k, label="borrow")


# =============================================================================
# The loops
# =============================================================================
def _win(reg, j, w):
    return Reg(list(reg[j * w:(j + 1) * w]))


def loop1(m, conf, e, dlog, i):
    """dlog += lookup1[i][j][e_j] for every exponent window: prime i-1's dlog
    becomes prime i's ([G25]: merged compute/uncompute).  Repairs deferred to
    one phaseup per window at the end."""
    par, D = conf.par, conf.D
    for j in range(par.W1):
        addr = _win(e, j, par.w1)
        T = m.anc(D, "T1")
        MB.lookup(m, addr, T, conf.lookup1[i][j], anf=True)
        sc = m.anc(D - 1, "ad")
        A.add(m.ctx, T, dlog, sc)
        m.free(sc)
        MB.unlookup(m, addr, T, conf.lookup1[i][j], group=("rns-l1", j))
        m.free(T)


def loop2(m, reg, modulus, c):
    """Compress reg mod `modulus` into reg[:c] by binary long division; the
    quotient flags stay in the high bits until `unloop2`."""
    ctx, n = m.ctx, len(reg)
    while n > c:
        n -= 1
        thr = modulus << (n - c)
        low = Reg(reg[:n + 1])
        cr, sc = m.anc(n + 1, "cr"), m.anc(n + 1, "sc")
        A.add_const(ctx, low, (-thr) % (1 << (n + 1)), cr, sc)
        m.free(cr, sc)
        cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
        A.cadd_const(ctx, reg[n], Reg(reg[:n]), thr, cp, sc)
        m.free(cp, sc)


def unloop2(m, reg, modulus, c):
    ctx, n = m.ctx, c
    while n < len(reg):
        thr = modulus << (n - c)
        cp, sc = m.anc(n, "cp"), m.anc(n, "sc")
        A.cadd_const(ctx, reg[n], Reg(reg[:n]), (-thr) % (1 << n), cp, sc)
        m.free(cp, sc)
        low = Reg(reg[:n + 1])
        cr, sc = m.anc(n + 1, "cr"), m.anc(n + 1, "sc")
        A.add_const(ctx, low, thr, cr, sc)
        m.free(cr, sc)
        n += 1


def _mult_windows(m, conf, i, j, l1, src, dst, table, flip, fix, reverse=False):
    """dst (+)= src * X_j(l1) mod p window by window: for each window k of
    src, dst -= T[l0 + (l1 << a0)] with T the table (flip: p - table, so the
    subtraction adds)."""
    par, p = conf.par, conf.primes[i]
    ks = range(par.W3)
    for k in (reversed(ks) if reverse else ks):
        a0 = RC.window_width(par.ell, k, par.w3)
        addr = Reg(list(src[k * par.w3:k * par.w3 + a0]) + list(l1))
        tab = table[j, k]
        sub_wrap(m, dst, addr, [p - v for v in tab] if flip else tab, p, fix)


def loop3(m, conf, dlc, i):
    """res <- g_p^dlc mod p: the first two windows by one lookup, then one
    windowed multiplication per window, each old result measured out."""
    par, p, w3, ell = conf.par, conf.primes[i], conf.par.w3, conf.par.ell
    res = m.anc(ell + 1, "V")
    a2 = min(2 * w3, ell)
    MB.lookup(m, Reg(dlc[:a2]), Reg(res[:ell]), conf.lookup3c[i], anf=True)
    for j in range(2, par.W3):
        a1 = RC.window_width(ell, j, w3)
        l1 = Reg(dlc[j * w3:j * w3 + a1])
        helper = m.anc(ell + 1, "H")
        _mult_windows(m, conf, i, j, l1, res, helper, conf.lookup3a[i], flip=True, fix=False)
        old, res = res, helper
        k = _lk_anc(ell + a1)

        def recompute(qc, qs, j=j, a1=a1, k=k):
            # old = res * X_j(l1)^-1 mod p, looked up on (res, l1)
            vv, ll = qs[:ell], qs[ell:ell + a1]
            oo, lk = qs[ell + a1:2 * ell + a1], qs[2 * ell + a1:]
            X = [pow(conf.gens[i], l << (j * w3), p) for l in range(1 << a1)]
            tab = [(v * pow(X[l], -1, p)) % p if v < p else 0
                   for l in range(1 << a1) for v in range(1 << ell)]
            _table_xor(qc, list(vv) + list(ll), oo, tab, lk)

        # measured out; unloop3 recomputes it and applies the repair there
        clear(m, list(res[:ell]) + list(l1), list(old[:ell]), recompute, fix=0,
              p_fire=0.5, rc_cost=0, nanc=k, label="old result")
        m.free(old)
    return res


def loop4(m, conf, res, acc, i):
    """acc -= lookup4[i][j][V_p window j] mod trunc: adds the truncated
    contributions of V_p u_p."""
    par = conf.par
    for j in range(par.W4):
        addr = _win(Reg(res[:par.ell]), j, par.w4)
        sub_wrap(m, acc, addr, conf.lookup4[i][j], conf.trunc, fix=True)


def unloop3(m, conf, dlc, res, i):
    """Undo loop3: for each window, recompute the older result (res * X^-1)
    into a fresh register, then clear the newer one; last, unload the first
    lookup."""
    par, p, w3, ell = conf.par, conf.primes[i], conf.par.w3, conf.par.ell
    for j in reversed(range(2, par.W3)):
        a1 = RC.window_width(ell, j, w3)
        l1 = Reg(dlc[j * w3:j * w3 + a1])
        helper = m.anc(ell + 1, "H")
        _mult_windows(m, conf, i, j, l1, res, helper, conf.lookup3b[i], flip=True, fix=True,
                      reverse=True)                               # helper = res X^-1
        res, helper = helper, res                                 # res: the older value
        _mult_windows(m, conf, i, j, l1, res, helper, conf.lookup3a[i], flip=False, fix=True,
                      reverse=True)                               # helper -= res X
        m.free(helper)
    a2 = min(2 * w3, ell)
    MB.unlookup(m, Reg(dlc[:a2]), Reg(res[:ell]), conf.lookup3c[i])
    m.free(res)


def approx_modexp(m, conf, e, acc):
    """The whole shot: acc (f + 1 qubits, holding the mask) += the truncated
    approximation of g^e mod N, modulo trunc."""
    par = conf.par
    dlog = m.anc(conf.D, "dlog")
    for i, p in enumerate(conf.primes):
        loop1(m, conf, e, dlog, i)
        loop2(m, dlog, p - 1, par.ell)
        dlc = Reg(dlog[:par.ell])
        res = loop3(m, conf, dlc, i)
        loop4(m, conf, res, acc, i)
        unloop3(m, conf, dlc, res, i)
        unloop2(m, dlog, p - 1, par.ell)
    loop1(m, conf, e, dlog, len(conf.primes))
    m.free(dlog)
    for j in range(par.W1):
        MB.phase_fix(m, ("rns-l1", j))


def build(conf):
    """A Machine holding one shot: registers e (m) and acc (f + 1)."""
    m = Machine("and", "g25")
    e = m.alloc(conf.par.m, "e")
    acc = m.alloc(conf.par.f + 1, "acc")
    approx_modexp(m, conf, e, acc)
    return m, e, acc


def one_prime(conf, i=0):
    """One prime's iteration on its own (loop1 .. unloop2, dlog starting from
    prime i-1's value): the unit [G25]'s cost is |P| copies of."""
    par = conf.par
    m = Machine("and", "g25-prime")
    e = m.alloc(par.m, "e")
    acc = m.alloc(par.f + 1, "acc")
    dlog = m.alloc(conf.D, "dlog")
    loop1(m, conf, e, dlog, i)
    loop2(m, dlog, conf.primes[i] - 1, par.ell)
    dlc = Reg(dlog[:par.ell])
    res = loop3(m, conf, dlc, i)
    loop4(m, conf, res, acc, i)
    unloop3(m, conf, dlc, res, i)
    unloop2(m, dlog, conf.primes[i] - 1, par.ell)
    return m
