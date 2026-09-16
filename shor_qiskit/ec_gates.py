"""AND gates and the emission context used by every ECDLP arithmetic circuit.

Both source papers count *Toffoli-equivalents*, and both get their gate counts
by using Gidney's temporary AND [Gid18] wherever the target of a Toffoli starts
in |0>.  Compute costs 4 T (T-depth 1, one ancilla, 11 Clifford in the JNRV19
form); uncompute costs 0 T -- an X-basis measurement and a classically
controlled CZ.

Two facts drive the design of this module:

* the AND gadget is *not* a permutation gate-by-gate (it goes through H), even
  though the map |a,b,0> -> |a,b,a AND b> is exactly a permutation.  Simulating
  it gate-by-gate on a basis state is therefore impossible; simulating it
  *atomically* is trivial.
* the uncompute is measurement-based, so its literal circuit is not unitary and
  cannot be composed into a static circuit that Aer will evolve as one state.

So AND and AND-dagger ship as named `Gate`s whose `definition` is the real,
unitary 4T/7T decomposition -- Aer runs those, and the fast simulator in
`ec_sim` recognises the *names* and applies the logical map in one step.  The
cost model in `ec_cost` charges the measurement-based price.
"""

from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister


class AndGate(Gate):
    """|a,b,t=0> -> |a,b,a AND b>.  4 T gates.  Target MUST start clean."""

    def __init__(self):
        super().__init__("ecand", 3, [])

    def _define(self):
        q = QuantumRegister(3, "q")
        qc = QuantumCircuit(q, name="ecand")
        a, b, t = q[0], q[1], q[2]
        qc.h(t)
        qc.t(t)
        qc.cx(b, t)
        qc.tdg(t)
        qc.cx(a, t)
        qc.t(t)
        qc.cx(b, t)
        qc.tdg(t)
        qc.h(t)
        qc.sdg(t)
        self.definition = qc

    def inverse(self, annotated=False):
        return AndDgGate()


class AndDgGate(Gate):
    """|a,b,a AND b> -> |a,b,0>.  0 T gates via measurement-based uncompute.

    The `definition` below is the unitary inverse, so the gate composes into a
    static circuit Aer can evolve.  The realisation that the papers cost is
    `qrom.and_uncompute`: H, measure, then CZ(a,b) if the outcome was 1.
    """

    def __init__(self):
        super().__init__("ecand_dg", 3, [])

    def _define(self):
        self.definition = AndGate().definition.inverse()

    def inverse(self, annotated=False):
        return AndGate()


class Ctx:
    """Wraps a QuantumCircuit and decides how each AND is emitted.

    mode="and"     : temporary ANDs where the target is clean (the papers' cost)
    mode="toffoli" : plain Toffolis everywhere (a pure permutation circuit --
                     Aer can evolve it, and every gate is X/CX/CCX/CSWAP)

    Both modes implement the same map; only the gate set and the cost differ.
    """

    def __init__(self, qc, mode="and"):
        assert mode in ("and", "toffoli"), mode
        self.qc = qc
        self.mode = mode

    # --- clean-target AND: this is where the papers save their T gates ------
    def and_(self, a, b, t):
        """t (which must be |0>) becomes a AND b."""
        if self.mode == "and":
            self.qc.append(AndGate(), [a, b, t])
        else:
            self.qc.ccx(a, b, t)

    def and_dg(self, a, b, t):
        """t (which must hold a AND b) is returned to |0>."""
        if self.mode == "and":
            self.qc.append(AndDgGate(), [a, b, t])
        else:
            self.qc.ccx(a, b, t)

    # --- dirty-target Toffoli: no saving available -------------------------
    def ccx(self, a, b, t):
        self.qc.ccx(a, b, t)

    # --- pass-throughs, so builders can take a Ctx and nothing else --------
    def x(self, q):
        self.qc.x(q)

    def cx(self, a, b):
        self.qc.cx(a, b)

    def cz(self, a, b):
        self.qc.cz(a, b)

    def swap(self, a, b):
        self.qc.swap(a, b)

    def cswap(self, c, a, b):
        self.qc.cswap(c, a, b)

    def mcx(self, ctrls, t):
        ctrls = list(ctrls)
        if len(ctrls) == 0:
            self.qc.x(t)
        elif len(ctrls) == 1:
            self.qc.cx(ctrls[0], t)
        elif len(ctrls) == 2:
            self.qc.ccx(ctrls[0], ctrls[1], t)
        else:
            self.qc.mcx(ctrls, t)

    def barrier(self, *args):
        self.qc.barrier(*args)
