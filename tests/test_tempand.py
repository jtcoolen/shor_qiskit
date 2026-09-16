"""Temporary AND: correctness, then the T-count it actually saves."""
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from _fastsim import setv

_SIM = AerSimulator(method="statevector")

def qoutcome(qc, shots=64):
    """Value of the QUANTUM registers, for circuits that already contain a
    mid-circuit measurement: that measurement's outcome is random, so read
    only the freshly appended register and require IT to be deterministic."""
    m = qc.copy()
    m.measure_all(add_bits=True)                 # appends its own register
    res = _SIM.run(transpile(m, _SIM, optimization_level=0), shots=shots).result()
    vals = {k.split(" ")[0] for k in res.get_counts()}   # leftmost = newest reg
    assert len(vals) == 1, f"quantum registers not deterministic: {vals}"
    return int(vals.pop(), 2)
from qrom import and_compute, and_uncompute, lookup_ui, lookup_ui_ta

# --- 1. the AND pair is identity on the ancilla, and computes AND ----------
for a in (0,1):
    for b in (0,1):
        q = QuantumRegister(3); c = ClassicalRegister(1)
        qc = QuantumCircuit(q, c)
        if a: qc.x(q[0])
        if b: qc.x(q[1])
        and_compute(qc, q[0], q[1], q[2])
        v = qoutcome(qc)
        assert ((v>>2)&1) == (a & b), (a,b,"and_compute wrong")
        and_uncompute(qc, q[0], q[1], q[2], c[0])
        v = qoutcome(qc)
        assert (v & 1, (v>>1)&1, (v>>2)&1) == (a, b, 0), (a,b,"pair not clean")
print("and_compute / and_uncompute: OK all inputs; ancilla returned to |0>")

# --- 2. lookup with temporary ANDs == lookup without ----------------------
for m in (1,2,3):
    W = 4
    table = [(11*j+5) % (1<<W) for j in range(1<<m)]
    for ctrl_on in (True, False):
        for ad in range(1<<m):
            c = QuantumRegister(1); a = QuantumRegister(m)
            o = QuantumRegister(W); an = QuantumRegister(m)
            cb = ClassicalRegister(1)
            qc = QuantumCircuit(c, a, o, an, cb)
            if ctrl_on: qc.x(c[0])
            setv(qc, ad, a)
            lookup_ui_ta(qc, c[0], list(a), list(o), table, list(an), cb[0])
            v = qoutcome(qc)
            got = ((v>>1)&((1<<m)-1), (v>>(1+m))&((1<<W)-1), (v>>(1+m+W))&((1<<m)-1))
            want = (ad, table[ad] if ctrl_on else 0, 0)
            assert got == want, (m, ctrl_on, ad, got, want)
print("lookup_ui_ta: OK all addresses m=1,2,3, both controls, ancillas clean")

# --- 3. the T count ------------------------------------------------------
CT = ["h", "s", "sdg", "t", "tdg", "x", "z", "cx", "cz"]

def tcount(qc):
    """Count T/T-dagger, recursing into control-flow bodies."""
    n = 0
    for inst in qc.data:
        if inst.operation.name in ("t", "tdg"):
            n += 1
        for blk in getattr(inst.operation, "blocks", ()):
            n += tcount(blk)
    return n

print(f"\n{'m':>2} {'L':>4} {'T (plain)':>10} {'T (temp-AND)':>13} {'saving':>8}")
for m in (2, 3, 4, 5):
    W, table = 4, [1] * (1 << m)
    c = QuantumRegister(1); a = QuantumRegister(m); o = QuantumRegister(W)
    an = QuantumRegister(m); cb = ClassicalRegister(1)

    plain = QuantumCircuit(c, a, o, an)
    lookup_ui(plain, c[0], list(a), list(o), table, list(an))
    Tp = tcount(transpile(plain, basis_gates=CT, optimization_level=0))

    ta = QuantumCircuit(c, a, o, an, cb)          # already Clifford+T
    lookup_ui_ta(ta, c[0], list(a), list(o), table, list(an), cb[0])
    Tt = tcount(ta)

    print(f"{m:>2} {1<<m:>4} {Tp:>10} {Tt:>13} {Tp/max(Tt,1):>7.2f}x")
print("\nTEMPORARY AND: ALL PASSED")
