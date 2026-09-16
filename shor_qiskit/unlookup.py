"""Measurement-based uncomputation, wired in (Gidney arXiv:1905.07682 Fig. 3).

A lookup is uncomputed by measuring its output in the X basis and repairing the
phase this leaves behind.  The repair table depends on the measurement outcome,
so this is a genuinely hybrid operation: circuit, measure, classical table
build, circuit.  That is exactly how it runs on hardware -- the classical
controller sits in the middle -- and it is why it cannot be one static circuit.

  compute:    |a>|0>          -> |a>|T[a]>
  X-measure:  |a>|T[a]>       -> (-1)^{m . T[a]} |a> (x) |m>,  m random
  fixup:      (-1)^{m . T[a]} |a>  ->  |a>
"""

from qiskit.circuit import QuantumCircuit

from qrom import phase_fixup


def parity(v):
    return bin(v).count("1") & 1


def fixup_table(m, table):
    """F[a] = parity(m AND T[a]) -- the phase the X-basis measurement left."""
    return [parity(m & T) for T in table]


def xbasis_open(qc, out):
    """First half of an X-basis measurement: the measurement itself is the
    runner's job, because its outcome has to reach classical code."""
    for q in out:
        qc.h(q)


def apply_fixup(qc, addr, m, table, one, unary_lo, anc_hi, l):
    """Second half: undo the phase that outcome m left on the address."""
    F = fixup_table(m, table)
    if any(F):
        phase_fixup(qc, addr, F, one, unary_lo, anc_hi, l)


def uncompute_cost(L, W, l=None):
    """Toffoli-ish cost of the two ways to uncompute an L-entry, W-bit lookup."""
    m = L.bit_length() - 1
    l = m // 2 if l is None else l
    h = m - l
    recompute = 2 * (L - 1)                 # run the unary-iteration lookup again
    measured = (1 << l) - 1 + 2 * ((1 << h) - 1)   # unary conv + CZ walk
    return recompute, measured


# --- the windowed accumulate, with every uncompute done by measurement ------
def run_windowed_acc_mbu(sim, N, k, x_val, c_val, w, n, shots_seed=None):
    """Execute  y += c * k*x (mod N)  windowed, uncomputing each lookup by
    X-basis measurement + phase fixup instead of by re-running it.

    Returns (x_out, y_out, scratch_clean, n_measurements).  Hybrid by
    necessity: the fixup table is built in Python between the stages.
    """
    import numpy as np
    from qiskit import transpile
    from qiskit.circuit import (ClassicalRegister, QuantumCircuit,
                                QuantumRegister)
    from qrom import lookup_ui
    from windowed import add_quantum_mod

    l = max(w // 2, 0)
    h = w - l
    regs = dict(
        c=QuantumRegister(1, "c"), x=QuantumRegister(n, "x"),
        y=QuantumRegister(n, "y"), sf=QuantumRegister(2, "sf"),
        t=QuantumRegister(n, "t"), anc=QuantumRegister(max(w, 1), "anc"),
        one=QuantumRegister(1, "one"), ul=QuantumRegister(1 << l, "ul"),
        ah=QuantumRegister(max(h, 1), "ah"),
    )
    order = ["c", "x", "y", "sf", "t", "anc", "one", "ul", "ah"]
    state = None
    nmeas = 0

    for i in range(0, n, w):
        win_lo, win_hi = i, min(i + w, n)
        nw = win_hi - win_lo
        table = [((k * j) << i) % N for j in range(1 << nw)]

        creg = ClassicalRegister(n, "m")
        qc = QuantumCircuit(*[regs[r] for r in order], creg)
        if state is None:                       # first stage: prepare inputs
            if c_val:
                qc.x(regs["c"][0])
            for b in range(n):
                if (x_val >> b) & 1:
                    qc.x(regs["x"][b])
        else:
            qc.set_statevector(state)
        win = [regs["x"][b] for b in range(win_lo, win_hi)]
        lookup_ui(qc, regs["c"][0], win, list(regs["t"])[:n], table,
                  list(regs["anc"])[:nw])
        add_quantum_mod(qc, list(regs["t"])[:n], list(regs["y"]),
                        regs["sf"][0], regs["sf"][1], N)
        xbasis_open(qc, list(regs["t"])[:n])     # uncompute, part 1
        qc.measure(regs["t"], creg)
        qc.save_statevector(label="psi")
        res = sim.run(transpile(qc, sim), shots=1, memory=True).result()
        meas = int(res.get_memory()[0], 2)
        state = np.asarray(res.data()["psi"])
        nmeas += 1

        qc2 = QuantumCircuit(*[regs[r] for r in order])
        qc2.set_statevector(state)
        # the lookup was controlled on c, so the phase only appeared when c=1
        F = [parity(meas & T) for T in table]
        if any(F) and c_val:
            phase_fixup(qc2, win, F, regs["one"][0], list(regs["ul"]),
                        list(regs["ah"]), min(l, nw))
        for b in range(n):                       # return t to |0>
            if (meas >> b) & 1:
                qc2.x(regs["t"][b])
        qc2.save_statevector(label="phi")
        state = np.asarray(
            sim.run(transpile(qc2, sim), shots=1).result().data()["phi"])

    idx = int(np.argmax(np.abs(state)))
    x_out = (idx >> 1) & ((1 << n) - 1)
    y_out = (idx >> (1 + n)) & ((1 << n) - 1)
    scratch = idx >> (1 + 2 * n)
    return x_out, y_out, scratch, nmeas
