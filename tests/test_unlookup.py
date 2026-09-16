"""Wire measurement-based uncomputation into a real lookup and check that the
address register comes back exactly as it went in."""
import numpy as np
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qrom import lookup_ui
from unlookup import apply_fixup, fixup_table, uncompute_cost, xbasis_open

SIM = AerSimulator(method="statevector")
rng = np.random.default_rng(7)

for m_bits, W in ((2, 4), (3, 4)):
    L = 1 << m_bits
    table = [int(v) for v in rng.integers(0, 1 << W, size=L)]
    l = m_bits // 2
    h = m_bits - l
    # a non-uniform address superposition, so a leaked phase would show up
    amps = rng.normal(size=L) + 1j * rng.normal(size=L)
    amps /= np.linalg.norm(amps)

    addr = QuantumRegister(m_bits, "a"); one = QuantumRegister(1, "one")
    out = QuantumRegister(W, "o"); anc = QuantumRegister(m_bits, "anc")
    ul = QuantumRegister(1 << l, "ul"); ah = QuantumRegister(max(h, 1), "ah")
    creg = ClassicalRegister(W, "m")

    # ---- stage 1: prepare, look up, open the X-basis measurement, measure ----
    qc1 = QuantumCircuit(addr, one, out, anc, ul, ah, creg)
    qc1.initialize(amps, addr)
    qc1.x(one)
    lookup_ui(qc1, one[0], list(addr), list(out), table, list(anc))
    qc1.x(one)
    xbasis_open(qc1, list(out))
    qc1.measure(out, creg)
    qc1.save_statevector(label="psi")
    res = SIM.run(transpile(qc1, SIM), shots=1, memory=True).result()
    meas = int(res.get_memory()[0], 2)
    psi = np.asarray(res.data()["psi"])

    # ---- classical middle: build the fixup table from the outcome ----------
    F = fixup_table(meas, table)

    # ---- stage 2: repair the phase ----------------------------------------
    qc2 = QuantumCircuit(addr, one, out, anc, ul, ah)
    qc2.set_statevector(psi)
    apply_fixup(qc2, list(addr), meas, table, one[0], list(ul), list(ah), l)
    qc2.save_statevector(label="phi")
    phi = np.asarray(SIM.run(transpile(qc2, SIM), shots=1).result().data()["phi"])

    # ---- the address register must be exactly what we started with --------
    nq = qc2.num_qubits
    got = np.zeros(L, dtype=complex)
    for a in range(L):
        idx = a | (meas << (m_bits + 1))          # one=0 after the second X, anc=0
        got[a] = phi[idx]
    got = got / np.linalg.norm(got)
    ph = np.vdot(got, amps); ph /= abs(ph)        # ignore irrelevant global phase
    assert np.allclose(got * ph, amps, atol=1e-8), (m_bits, meas, got, amps)
    leak = np.linalg.norm(phi) ** 2 - np.linalg.norm([phi[a | (meas << (m_bits+1))]
                                                      for a in range(L)]) ** 2
    rec, msr = uncompute_cost(L, W, l)
    print(f"  L={L:2d} W={W}: outcome m={meas:2d}, {sum(F)} of {L} phases needed "
          f"repair -> address restored exactly (leak {abs(leak):.1e}); "
          f"uncompute {msr} vs {rec} by recompute")
print("measurement-based uncomputation, WIRED IN: ALL PASSED")
