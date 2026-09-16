"""phase_fixup must act as exactly diag((-1)^F[a]) on the address register,
leaving every ancilla in |0>."""
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qrom import phase_fixup

rng = np.random.default_rng(0)
for m in (2, 3, 4):
    F = list(rng.integers(0, 2, size=2**m))
    for l in range(0, m + 1):
        h = m - l
        nq = m + 1 + (1 << l) + max(h, 1)
        if nq > 22:
            continue
        for a in range(2**m):
            addr = QuantumRegister(m, "a"); one = QuantumRegister(1, "one")
            ul = QuantumRegister(1 << l, "ul"); ah = QuantumRegister(max(h, 1), "ah")
            qc = QuantumCircuit(addr, one, ul, ah)
            for i in range(m):
                if (a >> i) & 1:
                    qc.x(addr[i])
            phase_fixup(qc, list(addr), F, one[0], list(ul), list(ah), l)
            sv = Statevector.from_instruction(qc).data
            # the only nonzero amplitude must be at |a>|0...0>, with sign (-1)^F[a]
            nz = np.flatnonzero(np.abs(sv) > 1e-9)
            assert nz.tolist() == [a], (m, l, a, "ancillas not clean", nz[:5])
            assert abs(sv[a] - (-1.0)**F[a]) < 1e-9, \
                (m, l, a, sv[a], (-1.0)**F[a])
    best = m // 2
    print(f"  m={m} (L={2**m}): OK all addresses, every split l=0..{m}; "
          f"sqrt split l={best} costs 2^{best}+2^{m-best}={2**best + 2**(m-best)} "
          f"vs {2**m} for a direct phase lookup")
print("phase_fixup == diag((-1)^F), ancillas clean: ALL PASSED")
