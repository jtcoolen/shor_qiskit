"""Verify Gidney windowing, bottom up.  Exact: these are permutation circuits,
so a single Aer shot is deterministic."""
import math
import time

from qiskit.circuit import QuantumCircuit, QuantumRegister

from _fastsim import outcome, setv
from shor_essentials import c_mult_acc
from windowed import add_quantum_mod, lookup, windowed_c_mult_acc

# --- 1. QROM lookup: every address, ancillas clean -------------------------
t0 = time.time()
for m in (1, 2, 3):
    W = 4
    table = [(7 * j + 3) % (2**W) for j in range(2**m)]
    for a in range(2**m):
        addr = QuantumRegister(m); out = QuantumRegister(W)
        un = QuantumRegister(1 << m)
        qc = QuantumCircuit(addr, out, un)
        setv(qc, a, addr)
        lookup(qc, list(addr), list(out), table, list(un))
        v = outcome(qc)
        got = (v & (2**m - 1), (v >> m) & (2**W - 1), v >> (m + W))
        assert got == (a, table[a], 0), (m, a, got, table[a])
print(f"lookup: OK all addresses m=1,2,3; address preserved, unary clean "
      f"({time.time()-t0:.0f}s)", flush=True)

# --- 2. modular addition with a QUANTUM addend -----------------------------
for N in (9, 15):
    t0 = time.time()
    n = math.ceil(math.log2(N))
    for yv in range(N):
        for tv in range(N):
            y = QuantumRegister(n); t = QuantumRegister(n); sf = QuantumRegister(2)
            qc = QuantumCircuit(y, t, sf)
            setv(qc, yv, y); setv(qc, tv, t)
            add_quantum_mod(qc, list(t), list(y), sf[0], sf[1], N)
            v = outcome(qc)
            got = (v & (2**n - 1), (v >> n) & (2**n - 1), v >> (2 * n))
            assert got == ((yv + tv) % N, tv, 0), (N, yv, tv, got)
    print(f"add_quantum_mod N={N}: OK all {N*N} (y,t) pairs; addend preserved, "
          f"sign+flag clean ({time.time()-t0:.0f}s)", flush=True)

# --- 3. windowed accumulate == the shift-and-add definition ----------------
N, n = 9, 4
for w in (1, 2):
    t0 = time.time()
    for k in (1, 7):
        for ctrl_on in (True, False):
            for xv in range(N):
                c = QuantumRegister(1); x = QuantumRegister(n)
                y = QuantumRegister(n); sf = QuantumRegister(2)
                t = QuantumRegister(n); un = QuantumRegister(w)
                qc = QuantumCircuit(c, x, y, sf, t, un)
                if ctrl_on:
                    qc.x(c[0])
                setv(qc, xv, x)
                windowed_c_mult_acc(qc, c[0], k, list(x), list(y),
                                    sf[0], sf[1], N, list(t), list(un), w)
                v = outcome(qc)
                got = ((v >> 1) & (2**n - 1), (v >> (1 + n)) & (2**n - 1),
                       v >> (1 + 2 * n))
                want = (k * xv) % N if ctrl_on else 0
                assert got == (xv, want, 0), (w, k, ctrl_on, xv, got, want)
    print(f"windowed_c_mult_acc N=9 w={w}: OK all x, k in (1,7), control "
          f"on/off; x preserved, all scratch clean ({time.time()-t0:.0f}s)",
          flush=True)

# --- 4. windowed and unwindowed agree, gate for gate, as maps --------------
t0 = time.time()
for w in (1, 2):
    for k in (2, 7):
        for xv in range(N):
            # reference: shift-and-add
            c = QuantumRegister(1); x = QuantumRegister(n)
            y = QuantumRegister(n); sf = QuantumRegister(2)
            ref = QuantumCircuit(c, x, y, sf)
            ref.x(c[0]); setv(ref, xv, x)
            c_mult_acc(ref, c[0], k, list(x), list(y), sf[0], sf[1], N)
            rv = outcome(ref)
            r_y = (rv >> (1 + n)) & (2**n - 1)
            # windowed
            c2 = QuantumRegister(1); x2 = QuantumRegister(n)
            y2 = QuantumRegister(n); sf2 = QuantumRegister(2)
            t2 = QuantumRegister(n); un2 = QuantumRegister(w)
            win = QuantumCircuit(c2, x2, y2, sf2, t2, un2)
            win.x(c2[0]); setv(win, xv, x2)
            windowed_c_mult_acc(win, c2[0], k, list(x2), list(y2), sf2[0],
                                sf2[1], N, list(t2), list(un2), w)
            wv = outcome(win)
            w_y = (wv >> (1 + n)) & (2**n - 1)
            assert r_y == w_y == (k * xv) % N, (w, k, xv, r_y, w_y)
print(f"windowed == shift-and-add on every input tested ({time.time()-t0:.0f}s)")

print("\nWINDOWING: ALL TESTS PASSED")
