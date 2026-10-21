"""Gottesman's standard-form encoder for a stabiliser code, built and verified with Stim.

Given stabiliser generators as Pauli strings, bring the binary check matrix [X|Z] to the
standard form  [ I  A1 A2 | B 0 C ; 0 0 0 | D I E ]  by Gaussian elimination (allowing
qubit permutations), derive logical X operators  [0 E^T I | C^T 0 0], and emit the
encoding circuit:  data qubits are the last k; for each logical X_j apply its Pauli
(restricted to the first n-k qubits) controlled on data qubit j; then for i < r apply H
on qubit i followed by generator i's Pauli on the other qubits controlled on qubit i.
"""
import sys, numpy as np, stim

def pauli_to_bits(s):
    n=len(s); x=np.zeros(n,int); z=np.zeros(n,int)
    for i,c in enumerate(s):
        if c in "XY": x[i]=1
        if c in "ZY": z[i]=1
    return x,z

def standard_form(stabs):
    m=len(stabs); n=len(stabs[0])
    M=np.array([np.concatenate(pauli_to_bits(s)) for s in stabs])%2
    perm=list(range(n))
    def swapcols(i,j):
        nonlocal M,perm
        if i!=j:
            M[:,[i,j]]=M[:,[j,i]]; M[:,[n+i,n+j]]=M[:,[n+j,n+i]]
            perm[i],perm[j]=perm[j],perm[i]
    # eliminate X part
    r=0
    for col in range(n):
        if r>=m: break
        piv=[i for i in range(r,m) if M[i,col]]
        if not piv:
            # try to bring a column with a pivot here
            found=False
            for c2 in range(col+1,n):
                if any(M[i,c2] for i in range(r,m)):
                    swapcols(col,c2); found=True; break
            if not found: break
            piv=[i for i in range(r,m) if M[i,col]]
        i=piv[0]; M[[r,i]]=M[[i,r]]
        for i2 in range(m):
            if i2!=r and M[i2,col]: M[i2]^=M[r]
        r+=1
    # eliminate Z part of the remaining rows on columns r..n-1
    r2=r
    for col in range(r,n):
        if r2>=m: break
        piv=[i for i in range(r2,m) if M[i,n+col]]
        if not piv:
            found=False
            for c2 in range(col+1,n):
                if any(M[i,n+c2] for i in range(r2,m)):
                    swapcols(col,c2); found=True; break
            if not found: break
            piv=[i for i in range(r2,m) if M[i,n+col]]
        i=piv[0]; M[[r2,i]]=M[[i,r2]]
        for i2 in range(m):
            if i2!=r2 and M[i2,n+col]: M[i2]^=M[r2]
        r2+=1
    return M,perm,r

def encoder_circuit(stabs):
    m=len(stabs); n=len(stabs[0]); k=n-m
    M,perm,r=standard_form(stabs)
    X=M[:,:n]; Z=M[:,n:]
    # blocks
    A2=X[:r, n-k:]; C=Z[:r, n-k:]; E=Z[r:, n-k:]
    # logical X: [0 E^T I | C^T 0 0]
    circ=stim.Circuit()
    for j in range(k):
        dq=n-k+j
        for q in range(r, n-k):          # E^T part: X on qubits r..n-k-1
            if E[q-r, j]: circ.append("CX",[dq,q])
        for q in range(r):               # C^T part: Z on qubits 0..r-1
            if C[q, j]: circ.append("CZ",[dq,q])
    for i in range(r):
        circ.append("H",[i])
        for q in range(n):
            if q==i: continue
            if X[i,q] and Z[i,q]: circ.append("CY",[i,q])
            elif X[i,q]: circ.append("CX",[i,q])
            elif Z[i,q]: circ.append("CZ",[i,q])
        if Z[i,i]:                      # generator i has Y on its pivot: (I+Y_i M')/sqrt2 needs the phase i
            circ.append("S",[i])
    return circ, perm, r

def verify(name, stabs, xl, zl):
    n=len(stabs[0]); m=len(stabs); k=n-m
    circ,perm,r=encoder_circuit(stabs)
    # the circuit acts on permuted qubits: physical qubit perm[i] <- circuit qubit i
    inv=[0]*n
    for i,p in enumerate(perm): inv[p]=i
    def relabel(s):  # pauli string on physical qubits -> circuit qubit order
        out=["_"]*n
        for p,c in enumerate(s): out[inv[p]]=c
        return "".join(out)
    P=stim.PauliString
    S=[P(relabel(s)) for s in stabs]; XL=P(relabel(xl)); ZL=P(relabel(zl))
    dq=n-k
    sim=stim.TableauSimulator(); sim.do(circ)
    ok0=all(sim.peek_observable_expectation(s)==1 for s in S)
    zexp=sim.peek_observable_expectation(ZL)
    sim=stim.TableauSimulator(); sim.h(dq); sim.do(circ)
    okp=all(sim.peek_observable_expectation(s)==1 for s in S)
    xexp=sim.peek_observable_expectation(XL)
    print(f"{name}: r={r} perm={perm} stabilisers ok: {ok0 and okp}; <Z_L> on |0_L> = {zexp}, <X_L> on |+_L> = {xexp}; ops={len(circ)}")
    print(circ)
    return circ,perm

if __name__=="__main__":
    verify("[[5,1,3]]", ["XZZX_","_XZZX","X_XZZ","ZX_XZ"], "XXXXX", "ZZZZZ")
    verify("Steane", ["___XXXX","_XX__XX","X_X_X_X","___ZZZZ","_ZZ__ZZ","Z_Z_Z_Z"], "XXXXXXX","ZZZZZZZ")
    verify("Shor", ["ZZ_______","_ZZ______","___ZZ____","____ZZ___","______ZZ_","_______ZZ","XXXXXX___","___XXXXXX"],
           "ZZZZZZZZZ","XXXXXXXXX")
    # [[15,1,3]] punctured Reed-Muller: X checks = the four weight-8 coordinate hyperplanes of the
    # 4-cube minus the origin, Z checks = those four plus the six weight-4 pairwise products.
    import itertools
    xs=["".join("X" if ((i>>b)&1) else "_" for i in range(1,16)) for b in range(4)]
    zs=["".join("Z" if ((i>>b)&1) else "_" for i in range(1,16)) for b in range(4)]
    zs+=["".join("Z" if (((i>>b1)&1) and ((i>>b2)&1)) else "_" for i in range(1,16)) for b1,b2 in itertools.combinations(range(4),2)]
    verify("[[15,1,3]] RM", xs+zs, "X"*15, "Z"*15)
