"""Bivariate bicycle codes: build H_X=[A|B], H_Z=[B^T|A^T] from the polynomials and verify k = n - rank H_X - rank H_Z."""
import numpy as np
def shift(l):  # cyclic shift matrix S_l
    return np.roll(np.eye(l,dtype=int),1,axis=1)
def xy(l,m):
    x=np.kron(shift(l),np.eye(m,dtype=int)); y=np.kron(np.eye(l,dtype=int),shift(m)); return x,y
def mpow(M,e):
    R=np.eye(M.shape[0],dtype=int)
    for _ in range(e): R=(R@M)%2
    return R
def rank2(M):
    M=M.copy()%2; r=0; rows,cols=M.shape
    for c in range(cols):
        piv=[i for i in range(r,rows) if M[i,c]]
        if not piv: continue
        M[[r,piv[0]]]=M[[piv[0],r]]
        for i in range(rows):
            if i!=r and M[i,c]: M[i]^=M[r]
        r+=1
        if r==rows: break
    return r
codes={ # name: (l, m, A terms as (xe,ye), B terms)
 "[[72,12,6]]":  (6,6,  [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)]),
 "[[90,8,10]]":  (15,3, [(9,0),(0,1),(0,2)], [(0,0),(2,0),(7,0)]),
 "[[108,8,10]]": (9,6,  [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)]),
 "[[144,12,12]]":(12,6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)]),
 "[[288,12,18]]":(12,12,[(3,0),(0,2),(0,7)], [(0,3),(1,0),(2,0)]),
}
for name,(l,m,At,Bt) in codes.items():
    x,y=xy(l,m); n=2*l*m
    A=sum(mpow(x,a)@mpow(y,b) for a,b in At)%2; B=sum(mpow(x,a)@mpow(y,b) for a,b in Bt)%2
    HX=np.concatenate([A,B],axis=1); HZ=np.concatenate([B.T,A.T],axis=1)
    assert not ((HX@HZ.T)%2).any()
    k=n-rank2(HX)-rank2(HZ)
    wX=HX.sum(axis=1); wZ=HZ.sum(axis=1); colw=HX.sum(axis=0)+HZ.sum(axis=0)
    print(f"{name}: n={n} k={k} check weight X={set(wX)} Z={set(wZ)} data-qubit degree={set(colw)}")
