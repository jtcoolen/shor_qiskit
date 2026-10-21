lam = 1 << k
lo, hi = list(addr[:k]), list(addr[k:])
...
lookup(m, hi, [q for blk in blocks for q in blk], wide)
for i in reversed(range(k)):                  # block 0 <- word a_lo
    s = 1 << i
    for j in range(s):
        for qa, qb in zip(blocks[j], blocks[j + s]):
            m.ctx.cswap(lo[i], qa, qb)
...
unlookup(m, addr, junk, G, group=group)
