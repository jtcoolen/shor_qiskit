MA.is_nonzero(m, addr, c[0])                       # 1
_lookup(m, one[0], addr, x1, xs)                   # 2
_lookup(m, one[0], addr, y1, ys)
MA.modsub(m, x1, x2, p)                            # 3
MA.modsub(m, y1, y2, p)                            # 4
_lookup(m, one[0], addr, x1, xs)                   # 5
_lookup(m, one[0], addr, y1, ys)
E.inplace_div(m, x2, y2, p, iters)                 # 6  y2 <- y2 / x2
_lookup(m, one[0], addr, xx, x3s)                  # 7
MA.modadd(m, xx, x2, p)                            # 8
_lookup(m, one[0], addr, xx, x3s)                  # 9
_csub_square(m, c[0], y2, x2, p)                   # 10 if c: x2 -= y2^2
E.inplace_mul(m, x2, y2, p, iters)                 # 11 y2 <- y2 * x2
MA.cmodneg(m, c[0], x2, p)                         # 12 if c: x2 <- -x2
...                                                # 13-17: add P back, unload
