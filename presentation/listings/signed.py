for addr, table in zip(addrs, tables):
    w = len(addr)
    s, low = addr[w - 1], Reg(list(addr[:w - 1]), "l")
    ctx.x(s)                                    # s <- [d < 0]
    for q in low:
        ctx.cx(s, q)                            # low <- l~, the |d| index
    if prev is None:
        neg(m, s, y2, p)                        # R <- -R when d < 0
    else:
        ps, plow = prev                         # close the last window and
        ctx.cx(ps, s)                           # open this one: one negation,
        neg(m, s, y2, p)                        # controlled by the XOR
        ctx.cx(ps, s)
        ...
    W.windowed_point_add_cfg(m, low, x2, y2, table, p, cfg)
    prev = (s, low)
