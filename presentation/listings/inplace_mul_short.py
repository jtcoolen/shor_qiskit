recs, u, v = dialog(m, x, q, iters)                  # 1: x -> dialog
clear_dialog_end(m, u, v)

s = m.anc(n, "bz")                                   # 2: replay onto (y, 0)
bezout_replay(m, y, s, recs, q, dbl, cadd)
for a, b in zip(y, s):                               # r is 0, s is the answer
    m.ctx.swap(a, b)
m.free(s)
...                                                  # 3: dialog -> x
for rec in reversed(recs):
    m.emit_inverse(dialog_round, m, u2, v2, rec)
A.encode_const(m.ctx, u2, q)                         # and starts at u = q
