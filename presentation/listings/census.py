fired = set()
for inp in samples:
    ...
    for pos, (name, w) in enumerate(prog):
        ...
        elif name in ("ccx", "ecand"):
            if bits[w[0]] & bits[w[1]]:
                fired.add(pos)
                bits[w[2]] ^= 1
        ...
return fired
