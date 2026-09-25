votes, q = {}, 1 << bits
for (j1, j2), c in items:
    for da in range(-search, search + 1):
        for db in range(-search, search + 1):
            a1 = (round(j1 * order / q) + da) % order
            a2 = (round(j2 * order / q) + db) % order
            if gcd(a1, order) != 1:
                continue
            k = a2 * pow(a1, -1, order) % order
            w = c / (1 + abs(da) + abs(db)) ** 2
            votes[k] = votes.get(k, 0.0) + w
return sorted(votes.items(), key=lambda kv: -kv[1])
