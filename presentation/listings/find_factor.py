while True:
    a = random.randrange(2, N)
    if (d := math.gcd(a, N)) > 1:
        return d                         # lucky gcd
    t = 2 * math.ceil(math.log2(N))
    r = order_from_counts(run(order_circuit(a, N)), a, N, t)
    if r and r % 2 == 0:
        d = math.gcd(pow(a, r // 2, N) - 1, N)
        if 1 < d < N:
            return d
