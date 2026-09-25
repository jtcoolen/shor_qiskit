def order_from_counts(counts, A, N, t):
    for bits in sorted(counts, key=counts.get, reverse=True):
        y = int(bits, 2)
        if y == 0:
            continue
        r = Fraction(y, 2**t).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            return r
    return 0
