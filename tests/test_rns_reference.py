"""Check CRT promise and quantify Gidney's toy high-bit approximation."""

from rns_reference import ResidueOracle, modular_deviation


def main():
    assert modular_deviation(14, 1, 15) == 2 / 15
    for n, base, bits, primes in (
            (15, 2, 3, (7, 11, 13, 19)),
            (15, 7, 3, (7, 11, 13, 19)),
            (21, 2, 2, (5, 11, 13))):
        for kept in range(1, 4 if n == 15 else 5):
            oracle = ResidueOracle(n, base, bits, primes, kept)
            assert oracle.crt_modulus >= n ** bits
            assert modular_deviation(
                oracle.crt_modulus, 0, n) < 2 ** -kept
            assert all(u % p == int(p == target)
                       for u, target in zip(oracle.coefficients(), primes)
                       for p in primes)
            for exponent in range(1 << bits):
                assert oracle.exact(exponent) == pow(base, exponent, n)
                assert 0 <= oracle.approximate(exponent) < (
                    n >> (n.bit_length() - kept))
                assert 0 <= oracle.deviation(exponent) <= 0.5
    example = ResidueOracle(15, 2, 3, (7, 11, 13, 19), 3)
    assert max(example.deviation(i) for i in range(8)) == 4 / 15
    for kw in (
            {"modulus": 15, "base": 2, "exponent_bits": 3,
             "primes": (7, 11), "kept_bits": 3},
            {"modulus": 15, "base": 2, "exponent_bits": 3,
             "primes": (11, 13, 17, 19), "kept_bits": 3},
            {"modulus": 15, "base": 2, "exponent_bits": 3,
             "primes": (7, 11, 13, 15, 19), "kept_bits": 3},
            {"modulus": 15, "base": 2, "exponent_bits": 3,
             "primes": (7, 11, 13, 19), "kept_bits": 0},
    ):
        try:
            ResidueOracle(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)
    print("test_rns_reference: CRT toy equality and high-bit error measured")


if __name__ == "__main__":
    main()
