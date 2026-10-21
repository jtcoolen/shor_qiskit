"""Classical toy oracle for Gidney arXiv:2505.15917 Eqs. (8)-(22).

This validates CRT residue exponentiation and its *approximate high-bit*
dot product. It is not a quantum RNS period-finding circuit, nor the
least-significant-bit Ekerå-Håstad algorithm of CFS ePrint 2024/222.
"""

from dataclasses import dataclass
from math import gcd, prod


def modular_deviation(a, b, modulus):
    if modulus < 2:
        raise ValueError("modulus must exceed one")
    diff = (a - b) % modulus
    return min(diff, modulus - diff) / modulus


@dataclass(frozen=True)
class ResidueOracle:
    modulus: int
    base: int
    exponent_bits: int
    primes: tuple[int, ...]
    kept_bits: int

    def __post_init__(self):
        n = self.modulus.bit_length()
        if (self.modulus < 3 or self.exponent_bits < 1
                or not 1 <= self.kept_bits <= n
                or not self.primes or len(set(self.primes)) != len(self.primes)
                or any(p < 3 or any(p % d == 0 for d in range(
                    2, int(p ** 0.5) + 1)) for p in self.primes)
                or gcd(self.base, self.modulus) != 1
                or prod(self.primes) < self.modulus ** self.exponent_bits
                or modular_deviation(
                    prod(self.primes), 0, self.modulus) >= 2 ** -self.kept_bits):
            raise ValueError("requires distinct odd prime residues, unit base,"
                             " CRT product >= modulus ** exponent_bits, and "
                             "Delta_N(CRT product) < 2 ** -kept_bits")

    @property
    def crt_modulus(self):
        return prod(self.primes)

    def residues(self, exponent):
        if not 0 <= exponent < 1 << self.exponent_bits:
            raise ValueError("exponent outside configured register")
        factors = (pow(self.base, 1 << bit, self.modulus)
                   for bit in range(self.exponent_bits)
                   if exponent >> bit & 1)
        product = prod(factors)
        return tuple(product % p for p in self.primes)

    def coefficients(self):
        length = self.crt_modulus
        return tuple((length // p) * pow(length // p, -1, p)
                     for p in self.primes)

    def exact(self, exponent):
        length = self.crt_modulus
        return sum(r * u for r, u in zip(
            self.residues(exponent), self.coefficients())) % length % self.modulus

    def approximate(self, exponent):
        shift = self.modulus.bit_length() - self.kept_bits
        length = self.crt_modulus
        accum = 0
        for residue, coefficient in zip(
                self.residues(exponent), self.coefficients()):
            for bit in range(residue.bit_length()):
                if residue >> bit & 1:
                    accum += (((coefficient << bit) % length)
                              % self.modulus) >> shift
        return accum % (self.modulus >> shift)

    def deviation(self, exponent):
        shift = self.modulus.bit_length() - self.kept_bits
        return modular_deviation(
            self.exact(exponent), self.approximate(exponent) << shift,
            self.modulus)
