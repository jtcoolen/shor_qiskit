"""Toy RNS high-bit exponentiation oracle inspired by Gidney 2505.15917.

Residues of the N-reduced power factors are computed coherently per prime,
their CRT bit contributions accumulated into a truncated output, and the
residue workspace uncomputed. This is an executable toy circuit, not the
paper's low-space GHZ lookups, recycling, Ekerå-Håstad postprocessor, or
physical-code implementation.
"""

from math import gcd

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from rns_reference import ResidueOracle
from shor_essentials import c_add_mod, c_ua, qft


def rns_oracle(config):
    """Return a clean-work RNS |e>|0> -> |e>|approximate_high_bits(e)>."""
    if not isinstance(config, ResidueOracle):
        raise TypeError("config must be a validated ResidueOracle")
    if any(gcd(config.base, p) != 1 for p in config.primes):
        raise ValueError("reversible toy prime multipliers require unit factors")
    width = max(p.bit_length() for p in config.primes)
    exponent = QuantumRegister(config.exponent_bits, "exponent")
    residue = QuantumRegister(width, "residue")
    work = QuantumRegister(width, "work")
    flags = QuantumRegister(2, "flags")
    out_modulus = config.modulus >> (
        config.modulus.bit_length() - config.kept_bits)
    output = QuantumRegister(max(1, out_modulus.bit_length()), "high")
    qc = QuantumCircuit(exponent, residue, work, flags, output)
    L, shift = config.crt_modulus, config.modulus.bit_length() - config.kept_bits
    for p, coefficient in zip(config.primes, config.coefficients()):
        word, scratch = list(residue[:p.bit_length()]), list(work[:p.bit_length()])
        qc.x(word[0])
        factors = [pow(config.base, 1 << bit, config.modulus) % p
                   for bit in range(config.exponent_bits)]
        for bit, factor in enumerate(factors):
            c_ua(qc, exponent[bit], factor, word, scratch,
                 flags[0], flags[1], p)
        for bit in range(p.bit_length()):
            weight = (((coefficient << bit) % L) % config.modulus) >> shift
            if weight:
                c_add_mod(qc, [word[bit]], weight % out_modulus,
                          output, flags[0], flags[1], out_modulus)
        for bit in reversed(range(config.exponent_bits)):
            c_ua(qc, exponent[bit], pow(factors[bit], -1, p),
                 word, scratch, flags[0], flags[1], p)
        qc.x(word[0])
    return qc, (exponent, residue, work, flags, output)


def rns_order_circuit(config):
    """QFT period finder using the clean-work toy RNS high-bit oracle."""
    oracle, regs = rns_oracle(config)
    phase = regs[0]
    out = ClassicalRegister(len(phase), "frequency")
    qc = QuantumCircuit(*oracle.qregs, out)
    qc.h(phase)
    qc.compose(oracle, inplace=True)
    qc.append(qft(len(phase)).inverse(), phase)
    qc.measure(phase, out)
    return qc
