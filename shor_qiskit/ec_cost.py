"""Resource counting, and the analytic projections to cryptographic sizes.

Two kinds of number live here and they must not be confused.

*Measured*: counted off a circuit this package actually builds and this
package's tests actually verify.  Only reachable at toy widths, because
building a 256-bit point addition is fine but nothing here has been run at that
size.  `count()` produces these.

*Projected*: the papers' own formulas, evaluated at cryptographic n.  Not
verified by anything here.  They are included so the measured numbers can be
sanity-checked against the published ones for shape and scaling, and they are
labelled as projections wherever they are printed.

Cost model.  Both papers count Toffoli-equivalents: a Toffoli, a CCZ, and a
Gidney AND all count as one.  In Clifford+T that is 7 T for a Toffoli (T-depth
4, [AMM+13]) against 4 T for an AND compute (T-depth 1, [JNRV19]) and 0 T for
an AND uncompute, which is measured separately here because it is the whole
reason the AND gadget exists.
"""

TOFFOLI_T = 7          # [AMM+13]
AND_T = 4              # [JNRV19] compute
AND_DG_T = 0           # measurement-based uncompute


def count(obj):
    """Gate counts for a Machine or a QuantumCircuit."""
    qc = getattr(obj, "qc", obj)
    tally = {}
    for ci in qc.data:
        tally[ci.operation.name] = tally.get(ci.operation.name, 0) + 1

    ands = tally.get("ecand", 0)
    and_dgs = tally.get("ecand_dg", 0)
    toffs = tally.get("ccx", 0) + tally.get("mcx", 0)
    cswaps = tally.get("cswap", 0)
    return {
        "qubits": qc.num_qubits,
        "gates": len(qc.data),
        "depth": qc.depth(),
        "and": ands,
        "and_dg": and_dgs,
        "toffoli": toffs,
        "cswap": cswaps,
        # --- two metrics, and the difference between them matters ---------
        # `toffoli` is what both papers report: "the Toffolis column counts
        # together CCX, CCZ as well as And gates" [1128].  A measurement-based
        # AND-dagger contains no Toffoli and no T gate, so it is not counted.
        # This is the number to compare against published figures.
        "toffoli_paper": ands + toffs + cswaps,
        # `toffoli_equiv` charges for the uncompute too.  It is the honest
        # count of three-qubit *operations* performed, and it is the one to use
        # when comparing a construction that uncomputes a lot against one that
        # does not -- but it understates Gidney-style circuits relative to the
        # literature, so it is never the headline.
        "toffoli_equiv": ands + and_dgs + toffs + cswaps,
        "t": ands * AND_T + and_dgs * AND_DG_T + (toffs + cswaps) * TOFFOLI_T,
        "clifford": tally.get("cx", 0) + tally.get("x", 0) + tally.get("swap", 0),
        "by_name": tally,
    }


def table(rows, headers):
    w = [max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0))
         for i in range(len(headers))]
    out = ["  ".join(str(h).rjust(w[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        out.append("  ".join(str(c).rjust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


# =============================================================================
# Projections -- the papers' formulas, not this package's circuits
# =============================================================================
def proj_1128_qubits(n, variant="space"):
    """[1128]: 4.355n + O(sqrt n) space-optimized, more for the gate-optimized.

    2.355n for the dialog (Fig. 1 encoding) plus 2n for the pair of modular
    integers the Bezout reconstruction holds.  The gate-optimized variant adds
    about n ancillas so Gidney's adder can be used during the reconstruction.
    """
    import math
    base = 4.355 * n + 2.3 * math.sqrt(n)
    return int(base + (n if variant == "gate" else 0))


def proj_1128_shor_toffoli(n=256, w=16, per_add=2**21.19):
    """[1128] eq. (1): 28 * (3 * 2^w + Q_A) for secp256k1.

    This restates the paper: the window count and Q_A are the paper's own
    figures, so the result is its number recomputed, not an independent check.
    Nothing in this package measures a 256-bit point addition.
    """
    return int(28 * (3 * (1 << w) + per_add))


def proj_106_additions(n, w):
    """[106] Sec 4.2: windowed point additions, and the zig-zag register count."""
    from ec_classical import n_point_additions, zigzag_registers
    adds = n_point_additions(n, w)
    return adds, zigzag_registers(adds)


def proj_eea_space(n, c_iter=2.4, c_pad=2.3):
    """Record and working space for [1128]'s Euclidean algorithm.

    Returns (what this package spends, what Sec 3.1's register sharing would).
    The gap is entirely the sharing optimization, which is probabilistic and is
    deliberately not implemented -- see `ec_eea`.
    """
    from ec_eea import dialog_width, dialog_width_compressed
    import math
    return {
        "record_raw": dialog_width(n, c_iter),
        "record_compressed": dialog_width_compressed(n, c_iter),
        "record_compressed_per_n": dialog_width_compressed(n, c_iter) / n,
        "paper_asymptotic_per_n": 2.355,
        "with_register_sharing": int(2.12 * n + c_pad * math.sqrt(n)),
    }


def report(machines, title=""):
    """Print a comparison table over {label: Machine}."""
    rows = []
    for label, m in machines.items():
        c = count(m)
        rows.append([label, c["qubits"], c["toffoli_equiv"], c["and"],
                     c["and_dg"], c["t"], c["depth"]])
    hdr = ["circuit", "qubits", "Toffoli-eq", "AND", "AND-dg", "T", "depth"]
    if title:
        print(title)
    print(table(rows, hdr))
    return rows
