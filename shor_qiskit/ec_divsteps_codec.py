"""Exact finite-toy-state branch decoder for the divsteps circuit.

The coefficient registers in ec_divsteps hold residues modulo p, not the
unreduced integers assumed by Appendix A's bit-window formula. This decoder
instead compiles the reachable updated states of each toy round into a
reversible lookup. The active bit is retained to distinguish a last active
step from an already-frozen step. It is not a scalable implementation of
Appendix A's arithmetic decoder.
"""

from functools import lru_cache

from ec_divsteps_reference import State, step, walk


def _bits(state, active, p, widths):
    values = (state.z, state.f, state.g, state.fv % p, state.gv % p, active)
    return tuple((value >> i) & 1
                 for value, width in zip(values, widths)
                 for i in range(width))


def _projection(states):
    """Pick bit positions that distinguish every pair of differing decisions."""
    pairs = {
        (a, b) for a in range(len(states)) for b in range(a)
        if states[a][1] != states[b][1]
    }
    chosen = []
    while pairs:
        index = max(
            (i for i in range(len(states[0][0])) if i not in chosen),
            key=lambda i: sum(states[a][0][i] != states[b][0][i]
                              for a, b in pairs),
        )
        newly = {(a, b) for a, b in pairs
                 if states[a][0][index] != states[b][0][index]}
        if not newly:
            raise ValueError("updated state does not determine branch decision")
        chosen.append(index)
        pairs -= newly
    return tuple(sorted(chosen))


@lru_cache(maxsize=None)
def tables(p, widths):
    """Return (selected bit positions, branch lookup) for every toy round."""
    if widths[:3] != ((max(walk(w, p)[1] for w in range(1, p)) + 1).bit_length() + 1,
                       p.bit_length() + 1, p.bit_length() + 1):
        raise ValueError("decoder layout does not match divsteps circuit")
    rounds = max(walk(w, p)[1] for w in range(1, p))
    paths = []
    for w in range(1, p):
        state = State(0, p, w, 0, 1)
        path = []
        for _ in range(rounds):
            if state.g:
                state, decision = step(state)
                active = 1
            else:
                decision = (0, 0)
                active = 0
            path.append((_bits(state, active, p, widths), decision))
        paths.append(path)
    disabled = _bits(State(0, 0, 0, 0, 0), 0, p, widths)
    result = []
    for r in range(rounds):
        states = [path[r] for path in paths] + [(disabled, (0, 0))]
        selected = _projection(states)
        lookup = {}
        for bits, decision in states:
            key = tuple(bits[i] for i in selected)
            if key in lookup and lookup[key] != decision:
                raise ValueError(f"ambiguous divsteps branch for p={p}, round={r}")
            lookup[key] = decision
        result.append((selected, lookup))
    return tuple(result)


def toggle(m, registers, targets, table):
    """XOR decoded (swap, odd) into clean or already-populated target bits."""
    selected, lookup = table
    wires = [bit for reg in registers for bit in reg]
    controls = [wires[i] for i in selected]
    for pattern, decision in lookup.items():
        if not any(decision):
            continue
        for wire, value in zip(controls, pattern):
            if not value:
                m.ctx.x(wire)
        for target, value in zip(targets, decision):
            if value:
                m.ctx.mcx(controls, target)
        for wire, value in zip(controls, pattern):
            if not value:
                m.ctx.x(wire)
