"""Check the signed-coefficient Appendix A arithmetic decoder as a circuit."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from _ec_util import ok, section
from ec_divsteps_arith_codec import toggle
from ec_divsteps_reference import State, step, walk
from ec_sim import Machine, run


def _states(p):
    for w in range(1, p):
        state = State(0, p, w, 0, 1)
        while state.g:
            state, decision = step(state)
            yield state, decision


def _build(z_width, g_width, coefficient_width, mode):
    m = Machine(mode)
    z = m.alloc(z_width, "z")
    g = m.alloc(g_width, "g")
    fv, gv = m.alloc(coefficient_width, "fv"), m.alloc(coefficient_width, "gv")
    active, swap, odd = (m.alloc(1, name)
                         for name in ("active", "swap", "odd"))
    toggle(m, z, g, fv, gv, active[0], swap[0], odd[0])
    return m, (z, g, fv, gv, active, swap, odd)


def _inputs(regs, state, active=1):
    z, g, fv, gv, enable, _, _ = regs
    return {z: state.z, g: state.g, fv: state.fv, gv: state.gv,
            enable: active}


def main():
    section("Appendix A bit-window decoder, all signed toy states")
    for p in (3, 5, 7, 11, 13, 17, 31, 61, 127):
        states = list(_states(p))
        max_coefficient = max(max(abs(s.fv), abs(s.gv)) for s, _ in states)
        coefficient_width = max(4, max_coefficient.bit_length() + 2)
        rounds = max(walk(w, p)[1] for w in range(1, p))
        m, regs = _build((rounds + 1).bit_length() + 1,
                         p.bit_length() + 1, coefficient_width, "toffoli")
        z, g, fv, gv, active, swap, odd = regs
        for state, decision in states:
            result = run(m, _inputs(regs, state))
            assert (result(swap), result(odd)) == decision, (p, state, decision)
            assert (result(z), result(g), result(fv), result(gv), result(active)) == (
                state.z % (1 << len(z)), state.g % (1 << len(g)),
                state.fv % (1 << len(fv)), state.gv % (1 << len(gv)), 1)
        for state in (states[0][0], State(0, 0, 0, 0, 0)):
            result = run(m, _inputs(regs, state, active=0))
            assert (result(swap), result(odd)) == (0, 0)
        ok(f"p={p}: {len(states)} signed states, exact bits and clean scratch")

    section("superposition phase and untouched coefficient registers")
    m, regs = _build(2, 2, 4, "and")
    selector = m.alloc(1, "selector")
    qc = QuantumCircuit(*m.qc.qregs)
    z, g, fv, gv, active, swap, odd = regs
    qc.h(selector[0])
    qc.x(active[0])
    for reg, value in ((g, -1), (fv, 4), (gv, -1)):
        for i, wire in enumerate(reg):
            if (value >> i) & 1:
                qc.x(wire)
    for i in range(1, len(gv)):
        qc.cx(selector[0], gv[i])
    qc.compose(m.qc, inplace=True)
    qc.x(odd[0])
    qc.cx(selector[0], odd[0])
    for i in range(1, len(gv)):
        qc.cx(selector[0], gv[i])
    for reg, value in ((g, -1), (fv, 4), (gv, -1)):
        for i, wire in enumerate(reg):
            if (value >> i) & 1:
                qc.x(wire)
    qc.x(active[0])
    qc.h(selector[0])
    qc.measure_all()
    simulator = AerSimulator(method="matrix_product_state", device="CPU")
    counts = simulator.run(
        transpile(qc, simulator, optimization_level=0),
        shots=32).result().get_counts()
    assert {int(key.replace(" ", ""), 2) for key in counts} == {0}, counts
    ok("ambiguous odd decision preserves relative phase; all scratch clean")


if __name__ == "__main__":
    main()
    print("\ntest_ec_divsteps_arith_codec: all passed")
