def _maj(qc, c, b, a, ctrls=()):
    qc.mcx([*ctrls, a], b)               # plain CNOT when ctrls is empty
    qc.cx(a, c)
    qc.ccx(c, b, a)

def _uma(qc, c, b, a, ctrls=()):
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.mcx([*ctrls, c], b)               # plain CNOT when ctrls is empty
