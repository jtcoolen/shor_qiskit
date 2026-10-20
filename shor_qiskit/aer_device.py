"""One place to pick the Aer device: CUDA GPU when available, CPU otherwise.

`SHOR_AER_DEVICE` = "auto" (default), "GPU" or "CPU".  "GPU" fails loudly if
qiskit-aer-gpu is not installed (CUDA builds exist only for Linux/NVIDIA);
"auto" falls back to CPU and says so in `describe()`.

Method.  The ECDLP arithmetic oracle is 40+ qubits wide, far beyond a
statevector, but it is a permutation of basis states with only the 2m control
qubits in superposition, so its entanglement is tiny.  On CPU that is a job for
the matrix-product-state method; on GPU Aer offers `tensor_network`
(cuTensorNet) for the same purpose and `statevector` for narrow circuits.
"""

import os
from functools import lru_cache

STATEVECTOR_MAX = 28


@lru_cache(maxsize=None)
def available_devices():
    from qiskit_aer import AerSimulator
    return tuple(AerSimulator().available_devices())


def device():
    want = os.environ.get("SHOR_AER_DEVICE", "auto").upper()
    have = available_devices()
    if want == "AUTO":
        return "GPU" if "GPU" in have else "CPU"
    if want not in have:
        raise RuntimeError(
            f"SHOR_AER_DEVICE={want} but Aer offers {have}; install "
            "qiskit-aer-gpu on a CUDA machine or use auto/CPU")
    return want


def method_for(num_qubits, dev=None):
    dev = dev or device()
    if num_qubits <= STATEVECTOR_MAX:
        return "statevector"
    return "tensor_network" if dev == "GPU" else "matrix_product_state"


def simulator(num_qubits, seed=None, **kw):
    """An AerSimulator on the chosen device with a method that fits the width."""
    from qiskit_aer import AerSimulator
    dev = device()
    opts = {"method": method_for(num_qubits, dev), "device": dev}
    if seed is not None:
        opts["seed_simulator"] = seed
    opts.update(kw)
    return AerSimulator(**opts)


def run(qc, shots=2048, seed=None, **kw):
    """Transpile for, and sample on, the chosen device.  Returns counts."""
    from qiskit import transpile
    sim = simulator(qc.num_qubits, seed, **kw)
    return sim.run(transpile(qc, sim, optimization_level=0),
                   shots=shots).result().get_counts()


def describe(num_qubits=None):
    dev = device()
    s = f"Aer {dev}"
    if num_qubits is not None:
        s += f"/{method_for(num_qubits, dev)}"
    if dev == "CPU" and "GPU" not in available_devices():
        s += " (no CUDA GPU on this host)"
    return s
