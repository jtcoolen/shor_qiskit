"""Physical storage estimate from Gidney arXiv:2505.15917 Sec. 3.2.

This is not a surface-code circuit, noise simulator, or an arithmetic
optimization. The paper quotes 430 physical qubits per cold logical qubit
for a 2D-yoked code at physical error 0.1% and target logical error
1e-15 per round; hot distance-25 patches occupy 1352 each. No scaling
law for other physical/logical error targets is asserted here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageFootprint:
    cold: int
    hot: int
    compute: int
    no_yoking: int

    @property
    def total(self):
        return self.cold + self.hot + self.compute

    @property
    def saved(self):
        return self.no_yoking - self.total


def estimate(cold_logical, hot_logical, *, compute_patches=0,
             distance=25, physical_error=0.001, target_error=1e-15):
    """Estimate physical footprint under the paper's fixed error assumptions.

    `compute_patches` counts ordinary hot-distance-25 surface-code patches,
    including routing/factories if those are represented in the input. No
    hidden factory overhead is added.
    """
    if (any(type(value) is not int or value < 0 for value in (
            cold_logical, hot_logical, compute_patches))
            or distance != 25 or physical_error != 0.001
            or target_error != 1e-15):
        raise ValueError("requires nonnegative counts and Gidney Sec. 3.2 "
                         "distance-25, p=0.001, target=1e-15 assumptions")
    hot_patch = 2 * (distance + 1) ** 2
    compute = compute_patches * hot_patch
    return StorageFootprint(
        cold=cold_logical * 430,
        hot=hot_logical * hot_patch,
        compute=compute,
        no_yoking=(cold_logical + hot_logical) * hot_patch + compute)
