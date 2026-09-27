"""Physical cost estimates: logical counts -> noisy qubits and wall-clock time.

Everything else in this package stops at the logical layer: qubits, Toffolis,
depths.  Those decide what an error-corrected machine must do, but not how big
it is or how long it runs.  This module turns them into both, with the
surface-code model of Gidney 2025 (arXiv:2505.15917 Sec 3.2):

  hot storage    ordinary distance-d patches, 2 (d + 1)^2 qubits per logical
                 qubit; d is the smallest distance whose per-patch per-round
                 logical error meets the target (1e-15)
  cold storage   idle qubits in *yoked* surface codes -- about 430 physical
                 qubits per logical qubit at the same target, roughly triple
                 the density of hot patches
  compute        a 7 x 18 region of hot patches: six CCZ factories of 3 x 4
                 patches each (magic-state cultivation, then 8T -> CCZ
                 distillation) and three columns of lattice-surgery workspace
  timing         a CCZ state per factory every ~150 rounds (cultivation of 8
                 T states in the factory footprint, plus 6 layers of
                 temporally encoded surgery at 2d/3 rounds, rounded up); with
                 six factories, one CCZ per 25 us at d = 25

`estimate(profile)` applies the model to any `LogicalProfile` -- one measured
off this package's circuits or restated from a paper.  `g25_rsa2048()`
reproduces the paper's headline, 897,864 qubits and 4.96 days, from its own
inputs, and is the validation target in the tests.

QLDPC architectures (Pinnacle, Cain et al.) are not modelled: their costs
depend on code tables, processing-unit schedules and connectivity this module
does not have.  Their published figures are kept in REFERENCE for comparison.
"""

import math
from dataclasses import dataclass, field


@dataclass
class SurfaceCode:
    p: float = 1e-3               # physical error rate
    cycle_us: float = 1.0         # surface-code cycle
    reaction_us: float = 10.0     # control-system reaction time
    target: float = 1e-15         # per-patch per-round logical error
    # p_L(d) = a (p / p_th)^((d+1)/2), calibrated so that d = 25 is the
    # smallest distance reaching 1e-15 at p = 1e-3, as Gidney 2025 Fig. 6 reads
    a: float = 0.1
    p_th: float = 0.0122
    cold_per_logical: float = 430.0      # yoked storage at the same target
    factories: int = 6
    factory_patches: tuple = (3, 4)
    compute_patches: tuple = (7, 18)
    cultivation_qubit_rounds: float = 30000.0   # per T state at 1e-7 [GSJ24]
    factory_layers: int = 6
    rounds_per_ccz_slack: float = 150.0          # 114.7 rounded up for slack

    def logical_error(self, d):
        return self.a * (self.p / self.p_th) ** ((d + 1) / 2)

    def distance(self):
        d = 3
        while self.logical_error(d) > self.target:
            d += 2
        return d

    def hot_per_logical(self, d=None):
        d = d or self.distance()
        return 2 * (d + 1) ** 2

    def ccz_rounds(self, d=None):
        """(cultivation rounds, surgery rounds, total) per CCZ per factory."""
        d = d or self.distance()
        fx, fy = self.factory_patches
        footprint = fx * fy * self.hot_per_logical(d)
        cult = 8 * self.cultivation_qubit_rounds / footprint
        surgery = self.factory_layers * 2 * d / 3
        return cult, surgery, cult + surgery

    def ccz_period_us(self, d=None):
        return self.rounds_per_ccz_slack / self.factories * self.cycle_us


@dataclass
class LogicalProfile:
    label: str
    cold: int                       # idle logical qubits (yoked storage)
    hot: int                        # active logical qubits
    toffoli: float                  # Toffolis per shot
    reaction_depth: float = 0.0     # sequential reactions per shot (optional)
    hours_per_shot: float = None    # overrides the runtime model (a paper's own)
    shots: float = 1.0              # expected shots per solution
    # the logical-qubit x time budget the code distance is chosen to protect;
    # defaults to the actual counts, a paper may round it up
    protect_logical: float = None
    protect_hours: float = None
    extra: dict = field(default_factory=dict)


def estimate(prof, code=None):
    """{d, physical qubits (by region), hours per shot, no-error rate, days}."""
    code = code or SurfaceCode()
    d = code.distance()
    hot_q = code.hot_per_logical(d)
    cx, cy = code.compute_patches
    compute = cx * cy
    phys = {"cold": prof.cold * code.cold_per_logical,
            "hot": prof.hot * hot_q,
            "compute": compute * hot_q}
    total = sum(phys.values())
    if prof.hours_per_shot is not None:
        hours = prof.hours_per_shot
        runtime_model = "given"
    else:
        t_ccz = prof.toffoli * code.ccz_period_us(d) * 1e-6
        t_react = prof.reaction_depth * code.reaction_us * 1e-6
        hours = max(t_ccz, t_react) / 3600
        runtime_model = "CCZ-throughput" if t_ccz >= t_react else "reaction"
    logical_all = prof.cold + prof.hot + compute
    prot_q = prof.protect_logical or logical_all
    prot_h = prof.protect_hours or hours
    rounds = prot_h * 3600 / (code.cycle_us * 1e-6)
    no_error = (1 - code.target) ** (prot_q * rounds)
    days = hours * prof.shots / 24 / no_error
    return {"label": prof.label, "distance": d, "hot_per_logical": hot_q,
            "physical": phys, "physical_total": total,
            "logical_including_idle": logical_all, "hours_per_shot": hours,
            "runtime_model": runtime_model, "no_error_rate": no_error,
            "days": days, "ccz_rounds": code.ccz_rounds(d),
            "ccz_period_us": code.ccz_period_us(d)}


def g25_rsa2048(hot="stated"):
    """Gidney 2025's RSA-2048 inputs (Table 5 row n = 2048 and Sec 3.2):
    m = 1280 cold input qubits, 6.5e9 Toffolis per factoring, 12.07 hours per
    shot, 9.2 expected shots.

    The hot count is where the paper disagrees with itself: Sec 3.2 writes
    "3f + 2l + len m = 131", but with f = 33, l = 21 and len(1280) = 11 that
    formula is 152 (and 1280 + 152 = 1432, not the stated 1409 peak).  The
    897,864-qubit headline uses 131.  hot="stated" reproduces the paper;
    hot="formula" uses 152 and gives 926,256."""
    m, f, l = 1280, 33, 21
    h = 131 if hot == "stated" else 3 * f + 2 * l + m.bit_length()
    # the paper protects "fewer than 1600" logical qubits for "roughly 12
    # hours": 6.9e13 qubit-rounds -> 93.3% (the exact 1537 x 12.07 h gives 93.5%)
    return LogicalProfile("RSA-2048, Gidney 2025", cold=m, hot=h,
                          toffoli=6.5e9 / 9.2, hours_per_shot=12.07, shots=9.2,
                          protect_logical=1600, protect_hours=12,
                          extra={"peak_logical": m + h, "hot_source": hot})


def ecdlp_profile(label, qubits, toffoli, additions=None, cold_fraction=0.0):
    """A windowed ECDLP run: everything hot unless told otherwise (the point
    addition keeps almost all its qubits busy)."""
    cold = int(qubits * cold_fraction)
    return LogicalProfile(label, cold=cold, hot=qubits - cold, toffoli=toffoli,
                          extra={"additions": additions})


# Published QLDPC / neutral-atom / trapped-ion estimates, recorded, not modelled.
REFERENCE = {
    "Pinnacle (arXiv:2602.11457 v2, Table VI, p=1e-3, 1 us, 10 us)": {
        "RSA-2048": [("94k qubits", "~1 month"), ("135k", "~1 week"), ("381k", "~1 day")]},
    "Cain et al. (arXiv:2603.28627)": {
        "ECC-256": [("9,739", "~1000 days"), ("11,961", "264 days"),
                    ("19k", "52 days"), ("26k", "10 days")],
        "RSA-2048": [("11,033", "4.3e4 days"), ("13,255", "1.0e4 days"),
                     ("68k", "870 days"), ("102k", "97 days")]},
    "IonQ (arXiv:2609.05625)": {"ECDLP secp256k1": [("19,397 ions", "25.7 days/attempt")]},
    "Babbush et al. (arXiv:2603.28846)": {"ECDLP secp256k1": [("<500k", "18-23 min")]},
}
