"""Rotated surface-code memory (Z basis) under circuit-level depolarising noise,
decoded with minimum-weight perfect matching (PyMatching / sparse blossom).

For each distance d and physical error rate p the circuit runs d rounds; the
logical error per shot eps is converted to an error per round via
    eps_round = (1 - (1 - 2 eps)^(1/d)) / 2 .
Output: data/surface_threshold.dat with columns d p shots errors eps eps_round
"""
import sys, time, numpy as np, stim, pymatching

out = sys.argv[1] if len(sys.argv) > 1 else "surface_threshold.dat"
distances = [3, 5, 7, 9]
ps = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.015]
MAX_SHOTS = 12_000_000
MIN_ERRORS = 1000
BATCH = 50_000
rng = np.random.default_rng(12345)
rows = []
t0 = time.time()
for d in distances:
    for p in ps:
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=p, before_round_data_depolarization=p,
            before_measure_flip_probability=p, after_reset_flip_probability=p)
        dem = circ.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circ.compile_detector_sampler(seed=int(rng.integers(1 << 30)))
        shots = errors = 0
        while shots < MAX_SHOTS and errors < MIN_ERRORS:
            dets, obs = sampler.sample(shots=BATCH, separate_observables=True)
            pred = matcher.decode_batch(dets)
            errors += int(np.sum(np.any(pred != obs, axis=1)))
            shots += BATCH
        eps = errors / shots
        eps_round = (1 - (1 - 2 * eps) ** (1 / d)) / 2 if eps < 0.5 else 0.5
        rows.append((d, p, shots, errors, eps, eps_round))
        print(f"d={d} p={p:.4f} shots={shots} errors={errors} eps={eps:.3e} "
              f"eps_round={eps_round:.3e}  [{time.time()-t0:.0f}s]", flush=True)
with open(out, "w") as f:
    f.write("d p shots errors eps eps_round\n")
    for r in rows:
        f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]:.6e} {r[5]:.6e}\n")
print("wrote", out)
