"""Pre-run paired-outcome power simulation (Step 1A protocol §8).

For each feasible baseline/disagreement/effect setting, simulate 10,000
independent sets of 140 paired episodes. Apply the one-sided exact discordant-
pair binomial test (Matched > Control) at Bonferroni alpha=0.00625. This is
only a power diagnostic, not the paired-bootstrap analysis of real outcomes.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from step1.common import write_json

N = 140
N_SIM = 10_000
SEED = 20260820
ALPHA = 0.00625
BASELINES = [0.10, 0.25, 0.40, 0.55]
DISCORDANCE = [0.10, 0.20, 0.30, 0.40]
EFFECTS = [round(0.01 * k, 2) for k in range(1, 21)]
POWER_TARGET = 0.80


def joint_probs(baseline: float, discordance: float, effect: float):
    """Order: both win, matched-only, control-only, both fail."""
    p10 = (discordance + effect) / 2
    p01 = (discordance - effect) / 2
    probs = (baseline - p01, p10, p01, 1 - baseline - p10)
    if min(probs) < -1e-10 or max(probs) > 1 + 1e-10:
        return None
    return tuple(max(0., p) for p in probs)


def critical_values(n: int, alpha: float):
    """Smallest favorable-discordance count with exact upper-tail p <= alpha."""
    result = []
    for m in range(n + 1):
        tail = 0.
        critical = m + 1
        for k in range(m, -1, -1):
            tail += math.comb(m, k) * 2. ** (-m)
            if tail <= alpha + 1e-15:
                critical = k
            else:
                break
        result.append(critical)
    return np.array(result, dtype=np.int16)


def simulate_power(rng, probabilities, critical):
    counts = rng.multinomial(N, probabilities, size=N_SIM)
    favorable = counts[:, 1]
    discordant = counts[:, 1] + counts[:, 2]
    return float(np.mean(favorable >= critical[discordant]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    critical = critical_values(N, ALPHA)
    table = []
    for b in BASELINES:
        for d in DISCORDANCE:
            for e in EFFECTS:
                probs = joint_probs(b, d, e)
                table.append({
                    "control_success": b, "discordance": d, "effect": e,
                    "joint_probs_11_10_01_00": probs,
                    "power": None if probs is None else simulate_power(rng, probs, critical),
                    "note": "skipped_invalid_joint_distribution" if probs is None else None,
                })
    min_effect = {}
    for b in BASELINES:
        for d in DISCORDANCE:
            key = f"control={b},discordance={d}"
            min_effect[key] = next((r["effect"] for r in table
                                    if r["control_success"] == b and r["discordance"] == d
                                    and r["power"] is not None and r["power"] >= POWER_TARGET), None)

    artifact = {
        "n_episodes": N, "n_simulations_per_scenario": N_SIM, "seed": SEED,
        "alpha_one_sided": ALPHA, "power_target": POWER_TARGET,
        "baselines": BASELINES, "discordance": DISCORDANCE, "effects": EFFECTS,
        "min_detectable_effect_at_80pct_power": min_effect, "table": table,
        "note": "Exact binomial test on discordant pairs; independent paired samples per scenario. Not the formal paired-bootstrap CI.",
    }
    write_json(os.path.join(args.out_dir, "power_check.json"), artifact)
    print("[OK] power_check.json:", min_effect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
