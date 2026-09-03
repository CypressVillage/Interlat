"""Minimum detectable effect for the 140-episode gate (protocol §8).

Fixed scenario grid: baseline success b in {0.10, 0.25, 0.40, 0.55} x
paired-discordance rate d in {0.10, 0.20, 0.30, 0.40} x effect e in 0.01..0.20
(step 0.01). For each (d, e): discordant pairs m = round(n*d); favor
probability p = 0.5 + e/(2d); one-sided exact binomial test with
alternative "Matched > Control" at alpha = 0.00625 (Bonferroni-adjusted
98.75% two-sided). Critical value k* = min{k : P_{Bin(m,0.5)}(X >= k) <= alpha};
power = P_{Bin(m,p)}(X >= k*). Combos with e > d violate the binary paired
probability constraint (p01 = (d-e)/2 < 0) and are skipped. Reported: the
smallest e achieving power >= 0.80 for each (b, d); b does not enter the
discordant-pair test and is kept in the artifact for the preregistered grid.

Runs with no arguments and writes a fixed example artifact.
"""

from __future__ import annotations

import argparse
import math
import os

from step1.common import write_json

N = 140
ALPHA = 0.00625
BASELINES = [0.10, 0.25, 0.40, 0.55]
DISCORDANCE = [0.10, 0.20, 0.30, 0.40]
EFFECTS = [round(0.01 * k, 2) for k in range(1, 21)]
POWER_TARGET = 0.80


def binom_sf(k: int, m: int, p: float) -> float:
    if k > m:
        return 0.0
    total = 0.0
    for i in range(k, m + 1):
        total += math.comb(m, i) * (p ** i) * ((1 - p) ** (m - i))
    return min(total, 1.0)


def crit_k(m: int) -> int:
    for k in range(m + 1):
        if binom_sf(k, m, 0.5) <= ALPHA:
            return k
    return m + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    table = []
    for d in DISCORDANCE:
        m = round(N * d)
        k = crit_k(m)
        for e in EFFECTS:
            if e > d:
                table.append({"discordance": d, "effect": e, "m_discordant": m,
                              "favor_p": None, "power": None, "note": "skipped_invalid_constraint"})
                continue
            p = 0.5 + e / (2 * d)
            power = binom_sf(k, m, p)
            table.append({"discordance": d, "effect": e, "m_discordant": m,
                          "crit_k": k, "favor_p": round(p, 6), "power": round(power, 4),
                          "note": None})
    min_effect = {}
    for d in DISCORDANCE:
        found = next((r["effect"] for r in table
                      if r["discordance"] == d and r["power"] is not None and r["power"] >= POWER_TARGET),
                     None)
        min_effect[f"d={d}"] = found

    artifact = {
        "n_episodes": N, "alpha_one_sided": ALPHA, "power_target": POWER_TARGET,
        "baselines": BASELINES, "discordance": DISCORDANCE, "effects": EFFECTS,
        "min_detectable_effect_at_80pct_power": min_effect,
        "note": "baseline success rate does not enter the discordant-pair exact binomial test; kept for grid fidelity",
        "table": table,
    }
    write_json(os.path.join(args.out_dir, "power_check.json"), artifact)
    print("[OK] power_check.json:", min_effect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
