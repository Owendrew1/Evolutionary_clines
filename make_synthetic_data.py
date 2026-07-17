"""
Generate synthetic SEC-style TSV files so the analysis pipeline can be tested
end-to-end without the real simulation output.

The synthetic data deliberately encodes the biology we expect:
  - pop_id indexes spatial position along a transect.
  - pop_size decreases with pop_id (drift strengthens down-gradient).
  - phenotype = clinal_signal(pop_id) * regime_factor + drift_noise
    where drift_noise grows as pop_size shrinks, and regime_factor shrinks the
    cline as migration increases (higher migration smooths the cline).

This lets you sanity-check that the analysis recovers:
  - steeper slopes at lower migration,
  - weaker/noisier clines where pop_size is small,
  - class-specific slope magnitudes.

Usage:
    python make_synthetic_data.py --out data --n-files 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPISTASIS_PREFIXES = ["RE", "DE", "DI", "DRE", "DDE", "PO", "NE"]

# Per-class "true" clinal slope strength (arbitrary units) -- some classes show
# stronger spatial structure than others.
CLASS_STRENGTH = {
    "RE": 1.0,
    "DE": 0.7,
    "DI": -0.5,   # inhibitory -> negative direction
    "DRE": 0.9,
    "DDE": 0.4,
    "PO": 1.2,
    "NE": -0.8,
}

# Higher migration -> smaller factor -> flatter cline.
REGIME_FACTOR = {"mig0": 1.0, "mig0.01": 0.6, "mig0.05": 0.25}


def make_one_file(rng: np.random.Generator, regime: str,
                  n_pops: int = 20, n_gens: int = 10, cols_per_class: int = 2) -> pd.DataFrame:
    pop_ids = np.arange(1, n_pops + 1)
    # pop_size decreases along the transect (linear-ish with a little noise).
    base_size = np.linspace(1000, 100, n_pops)
    factor = REGIME_FACTOR[regime]

    records = []
    for gen in range(1, n_gens + 1):
        size_noise = rng.normal(0, 15, n_pops)
        pop_size = np.clip(base_size + size_noise, 20, None).astype(int)
        # drift SD is larger where pop_size is small.
        drift_sd = 30.0 / np.sqrt(pop_size)

        row_base = {}
        for pid, psize, dsd in zip(pop_ids, pop_size, drift_sd):
            rec = {"generation": gen, "pop_id": int(pid), "pop_size": int(psize)}
            # normalized spatial coordinate in [0, 1]
            x = (pid - 1) / (n_pops - 1)
            for cls in EPISTASIS_PREFIXES:
                strength = CLASS_STRENGTH[cls] * factor
                signal = 50 + strength * 40 * x  # linear cline
                for k in range(1, cols_per_class + 1):
                    noise = rng.normal(0, dsd * 20)
                    rec[f"{cls}_{k}"] = signal + noise
            # additive dosage columns to be ignored by the analysis
            rec["add_dosage_1"] = rng.normal(10, 1)
            rec["additive_2"] = rng.normal(5, 0.5)
            records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--n-files", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    for regime in REGIME_FACTOR:
        d = a.out / regime
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, a.n_files + 1):
            df = make_one_file(rng, regime)
            fp = d / f"{regime}_rep{i:02d}.tsv"
            df.to_csv(fp, sep="\t", index=False)
            print(f"[write] {fp}")


if __name__ == "__main__":
    main()
