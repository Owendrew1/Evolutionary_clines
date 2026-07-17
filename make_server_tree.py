"""
Build a fake 'server' directory to test select_files.py + analyze_clines.py
without touching the real cluster.

Mimics the real layout: thousands of files named `sim_mig{X}_rep{N}.tsv`, each
file containing ALL epistasis classes as columns. Regime is encoded only in the
filename; files sit flat under results/slim (the hardest case for detection).

    fake_server/results/slim/sim_mig0_rep0007.tsv
    fake_server/results/slim/sim_mig0.01_rep0031.tsv
    fake_server/results/slim/sim_mig0.05_rep0123.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from make_synthetic_data import make_one_file, REGIME_FACTOR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("fake_server/results/slim"))
    p.add_argument("--per-regime", type=int, default=300, help="files per migration regime")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n = 0
    for regime in REGIME_FACTOR:
        for rep in range(a.per_regime):
            df = make_one_file(rng, regime)  # all epistasis columns included
            df.to_csv(a.out / f"sim_{regime}_rep{rep:04d}.tsv", sep="\t", index=False)
            n += 1
    print(f"[done] wrote {n} files under {a.out.resolve()}")


if __name__ == "__main__":
    main()
