"""
Randomly select a subset of SEC simulation files from the server, per regime.

Why this exists
---------------
The real output on the cluster (under
`/scratch/jsantang/epistasis_clines/results/slim`) has thousands of files named
like `sim_mig0_rep983.tsv`. For a first pass we want a small, random, but
balanced subset so the analysis is quick and not biased toward whatever sorts
first.

What "balanced" means here
--------------------------
Each simulation file already contains ALL epistasis classes as columns
(RE, DE, DI, DRE, DDE, PO, NE). So we cannot -- and do not need to -- pick files
"by epistasis type": every file we draw covers all classes. The epistasis
SEPARATION happens later, in the analysis, which reports each class on its own
(see analyze_clines.py summaries/figures).

Therefore selection is stratified by the ONE file-level factor that varies:

    migration regime  (mig0, mig0.01, mig0.05)  ->  N random files each

with a fixed seed so the selection is reproducible.

Staging
-------
Selected files are staged into:

    <out>/mig0/<file>
    <out>/mig0.01/<file>
    <out>/mig0.05/<file>

By default it SYMLINKS (instant, no copy) so it is cheap on the cluster. Use
`--copy` to physically copy instead.

Typical use on the server
--------------------------
    # 1) Inspect what's there and what WOULD be selected (touches nothing):
    python select_files.py \
        --root /scratch/jsantang/epistasis_clines/results/slim \
        --n 10 --dry-run

    # 2) Stage 10 random files per regime into ./selected:
    python select_files.py \
        --root /scratch/jsantang/epistasis_clines/results/slim \
        --n 10 --out selected

    # 3) Run the analysis on the staged subset:
    python analyze_clines.py \
        --mig0 selected/mig0 --mig0.01 selected/mig0.01 --mig0.05 selected/mig0.05 \
        --out outputs --n-files -1
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Migration regimes, most-specific first so "mig0" doesn't swallow
# "mig0.01"/"mig0.05".
DEFAULT_REGIME_TOKENS: List[str] = ["mig0.05", "mig0.01", "mig0"]


def detect_regime(rel_path: str, regime_tokens: List[str]) -> Optional[str]:
    """Return the migration regime token found in the path/filename, or None.

    Matched most-specific-first, and we forbid a trailing digit/dot so 'mig0'
    does not match inside 'mig0.05'/'mig0.01'.
    """
    for tok in regime_tokens:  # already ordered specific -> general
        pat = re.escape(tok) + r"(?![0-9.])"
        if re.search(pat, rel_path):
            return tok
    return None


def scan(root: Path, glob: str, regime_tokens: List[str]) -> Dict[str, List[Path]]:
    """Walk `root`, bucketing files by migration regime."""
    cells: Dict[str, List[Path]] = defaultdict(list)
    n_seen = n_bucketed = 0
    for f in root.rglob(glob):
        if not f.is_file():
            continue
        n_seen += 1
        regime = detect_regime(str(f.relative_to(root)), regime_tokens)
        if regime is None:
            continue
        cells[regime].append(f)
        n_bucketed += 1
    print(f"[scan] saw {n_seen} file(s), assigned {n_bucketed} to a migration regime",
          file=sys.stderr)
    return cells


def sample_cells(cells: Dict[str, List[Path]], n: int, seed: int) -> Dict[str, List[Path]]:
    """Randomly pick up to `n` files per regime, reproducibly."""
    rng = random.Random(seed)
    picked: Dict[str, List[Path]] = {}
    for regime, paths in cells.items():
        ordered = sorted(paths)  # deterministic base order before shuffling
        rng.shuffle(ordered)
        picked[regime] = ordered[: min(n, len(ordered))]
    return picked


def stage(picked: Dict[str, List[Path]], out: Path, copy: bool) -> None:
    """Materialize the selection into out/<regime>/<file>."""
    for regime, paths in sorted(picked.items()):
        dest_dir = out / regime
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dest = dest_dir / src.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if copy:
                shutil.copy2(src, dest)
            else:
                os.symlink(src.resolve(), dest)  # absolute link resolves from any cwd


def write_manifest(picked: Dict[str, List[Path]], out_csv: Path) -> None:
    """Record exactly which files were chosen (auditable, reproducible)."""
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "selected_path", "filename"])
        for regime, paths in sorted(picked.items()):
            for p in paths:
                w.writerow([regime, str(p), p.name])


def print_report(cells: Dict[str, List[Path]], picked: Dict[str, List[Path]],
                 regime_tokens: List[str]) -> None:
    """Human-readable available-vs-selected counts per regime."""
    print("\n=== files per migration regime: available (selected) ===")
    for regime in regime_tokens:
        avail = len(cells.get(regime, []))
        sel = len(picked.get(regime, []))
        if avail:
            print(f"  {regime:>8}: {avail} available  ->  {sel} selected")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Random per-regime file selection for SEC analysis")
    p.add_argument("--root", type=Path, required=True,
                   help="root dir to search recursively (e.g. .../results/slim)")
    p.add_argument("--out", type=Path, default=Path("selected"),
                   help="staging dir; files go to <out>/<regime>/")
    p.add_argument("--n", type=int, default=10, help="files per migration regime")
    p.add_argument("--glob", default="*.tsv", help="filename pattern (recursive)")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for reproducible selection")
    p.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    p.add_argument("--regime-tokens", nargs="+", default=DEFAULT_REGIME_TOKENS,
                   help="migration tokens, most-specific first")
    p.add_argument("--dry-run", action="store_true",
                   help="scan and report only; do not stage any files")
    a = p.parse_args()

    if not a.root.exists():
        sys.exit(f"[error] root not found: {a.root}")

    cells = scan(a.root, a.glob, a.regime_tokens)
    if not cells:
        sys.exit("[error] no files matched a migration regime. "
                 "Run with --dry-run and check --glob / --regime-tokens / filenames.")

    picked = sample_cells(cells, a.n, a.seed)
    print_report(cells, picked, a.regime_tokens)

    if a.dry_run:
        print("[dry-run] nothing staged. Re-run without --dry-run to materialize.")
        return

    stage(picked, a.out, a.copy)
    write_manifest(picked, a.out / "selection_manifest.csv")
    total = sum(len(v) for v in picked.values())
    mode = "copied" if a.copy else "symlinked"
    print(f"[done] {mode} {total} file(s) into {a.out.resolve()}")
    print(f"[done] manifest -> {(a.out / 'selection_manifest.csv').resolve()}")


if __name__ == "__main__":
    main()
