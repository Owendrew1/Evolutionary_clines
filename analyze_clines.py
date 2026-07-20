"""
Simulating Evolutionary Clines (SEC) -- first-pass validation workflow.

Goal of this script
-------------------
Take simulation output (TSV files) for three migration regimes (mig0, mig0.01,
mig0.05), and answer a few concrete questions before any deeper analysis:

  1. Are the files structurally sane (columns, generations, spatial range,
     decreasing population size, numeric phenotypes)?
  2. Does phenotype change smoothly with spatial position (pop_id)? -> slope
  3. Is the slope direction consistent across generations / replicates?
  4. Does lower migration show stronger spatial structure (steeper slope)?
  5. Do the epistasis classes differ in slope strength or variability?

Design
------
- Pure functions for load / validate / slope / summarize, orchestrated in main().
- You point it at one folder per migration regime. It grabs the first N files
  (default 5) so the first pass is cheap; bump --n-files to run everything.
- Epistasis phenotype columns are auto-detected by prefix. Additive dosage
  columns are ignored for the main analysis (matched separately and skipped).
- Slope = ordinary least squares of phenotype vs pop_id, computed *within each
  generation* of each file, then summarized across generations.

Run
---
    python analyze_clines.py \
        --mig0 data/mig0 --mig0.01 data/mig0.01 --mig0.05 data/mig0.05 \
        --out outputs --n-files 5

Everything below is written to be readable and defensible, not clever.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Matplotlib needs a writable config dir; keep it inside the project so the
# script runs on machines where $HOME/.matplotlib is not writable.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    _HAS_SEABORN = True
    sns.set_theme(style="whitegrid", context="talk")
except Exception:  # seaborn optional -- fall back to plain matplotlib
    _HAS_SEABORN = False
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Epistasis class prefixes. Phenotype columns are expected to look like
# "RE_1", "DE_2", "DRE", etc. The regex matches a prefix only when followed by
# an underscore, a digit, or the end of the name, so "DE" does NOT swallow
# "DDE" or "DRE".
EPISTASIS_PREFIXES: List[str] = ["RE", "DE", "DI", "DRE", "DDE", "PO", "NE"]

EPISTASIS_LABELS: Dict[str, str] = {
    "RE": "recessive epistasis",
    "DE": "dominant epistasis",
    "DI": "dominant inhibitory",
    "DRE": "dominant recessive epistasis",
    "DDE": "duplicate dominant epistasis",
    "PO": "positive epistasis",
    "NE": "negative epistasis",
}

# Columns we expect to key the analysis on. Adjust here if your headers differ.
COL_GENERATION = "generation"
COL_POP_ID = "pop_id"
COL_POP_SIZE = "pop_size"

# Columns to ignore in the phenotype slope analysis (additive dosages, allele
# freqs, metadata). Real files use ADD_d0..ADD_d4 and freq_domA/freq_domB.
SKIP_COLUMN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [r"^ADD_", r"add", r"dosage", r"dose", r"^freq_", r"^max_mig", r"^replicate$", r"^pop_max$", r"^pop_min$"]
]


@dataclass
class ClineConfig:
    """Where the data lives and where results go."""

    regimes: Dict[str, Path]  # e.g. {"mig0": Path("data/mig0"), ...}
    out_dir: Path
    n_files: int = 5
    file_glob: str = "*.tsv"
    representative_only: bool = False  # True -> one column per class only


# ---------------------------------------------------------------------------
# 1. File handling
# ---------------------------------------------------------------------------

def discover_files(folder: Path, glob: str, limit: Optional[int]) -> List[Path]:
    """Return up to `limit` files from `folder`, sorted for reproducibility."""
    if not folder.exists():
        raise FileNotFoundError(f"Regime folder not found: {folder}")
    files = sorted(folder.glob(glob))
    if not files:
        # Fall back to .txt if the TSVs use another extension.
        files = sorted(folder.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    return files


def load_file(path: Path) -> pd.DataFrame:
    """Load one TSV, tagging it with its source filename for traceability."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df["__source_file"] = path.name
    return df


# ---------------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------------

def should_skip_column(name: str) -> bool:
    return any(p.search(name) for p in SKIP_COLUMN_PATTERNS)


def classify_epistasis_column(name: str) -> Optional[str]:
    """Return the epistasis class prefix a column belongs to, or None.

    Real SEC columns look like RE_pheno1, DDE_pheno2. Matching is
    longest-prefix-first so "DE" never claims "DDE_*"/"DRE_*".
    """
    if should_skip_column(name):
        return None
    # Longest prefixes first (DRE, DDE before DE/DI/RE).
    for prefix in sorted(EPISTASIS_PREFIXES, key=len, reverse=True):
        # RE_pheno1, RE_1, or bare RE
        if re.match(rf"^{prefix}(_pheno|_|\d|$)", name, re.IGNORECASE):
            return prefix
    return None


def identify_phenotype_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map each epistasis class -> list of its phenotype column names."""
    mapping: Dict[str, List[str]] = {p: [] for p in EPISTASIS_PREFIXES}
    for col in df.columns:
        cls = classify_epistasis_column(col)
        if cls is not None:
            mapping[cls].append(col)
    return {k: v for k, v in mapping.items() if v}


def representative_columns(pheno_map: Dict[str, List[str]]) -> Dict[str, str]:
    """Pick one representative column per class (first, sorted)."""
    return {cls: sorted(cols)[0] for cls, cols in pheno_map.items()}


# ---------------------------------------------------------------------------
# 2. Data-structure checks
# ---------------------------------------------------------------------------

def validate_file(df: pd.DataFrame, path: Path, pheno_map: Dict[str, List[str]]) -> Dict:
    """Run structural sanity checks. Returns a flat dict (one row per file)."""
    checks: Dict[str, object] = {"file": path.name, "n_rows": len(df)}

    # Required key columns present?
    required = [COL_GENERATION, COL_POP_ID, COL_POP_SIZE]
    missing = [c for c in required if c not in df.columns]
    checks["missing_key_cols"] = ",".join(missing) if missing else ""
    checks["has_key_cols"] = not missing

    # Any phenotype columns found?
    all_pheno = [c for cols in pheno_map.values() for c in cols]
    checks["n_pheno_cols"] = len(all_pheno)
    checks["epistasis_classes_found"] = ",".join(sorted(pheno_map.keys()))

    # Generation sanity.
    if COL_GENERATION in df.columns:
        g = pd.to_numeric(df[COL_GENERATION], errors="coerce")
        checks["n_generations"] = int(g.nunique())
        checks["gen_min"] = float(g.min())
        checks["gen_max"] = float(g.max())
        checks["gen_has_nan"] = bool(g.isna().any())
    # pop_id spatial range.
    if COL_POP_ID in df.columns:
        pid = pd.to_numeric(df[COL_POP_ID], errors="coerce")
        checks["n_pop_id"] = int(pid.nunique())
        checks["pop_id_min"] = float(pid.min())
        checks["pop_id_max"] = float(pid.max())

    # pop_size should decrease along the gradient (pop_id increasing).
    # Test with a per-file regression of pop_size on pop_id and its correlation.
    if COL_POP_ID in df.columns and COL_POP_SIZE in df.columns:
        sub = df[[COL_POP_ID, COL_POP_SIZE]].apply(pd.to_numeric, errors="coerce").dropna()
        # collapse to one pop_size per pop_id (mean) so replicate rows don't skew it
        agg = sub.groupby(COL_POP_ID)[COL_POP_SIZE].mean().reset_index()
        if len(agg) >= 3:
            r = np.corrcoef(agg[COL_POP_ID], agg[COL_POP_SIZE])[0, 1]
            checks["popsize_vs_popid_corr"] = float(r)
            checks["popsize_decreases"] = bool(r < 0)
        checks["pop_size_min"] = float(sub[COL_POP_SIZE].min())
        checks["pop_size_max"] = float(sub[COL_POP_SIZE].max())

    # Phenotype columns numeric & not all-NaN?
    non_numeric = []
    all_nan = []
    for c in all_pheno:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.isna().all():
            all_nan.append(c)
        elif coerced.isna().mean() > 0.0 and not np.issubdtype(df[c].dtype, np.number):
            non_numeric.append(c)
    checks["pheno_non_numeric"] = ",".join(non_numeric)
    checks["pheno_all_nan"] = ",".join(all_nan)
    checks["pheno_ok"] = (not non_numeric) and (not all_nan)

    # Overall pass flag: everything we care about is satisfied.
    checks["PASS"] = bool(
        checks.get("has_key_cols")
        and checks.get("n_pheno_cols", 0) > 0
        and checks.get("pheno_ok")
        and checks.get("popsize_decreases", True)
    )
    return checks


# ---------------------------------------------------------------------------
# 3. Slope analysis
# ---------------------------------------------------------------------------

def _ols_slope(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """OLS slope of y on x. Returns None if too few / degenerate points."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r": float(res.rvalue),
        "r2": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "n_points": int(len(x)),
    }


def slopes_for_file(
    df: pd.DataFrame,
    pheno_cols: Sequence[str],
    regime: str,
    file_name: str,
) -> pd.DataFrame:
    """Per-generation slope of each phenotype column vs pop_id, for one file."""
    rows: List[Dict] = []
    if COL_GENERATION not in df.columns or COL_POP_ID not in df.columns:
        return pd.DataFrame(rows)

    df = df.copy()
    df[COL_POP_ID] = pd.to_numeric(df[COL_POP_ID], errors="coerce")

    for gen, gdf in df.groupby(COL_GENERATION):
        x = gdf[COL_POP_ID].to_numpy(dtype=float)
        for col in pheno_cols:
            y = pd.to_numeric(gdf[col], errors="coerce").to_numpy(dtype=float)
            fit = _ols_slope(x, y)
            if fit is None:
                continue
            rows.append(
                {
                    "regime": regime,
                    "file": file_name,
                    "generation": gen,
                    "epistasis_class": classify_epistasis_column(col),
                    "column": col,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def summarize_slopes(slope_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate slopes over generations/files within the requested grouping."""
    if slope_df.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        s = g["slope"]
        return pd.Series(
            {
                "n_slopes": len(s),
                "mean_slope": s.mean(),
                "median_slope": s.median(),
                "std_slope": s.std(ddof=1) if len(s) > 1 else 0.0,
                "mean_abs_slope": s.abs().mean(),
                # fraction of slopes sharing the majority sign = directional consistency
                "frac_same_sign": max((s > 0).mean(), (s < 0).mean()),
                "mean_r2": g["r2"].mean(),
                "frac_p_lt_0.05": (g["p_value"] < 0.05).mean(),
            }
        )

    grouped = slope_df.groupby(list(group_cols), dropna=False)
    try:  # pandas >= 2.2 wants include_groups to avoid a FutureWarning
        out = grouped.apply(_agg, include_groups=False).reset_index()
    except TypeError:  # older pandas has no include_groups kwarg
        out = grouped.apply(_agg).reset_index()
    return out


# ---------------------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------------------

REGIME_ORDER = ["mig0", "mig0.01", "mig0.05"]


def _regime_sort_key(r: str) -> float:
    m = re.search(r"([0-9]*\.?[0-9]+)", r)
    return float(m.group(1)) if m else 0.0


def plot_class_cline(
    regime: str, cls: str, long_df: pd.DataFrame, out_path: Path
) -> None:
    """One figure per (migration regime x epistasis class).

    Pools all selected files, all generations, and ALL columns of the class,
    then shows phenotype vs pop_id and the fitted clinal slope:
      - each column of the class as a faint mean line (columns "put together"),
      - the pooled points summarized as mean +/- SD per pop_id,
      - a bold OLS fit line whose slope/R^2 are annotated.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Faint per-column mean lines so every column of the class is visible.
    cols = sorted(long_df["column"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(cols), 1)))
    for color, col in zip(palette, cols):
        cdf = long_df[long_df["column"] == col]
        line = cdf.groupby(COL_POP_ID)["value"].mean().reset_index()
        ax.plot(line[COL_POP_ID], line["value"], "-", lw=1, alpha=0.5,
                color=color, label=col)

    # Pooled mean +/- SD per pop_id (the summary cloud).
    summ = long_df.groupby(COL_POP_ID)["value"].agg(["mean", "std"]).reset_index()
    ax.errorbar(summ[COL_POP_ID], summ["mean"], yerr=summ["std"], fmt="o",
                ms=4, capsize=3, color="0.25", alpha=0.8, zorder=4,
                label="pooled mean +/- SD")

    # Bold OLS fit through ALL pooled points -> the clinal slope.
    x = long_df[COL_POP_ID].to_numpy(dtype=float)
    y = long_df["value"].to_numpy(dtype=float)
    fit = _ols_slope(x, y)
    if fit is not None:
        xs = np.array([np.nanmin(x), np.nanmax(x)])
        ys = fit["intercept"] + fit["slope"] * xs
        ax.plot(xs, ys, "-", color="crimson", lw=3, zorder=5,
                label=f"OLS fit (slope={fit['slope']:.3g})")
        ax.text(0.02, 0.98,
                f"slope = {fit['slope']:.3g}\nR^2 = {fit['r2']:.2f}\n"
                f"n = {fit['n_points']}  (p = {fit['p_value']:.1e})",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlabel("pop_id (spatial position)")
    ax.set_ylabel("phenotype frequency")
    ax.set_title(f"{regime} - {cls} ({EPISTASIS_LABELS.get(cls, cls)})")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ensure_dirs(out: Path) -> Dict[str, Path]:
    dirs = {
        "root": out,
        "tables": out / "tables",
        "summaries": out / "summaries",
        "figures": out / "figures",
        "cleaned": out / "cleaned",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run(config: ClineConfig) -> None:
    dirs = ensure_dirs(config.out_dir)

    all_slopes: List[pd.DataFrame] = []
    all_validation: List[Dict] = []
    # Tidy phenotype values pooled across files, for the per-(regime x class) plots.
    all_long: List[pd.DataFrame] = []

    for regime, folder in config.regimes.items():
        try:
            files = discover_files(folder, config.file_glob, config.n_files)
        except FileNotFoundError as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue
        if not files:
            print(f"[warn] no files found in {folder}", file=sys.stderr)
            continue
        print(f"[{regime}] analyzing {len(files)} file(s) from {folder}")

        for path in files:
            df = load_file(path)
            pheno_map = identify_phenotype_columns(df)

            # Validation.
            checks = validate_file(df, path, pheno_map)
            checks["regime"] = regime
            all_validation.append(checks)

            # Choose columns: representative-per-class or all.
            rep = representative_columns(pheno_map)
            if config.representative_only:
                cols = list(rep.values())
            else:
                cols = [c for cols in pheno_map.values() for c in cols]

            # Slopes.
            sdf = slopes_for_file(df, cols, regime, path.name)
            if not sdf.empty:
                all_slopes.append(sdf)

            # Pool tidy phenotype values (regime, class, column, pop_id, value)
            # across every selected file for the per-(regime x class) cline plots.
            if COL_POP_ID in df.columns and cols:
                melt = df[[COL_POP_ID] + cols].copy()
                melt[COL_POP_ID] = pd.to_numeric(melt[COL_POP_ID], errors="coerce")
                melt = melt.melt(id_vars=[COL_POP_ID], var_name="column", value_name="value")
                melt["value"] = pd.to_numeric(melt["value"], errors="coerce")
                melt["regime"] = regime
                melt["epistasis_class"] = melt["column"].map(classify_epistasis_column)
                all_long.append(melt.dropna(subset=[COL_POP_ID, "value", "epistasis_class"]))

    # ---- Assemble master tables ----
    validation_df = pd.DataFrame(all_validation)
    validation_df.to_csv(dirs["tables"] / "validation_report.csv", index=False)
    print(f"[write] {dirs['tables'] / 'validation_report.csv'}")

    if not all_slopes:
        print("[error] No slopes computed -- check that files/columns were found.", file=sys.stderr)
        return

    slope_df = pd.concat(all_slopes, ignore_index=True)
    slope_df.to_csv(dirs["cleaned"] / "per_generation_slopes.csv", index=False)
    print(f"[write] {dirs['cleaned'] / 'per_generation_slopes.csv'}")

    # ---- Summaries at several grouping levels ----
    summaries = {
        "summary_by_regime.csv": ["regime"],
        "summary_by_regime_class.csv": ["regime", "epistasis_class"],
        "summary_by_regime_class_column.csv": ["regime", "epistasis_class", "column"],
        "summary_by_regime_generation.csv": ["regime", "generation"],
    }
    for fname, gcols in summaries.items():
        summ = summarize_slopes(slope_df, gcols)
        summ.to_csv(dirs["summaries"] / fname, index=False)
        print(f"[write] {dirs['summaries'] / fname}")

    # Add human-readable epistasis labels to the class-level summary.
    class_summary = summarize_slopes(slope_df, ["regime", "epistasis_class"])
    class_summary["epistasis_label"] = class_summary["epistasis_class"].map(EPISTASIS_LABELS)
    class_summary.to_csv(dirs["summaries"] / "summary_by_regime_class_labeled.csv", index=False)

    # ---- Figures: one per (migration regime x epistasis class) ----
    # 3 regimes x 7 classes = 21 figures, each pooling the 10 selected files,
    # showing phenotype vs pop_id with the fitted clinal slope.
    figs = dirs["figures"]
    long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    n_written = 0
    if not long_df.empty:
        regimes = sorted(long_df["regime"].unique(), key=_regime_sort_key)
        for regime in regimes:
            reg_dir = figs / re.sub(r"[^A-Za-z0-9._-]", "_", regime)
            reg_dir.mkdir(parents=True, exist_ok=True)
            for cls in EPISTASIS_PREFIXES:
                cell = long_df[(long_df["regime"] == regime) & (long_df["epistasis_class"] == cls)]
                if cell.empty:
                    continue
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", regime)
                plot_class_cline(regime, cls, cell, reg_dir / f"cline_{safe}_{cls}.png")
                n_written += 1
    print(f"[write] {n_written} figure(s) -> {figs} (grouped by regime, one per epistasis class)")

    print("\n[done] Outputs written under:", config.out_dir.resolve())
    _print_headline(slope_df, class_summary)


def _print_headline(slope_df: pd.DataFrame, class_summary: pd.DataFrame) -> None:
    """Print a short, plain-language read on the data to stdout."""
    print("\n================ FIRST-PASS READOUT ================")
    reg = summarize_slopes(slope_df, ["regime"]).sort_values("regime", key=lambda s: s.map(_regime_sort_key))
    for _, row in reg.iterrows():
        print(
            f"  {row['regime']:>8}: mean|slope|={row['mean_abs_slope']:.4g}  "
            f"dir-consistency={row['frac_same_sign']:.2f}  mean R^2={row['mean_r2']:.2f}"
        )
    print("  (Expectation: mean|slope| and R^2 should DECREASE as migration increases.)")
    print("====================================================")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> ClineConfig:
    p = argparse.ArgumentParser(description="SEC clinal-signal first-pass analysis")
    p.add_argument("--mig0", type=Path, default=Path("data/mig0"))
    p.add_argument("--mig0.01", dest="mig0_01", type=Path, default=Path("data/mig0.01"))
    p.add_argument("--mig0.05", dest="mig0_05", type=Path, default=Path("data/mig0.05"))
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--n-files", type=int, default=5, help="files per regime (first pass). Use -1 for all.")
    p.add_argument("--glob", default="*.tsv", help="filename pattern within each regime folder")
    p.add_argument("--representative-only", action="store_true",
                   help="analyze only one representative column per epistasis class")
    a = p.parse_args(argv)
    return ClineConfig(
        regimes={"mig0": a.mig0, "mig0.01": a.mig0_01, "mig0.05": a.mig0_05},
        out_dir=a.out,
        n_files=None if a.n_files is not None and a.n_files < 0 else a.n_files,
        file_glob=a.glob,
        representative_only=a.representative_only,
    )


if __name__ == "__main__":
    run(parse_args())
