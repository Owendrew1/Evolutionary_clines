# Simulating Evolutionary Clines (SEC) — first-pass validation

A small, reproducible workflow to check whether SEC simulation output shows a
**coherent clinal signal** before investing in a full analysis or paper.

It answers five concrete questions:

1. Are the files structurally sane?
2. Does phenotype change smoothly with spatial position (`pop_id`)?
3. Is the slope direction consistent across generations/replicates?
4. Does **lower migration** produce **stronger** spatial structure?
5. Do epistasis classes differ in slope strength or variability?

---

## 1. Recommended analysis plan

The logic is intentionally simple so it is easy to defend later.

- **Unit of signal = the slope of phenotype vs `pop_id`.** `pop_id` is the
  spatial position along the transect, so a non-zero slope *is* a cline.
- **Computed within each generation** of each file (so time is not confounded),
  then **summarized across generations and replicates**.
- **Method = ordinary least squares** (`scipy.stats.linregress`). One number per
  (file, generation, phenotype column): slope, plus R², p-value for context.
- **Epistasis classes** (`RE, DE, DI, DRE, DDE, PO, NE`) are auto-detected from
  column prefixes. Additive-dosage columns are detected and **ignored**.
- **Migration comparison** is the payoff: mean |slope|, directional consistency,
  and mean R² should all **decrease** as migration increases if the model is
  behaving sensibly (more gene flow smooths the cline).

First pass uses **5 files per regime**; flip one flag to run all files later.

---

## 2. The code

| File | Purpose |
|------|---------|
| `select_files.py` | Randomly selects N files **per migration regime** from thousands of server files, seeded/reproducible, staged into a clean tree. Run this first on the cluster. |
| `analyze_clines.py` | The full workflow: load → validate → slopes → summaries → figures. Modular pure functions orchestrated in `run()`. |
| `make_synthetic_data.py` | Generates SEC-style test data encoding the expected biology. Use it to verify the pipeline and to compare against your real file schema. |
| `make_server_tree.py` | Builds a fake `results/slim` of many `sim_mig{X}_rep{N}.tsv` files to test `select_files.py` locally. |
| `requirements.txt` | Dependencies (`seaborn` is optional; matplotlib fallback is built in). |

## How the sampling is structured (important)

Each simulation file (`sim_mig0_rep983.tsv`) contains **all epistasis classes as
columns** (RE, DE, DI, DRE, DDE, PO, NE). So a file is not "an epistasis type" —
every file covers all classes. That means:

- **File selection is stratified by the one file-level factor that varies:
  migration regime.** We draw N random files from each of `mig0`, `mig0.01`,
  `mig0.05`.
- **Epistasis separation happens in the analysis, not the file picking.** Every
  summary and figure reports each epistasis class on its own (the "5 from each
  epistasis type" separation you wanted is delivered per-class in the outputs,
  and each class is measured on all selected files rather than a different 5).

## Selecting a subset on the server (do this first)

The real output lives on the cluster (e.g. `/scratch/jsantang/epistasis_clines/results/slim`)
with thousands of files. `select_files.py` draws a **random but balanced**
subset (N files per migration regime) so the first pass is fast and unbiased:

```bash
# ssh into the cluster, then:

# 1) Inspect first — shows available/selected counts per regime, touches nothing:
python select_files.py --root /scratch/jsantang/epistasis_clines/results/slim --n 10 --dry-run

# 2) Stage 10 random files per regime into ./selected (symlinks, instant):
python select_files.py --root /scratch/jsantang/epistasis_clines/results/slim --n 10 --out selected
```

This creates:

```
selected/
├── selection_manifest.csv     # exactly which files were chosen (auditable)
├── mig0/       sim_mig0_rep0431.tsv ...     (10 random files)
├── mig0.01/    sim_mig0.01_rep0088.tsv ...
└── mig0.05/    sim_mig0.05_rep0210.tsv ...
```

Key flags: `--n` (files per regime; use 5 for a quick look, 10 for more signal),
`--seed` (reproducible selection; same seed → same files), `--copy` (physically
copy instead of symlink), `--glob` (pattern, default `*.tsv`), `--regime-tokens`
(if your migration labels differ), `--dry-run`.

Detection is boundary-aware: `mig0` never matches inside `mig0.05`/`mig0.01`.
**Always run `--dry-run` first** to confirm the counts look right before staging.

### Install & run

```bash
pip install -r requirements.txt
```

**Analyze the staged selection:**

```bash
python analyze_clines.py \
    --mig0 selected/mig0 --mig0.01 selected/mig0.01 --mig0.05 selected/mig0.05 \
    --out outputs --n-files -1
```

`--n-files -1` uses every file you staged. Epistasis classes are auto-detected
from the columns and each is reported separately in the summaries/figures.

Useful flags:

- `--n-files -1` → analyze all staged files (use a positive number to cap per folder).
- `--glob "*.tsv"` → change if your files use a different extension/pattern.
- `--representative-only` → analyze **one representative column per class** instead of every column.

Test the whole chain locally without the cluster:

```bash
python make_server_tree.py --out fake_server/results/slim --per-regime 300
python select_files.py --root fake_server/results/slim --n 10 --out selected
python analyze_clines.py --mig0 selected/mig0 --mig0.01 selected/mig0.01 \
    --mig0.05 selected/mig0.05 --out outputs --n-files -1
```

### Expected input schema

Tab-separated, one row per (generation × population), with at least:

- `generation` — integer generation index
- `pop_id` — spatial position along the transect (increasing = down-gradient)
- `pop_size` — population size (should **decrease** as `pop_id` increases)
- phenotype columns named by epistasis prefix, e.g. `RE_1`, `DE_2`, `DRE_1`, `PO_3`
- additive-dosage columns (containing `add`/`dosage`/`dose`) — ignored automatically

If your real headers differ, adjust the constants at the top of
`analyze_clines.py` (`COL_GENERATION`, `COL_POP_ID`, `COL_POP_SIZE`,
`EPISTASIS_PREFIXES`, `ADDITIVE_PATTERNS`). Everything else keys off those.

---

## 3. Exact outputs produced

All under `outputs/`, organized by type and named by regime/metric:

```
outputs/
├── tables/
│   └── validation_report.csv          # one row per file: structural checks + PASS flag
├── cleaned/
│   └── per_generation_slopes.csv      # tidy: one row per (regime, file, generation, column) slope
├── summaries/
│   ├── summary_by_regime.csv          # headline comparison across migration regimes
│   ├── summary_by_regime_class.csv    # regime × epistasis class
│   ├── summary_by_regime_class_labeled.csv  # + human-readable class names
│   ├── summary_by_regime_class_column.csv   # per-column detail
│   └── summary_by_regime_generation.csv     # slope stability over time
└── figures/
    ├── slope_distribution_by_regime.png     # is signal steeper at low migration?
    ├── slope_distribution_by_class.png      # do classes differ?
    ├── mean_slope_over_generation.png       # is the cline stable over time?
    ├── meanpheno_<regime>_<class>_<col>.png # mean phenotype ± SD vs pop_id
    └── lines_<regime>_<class>_<col>.png     # phenotype vs pop_id, several generations
```

Each summary table carries these metrics (per group):

- `mean_slope`, `median_slope`, `std_slope`, `mean_abs_slope`
- `frac_same_sign` — fraction of slopes sharing the majority sign (**directional consistency**, 0.5 = random, 1.0 = perfectly consistent)
- `mean_r2` — how linear/tight the cline is on average
- `frac_p_lt_0.05` — fraction of individual slopes that are significant

---

## 4. How to interpret the results

The script also prints a short readout to the terminal. Read it against these
expectations.

### Signs the data are GOOD (worth continuing to a short paper)

- **Slope direction is consistent**: `frac_same_sign` well above 0.5 (say > 0.7)
  within a regime/class. The cline points the same way across generations and
  replicates rather than flipping randomly.
- **Migration ordering holds**: `mean_abs_slope` and `mean_r2` **decrease** from
  `mig0` → `mig0.01` → `mig0.05`. Lower migration = stronger, tighter cline.
  This is the single most important sanity check.
- **`mean_phenotype vs pop_id` looks smooth/monotone**, not jagged noise, in the
  `meanpheno_*` figures — especially for `mig0`.
- **Classes differ sensibly**: e.g. inhibitory (`DI`) and negative (`NE`) classes
  show negative slopes; positive/recessive show positive; magnitudes differ but
  are stable within a class.
- **`validation_report.csv` is all `PASS = True`** (`pop_size` decreases, phenotypes numeric, columns present).

### Signs the data are NOT ready (fix the sim or investigate first)

- `frac_same_sign` near 0.5 everywhere → slopes are basically noise; no coherent cline.
- `mean_abs_slope`/`mean_r2` **flat or inverted** across migration regimes → the
  migration mechanism isn't shaping spatial structure as expected.
- `mean_r2` near 0 even at `mig0` → phenotype does not vary smoothly with space
  (drift may be swamping the signal, or `pop_id` isn't the right spatial axis).
- Validation failures (`pop_size` not decreasing, non-numeric phenotypes,
  missing columns) → a data/export problem to resolve before any analysis.

### The bottom line for a short paper

You have a defensible short paper if, across the first 5 files per regime:

1. the migration ordering holds (steeper/tighter cline at lower migration),
2. slope direction is consistent within class/regime, and
3. at least a few epistasis classes show a clear, smooth cline at `mig0`.

If those hold, expand to all files (`--n-files -1`) and the same three figures
(`slope_distribution_by_regime`, `slope_distribution_by_class`,
`meanpheno_*`) become your core paper figures.

---

## Reproducibility notes

- Files are processed in sorted order; the synthetic generator is seeded.
- Matplotlib runs headless (`Agg`) and writes its cache into `.mplcache/` inside
  the project, so it works on machines where `$HOME/.matplotlib` isn't writable.
- No hidden state: rerunning with the same inputs reproduces the same outputs.
