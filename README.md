# Bank Failure Early-Warning Study

This repository contains the processed data, saved out-of-time predictions, analysis code, and result tables for the paper. The study covers 11,046 FDIC-insured institutions from 2001Q1 to 2024Q3 and evaluates failure risk at 1Q, 2Q, 4Q, and 8Q horizons.

## Repository contents

- `data/model_panel.parquet`: processed institution-quarter panel used to fit the models.
- `data/oot_predictions/`: saved strict out-of-time predictions for the Random Forest, Economic Logit, five-seed 4Q comparison, and STRUCTURAL-plus-Capital ablation.
- `data/eligible_failure_paths.csv`: the 515 failure paths used in the persistent-warning analysis.
- `data/variable_dictionary.csv` and `data/feature_groups.json`: feature definitions and information-set membership.
- `data/data_manifest.csv`: source and construction information for the processed data.
- `data/model_metadata/` and `config/`: model settings, split details, and predictor definitions.
- `results/tables/`: tables used in the paper and diagnostics.
- `scripts/`: scripts for reproducing results, bootstrapping comparisons, and refitting models.

The raw FDIC downloads are not included. `data/model_panel.parquet` is the input for refitting. Historical Call Report release dates and macroeconomic vintages are not available, so quarter-aligned predictions are pseudo-real-time.

## Setup

The Parquet files use Git LFS. Install Git LFS before cloning or downloading the data. Then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the main results

From the repository root, run:

```bash
python scripts/reproduce_paper_outputs.py
```

The script writes regenerated tables and figures to `results/reproduced/`. It uses the saved OOT predictions and checks that compared systems share the same bank-quarter observations and labels.

Other analyses can be regenerated with:

```bash
python scripts/bootstrap_fixed_workload.py --reps 2000 --seed 20261010
python scripts/summarize_4q_seeds.py
python scripts/training_sample_diagnostics.py
```

To refit the prediction models from the processed panel, run:

```bash
python scripts/refit_strict_predictions.py
```

Use `--model rf`, `--model economic-logit`, or `--model structural-plus-capital` to refit one model family. New predictions are written to `results/refit/`; the saved prediction files are not modified. The refit uses the included model settings without retuning. Numerical results may differ slightly across software versions or platforms.
