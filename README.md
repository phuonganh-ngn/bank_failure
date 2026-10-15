# Paper reproducibility package

This folder contains only the processed analytical panel, frozen prediction inputs, model-fit manifests, and scripts needed to reproduce the main predictive and capacity-constrained screening results for the bank-failure paper.

## Contents

- `data/model_panel.parquet`: final 671,414-row institution-quarter panel used for modeling. It includes the frozen features and 1Q, 2Q, 4Q, and 8Q outcomes.
- `data/variable_dictionary.csv` and `data/frozen_feature_groups.json`: predictor definitions and information-stage membership.
- `data/frozen_predictions/strict_rf_oot_predictions.parquet`: frozen balanced Random Forest OOT scores for STRUCTURAL, EARLY, and FULL, at 0Q, 1Q, and 2Q predictor lags and all four horizons.
- `data/frozen_predictions/strict_economic_logit_oot_predictions.parquet`: frozen strict Economic Logit scores for all four horizons.
- `data/frozen_predictions/structural_plus_capital_8q_predictions.parquet`: the prespecified 8Q STRUCTURAL-plus-equity/assets ablation predictions.
- `data/frozen_predictions/rf_multiseed_4q_early_full.parquet`: the saved five-seed 4Q EARLY/FULL predictions used for the seed-sensitivity check.
- `data/eligible_failure_paths.csv`: the fixed 515-failure operational cohort and T−4 asset/deposit weights.
- `data/frozen_metadata/`: the original RF and Economic Logit fit records, including seeds, selected RF hyperparameters, and training/test partition counts.
- `data/frozen_metadata/structural_plus_capital_fit_manifest.json`: selected settings and sample counts for the STRUCTURAL-plus-capital refit.
- `config/`: the ten-variable Economic Logit feature list and the frozen study design.
- `scripts/reproduce_paper_outputs.py`: creates pooled and OOT-block predictive metrics, the five-system workload frontier, fixed-workload table, and figure from frozen predictions.
- `scripts/bootstrap_fixed_workload.py`: recomputes paired institution-cluster bootstrap intervals at fixed confirmed workload.
- `scripts/refit_strict_predictions.py`: refits the strict RF, Economic Logit, and STRUCTURAL-plus-capital systems from the processed panel using archived fit settings. It writes new predictions under `results/refit/` and does not modify the frozen inputs.
- `scripts/summarize_4q_seeds.py`: recreates the five-seed 4Q EARLY/FULL pooled and block-level AP gap summaries.
- `scripts/training_sample_diagnostics.py`: reports mature eligible training rows separately from the smaller rows used to fit each fold, alongside positive-label and distinct-bank counts.
- `results/reference/paired_workload_differences.csv`: archived paired-bootstrap summary used for the manuscript's fixed-workload comparisons.
- `results/reference/`: compact audited result tables for pooled/block performance, the fixed-workload frontier, the STRUCTURAL-plus-capital ablation, the five-seed 4Q check, and fitted-versus-mature training counts.

Raw downloaded FDIC extracts are not included: the final processed panel is the input to the model and evaluation scripts. `data/data_manifest.csv` records the source-file provenance. The package does not reconstruct historical Call Report publication dates or macroeconomic vintages; the quarter-aligned analysis remains pseudo-real-time.

## Environment

The dependency versions below match the analysis environment used to prepare this package. Install Git LFS before adding the Parquet files to a remote repository; the panel and prediction files are stored using Git LFS attributes.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce reported results

From this directory, run:

```bash
python scripts/reproduce_paper_outputs.py
```

Generated CSVs and PDF/SVG/PNG figures are written to `results/reproduced/` (ignored by Git because the scripts recreate them). The script aligns systems on common bank-quarter keys and checks that labels match before calculating results. The five-system frontier and primary figure use `Capital Screen` as the display label; the saved numeric system key remains `CAPITAL_SCREEN`.

Recreate the 4Q seed summary and training-partition reconciliation with:

```bash
python scripts/summarize_4q_seeds.py
python scripts/training_sample_diagnostics.py
```

To regenerate the paired fixed-workload bootstrap intervals after running the output script:

```bash
python scripts/bootstrap_fixed_workload.py --reps 2000 --seed 20261010
```

The bootstrap conditions on the saved model scores. It resamples institution clusters, rebuilds each system's workload frontier, and reselects the nearest capacity to the target in each draw. It does not include training or model-selection uncertainty.

To refit the prediction systems:

```bash
python scripts/refit_strict_predictions.py
```

Use `--model rf`, `--model economic-logit`, or `--model structural-plus-capital` to run one model family. Refit predictions are saved separately in `results/refit/`. RF hyperparameters and seeds are read from the included frozen fit manifests; this runner does not retune them. Small score differences can arise if package versions, platform, or parallel numerical behavior differ from the original fit environment. The archived OOT predictions remain the exact inputs for the paper's reported tables and figure.

## Data and file integrity

`SHA256SUMS.txt` records the SHA-256 digest for each tracked package file other than the checksum file itself. It covers the data, code, configuration, and reference result tables; generated `results/reproduced/` files are excluded. Verify hashes from this directory with:

```bash
sha256sum -c SHA256SUMS.txt
```

The folder is prepared for Git tracking but is not initialized as a Git repository. The parent research workspace did not contain a `.git` directory when this package was created.
