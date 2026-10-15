#!/usr/bin/env python3
"""Reconcile mature eligible training rows with actual model-fit rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from refit_strict_predictions import (
    FOLDS, availability, economic_partitions, feature_sets,
    matured_train_mask, rf_partitions,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduced"


def counts(frame: pd.DataFrame, target: str) -> tuple[int, int, int]:
    y = frame[target]
    positive = y.eq(1)
    return len(frame), int(positive.sum()), int(frame.loc[positive, "CERT"].nunique())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data/model_panel.parquet")
    panel["CERT"] = panel.CERT.astype(str)
    panel["quarter"] = pd.PeriodIndex(panel.quarter, freq="Q")
    sets, _ = feature_sets()
    rf_manifest = [json.loads(line) for line in (ROOT / "data/frozen_metadata/rf_fit_manifest.jsonl").read_text().splitlines()]
    econ_manifest = [json.loads(line) for line in (ROOT / "data/frozen_metadata/economic_logit_fit_manifest.jsonl").read_text().splitlines()]
    records = []

    for spec in rf_manifest:
        if int(spec["lag"]) != 0:
            continue
        info, horizon, fold = spec["information_set"], int(spec["horizon"]), spec["fold"]
        target = f"fail_{horizon}q"
        start_text = next(start for name, start, _ in FOLDS if name == fold)
        end_text = next(end for name, _, end in FOLDS if name == fold)
        start, end = pd.Period(start_text, freq="Q"), pd.Period(end_text, freq="Q")
        features = sets[info]
        avail = availability(panel, features, 0)
        mature_mask = matured_train_mask(panel.quarter, start, horizon) & panel[target].notna() & avail
        outer = panel.loc[mature_mask].copy()
        fit, cal = rf_partitions(outer, target, horizon, features, 0)
        q = pd.PeriodIndex(panel.quarter, freq="Q")
        test_mask = pd.Series((q >= start) & (q <= pd.Period(end_text, freq="Q")), index=panel.index) & panel[target].notna() & avail
        test = panel.loc[test_mask]
        outer_counts, fit_counts, cal_counts, test_counts = (
            counts(outer, target), counts(fit, target), counts(cal, target), counts(test, target))
        expected = (int(spec["n_outer_train"]), int(spec["n_fit"]), int(spec["n_calibration"]), int(spec["n_test"]))
        actual = (outer_counts[0], fit_counts[0], cal_counts[0], test_counts[0])
        if actual != expected:
            raise RuntimeError(f"RF count reconciliation failed for {info}/{horizon}Q/{fold}: {actual} != {expected}")
        records.append({"system": "Random Forest", "information_set": info,
            "predictor_lag_quarters": 0, "horizon_quarters": horizon, "fold": fold,
            "mature_eligible_train_rows": outer_counts[0], "mature_eligible_positive_labels": outer_counts[1],
            "mature_eligible_distinct_failure_banks": outer_counts[2],
            "fitted_train_rows": fit_counts[0], "fitted_train_positive_labels": fit_counts[1],
            "fitted_train_distinct_failure_banks": fit_counts[2],
            "calibration_rows": cal_counts[0], "calibration_positive_labels": cal_counts[1],
            "test_rows": test_counts[0], "test_positive_labels": test_counts[1],
            "strict_embargo": "q + h < test_start"})

    for spec in econ_manifest:
        horizon, fold = int(spec["horizon"]), spec["fold"]
        target = f"fail_{horizon}q"
        start_text = next(start for name, start, _ in FOLDS if name == fold)
        end_text = next(end for name, _, end in FOLDS if name == fold)
        start, end = pd.Period(start_text, freq="Q"), pd.Period(end_text, freq="Q")
        mature_mask = matured_train_mask(panel.quarter, start, horizon) & panel[target].notna()
        outer = panel.loc[mature_mask].copy()
        fit, cal = economic_partitions(outer, target, horizon)
        q = pd.PeriodIndex(panel.quarter, freq="Q")
        test_mask = pd.Series((q >= start) & (q <= end), index=panel.index) & panel[target].notna()
        test = panel.loc[test_mask]
        outer_counts, fit_counts, cal_counts, test_counts = (
            counts(outer, target), counts(fit, target), counts(cal, target), counts(test, target))
        expected = (int(spec["outer_train_rows"]), int(spec["fit_rows"]), int(spec["calibration_rows"]), int(spec["test_rows"]))
        actual = (outer_counts[0], fit_counts[0], cal_counts[0], test_counts[0])
        if actual != expected:
            raise RuntimeError(f"Economic Logit count reconciliation failed for {horizon}Q/{fold}: {actual} != {expected}")
        records.append({"system": "Economic Logit", "information_set": "ECONOMIC_LOGIT",
            "predictor_lag_quarters": 0, "horizon_quarters": horizon, "fold": fold,
            "mature_eligible_train_rows": outer_counts[0], "mature_eligible_positive_labels": outer_counts[1],
            "mature_eligible_distinct_failure_banks": outer_counts[2],
            "fitted_train_rows": fit_counts[0], "fitted_train_positive_labels": fit_counts[1],
            "fitted_train_distinct_failure_banks": fit_counts[2],
            "calibration_rows": cal_counts[0], "calibration_positive_labels": cal_counts[1],
            "test_rows": test_counts[0], "test_positive_labels": test_counts[1],
            "strict_embargo": "q + h < test_start"})

    result = pd.DataFrame(records).sort_values(["system", "horizon_quarters", "fold", "information_set"])
    result.to_csv(OUT / "training_sample_diagnostics.csv", index=False)
    print(f"Reconciled {len(result)} model-fold partitions; mature eligible and fitted training rows are reported separately.")


if __name__ == "__main__":
    main()
