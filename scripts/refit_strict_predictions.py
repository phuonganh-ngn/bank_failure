#!/usr/bin/env python3
"""Refit the paper's strict OOT Random Forest and Economic Logit systems.

This runner uses the archived selected RF hyperparameters and fold seeds in
data/frozen_metadata. It does not retune models. Outputs go to a new directory
under results/refit/ and never replace the frozen prediction artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FOLDS = (("gfc", "2007Q1", "2009Q4"),
         ("post_gfc", "2010Q1", "2018Q4"),
         ("covid_modern", "2019Q1", "2024Q3"))
HORIZONS = (1, 2, 4, 8)
LAGS = (0, 1, 2)
LOW_Q, HIGH_Q = 0.01, 0.99
THREADS = 4


def matured_train_mask(quarter: pd.Series, start: pd.Period, horizon: int) -> pd.Series:
    q = pd.PeriodIndex(quarter, freq="Q")
    return pd.Series((q < start) & ((q + horizon) < start), index=quarter.index)


def feature_sets() -> tuple[dict[str, list[str]], list[str]]:
    dictionary = pd.read_csv(ROOT / "data/variable_dictionary.csv")
    kept = dictionary.loc[dictionary.retained_for_modeling.astype(bool)]
    all_features = kept.final_feature_name.tolist()
    tier = dict(zip(kept.final_feature_name, kept.tier))
    sets = {
        "STRUCTURAL": [f for f in all_features if tier[f] in ("structural_vulnerability", "macro_control")],
        "EARLY": [f for f in all_features if tier[f] != "proximal_symptoms_strict"],
        "FULL": all_features,
    }
    return sets, all_features


def preprocess_fit(train: pd.DataFrame, features: list[str]):
    thresholds: dict[str, tuple[float, float]] = {}
    medians: dict[str, float] = {}
    for feature in features:
        x = pd.to_numeric(train[feature], errors="coerce")
        nonmissing = x.dropna()
        if nonmissing.empty:
            thresholds[feature] = (float("nan"), float("nan"))
            medians[feature] = 0.0
        else:
            thresholds[feature] = (float(nonmissing.quantile(LOW_Q)), float(nonmissing.quantile(HIGH_Q)))
            medians[feature] = float(nonmissing.median())
    return thresholds, medians


def preprocess_transform(frame: pd.DataFrame, features: list[str], state) -> pd.DataFrame:
    thresholds, medians = state
    result = pd.DataFrame(index=frame.index)
    for feature in features:
        x = pd.to_numeric(frame[feature], errors="coerce")
        lo, hi = thresholds[feature]
        if np.isfinite(lo) and np.isfinite(hi):
            x = x.clip(lo, hi)
        result[feature] = x.fillna(medians[feature])
    return result


def lag_panel(base: pd.DataFrame, features: list[str], lag: int) -> pd.DataFrame:
    if lag == 0:
        result = base.copy()
        result["lag_source_found"] = True
        return result
    keys = base[["CERT", "quarter"]].copy()
    keys["quarter"] += lag
    source = base[["CERT", "quarter", *features]].copy()
    source["quarter"] += lag
    source = source.rename(columns={f: f"__lagged__{f}" for f in features})
    mapped = keys.merge(source, on=["CERT", "quarter"], how="left", validate="one_to_one")
    mapped["lag_source_found"] = mapped[[f"__lagged__{f}" for f in features]].notna().any(axis=1)
    result = base.drop(columns=features).merge(mapped, on=["CERT", "quarter"], how="left", validate="one_to_one")
    for feature in features:
        result[feature] = result.pop(f"__lagged__{feature}")
    result["lag_source_found"] = result.lag_source_found.fillna(False).astype(bool)
    return result


def availability(panel: pd.DataFrame, features: list[str], lag: int) -> pd.Series:
    source_ok = panel.lag_source_found.astype(bool) if lag else pd.Series(True, index=panel.index)
    return source_ok & panel[features].notna().any(axis=1)


def rf_partitions(outer: pd.DataFrame, target: str, horizon: int, features: list[str], lag: int):
    q = pd.PeriodIndex(outer.quarter, freq="Q")
    end = q.max()
    cal_start = end - 3
    avail = availability(outer, features, lag)
    cal_mask = pd.Series((q >= cal_start) & (q <= end), index=outer.index) & outer[target].notna() & avail
    cal = outer.loc[cal_mask]
    fit_mask = matured_train_mask(outer.quarter, cal_start, horizon) & outer[target].notna() & avail
    fit = outer.loc[fit_mask]
    if len(fit) >= 2000 and cal[target].nunique() == 2 and int(cal[target].sum()) >= 10 and len(cal) - int(cal[target].sum()) >= 20:
        return fit, cal
    return outer.loc[outer[target].notna() & avail], outer.iloc[0:0]


def economic_partitions(outer: pd.DataFrame, target: str, horizon: int):
    q = pd.PeriodIndex(outer.quarter, freq="Q")
    cal_start = q.max() - 3
    cal_mask = pd.Series((q >= cal_start) & (q <= q.max()), index=outer.index) & outer[target].notna()
    cal = outer.loc[cal_mask]
    fit_mask = matured_train_mask(outer.quarter, cal_start, horizon) & outer[target].notna()
    fit = outer.loc[fit_mask]
    enough = (len(fit) >= 2000 and cal[target].nunique() == 2 and int(cal[target].sum()) >= 10
              and len(cal) - int(cal[target].sum()) >= 20)
    return (fit, cal) if enough else (outer, outer.iloc[0:0])


def calibration_result(model, scaler, pre_state, cal: pd.DataFrame, test: pd.DataFrame,
                       features: list[str], target: str, fit_prevalence: float):
    if len(cal) and cal[target].nunique() == 2 and int(cal[target].sum()) >= 5:
        dtype = np.float32 if scaler is None else np.float64
        xcal = preprocess_transform(cal[features], features, pre_state).to_numpy(dtype=dtype)
        if scaler is not None:
            xcal = scaler.transform(xcal)
        cal_scores = model.predict_proba(xcal)[:, 1]
        cal_model = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
        cal_model.fit(logit(np.clip(cal_scores, 1e-7, 1 - 1e-7)).reshape(-1, 1), cal[target].astype(int))
        slope = float(cal_model.coef_[0, 0])
        if slope > 0:
            status = "positive_slope_platt"
            raw_test = test["_raw_score"].to_numpy(float)
            calibrated = cal_model.predict_proba(logit(np.clip(raw_test, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1]
        else:
            status = ("negative_slope_prevalence_fallback" if scaler is not None else
                      "negative_slope_replaced_by_calibration_prevalence_prior")
            calibrated = np.full(len(test), float(cal[target].mean()))
        return np.clip(calibrated, 1e-7, 1 - 1e-7), status, slope
    status = ("prevalence_prior_no_eligible_calibration" if scaler is not None else
              "no_eligible_calibration_fit_prevalence_prior")
    return np.full(len(test), fit_prevalence), status, float("nan")


def output_frame(test: pd.DataFrame, horizon: int, fold: str, model: str, info: str,
                 lag: int, raw: np.ndarray, calibrated: np.ndarray, status: str, slope: float):
    return pd.DataFrame({
        "CERT": test.CERT.astype(str).to_numpy(),
        "quarter": pd.PeriodIndex(test.quarter, freq="Q").astype(str),
        "horizon_quarters": horizon, "fold": fold, "model": model,
        "information_set": info, "predictor_lag_quarters": lag,
        "weighting": "balanced", "label": test[f"fail_{horizon}q"].astype(int).to_numpy(),
        "raw_score": raw, "calibrated_probability": calibrated,
        "calibration_status": status, "calibration_platt_slope": slope,
    })


def write_parquet_chunk(writer, path: Path, frame: pd.DataFrame):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd", compression_level=9)
    writer.write_table(table)
    return writer


def refit_rf(panel: pd.DataFrame, sets: dict[str, list[str]], all_features: list[str], out_dir: Path):
    manifest = [json.loads(line) for line in (ROOT / "data/frozen_metadata/rf_fit_manifest.jsonl").read_text().splitlines()]
    records = {(int(x["lag"]), x["information_set"], int(x["horizon"]), x["fold"]): x for x in manifest}
    writer = None
    out_path = out_dir / "strict_rf_oot_predictions.parquet"
    panel_cache = {lag: lag_panel(panel, all_features, lag) for lag in LAGS}
    for lag in LAGS:
        current = panel_cache[lag]
        avail_cache = {name: availability(current, features, lag) for name, features in sets.items()}
        q = pd.PeriodIndex(current.quarter, freq="Q")
        for info, features in sets.items():
            for horizon in HORIZONS:
                target = f"fail_{horizon}q"
                for fold_index, (fold, start_text, end_text) in enumerate(FOLDS):
                    spec = records[(lag, info, horizon, fold)]
                    start, end = pd.Period(start_text, freq="Q"), pd.Period(end_text, freq="Q")
                    available = avail_cache[info]
                    train_mask = matured_train_mask(current.quarter, start, horizon) & current[target].notna() & available
                    test_mask = pd.Series((q >= start) & (q <= end), index=current.index) & current[target].notna() & available
                    outer, test = current.loc[train_mask].copy(), current.loc[test_mask].copy()
                    fit, cal = rf_partitions(outer, target, horizon, features, lag)
                    actual = (len(outer), len(fit), len(cal), len(test))
                    expected = (int(spec["n_outer_train"]), int(spec["n_fit"]), int(spec["n_calibration"]), int(spec["n_test"]))
                    if actual != expected:
                        raise RuntimeError(f"RF partition mismatch for {lag}Q/{info}/{horizon}Q/{fold}: {actual} != {expected}")
                    pre_state = preprocess_fit(fit[features], features)
                    xfit = preprocess_transform(fit[features], features, pre_state).to_numpy(dtype=np.float32)
                    xtest = preprocess_transform(test[features], features, pre_state).to_numpy(dtype=np.float32)
                    params = spec["selected_params"]
                    model = RandomForestClassifier(
                        n_estimators=60, max_samples=0.20, max_depth=params["max_depth"],
                        min_samples_leaf=int(params["min_samples_leaf"]), max_features=params["max_features"],
                        class_weight="balanced_subsample", n_jobs=THREADS, random_state=int(spec["seed"]),
                        bootstrap=True,
                    )
                    model.fit(xfit, fit[target].astype(int))
                    raw = np.asarray(model.predict_proba(xtest)[:, 1], dtype=float)
                    # The calibration helper expects the already computed test scores.
                    test_for_cal = test.copy()
                    test_for_cal["_raw_score"] = raw
                    calibrated, status, slope = calibration_result(
                        model, None, pre_state, cal, test_for_cal, features, target, float(fit[target].mean())
                    )
                    result = output_frame(test, horizon, fold, "random_forest", info, lag,
                                          raw, calibrated, status, slope)
                    writer = write_parquet_chunk(writer, out_path, result)
                    print(f"RF lag={lag} info={info} h={horizon} fold={fold}: fit={len(fit):,}, test={len(test):,}", flush=True)
    if writer:
        writer.close()
    return out_path


def refit_economic_logit(panel: pd.DataFrame, out_dir: Path):
    variables = pd.read_csv(ROOT / "config/economic_logit_predictors.csv").variable.tolist()
    manifest = [json.loads(line) for line in (ROOT / "data/frozen_metadata/economic_logit_fit_manifest.jsonl").read_text().splitlines()]
    records = {(int(x["horizon"]), x["fold"]): x for x in manifest}
    writer = None
    out_path = out_dir / "strict_economic_logit_oot_predictions.parquet"
    q = pd.PeriodIndex(panel.quarter, freq="Q")
    for horizon in HORIZONS:
        target = f"fail_{horizon}q"
        for fold_index, (fold, start_text, end_text) in enumerate(FOLDS):
            spec = records[(horizon, fold)]
            start, end = pd.Period(start_text, freq="Q"), pd.Period(end_text, freq="Q")
            train_mask = matured_train_mask(panel.quarter, start, horizon) & panel[target].notna()
            test_mask = pd.Series((q >= start) & (q <= end), index=panel.index) & panel[target].notna()
            outer, test = panel.loc[train_mask].copy(), panel.loc[test_mask].copy()
            fit, cal = economic_partitions(outer, target, horizon)
            actual = (len(outer), len(fit), len(cal), len(test))
            expected = (int(spec["outer_train_rows"]), int(spec["fit_rows"]), int(spec["calibration_rows"]), int(spec["test_rows"]))
            if actual != expected:
                raise RuntimeError(f"Economic Logit partition mismatch for {horizon}Q/{fold}: {actual} != {expected}")
            pre_state = preprocess_fit(fit[variables], variables)
            xfit = preprocess_transform(fit[variables], variables, pre_state).to_numpy(dtype=np.float64)
            x_test = preprocess_transform(test[variables], variables, pre_state).to_numpy(dtype=np.float64)
            scaler = StandardScaler().fit(xfit)
            seed = 20261007 + horizon * 100 + fold_index
            model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed)
            model.fit(scaler.transform(xfit), fit[target].astype(int))
            raw = np.asarray(model.predict_proba(scaler.transform(x_test))[:, 1], dtype=float)
            test_for_cal = test.copy()
            test_for_cal["_raw_score"] = raw
            calibrated, status, slope = calibration_result(
                model, scaler, pre_state, cal, test_for_cal, variables, target, float(fit[target].mean())
            )
            result = output_frame(test, horizon, fold, "Economic Logit", "ECONOMIC_LOGIT", 0,
                                  raw, calibrated, status, slope)
            writer = write_parquet_chunk(writer, out_path, result)
            print(f"Economic Logit h={horizon} fold={fold}: fit={len(fit):,}, test={len(test):,}", flush=True)
    if writer:
        writer.close()
    return out_path


def refit_structural_plus_capital(panel: pd.DataFrame, sets: dict[str, list[str]], out_dir: Path):
    structural = sets["STRUCTURAL"]
    if "equity_to_assets" in structural:
        raise RuntimeError("equity_to_assets is already in STRUCTURAL; refusing to duplicate it")
    features = structural + ["equity_to_assets"]
    if panel.equity_to_assets.isna().all():
        raise RuntimeError("equity_to_assets is unavailable in the analytical panel")
    manifest = json.loads((ROOT / "data/frozen_metadata/structural_plus_capital_fit_manifest.json").read_text())
    records = {row["fold"]: row for row in manifest}
    horizon, lag = 8, 0
    target = f"fail_{horizon}q"
    q = pd.PeriodIndex(panel.quarter, freq="Q")
    available = availability(panel, structural, lag)
    writer = None
    out_path = out_dir / "structural_plus_capital_8q_predictions.parquet"
    for fold, start_text, end_text in FOLDS:
        spec = records[fold]
        start, end = pd.Period(start_text, freq="Q"), pd.Period(end_text, freq="Q")
        train_mask = matured_train_mask(panel.quarter, start, horizon) & panel[target].notna() & available
        test_mask = pd.Series((q >= start) & (q <= end), index=panel.index) & panel[target].notna() & available
        outer, test = panel.loc[train_mask].copy(), panel.loc[test_mask].copy()
        fit, cal = rf_partitions(outer, target, horizon, features, lag)
        actual = (len(outer), len(fit), len(cal), len(test))
        expected = (int(spec["n_outer_train"]), int(spec["n_fit"]),
                    int(spec["n_calibration"]), int(spec["n_test"]))
        if actual != expected:
            raise RuntimeError(f"STRUCTURAL_PLUS_CAPITAL partition mismatch for {fold}: {actual} != {expected}")
        if fit.equity_to_assets.isna().any() or test.equity_to_assets.isna().any():
            raise RuntimeError(f"Capital ratio missing in the frozen fit/test rows for {fold}")
        pre_state = preprocess_fit(fit[features], features)
        xfit = preprocess_transform(fit[features], features, pre_state).to_numpy(dtype=np.float32)
        xtest = preprocess_transform(test[features], features, pre_state).to_numpy(dtype=np.float32)
        params = spec["params"]
        model = RandomForestClassifier(
            n_estimators=60, max_samples=0.20, max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]), max_features=params["max_features"],
            class_weight="balanced_subsample", n_jobs=THREADS, random_state=int(spec["seed"]),
            bootstrap=True,
        )
        model.fit(xfit, fit[target].astype(int))
        raw = np.asarray(model.predict_proba(xtest)[:, 1], dtype=float)
        test_for_cal = test.copy()
        test_for_cal["_raw_score"] = raw
        calibrated, status, slope = calibration_result(
            model, None, pre_state, cal, test_for_cal, features, target, float(fit[target].mean())
        )
        result = output_frame(test, horizon, fold, "random_forest", "STRUCTURAL_PLUS_CAPITAL", lag,
                              raw, calibrated, status, slope)
        writer = write_parquet_chunk(writer, out_path, result)
        print(f"STRUCTURAL_PLUS_CAPITAL h=8 fold={fold}: fit={len(fit):,}, test={len(test):,}", flush=True)
    if writer:
        writer.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("all", "rf", "economic-logit", "structural-plus-capital"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/refit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(ROOT / "data/model_panel.parquet")
    panel["CERT"] = panel.CERT.astype(str)
    panel["quarter"] = pd.PeriodIndex(panel.quarter, freq="Q")
    sets, all_features = feature_sets()
    if args.model in ("all", "rf"):
        refit_rf(panel, sets, all_features, args.output_dir)
    if args.model in ("all", "economic-logit"):
        refit_economic_logit(panel, args.output_dir)
    if args.model in ("all", "structural-plus-capital"):
        refit_structural_plus_capital(panel, sets, args.output_dir)


if __name__ == "__main__":
    main()
