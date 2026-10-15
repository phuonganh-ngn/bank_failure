#!/usr/bin/env python3
"""Paired bank-cluster bootstrap for fixed confirmed-workload contrasts.

Resamples banks, rebuilds every system's capacity frontier, and chooses the
nearest observed capacity to each target workload in every replicate. Saved
model scores are fixed; the intervals exclude training and selection uncertainty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce_paper_outputs import (
    CAPACITY_GRID, SYSTEMS, _feature_rows, _operational_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduced"
SEED_DEFAULT = 20261010
REPS_DEFAULT = 2000
TARGETS = (50.0, 100.0, 175.0, 250.0)
PAIRS = (("RF EARLY", "CAPITAL_SCREEN"), ("RF FULL", "CAPITAL_SCREEN"),
         ("Economic Logit", "CAPITAL_SCREEN"), ("RF EARLY", "Economic Logit"))
METRICS = ("failure_recall", "recall_lead_ge_4Q", "asset_weighted_recall_T4",
           "false_alerts_per_detected_failure")


def precompute(aligned: dict[str, pd.DataFrame], cases: pd.DataFrame):
    all_banks = sorted(set().union(*(set(frame.CERT.astype(str)) for frame in aligned.values())))
    bank_map = {bank: index for index, bank in enumerate(all_banks)}
    quarters = sorted(aligned["CAPITAL_SCREEN"].quarter.unique())
    q_lookup = {int(q.ordinal): index for index, q in enumerate(quarters)}
    systems = {}
    for name, frame in aligned.items():
        qdata = []
        for quarter in quarters:
            group = frame.loc[frame.quarter.eq(quarter)].sort_values("panel_order", kind="mergesort")
            bank_ids = np.asarray([bank_map[x] for x in group.CERT.astype(str)], dtype=np.int32)
            score = group.raw_score.to_numpy(float)
            order = np.argsort(-score, kind="stable")
            qdata.append({"banks_ranked": bank_ids[order],
                          "labels_ranked": group.label.to_numpy(np.int8)[order],
                          "n_rows": len(group)})
        systems[name] = qdata
    case = cases.copy()
    case["bank_idx"] = case.CERT.astype(str).map(bank_map).astype(int)
    case["failure_ord"] = case.failure_quarter.astype("int64")
    if case.bank_idx.isna().any():
        raise RuntimeError("A failure cluster is absent from the common OOT universe")
    if len(all_banks) != aligned["CAPITAL_SCREEN"].CERT.nunique():
        raise RuntimeError("Bank mapping is not unique")
    return all_banks, quarters, q_lookup, systems, case


def system_bootstrap(multiplicity, offsets, q_lookup, qdata, case, targets):
    bank_count = len(multiplicity)
    grid_count = len(CAPACITY_GRID)
    quarter_count = len(qdata)
    confirmed_by_quarter = np.full((quarter_count, bank_count), grid_count, dtype=np.uint8)
    confirmed_count = np.zeros(grid_count, dtype=np.int64)
    false_count = np.zeros(grid_count, dtype=np.int64)
    previous = None

    for q_index, item in enumerate(qdata):
        bank_ids = item["banks_ranked"]
        labels = item["labels_ranked"]
        weights = multiplicity[bank_ids]
        active = weights > 0
        bank_ids, weights, labels = bank_ids[active], weights[active], labels[active]
        expanded_n = int(weights.sum())
        current = np.full(bank_count, grid_count, dtype=np.uint8)
        row_label = np.ones(bank_count, dtype=np.int8)
        if expanded_n:
            starts = np.cumsum(weights, dtype=np.int64) - weights
            copies_start = np.repeat(starts, weights)
            copies_index = np.arange(expanded_n) - np.repeat(starts, weights)
            ranks = copies_start + copies_index + 1
            slots = np.repeat(offsets[bank_ids], weights) + copies_index
            thresholds = np.ceil(CAPACITY_GRID * expanded_n).astype(np.int64)
            capacity_index = np.searchsorted(thresholds, ranks, side="left").clip(0, grid_count)
            current[slots] = capacity_index.astype(np.uint8)
            row_label[slots] = np.repeat(labels, weights)
        confirmed = current if previous is None else np.maximum(current, previous)
        confirmed_by_quarter[q_index] = confirmed
        valid = confirmed < grid_count
        if valid.any():
            confirmed_count += np.bincount(confirmed[valid], minlength=grid_count)[:grid_count]
            false_mask = valid & (row_label == 0)
            if false_mask.any():
                false_count += np.bincount(confirmed[false_mask], minlength=grid_count)[:grid_count]
        previous = current

    workload = np.cumsum(confirmed_count) / quarter_count
    false_alerts = np.cumsum(false_count)
    event_ids = case.bank_idx.to_numpy(int)
    event_multiplicity = multiplicity[event_ids]
    total_failures = int(event_multiplicity.sum())
    assets = case.assets_t_minus_4.to_numpy(float)
    total_assets = float(np.dot(event_multiplicity, assets))
    failure_ord = case.failure_ord.to_numpy(np.int64)
    lead_values = np.arange(8, 0, -1, dtype=np.int64)
    window_ord = failure_ord[:, None] - lead_values[None, :]
    window_index = np.vectorize(q_lookup.__getitem__)(window_ord)
    max_copies = int(event_multiplicity.max()) if len(event_multiplicity) else 0
    if max_copies:
        event_slots = np.minimum(offsets[event_ids, None]
                                 + np.arange(max_copies, dtype=np.int64)[None, :], bank_count - 1)
        valid_copy = np.arange(max_copies)[None, :] < event_multiplicity[:, None]
        event_alert_index = confirmed_by_quarter[window_index[:, :, None], event_slots[:, None, :]]
    else:
        valid_copy = np.zeros((len(event_ids), 0), dtype=bool)
        event_alert_index = np.empty((len(event_ids), 8, 0), dtype=np.uint8)

    results = {}
    for target in targets:
        index = int(np.abs(workload - target).argmin())
        low, high = float(workload.min()), float(workload.max())
        result = {"capacity_k": float(CAPACITY_GRID[index]), "attained_workload": float(workload[index]),
                  "workload_mismatch": float(workload[index] - target), "workload_min": low,
                  "workload_max": high, "false_alerts": int(false_alerts[index])}
        if not low <= target <= high or total_failures == 0:
            result.update({metric: np.nan for metric in METRICS})
            results[target] = result
            continue
        hit = (event_alert_index <= index) & valid_copy[:, None, :]
        any_hit = hit.any(axis=1)
        lead4_hit = hit[:, lead_values >= 4, :].any(axis=1)
        detected_count = int(any_hit.sum())
        lead4_count = int(lead4_hit.sum())
        detected_per_event = any_hit.sum(axis=1).astype(np.int64)
        asset_numerator = float(np.dot(detected_per_event, assets))
        first_position = np.argmax(hit, axis=1)
        lead_hit = np.where(any_hit, lead_values[first_position], -1)
        observed_leads = lead_hit[any_hit]
        result.update({
            "failure_recall": detected_count / total_failures,
            "recall_lead_ge_4Q": lead4_count / total_failures,
            "asset_weighted_recall_T4": asset_numerator / total_assets if total_assets else np.nan,
            "false_alerts_per_detected_failure": int(false_alerts[index]) / max(1, detected_count),
            "median_lead_quarters": float(np.median(observed_leads)) if len(observed_leads) else np.nan,
            "detected_failures_bootstrap": detected_count,
            "eligible_failures_bootstrap": total_failures,
        })
        results[target] = result
    return workload, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(ROOT / "data/model_panel.parquet", columns=[
        "CERT", "quarter", "fail_1q", "fail_2q", "fail_4q", "fail_8q", "equity_to_assets"])
    rf = pd.read_parquet(ROOT / "data/oot_predictions/strict_rf_oot_predictions.parquet")
    econ = pd.read_parquet(ROOT / "data/oot_predictions/strict_economic_logit_oot_predictions.parquet")
    cases = pd.read_csv(ROOT / "data/eligible_failure_paths.csv", dtype={"CERT": str})
    full = _feature_rows(panel, rf, econ)
    operational = {name: full[(name, 8)] for name in SYSTEMS}
    aligned, cases = _operational_inputs(operational, panel, cases)
    all_banks, quarters, q_lookup, systems, case = precompute(aligned, cases)
    bank_count = len(all_banks)
    rng = np.random.default_rng(args.seed)

    point = pd.read_csv(OUT / "same_workload_comparison.csv")
    estimable = {}
    needed = {}
    for target in TARGETS:
        estimable[target] = []
        for first, second in PAIRS:
            a = point[(point.target_workload == target) & (point.system == first)].iloc[0]
            b = point[(point.target_workload == target) & (point.system == second)].iloc[0]
            if bool(a.within_2_alert_tolerance and b.within_2_alert_tolerance):
                estimable[target].append((first, second))
        needed[target] = {name for pair in estimable[target] for name in pair}
    active = set().union(*needed.values()) if needed else set()
    differences = []
    system_points = []

    for replicate in range(args.reps):
        draw = rng.integers(0, bank_count, size=bank_count)
        multiplicity = np.bincount(draw, minlength=bank_count).astype(np.int16)
        offsets = np.concatenate(([0], np.cumsum(multiplicity[:-1], dtype=np.int64)))
        replicate_results = {}
        for system in active:
            targets = [target for target in TARGETS if system in needed[target]]
            workload, metrics = system_bootstrap(
                multiplicity, offsets, q_lookup, systems[system], case, targets)
            replicate_results[system] = {"workload": workload, "metrics": metrics}
        for target in TARGETS:
            for system in needed[target]:
                result = replicate_results[system]["metrics"][target]
                system_points.append({"replicate": replicate, "target_workload": target,
                    "system": system, **{k: result.get(k, np.nan) for k in
                    ("capacity_k", "attained_workload", "workload_mismatch", *METRICS)}})
            for first, second in estimable[target]:
                first_result = replicate_results[first]["metrics"][target]
                second_result = replicate_results[second]["metrics"][target]
                low = max(first_result["workload_min"], second_result["workload_min"])
                high = min(first_result["workload_max"], second_result["workload_max"])
                valid = low <= target <= high
                for metric in METRICS:
                    a, b = first_result.get(metric, np.nan), second_result.get(metric, np.nan)
                    value = a - b if valid and np.isfinite(a) and np.isfinite(b) else np.nan
                    differences.append({"replicate": replicate, "target_workload": target,
                        "contrast": f"{first} - {second}", "metric": metric,
                        "first_attained_workload": first_result["attained_workload"],
                        "second_attained_workload": second_result["attained_workload"],
                        "replicate_common_support_min": low, "replicate_common_support_max": high,
                        "within_replicate_common_support": valid, "difference_first_minus_second": value})
        if (replicate + 1) % 100 == 0:
            print(f"Bootstrap {replicate + 1}/{args.reps}", flush=True)

    draws = pd.DataFrame(differences)
    draws.to_csv(OUT / "paired_workload_bootstrap_draws.csv", index=False)
    pd.DataFrame(system_points).to_csv(OUT / "bootstrap_system_workload_draws.csv", index=False)
    summaries = []
    for target in TARGETS:
        for first, second in PAIRS:
            contrast = f"{first} - {second}"
            allowed = (first, second) in estimable[target]
            first_point = point[(point.target_workload == target) & (point.system == first)]
            second_point = point[(point.target_workload == target) & (point.system == second)]
            for metric in METRICS:
                group = draws.loc[(draws.target_workload == target) & (draws.contrast == contrast)
                                  & (draws.metric == metric)] if allowed else pd.DataFrame()
                values = group.difference_first_minus_second.dropna() if len(group) else pd.Series(dtype=float)
                point_diff = np.nan
                if allowed and len(first_point) == 1 and len(second_point) == 1:
                    first_value, second_value = first_point.iloc[0][metric], second_point.iloc[0][metric]
                    if pd.notna(first_value) and pd.notna(second_value):
                        point_diff = float(first_value - second_value)
                summaries.append({"target_workload": target, "contrast": contrast, "metric": metric,
                    "full_sample_difference": point_diff, "bootstrap_mean": float(values.mean()) if len(values) else np.nan,
                    "bootstrap_median": float(values.median()) if len(values) else np.nan,
                    "ci_2_5": float(values.quantile(.025)) if len(values) else np.nan,
                    "ci_97_5": float(values.quantile(.975)) if len(values) else np.nan,
                    "valid_replicates": len(values), "attempted_replicates": args.reps,
                    "status": "PASS" if len(values) >= 0.8 * args.reps else "NOT_ESTIMABLE_OR_LOW_VALID_DRAWS"})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "paired_workload_differences.csv", index=False)
    (OUT / "bootstrap_method.txt").write_text(
        f"Paired institution-cluster bootstrap; seed={args.seed}; attempted replicates={args.reps}; "
        "each bank's quarterly rows and failure history resampled together; model scores fixed; "
        "each system's dense workload frontier rebuilt within replicate; no extrapolation outside "
        "replicate pairwise common support. Percentile intervals condition on fitted models.\n")
    print(f"Saved fixed-workload bootstrap summaries to {OUT}")


if __name__ == "__main__":
    main()
