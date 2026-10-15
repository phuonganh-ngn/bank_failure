#!/usr/bin/env python3
"""Recreate the paper's pooled metrics, block metrics, capacity frontier, and plot.

The default mode uses archived fitted OOT predictions. It does not train models.
All generated files are placed under results/reproduced/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduced"
FOLDS = (("gfc", "2007Q1", "2009Q4"),
         ("post_gfc", "2010Q1", "2018Q4"),
         ("covid_modern", "2019Q1", "2024Q3"))
HORIZONS = (1, 2, 4, 8)
CAPACITY_GRID = np.round(np.arange(0.005, 0.1500001, 0.0025), 6)
SYSTEMS = ("CAPITAL_SCREEN", "Economic Logit", "RF STRUCTURAL", "RF EARLY", "RF FULL")


def _top5_recall(frame: pd.DataFrame) -> float:
    positives = int(frame.label.sum())
    if positives == 0:
        return float("nan")
    picked = []
    for _, group in frame.groupby("quarter", sort=False):
        k = max(1, int(math.ceil(len(group) * 0.05)))
        picked.extend(group.nlargest(k, "raw_score").index.tolist())
    return float(frame.loc[picked, "label"].sum() / positives)


def _unique_index(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.duplicated(keys).any():
        raise RuntimeError(f"Duplicate observation identifiers for keys {keys}")
    return frame.set_index(keys)


def _metric_row(frame: pd.DataFrame, system: str, horizon: int, fold: str) -> dict:
    y = frame.label.to_numpy(dtype=int)
    score = frame.raw_score.to_numpy(dtype=float)
    prevalence = float(y.mean()) if len(y) else float("nan")
    ap = float(average_precision_score(y, score)) if y.sum() and len(np.unique(y)) > 1 else float("nan")
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    row = {"system": system, "horizon_quarters": horizon, "fold": fold,
           "n_test": len(frame), "positive_labels": int(y.sum()), "prevalence": prevalence,
           "average_precision": ap, "ap_prevalence_ratio": ap / prevalence if prevalence else np.nan,
           "normalized_ap": (ap - prevalence) / (1 - prevalence) if prevalence < 1 else np.nan,
           "roc_auc": auc, "recall_top5_per_quarter": _top5_recall(frame)}
    if "calibrated_probability" in frame and frame.calibrated_probability.notna().all():
        row["brier"] = float(brier_score_loss(y, frame.calibrated_probability.to_numpy(float)))
    else:
        row["brier"] = np.nan
    return row


def _feature_rows(panel: pd.DataFrame, predictions_rf: pd.DataFrame, predictions_econ: pd.DataFrame):
    panel = panel.copy()
    panel["CERT"] = panel.CERT.astype(str)
    panel["quarter"] = pd.PeriodIndex(panel.quarter, freq="Q")
    panel["panel_order"] = np.arange(len(panel), dtype=np.int64)
    panel["equity_to_assets"] = pd.to_numeric(panel.equity_to_assets, errors="coerce")

    rf = predictions_rf.copy()
    rf["CERT"] = rf.CERT.astype(str)
    rf["quarter"] = pd.PeriodIndex(rf.quarter, freq="Q")
    rf = rf[(rf.predictor_lag_quarters == 0) & (rf.weighting == "balanced")]
    econ = predictions_econ.copy()
    econ["CERT"] = econ.CERT.astype(str)
    econ["quarter"] = pd.PeriodIndex(econ.quarter, freq="Q")

    aligned: dict[tuple[str, int], pd.DataFrame] = {}
    for h in HORIZONS:
        pred_parts = {}
        for info in ("STRUCTURAL", "EARLY", "FULL"):
            g = rf[(rf.horizon_quarters == h) & (rf.information_set == info)]
            pred_parts[f"RF {info}"] = g
        pred_parts["Economic Logit"] = econ[econ.horizon_quarters == h]
        panel_h = panel.loc[panel[f"fail_{h}q"].notna()].copy()
        panel_h["fold"] = ""
        for fold, start_text, end_text in FOLDS:
            start, end = pd.Period(start_text, freq="Q"), pd.Period(end_text, freq="Q")
            panel_h.loc[panel_h.quarter.between(start, end), "fold"] = fold
        panel_h = panel_h[panel_h.fold.ne("")].copy()
        key_cols = ["CERT", "quarter"]
        key_sets = [set(zip(g.CERT, g.quarter)) for g in pred_parts.values()]
        common = set.intersection(*key_sets)
        cap = panel_h.loc[panel_h.equity_to_assets.notna(), [*key_cols, "fold", f"fail_{h}q", "panel_order", "equity_to_assets"]].copy()
        common &= set(zip(cap.CERT, cap.quarter))
        if not common:
            raise RuntimeError(f"No common OOT rows for horizon {h}")
        cap = cap.loc[[key in common for key in zip(cap.CERT, cap.quarter)]].copy()
        cap = _unique_index(cap, key_cols).reset_index()
        cap = cap.rename(columns={f"fail_{h}q": "label"})
        cap["label"] = cap.label.astype(int)
        cap["raw_score"] = -cap.equity_to_assets.astype(float)
        for sysname, g in pred_parts.items():
            if g.duplicated(key_cols).any():
                raise RuntimeError(f"Duplicate saved prediction keys for {sysname}/{h}Q")
            z = _unique_index(g, key_cols).reindex(pd.MultiIndex.from_frame(cap[key_cols]))
            if z.raw_score.isna().any() or z.label.isna().any():
                raise RuntimeError(f"Incomplete prediction alignment for {sysname}/{h}Q")
            if not np.array_equal(z.label.astype(int).to_numpy(), cap.label.to_numpy()):
                raise RuntimeError(f"Outcome labels differ for {sysname}/{h}Q")
            aligned[(sysname, h)] = pd.DataFrame({
                "CERT": cap.CERT.astype(str).to_numpy(), "quarter": cap.quarter.to_numpy(),
                "fold": cap.fold.to_numpy(), "label": z.label.astype(int).to_numpy(),
                "raw_score": z.raw_score.astype(float).to_numpy(),
                **({"calibrated_probability": z.calibrated_probability.to_numpy(float)}
                   if "calibrated_probability" in z.columns else {}),
                "panel_order": cap.panel_order.to_numpy(np.int64),
            })
        aligned[("CAPITAL_SCREEN", h)] = cap[["CERT", "quarter", "fold", "label", "raw_score", "panel_order"]].copy()
    return aligned


def _operational_inputs(aligned: dict, panel: pd.DataFrame, cases: pd.DataFrame):
    h = 8
    chosen = {name: aligned[(name, h)].copy() for name in SYSTEMS}
    key_sets = [set(zip(frame.CERT.astype(str), frame.quarter)) for frame in chosen.values()]
    common = set.intersection(*key_sets)
    if not common:
        raise RuntimeError("The five operational systems have no common 8Q prediction universe")
    panel_keys = panel[["CERT", "quarter"]].copy()
    panel_keys["CERT"] = panel_keys.CERT.astype(str)
    panel_keys["quarter"] = pd.PeriodIndex(panel_keys.quarter, freq="Q")
    order_map = dict(zip(zip(panel_keys.CERT, panel_keys.quarter), np.arange(len(panel_keys), dtype=np.int64)))
    common_ordered = sorted(common, key=lambda k: order_map[k])
    common_index = pd.MultiIndex.from_tuples(common_ordered, names=["CERT", "quarter"])
    for name, frame in chosen.items():
        frame["CERT"] = frame.CERT.astype(str)
        frame = _unique_index(frame, ["CERT", "quarter"]).reindex(common_index).reset_index()
        frame["panel_order"] = [order_map[(c, q)] for c, q in zip(frame.CERT, frame.quarter)]
        chosen[name] = frame

    cases = cases.copy()
    cases["CERT"] = cases.CERT.astype(str)
    cases["failure_quarter"] = pd.PeriodIndex(cases.failure_quarter, freq="Q")
    cases["failure_ord"] = cases.failure_quarter.astype("int64")
    if len(cases) != 515 or cases.CERT.nunique() != 515:
        raise RuntimeError(f"Expected the saved 515-failure cohort, found {len(cases)}")
    keys_by_system = {name: set(zip(g.CERT.astype(str), g.quarter)) for name, g in chosen.items()}
    for case in cases.itertuples(index=False):
        window = [(case.CERT, case.failure_quarter - j) for j in range(1, 9)]
        if any(not set(window).issubset(keys_by_system[name]) for name in SYSTEMS):
            raise RuntimeError(f"Incomplete eight-quarter path for CERT {case.CERT}")
    return chosen, cases


def score_frontier(frame: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    d = frame.sort_values(["quarter", "panel_order"], kind="mergesort").copy().reset_index(drop=True)
    d["q_ord"] = pd.PeriodIndex(d.quarter, freq="Q").astype("int64")
    qsize = d.groupby("quarter", sort=False).CERT.transform("size").to_numpy()
    quarter_sizes = d.groupby("quarter", sort=False).CERT.size().to_numpy()
    ranks = d.groupby("quarter", sort=False).raw_score.rank(method="first", ascending=False).to_numpy()
    top = ranks[:, None] <= np.ceil(qsize[:, None] * CAPACITY_GRID[None, :])
    d["rowpos"] = np.arange(len(d))
    ordered = d.sort_values(["CERT", "q_ord"], kind="mergesort")
    bank_groups = ordered.groupby("CERT", sort=False).indices
    order_pos = ordered.rowpos.to_numpy()
    confirmed = np.zeros_like(top)
    for positions in bank_groups.values():
        p = order_pos[positions]
        qs = d.q_ord.to_numpy()[p]
        if len(p) < 2:
            continue
        consecutive = np.diff(qs) == 1
        cur, prior = p[1:][consecutive], p[:-1][consecutive]
        confirmed[cur, :] = top[cur, :] & top[prior, :]

    quarters = d.quarter.drop_duplicates().tolist()
    y = d.label.to_numpy(int)
    assets = cases.assets_t_minus_4.to_numpy(float)
    deposits = cases.deposits_t_minus_4.to_numpy(float)
    records = []
    for j, capacity in enumerate(CAPACITY_GRID):
        confirmed_j = confirmed[:, j]
        q_count = pd.Series(confirmed_j).groupby(d.quarter, sort=False).sum().reindex(quarters, fill_value=0).to_numpy(float)
        false_count = int(np.sum(confirmed_j & (y == 0)))
        alert_map = d.loc[confirmed_j].groupby("CERT", sort=False).q_ord.apply(lambda x: set(x.astype(int))).to_dict()
        lead = np.asarray([
            max((int(case.failure_ord) - q for q in alert_map.get(case.CERT, set())
                 if int(case.failure_ord) - 8 <= q < int(case.failure_ord)), default=np.nan)
            for case in cases.itertuples(index=False)
        ], dtype=float)
        detected = np.isfinite(lead)
        records.append({
            "system": frame.system.iloc[0], "capacity_k": float(capacity),
            "mean_raw_candidates_per_quarter": float(np.ceil(quarter_sizes * capacity).mean()),
            "mean_confirmed_alert_institutions_per_quarter": float(q_count.mean()),
            "failure_recall": float(detected.mean()),
            "median_lead_quarters": float(np.nanmedian(lead)) if detected.any() else np.nan,
            "mean_lead_quarters": float(np.nanmean(lead)) if detected.any() else np.nan,
            "recall_lead_ge_1Q": float(np.sum(detected & (lead >= 1)) / len(cases)),
            "recall_lead_ge_2Q": float(np.sum(detected & (lead >= 2)) / len(cases)),
            "recall_lead_ge_4Q": float(np.sum(detected & (lead >= 4)) / len(cases)),
            "recall_lead_ge_6Q": float(np.sum(detected & (lead >= 6)) / len(cases)),
            "asset_weighted_recall_T4": float(assets[detected].sum() / assets.sum()),
            "deposit_weighted_recall_T4": float(deposits[detected].sum() / deposits.sum()),
            "false_alert_decisions": false_count,
            "detected_failures": int(detected.sum()),
            "false_alerts_per_detected_failure": false_count / max(1, int(detected.sum())),
            "eligible_failure_paths": len(cases), "quarter_count": len(quarters),
        })
    return pd.DataFrame(records)


def render_frontier(frontier: pd.DataFrame, support: tuple[float, float]):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"CAPITAL_SCREEN": "#555555", "Economic Logit": "#9c755f",
              "RF STRUCTURAL": "#4e79a7", "RF EARLY": "#e15759", "RF FULL": "#59a14f"}
    display_names = {"CAPITAL_SCREEN": "Capital Screen"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for system, group in frontier.groupby("system", sort=False):
        group = group.sort_values("mean_confirmed_alert_institutions_per_quarter")
        ax.plot(group.mean_confirmed_alert_institutions_per_quarter, group.failure_recall,
                marker="o", markersize=2.5, linewidth=1.6,
                label=display_names.get(system, system), color=colors[system])
    ax.axvspan(*support, color="#d9eaf7", alpha=0.45, zorder=0, label="Common support")
    ax.set_xlabel("Mean confirmed-alert institutions per quarter")
    ax.set_ylabel("Failure recall")
    ax.set_ylim(0, 1)
    ax.grid(axis="both", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"Figure_Workload_vs_Failure_Recall.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data/model_panel.parquet", columns=["CERT", "quarter", "fail_1q", "fail_2q", "fail_4q", "fail_8q", "equity_to_assets"])
    rf = pd.read_parquet(ROOT / "data/oot_predictions/strict_rf_oot_predictions.parquet")
    econ = pd.read_parquet(ROOT / "data/oot_predictions/strict_economic_logit_oot_predictions.parquet")
    structural_capital = pd.read_parquet(ROOT / "data/oot_predictions/structural_plus_capital_8q_predictions.parquet")
    cases = pd.read_csv(ROOT / "data/eligible_failure_paths.csv", dtype={"CERT": str})
    aligned = _feature_rows(panel, rf, econ)

    metric_rows = []
    for (system, horizon), frame in aligned.items():
        metric_rows.append(_metric_row(frame, system, horizon, "pooled"))
        for fold, _, _ in FOLDS:
            metric_rows.append(_metric_row(frame.loc[frame.fold == fold], system, horizon, fold))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "predictive_performance.csv", index=False)

    structural_capital = structural_capital.copy()
    structural_capital["CERT"] = structural_capital.CERT.astype(str)
    structural_capital["quarter"] = pd.PeriodIndex(structural_capital.quarter, freq="Q")
    if structural_capital.duplicated(["CERT", "quarter"]).any():
        raise RuntimeError("Duplicate STRUCTURAL_PLUS_CAPITAL prediction keys")
    panel["CERT"] = panel.CERT.astype(str)
    panel["quarter"] = pd.PeriodIndex(panel.quarter, freq="Q")
    panel["panel_order"] = np.arange(len(panel), dtype=np.int64)
    sc_labels = panel.set_index(["CERT", "quarter"]).fail_8q
    sc_keys = pd.MultiIndex.from_frame(structural_capital[["CERT", "quarter"]])
    sc_expected = sc_labels.reindex(sc_keys).to_numpy()
    if pd.isna(sc_expected).any() or not np.array_equal(structural_capital.label.astype(int).to_numpy(), sc_expected.astype(int)):
        raise RuntimeError("STRUCTURAL_PLUS_CAPITAL labels do not match the analytical panel")
    sc_frame = structural_capital[["CERT", "quarter", "fold", "label", "raw_score", "calibrated_probability"]].copy()
    order_lookup = panel.set_index(["CERT", "quarter"]).panel_order
    sc_frame["panel_order"] = order_lookup.reindex(pd.MultiIndex.from_frame(sc_frame[["CERT", "quarter"]])).to_numpy()
    sc_metrics = _metric_row(sc_frame, "RF STRUCTURAL_PLUS_CAPITAL", 8, "pooled")
    pd.DataFrame([sc_metrics]).to_csv(OUT / "performance_structural_plus_capital.csv", index=False)

    # Re-align the operational cohort across all five systems before scoring.
    operational_panel = panel.copy()
    operational_panel["CERT"] = operational_panel.CERT.astype(str)
    operational_panel["quarter"] = pd.PeriodIndex(operational_panel.quarter, freq="Q")
    chosen, cases = _operational_inputs(aligned, operational_panel, cases)
    common_keys = set(zip(chosen["RF STRUCTURAL"].CERT.astype(str), chosen["RF STRUCTURAL"].quarter))
    sc_key_set = set(zip(sc_frame.CERT.astype(str), sc_frame.quarter))
    if common_keys != sc_key_set:
        raise RuntimeError(f"STRUCTURAL_PLUS_CAPITAL does not share the operational universe: {len(common_keys - sc_key_set)} missing, {len(sc_key_set - common_keys)} extra")
    sc_frame = sc_frame.sort_values("panel_order", kind="mergesort").reset_index(drop=True)
    sc_frame["system"] = "RF STRUCTURAL_PLUS_CAPITAL"
    chosen["RF STRUCTURAL_PLUS_CAPITAL"] = sc_frame
    frontier_rows = []
    for system in (*SYSTEMS, "RF STRUCTURAL_PLUS_CAPITAL"):
        frame = chosen[system].copy()
        frame["system"] = system
        frontier_rows.append(score_frontier(frame, cases))
    frontier = pd.concat(frontier_rows, ignore_index=True)
    five_system_frontier = frontier.loc[frontier.system.isin(SYSTEMS)].copy()
    six_system_frontier = frontier.copy()
    five_system_frontier.to_csv(OUT / "workload_frontier.csv", index=False)
    six_system_frontier.to_csv(OUT / "workload_frontier_with_capital_ablation.csv", index=False)
    frontier.loc[frontier.system.eq("RF STRUCTURAL_PLUS_CAPITAL")].to_csv(
        OUT / "capital_ablation_frontier.csv", index=False)

    workload = five_system_frontier.groupby("system").mean_confirmed_alert_institutions_per_quarter.agg(["min", "max"])
    support = (float(workload["min"].max()), float(workload["max"].min()))
    if support[0] >= support[1]:
        raise RuntimeError(f"No common confirmed-workload support: {support}")
    pd.DataFrame([{"system": name, "workload_min": workload.loc[name, "min"], "workload_max": workload.loc[name, "max"],
                   "common_support_min": support[0], "common_support_max": support[1]} for name in workload.index]).to_csv(
                       OUT / "common_workload_range.csv", index=False)
    fixed_rows = []
    for target in (50.0, 100.0, 175.0, 250.0):
        if not support[0] <= target <= support[1]:
            continue
        for system, group in five_system_frontier.groupby("system", sort=False):
            r = group.loc[(group.mean_confirmed_alert_institutions_per_quarter - target).abs().idxmin()]
            mismatch = float(r.mean_confirmed_alert_institutions_per_quarter - target)
            valid = abs(mismatch) <= 2.0
            fixed_rows.append({"target_workload": target, "system": system,
                "attained_workload": r.mean_confirmed_alert_institutions_per_quarter,
                "capacity_k": r.capacity_k, "workload_difference": mismatch,
                "within_2_alert_tolerance": valid,
                "failure_recall": r.failure_recall if valid else np.nan,
                "recall_lead_ge_4Q": r.recall_lead_ge_4Q if valid else np.nan,
                "median_lead_quarters": r.median_lead_quarters if valid else np.nan,
                "asset_weighted_recall_T4": r.asset_weighted_recall_T4 if valid else np.nan,
                "deposit_weighted_recall_T4": r.deposit_weighted_recall_T4 if valid else np.nan,
                "false_alerts_per_detected_failure": r.false_alerts_per_detected_failure if valid else np.nan})
    pd.DataFrame(fixed_rows).to_csv(OUT / "same_workload_comparison.csv", index=False)
    pd.DataFrame([frontier.loc[(frontier.system == system) & np.isclose(frontier.capacity_k, 0.05)].iloc[0]
                  for system in ("RF STRUCTURAL", "RF STRUCTURAL_PLUS_CAPITAL", "RF EARLY")]).to_csv(
                      OUT / "operational_ablation_at_5pct.csv", index=False)
    fixed_sc = frontier.loc[frontier.system.eq("RF STRUCTURAL_PLUS_CAPITAL")].copy()
    fixed_sc["target_workload"] = 100.0
    fixed_sc["workload_difference"] = fixed_sc.mean_confirmed_alert_institutions_per_quarter - 100.0
    fixed_sc = fixed_sc.loc[[fixed_sc.workload_difference.abs().idxmin()]]
    fixed_sc.to_csv(OUT / "structural_plus_capital_workload_diagnostic_100.csv", index=False)
    render_frontier(five_system_frontier, support)
    summary = {
        "common_workload_support": {"min_confirmed_alerts_per_quarter": support[0], "max_confirmed_alerts_per_quarter": support[1]},
        "frontier_rows": len(five_system_frontier), "six_system_frontier_rows": len(six_system_frontier),
        "capacity_points_per_system": len(CAPACITY_GRID),
        "fixed_workload_targets": [50, 100, 175, 250],
        "matching_tolerance_alerts": 2,
        "primary_bootstrap_summary_file": "../tables/paired_workload_differences.csv",
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote pooled and block metrics; five-system frontier rows={len(five_system_frontier)}; "
          f"six-system diagnostic rows={len(six_system_frontier)}; common workload support "
          f"{support[0]:.3f}–{support[1]:.3f}.")


if __name__ == "__main__":
    main()
