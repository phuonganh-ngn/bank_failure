#!/usr/bin/env python3
"""Recreate the five-seed 4Q EARLY-versus-FULL sensitivity summary."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduced"


def ap(frame: pd.DataFrame) -> float:
    return float(average_precision_score(frame.label.astype(int), frame.raw_score.astype(float)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(ROOT / "data/frozen_predictions/rf_multiseed_4q_early_full.parquet")
    if set(data.information_set.unique()) != {"EARLY", "FULL"} or set(data.seed_index.unique()) != set(range(5)):
        raise RuntimeError("Expected five paired seed indices for 4Q EARLY and FULL")
    seed_rows = []
    fold_rows = []
    for seed_index, seed_data in data.groupby("seed_index", sort=True):
        values = {}
        for info, frame in seed_data.groupby("information_set", sort=True):
            values[info] = ap(frame)
        seed_rows.append({
            "seed_index": int(seed_index),
            "early_seed": int(seed_data.loc[seed_data.information_set.eq("EARLY"), "seed"].iloc[0]),
            "full_seed": int(seed_data.loc[seed_data.information_set.eq("FULL"), "seed"].iloc[0]),
            "early_ap_pooled": values["EARLY"], "full_ap_pooled": values["FULL"],
            "early_minus_full_ap": values["EARLY"] - values["FULL"],
        })
        for fold, group in seed_data.groupby("fold", sort=True):
            fold_ap = {info: ap(frame) for info, frame in group.groupby("information_set", sort=True)}
            fold_rows.append({"seed_index": int(seed_index), "fold": fold,
                "early_ap": fold_ap["EARLY"], "full_ap": fold_ap["FULL"],
                "early_minus_full_ap": fold_ap["EARLY"] - fold_ap["FULL"],
                "positive_labels": int(group.loc[group.information_set.eq("EARLY"), "label"].sum()),
                "test_rows": int(group.loc[group.information_set.eq("EARLY")].shape[0])})
    by_seed = pd.DataFrame(seed_rows)
    by_fold = pd.DataFrame(fold_rows)
    diffs = by_seed.early_minus_full_ap.to_numpy(float)
    summary = pd.DataFrame([{
        "horizon_quarters": 4, "seed_count": len(by_seed),
        "early_minus_full_ap_mean": float(diffs.mean()),
        "early_minus_full_ap_sd": float(diffs.std(ddof=1)),
        "early_minus_full_ap_min": float(diffs.min()),
        "early_minus_full_ap_max": float(diffs.max()),
        "share_seeds_early_exceeds_full": float(np.mean(diffs > 0)),
        "pooled_raw_score_ap": True,
        "interpretation": "Seed variation is conditional on the saved folds and test cases; pooled raw-score AP can depend on cross-block score comparability.",
    }])
    by_seed.to_csv(OUT / "multiseed_4q_by_seed.csv", index=False)
    by_fold.to_csv(OUT / "multiseed_4q_by_fold.csv", index=False)
    summary.to_csv(OUT / "multiseed_4q_gap_summary.csv", index=False)
    print(f"Wrote five-seed 4Q summary: mean gap={diffs.mean():.6f}, SD={diffs.std(ddof=1):.6f}, "
          f"range=[{diffs.min():.6f}, {diffs.max():.6f}].")


if __name__ == "__main__":
    main()
