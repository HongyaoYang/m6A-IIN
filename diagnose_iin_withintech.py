#!/usr/bin/env python3
"""Read-only diagnostics for low within-technology IIN regression performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from iin_protocol_common import parse_split


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("tech_GT"))
    parser.add_argument("--model-dir", type=Path, default=Path("model_IIN/final_author_elmo"))
    parser.add_argument("--min-total", type=int, default=10)
    parser.add_argument(
        "--techs", nargs="+",
        default=["MeRIP_seq", "m6ACE_seq", "MAZTER_seq", "scDART_seq", "GLORI"],
    )
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    args = arguments()
    if args.min_total < 1:
        raise ValueError("--min-total must be positive")
    train_split = parse_split(args.base_dir / "train_site.csv")
    valid_split = parse_split(args.base_dir / "valid_site.csv")
    reports = []
    for tech in args.techs:
        gt = pd.read_csv(
            args.base_dir / f"{tech}.csv",
            usecols=["site_idx", "ratio", "Total", "class"],
        )
        predictions = pd.read_csv(
            args.model_dir / f"{tech}_allSite_preds.csv",
            usecols=["site_idx", "pred"],
        )
        if gt["site_idx"].duplicated().any() or predictions["site_idx"].duplicated().any():
            raise ValueError(f"Duplicate site_idx in {tech}")
        test = predictions.merge(gt, on="site_idx", how="inner", validate="one_to_one")
        # allSite_preds contains sites from every technology; an inner join is
        # intentional. Only this technology's held-out GT rows are scored.
        train = gt.loc[gt["site_idx"].isin(train_split[tech]) & (gt["Total"] > 0)]
        valid = gt.loc[gt["site_idx"].isin(valid_split[tech]) & (gt["Total"] > 0)]
        heldout_gt = gt.loc[
            ~gt["site_idx"].isin(set(train_split[tech]) | set(valid_split[tech]))
        ]
        missing_heldout_ids = set(heldout_gt["site_idx"]) - set(test["site_idx"])
        if set(test["site_idx"]) & (set(train["site_idx"]) | set(valid["site_idx"])):
            raise ValueError(f"Train/valid leakage into {tech} same-technology test")
        eligible = test.loc[test["Total"] >= args.min_total]
        if len(eligible) < 2:
            raise ValueError(f"Too few eligible sites for {tech}")
        y = eligible["ratio"].to_numpy(dtype=float)
        p = eligible["pred"].to_numpy(dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError(f"Nonfinite observed/predicted values for {tech}")
        train_mean = float(train["ratio"].mean())
        test_mean = float(y.mean())
        report = {
            "technology": tech,
            "n_train_total_gt0": int(len(train)),
            "n_valid_total_gt0": int(len(valid)),
            "n_test_all_covered": int(len(test)),
            "n_test_gt_uncovered": int(len(missing_heldout_ids)),
            "n_test_total_ge_threshold": int(len(eligible)),
            "train_ratio_mean": train_mean,
            "valid_ratio_mean": float(valid["ratio"].mean()),
            "test_ratio": summarize(y),
            "test_prediction": summarize(p),
            "mean_prediction_minus_observation": float(p.mean() - test_mean),
            "test_mse": float(mean_squared_error(y, p)),
            "test_mean_baseline_mse": float(mean_squared_error(y, np.full_like(y, test_mean))),
            "train_mean_baseline_mse": float(mean_squared_error(y, np.full_like(y, train_mean))),
            "test_R2": float(r2_score(y, p)),
            "train_mean_baseline_R2": float(r2_score(y, np.full_like(y, train_mean))),
            "test_class_positive_fraction": float(eligible["class"].mean()),
            "train_class_positive_fraction_total_ge_threshold": float(
                train.loc[train["Total"] >= args.min_total, "class"].mean()
            ),
        }
        config_path = args.model_dir / f"{tech}_run_config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            report["best_epoch"] = config.get("best_epoch_value")
            report["max_epochs"] = config.get("epochs")
            report["learning_rate"] = config.get("learning_rate")
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    print(f"Completed {len(reports)} read-only technology diagnostics", flush=True)


if __name__ == "__main__":
    main()
