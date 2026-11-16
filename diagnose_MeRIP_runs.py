#!/usr/bin/env python3
"""Paired diagnostics for the existing standard and cap10 MeRIP runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from iin_protocol_common import (
    TECHNOLOGIES,
    calculate_metrics,
    join_predictions,
    parse_split,
    read_ground_truth,
)

PROJECT = Path("/gpfs/work/bio/hongyaoyang23/m6A_IIN")
BASE = PROJECT / "tech_GT"
TECH = "MeRIP_seq"
OUT = PROJECT / "model_IIN/diagnostics_MeRIP_20261006"
RUNS = {
    "standard_uncapped": PROJECT / "model_IIN/standard_binomial_MeRIP_control",
    "cap10_sensitivity": PROJECT / "model_IIN/cap10_MeRIP",
}
MIN_TOTAL = 10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_run(name: str, run_dir: Path, gt: pd.DataFrame, valid_ids: set[int]):
    config_path = run_dir / f"{TECH}_run_config.json"
    log_path = run_dir / f"{TECH}_train_valid_log.csv"
    pred_path = run_dir / f"{TECH}_best_valid_predictions.csv"
    config = json.loads(config_path.read_text())
    log = pd.read_csv(log_path)
    pred = pd.read_csv(pred_path)
    required_pred = {"site_idx", "pred", "ratio", "m6A", "Total", "class"}
    require(required_pred.issubset(pred.columns), f"{pred_path} missing fields")
    pred["site_idx"] = pd.to_numeric(pred["site_idx"], errors="raise").astype("int64")
    for col in ("pred", "ratio", "m6A", "Total", "class"):
        pred[col] = pd.to_numeric(pred[col], errors="raise")
    require(not pred["site_idx"].duplicated().any(), f"duplicate prediction site_idx: {name}")
    require(np.isfinite(pred[["pred", "ratio", "m6A", "Total"]].to_numpy()).all(),
            f"non-finite prediction/label values: {name}")
    require(pred["class"].dropna().isin([0, 1]).all(),
            f"invalid non-null binary class values: {name}")
    require(pred.loc[pred["Total"] >= MIN_TOTAL, "class"].notna().all(),
            f"missing class labels for Total>={MIN_TOTAL}: {name}")
    require(pred["pred"].between(0.0, 1.0).all(), f"pred outside [0,1]: {name}")
    require((pred["m6A"] >= 0).all() and (pred["m6A"] <= pred["Total"]).all(),
            f"invalid count labels: {name}")

    source = pred.merge(gt, on="site_idx", how="left", suffixes=("_saved", "_gt"),
                        validate="one_to_one", indicator=True)
    require(source["_merge"].eq("both").all(), f"predictions not all matched to GT: {name}")
    for col in ("m6A", "Total"):
        require(np.array_equal(source[f"{col}_saved"].to_numpy(), source[f"{col}_gt"].to_numpy()),
                f"saved {col} differs from GT: {name}")
    require(np.array_equal(source["class_saved"].to_numpy(),
                           source["class_gt"].to_numpy(), equal_nan=True),
            f"saved class differs from GT: {name}")
    require(np.allclose(source["ratio_saved"], source["ratio_gt"], rtol=0, atol=1e-7),
            f"saved ratio differs from GT: {name}")
    require(set(pred["site_idx"]) <= valid_ids, f"prediction contains a non-validation site: {name}")

    expected = set(gt.loc[gt["site_idx"].isin(valid_ids) & (gt["Total"] > 0), "site_idx"])
    observed = set(pred["site_idx"])
    coverage = {
        "run": name,
        "valid_split_ids": len(valid_ids),
        "expected_valid_total_gt0": len(expected),
        "predicted_rows": len(pred),
        "predicted_unique_site_idx": pred["site_idx"].nunique(),
        "missing_expected_sites": len(expected - observed),
        "unexpected_sites": len(observed - expected),
        "n_total_ge_10": int((pred["Total"] >= MIN_TOTAL).sum()),
    }

    loss_col = "valid_binomial_nll" if "valid_binomial_nll" in log.columns else "valid_capped_binomial_nll"
    require(loss_col in log.columns, f"no validation loss column for {name}")
    best_log = log.loc[log[loss_col].idxmin()]
    best_epoch_log = int(best_log["epoch"])
    best_epoch_config = int(config.get("best_epoch_value", best_epoch_log))
    require(best_epoch_log == best_epoch_config,
            f"best epoch mismatch in {name}: log={best_epoch_log}, config={best_epoch_config}")
    coverage.update({
        "config_best_epoch": best_epoch_config,
        "min_logged_validation_loss_epoch": best_epoch_log,
        "logged_best_loss_column": loss_col,
        "logged_best_loss": float(best_log[loss_col]),
        "completed_epochs": int(config.get("completed_epochs", len(log))),
        "logged_lr_at_best_epoch": float(best_log["learning_rate"]),
        "config": config,
    })
    return pred[["site_idx", "pred"]], coverage, config, log


def ordinary_binomial_nll(y: np.ndarray, total: np.ndarray, pred: np.ndarray) -> Tuple[float, float, int]:
    # Only numerical protection for log(0); raw saved predictions remain unchanged.
    eps = np.finfo(np.float64).eps
    p = np.clip(pred.astype(np.float64), eps, 1.0 - eps)
    per_site = -(y * np.log(p) + (total - y) * np.log1p(-p))
    return float(per_site.mean()), float(per_site.sum()), int(((pred <= 0) | (pred >= 1)).sum())


def metric_row(name: str, frame: pd.DataFrame) -> Dict[str, object]:
    result = calculate_metrics(frame, min_total=MIN_TOTAL)
    y = frame["ratio"].to_numpy(dtype=np.float64)
    p = frame["pred"].to_numpy(dtype=np.float64)
    total = frame["Total"].to_numpy(dtype=np.float64)
    nll_mean, nll_sum, n_clipped = ordinary_binomial_nll(
        frame["m6A"].to_numpy(dtype=np.float64), total, p
    )
    w_mse = float(np.average(np.square(y - p), weights=total))
    return {
        "run": name,
        **result,
        "MSE": float(np.mean(np.square(y - p))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Total_weighted_MSE": w_mse,
        "ordinary_binomial_nll_mean_on_Total_ge_10": nll_mean,
        "ordinary_binomial_nll_sum_on_Total_ge_10": nll_sum,
        "n_pred_at_0_or_1_numerically_clipped_for_NLL_only": n_clipped,
    }


def describe(values: np.ndarray, name: str, kind: str) -> Dict[str, object]:
    q = np.quantile(values.astype(np.float64), [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {
        "run": name, "quantity": kind, "n": int(values.size),
        "mean": float(np.mean(values)), "sd": float(np.std(values)),
        "min": float(q[0]), "q01": float(q[1]), "q05": float(q[2]),
        "q25": float(q[3]), "median": float(q[4]), "q75": float(q[5]),
        "q95": float(q[6]), "q99": float(q[7]), "max": float(q[8]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    train = parse_split(BASE / "train_site.csv")
    valid = parse_split(BASE / "valid_site.csv")
    require(TECH in TECHNOLOGIES, f"unknown technology: {TECH}")
    train_ids, valid_ids = set(train[TECH]), set(valid[TECH])
    require(not train_ids & valid_ids, "train/valid site overlap")
    gt = read_ground_truth(BASE / f"{TECH}.csv")
    require(not gt["site_idx"].duplicated().any(), "duplicate site_idx in GT")
    require(((gt["m6A"] >= 0) & (gt["m6A"] <= gt["Total"])).all(), "invalid GT counts")
    require(np.allclose(gt.loc[gt["Total"] > 0, "ratio"],
                        gt.loc[gt["Total"] > 0, "m6A"] / gt.loc[gt["Total"] > 0, "Total"],
                        rtol=0, atol=1e-7), "GT ratio != m6A/Total")

    loaded = {}
    audit = {

        "technology": TECH,
        "metrics_threshold_total": MIN_TOTAL,
        "train_split_unique_ids": len(train_ids),
        "valid_split_unique_ids": len(valid_ids),
        "train_valid_overlap": len(train_ids & valid_ids),
        "gt_unique_rows": len(gt),
        "gt_total_gt0_rows": int((gt["Total"] > 0).sum()),
        "runs": {},
    }
    for name, run_dir in RUNS.items():
        pred, coverage, config, log = read_run(name, run_dir, gt, valid_ids)
        loaded[name] = pred
        audit["runs"][name] = coverage

    std = loaded["standard_uncapped"]
    cap = loaded["cap10_sensitivity"]
    require(set(std.site_idx) == set(cap.site_idx), "runs do not cover identical validation site IDs")
    paired = std.rename(columns={"pred": "pred_standard"}).merge(
        cap.rename(columns={"pred": "pred_cap10"}), on="site_idx", how="inner", validate="one_to_one"
    ).merge(gt, on="site_idx", how="inner", validate="one_to_one")
    require(not paired.empty, "no common prediction/GT rows")
    paired.to_csv(OUT / "paired_all_valid_predictions.csv", index=False)
    eval_base = paired.loc[paired["Total"] >= MIN_TOTAL].copy()
    require(not eval_base.empty, "no common Total>=10 sites")
    eval_base.to_csv(OUT / "paired_Total_ge_10_predictions.csv", index=False)
    audit["paired_common_valid_sites"] = len(paired)
    audit["paired_common_Total_ge_10_sites"] = len(eval_base)

    summary_rows = []
    distribution_rows = [describe(eval_base["ratio"].to_numpy(), "ground_truth", "ratio")]
    total_rows = []
    for name, pred_col in (("standard_uncapped", "pred_standard"), ("cap10_sensitivity", "pred_cap10")):
        frame = eval_base[["site_idx", "ratio", "m6A", "Total", "class", pred_col]].rename(columns={pred_col: "pred"})
        summary_rows.append(metric_row(name, frame))
        distribution_rows.append(describe(frame["pred"].to_numpy(), name, "prediction"))
        for label, low, high in (("10_29", 10, 29), ("30_99", 30, 99), ("100_999", 100, 999), ("1000_plus", 1000, None)):
            mask = frame["Total"] >= low if high is None else frame["Total"].between(low, high)
            group = frame.loc[mask]
            if group.empty:
                continue
            delta = group["ratio"].to_numpy() - group["pred"].to_numpy()
            total_rows.append({
                "run": name, "Total_group": label, "n": len(group),
                "truth_mean": float(group["ratio"].mean()), "prediction_mean": float(group["pred"].mean()),
                "MSE": float(np.mean(delta ** 2)), "MAE": float(np.mean(np.abs(delta))),
                "Total_weighted_MSE": float(np.average(delta ** 2, weights=group["Total"])),
            })

    # Ordinary uncapped NLL on all common valid sites, irrespective of the metric threshold.
    for name, col in (("standard_uncapped", "pred_standard"), ("cap10_sensitivity", "pred_cap10")):
        nll_mean, nll_sum, clipped = ordinary_binomial_nll(
            paired["m6A"].to_numpy(dtype=np.float64),
            paired["Total"].to_numpy(dtype=np.float64),
            paired[col].to_numpy(dtype=np.float64),
        )
        row = next(item for item in summary_rows if item["run"] == name)
        row["ordinary_binomial_nll_mean_all_common_valid"] = nll_mean
        row["ordinary_binomial_nll_sum_all_common_valid"] = nll_sum
        row["n_pred_at_0_or_1_numerically_clipped_for_all_valid_NLL_only"] = clipped

    train_gt = gt.loc[gt["site_idx"].isin(train_ids) & (gt["Total"] > 0)]
    require(not train_gt.empty, "no eligible training labels for constant baselines")
    constants = {
        "training_arithmetic_mean_ratio": float(train_gt["ratio"].mean()),
        "training_pooled_modified_over_total": float(train_gt["m6A"].sum() / train_gt["Total"].sum()),
    }
    audit["train_baseline_training_sites_total_gt0"] = len(train_gt)
    audit["training_only_constant_baselines"] = constants
    for name, value in constants.items():
        frame = eval_base[["site_idx", "ratio", "m6A", "Total", "class"]].copy()
        frame["pred"] = value
        summary_rows.append(metric_row(name, frame))
        distribution_rows.append(describe(frame["pred"].to_numpy(), name, "prediction"))

    pd.DataFrame(summary_rows).to_csv(OUT / "paired_metrics_and_baselines.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUT / "prediction_truth_distributions.csv", index=False)
    pd.DataFrame(total_rows).to_csv(OUT / "metrics_by_total_group.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + chr(10))
    print(f"Diagnostic outputs: {OUT}")
    print(f"Common valid prediction rows: {len(paired)}; Total>=10 rows: {len(eval_base)}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Training-only baselines: {constants}")
    for name, info in audit["runs"].items():
        print(f"{name} coverage: predicted={info['predicted_rows']} expected={info['expected_valid_total_gt0']} missing={info['missing_expected_sites']} unexpected={info['unexpected_sites']} best_epoch={info['config_best_epoch']}")


if __name__ == "__main__":
    main()
