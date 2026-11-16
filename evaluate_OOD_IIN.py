"""Calculate cross-technology IIN OOD metrics from saved all-site predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from iin_protocol_common import (
    TECHNOLOGIES,
    calculate_metrics,
    join_predictions,
    read_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/gpfs/work/bio/hongyaoyang23/m6A_IIN/model_IIN"),
    )
    parser.add_argument("--min-total", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_total < 1:
        raise ValueError("min total must be positive")
    rows: List[Dict[str, object]] = []
    for source in TECHNOLOGIES:
        prediction_path = args.model_dir / f"{source}_allSite_preds.csv"
        predictions = pd.read_csv(prediction_path)
        required = {"site_idx", "pred"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"{prediction_path} must contain {sorted(required)}")
        predictions = predictions.loc[:, ["site_idx", "pred"]].copy()
        predictions["site_idx"] = pd.to_numeric(
            predictions["site_idx"], errors="raise"
        ).astype("int64")
        predictions["pred"] = pd.to_numeric(predictions["pred"], errors="raise")
        if predictions["site_idx"].duplicated().any():
            raise ValueError(f"Duplicate site_idx values in {prediction_path}")

        for target in TECHNOLOGIES:
            ground_truth = read_ground_truth(args.base_dir / f"{target}.csv")
            joined = join_predictions(predictions, ground_truth)
            metrics = calculate_metrics(joined, args.min_total)
            rows.append(
                {
                    "model_technology": source,
                    "target_technology": target,
                    "evaluation_type": (
                        "same_technology_holdout" if source == target else "cross_technology_OOD"
                    ),
                    **metrics,
                }
            )

    metrics_frame = pd.DataFrame(rows)
    output_path = args.model_dir / "IIN_OOD_metrics.csv"
    metrics_frame.to_csv(output_path, index=False)
    cross_ood = metrics_frame.loc[
        metrics_frame["evaluation_type"] == "cross_technology_OOD"
    ]
    cross_ood[["AUROC", "AUPRC", "R2", "rho", "cor"]].agg(
        ["mean", "std"]
    ).to_csv(args.model_dir / "IIN_OOD_metrics_summary.csv")
    print(f"OOD metrics saved to {output_path}")


if __name__ == "__main__":
    main()
