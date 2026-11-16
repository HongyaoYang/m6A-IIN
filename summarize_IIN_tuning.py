#!/usr/bin/env python3
"""Select IIN tuning candidates using validation Binomial NLL only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TECHNOLOGIES = ("MeRIP_seq", "scDART_seq", "m6ACE_seq", "MAZTER_seq")
CONFIGURATIONS = ("adam_3e-4", "adam_1e-4", "rmsprop_8e-5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tuning_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for configuration in CONFIGURATIONS:
        model_dir = args.tuning_dir / configuration
        for technology in TECHNOLOGIES:
            history_path = model_dir / f"{technology}_train_valid_log.csv"
            config_path = model_dir / f"{technology}_run_config.json"
            checkpoint_path = model_dir / f"{technology}_best_model.model"
            prediction_path = model_dir / f"{technology}_best_valid_predictions.csv"
            required = (history_path, config_path, checkpoint_path, prediction_path)
            if not all(path.is_file() and path.stat().st_size > 0 for path in required):
                rows.append(
                    {
                        "technology": technology,
                        "configuration": configuration,
                        "status": "incomplete",
                    }
                )
                continue

            history = pd.read_csv(history_path)
            with config_path.open(encoding="utf-8") as handle:
                run_config = json.load(handle)
            best_index = history["valid_binomial_nll"].idxmin()
            best = history.loc[best_index]
            rows.append(
                {
                    "technology": technology,
                    "configuration": configuration,
                    "status": "complete",
                    "optimizer": run_config["optimizer"],
                    "initial_learning_rate": run_config["learning_rate"],
                    "completed_epochs": run_config["completed_epochs"],
                    "best_epoch": int(best["epoch"]),
                    "best_valid_binomial_nll": float(best["valid_binomial_nll"]),
                    "valid_AUROC_at_best_nll": float(best["AUROC"]),
                    "valid_AUPRC_at_best_nll": float(best["AUPRC"]),
                    "valid_R2_at_best_nll": float(best["R2"]),
                    "valid_rho_at_best_nll": float(best["rho"]),
                    "valid_cor_at_best_nll": float(best["cor"]),
                    "checkpoint": str(checkpoint_path.resolve()),
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = args.tuning_dir / "tuning_validation_summary.csv"
    summary.to_csv(summary_path, index=False)
    complete = summary.loc[summary["status"] == "complete"].copy()
    if complete.empty:
        print(f"No completed tuning runs. Wrote status table to {summary_path}")
        return

    selected = complete.loc[
        complete.groupby("technology")["best_valid_binomial_nll"].idxmin()
    ].sort_values("technology")
    selected_path = args.tuning_dir / "selected_by_validation_nll.csv"
    selected.to_csv(selected_path, index=False)
    print(f"Wrote all tuning results to {summary_path}")
    print(f"Wrote validation-selected candidates to {selected_path}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
