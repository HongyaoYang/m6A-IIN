"""Predict all FASTA sites with each IIN model, excluding its own train/valid sites."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iin_protocol_common import (
    TECHNOLOGIES,
    PrecomputedELMoStore,
    PrecomputedStructureStore,
    build_unlabelled_frame,
    frame_to_dataset,
    load_checkpoint,
    make_loader,
    parse_split,
    predict_loader,
    read_center_windows_by_ids,
    read_fasta_ids,
    validate_split_pair,
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
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--structures", type=Path, default=None)
    parser.add_argument("--elmo-features", type=Path, default=None)
    parser.add_argument("--train-split", type=Path, default=None)
    parser.add_argument("--valid-split", type=Path, default=None)
    parser.add_argument("--tech", choices=("all", *TECHNOLOGIES), default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Maximum number of IIN graph samples materialized in memory at once",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-self-loops", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing completed or partial output instead of resuming/skipping",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_fasta = args.base_dir / "sequences_41nt.fa"
    args.fasta = args.fasta or (
        compact_fasta if compact_fasta.is_file() else args.base_dir / "sequences.fa"
    )
    args.train_split = args.train_split or args.base_dir / "train_site.csv"
    args.valid_split = args.valid_split or args.base_dir / "valid_site.csv"
    args.structures = args.structures or args.base_dir / "rnafold_structures.npy"
    args.elmo_features = args.elmo_features or args.base_dir / "author_elmo_features.npy"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("batch size and chunk size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    technologies = TECHNOLOGIES if args.tech == "all" else (args.tech,)
    train_split = parse_split(args.train_split)
    valid_split = parse_split(args.valid_split)
    validate_split_pair(train_split, valid_split, technologies)
    all_site_ids = read_fasta_ids(args.fasta)
    structures = PrecomputedStructureStore(args.structures)
    elmo_features = PrecomputedELMoStore(args.elmo_features)
    print(f"Total FASTA sites: {len(all_site_ids)}")

    for source_technology in technologies:
        print(f"\nLoading IIN model: {source_technology}")
        checkpoint_path = args.model_dir / f"{source_technology}_best_model.model"
        model, _ = load_checkpoint(checkpoint_path, device)
        excluded = set(train_split[source_technology]) | set(valid_split[source_technology])
        infer_ids = all_site_ids[~np.isin(all_site_ids, list(excluded))]
        output_path = args.model_dir / f"{source_technology}_allSite_preds.csv"
        partial_path = output_path.with_suffix(output_path.suffix + ".partial")

        if args.overwrite:
            output_path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)
        elif output_path.exists():
            completed = pd.read_csv(output_path, usecols=["model_technology", "site_idx"])
            if len(completed) != len(infer_ids):
                raise RuntimeError(
                    f"{output_path} exists but has {len(completed)} rows; "
                    f"expected {len(infer_ids)}. Use --overwrite to regenerate it."
                )
            if completed["site_idx"].duplicated().any():
                raise RuntimeError(f"Duplicate site_idx values in {output_path}")
            if not completed["model_technology"].eq(source_technology).all():
                raise RuntimeError(f"Model technology mismatch in {output_path}")
            if set(completed["site_idx"].astype("int64")) != set(map(int, infer_ids)):
                raise RuntimeError(f"site_idx coverage mismatch in {output_path}")
            print(f"Validated existing complete predictions: {output_path}; skipping")
            continue

        completed_ids = set()
        written = 0
        if partial_path.exists():
            completed = pd.read_csv(
                partial_path, usecols=["model_technology", "site_idx"]
            )
            if completed["site_idx"].duplicated().any():
                raise RuntimeError(f"Duplicate site_idx values in {partial_path}")
            if not completed["model_technology"].eq(source_technology).all():
                raise RuntimeError(f"Model technology mismatch in {partial_path}")
            completed_ids = set(completed["site_idx"].astype("int64"))
            unexpected = completed_ids - set(map(int, infer_ids))
            if unexpected:
                raise RuntimeError(
                    f"{partial_path} contains {len(unexpected)} unexpected site_idx values"
                )
            written = len(completed_ids)
            print(f"Resuming {source_technology} from {written}/{len(infer_ids)} sites")

        remaining_ids = infer_ids[
            ~np.isin(infer_ids, np.fromiter(completed_ids, dtype=np.int64))
        ]
        sequences = read_center_windows_by_ids(
            args.fasta, remaining_ids, sequence_length=41
        )
        if len(infer_ids) == 0:
            pd.DataFrame(columns=["model_technology", "site_idx", "pred"]).to_csv(
                partial_path, index=False
            )
        for start in range(0, len(remaining_ids), args.chunk_size):
            chunk_ids = remaining_ids[start : start + args.chunk_size]
            infer_frame = build_unlabelled_frame(sequences, chunk_ids, sequence_length=41)
            dataset = frame_to_dataset(
                infer_frame,
                structures,
                elmo_features,
                not args.no_self_loops,
            )
            loader = make_loader(dataset, args.batch_size, False, device, args.num_workers)
            predictions = predict_loader(model, loader, device)
            predictions.insert(0, "model_technology", source_technology)
            predictions.to_csv(
                partial_path,
                mode="a",
                header=written == 0,
                index=False,
            )
            written += len(predictions)
            print(f"  predicted {written}/{len(infer_ids)} sites")
            del infer_frame, dataset, loader, predictions
        if written != len(infer_ids):
            raise RuntimeError(
                f"{source_technology}: expected {len(infer_ids)} predictions, got {written}"
            )
        partial_path.replace(output_path)
        print(
            f"Saved {written} predictions to {output_path}; "
            f"excluded own train/valid sites={len(excluded)}"
        )


if __name__ == "__main__":
    main()
