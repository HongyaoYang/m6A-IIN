"""Train one fixed m6A-IIN model with the senior student's data protocol."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from iin_protocol_common import (
    M6AIINForwardRegressor,
    PrecomputedELMoStore,
    PrecomputedStructureStore,
    TrainConfig,
    binomial_nll,
    build_labelled_frame,
    calculate_metrics,
    frame_to_dataset,
    join_predictions,
    load_checkpoint,
    make_loader,
    model_logits,
    parse_split,
    predict_loader,
    read_center_windows_by_ids,
    read_ground_truth,
    save_checkpoint,
    validate_split_pair,
    validation_loss,
)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Keep all 11 technology runs reproducible under the shared seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech", required=True)
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
    parser.add_argument("--excluded-sites", type=Path, default=None)
    parser.add_argument("--structures", type=Path, default=None)
    parser.add_argument("--elmo-features", type=Path, default=None)
    parser.add_argument("--train-split", type=Path, default=None)
    parser.add_argument("--valid-split", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "rmsprop"),
        default="adam",
        help="Optimizer only; the architecture and Binomial NLL stay unchanged",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Optional global gradient-norm cap; disabled by default",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="none",
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; 0 disables",
    )
    parser.add_argument("--metric-min-total", type=int, default=10)
    parser.add_argument(
        "--max-train-sites",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke tests only; omit for formal training",
    )
    parser.add_argument(
        "--max-valid-sites",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke tests only; omit for formal training",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--no-self-loops", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    compact_fasta = args.base_dir / "sequences_41nt.fa"
    args.fasta = args.fasta or (
        compact_fasta if compact_fasta.is_file() else args.base_dir / "sequences.fa"
    )
    if args.excluded_sites is None and args.fasta.name == "sequences_41nt.fa":
        args.excluded_sites = args.base_dir / "sequences_41nt.excluded.tsv"
    args.train_split = args.train_split or args.base_dir / "train_site.csv"
    args.valid_split = args.valid_split or args.base_dir / "valid_site.csv"
    args.ground_truth = args.base_dir / f"{args.tech}.csv"
    args.structures = args.structures or args.base_dir / "rnafold_structures.npy"
    args.elmo_features = args.elmo_features or args.base_dir / "author_elmo_features.npy"
    args.model_dir.mkdir(parents=True, exist_ok=True)


def train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    n_sites = 0
    start = time.time()
    for batch_index, batch in enumerate(loader):
        logits, modified, total, _ = model_logits(model, batch, device)
        loss = binomial_nll(logits, modified, total)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}"
                )
        optimizer.step()
        batch_size = int(modified.numel())
        total_loss += float(loss.item()) * batch_size
        n_sites += batch_size
    average = total_loss / n_sites
    print(
        f"Epoch {epoch:02d}: train Binomial NLL={average:.6f}, "
        f"sites={n_sites}, time={time.time() - start:.1f}s"
    )
    return average


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive when supplied")
    if not 0 < args.scheduler_factor < 1:
        raise ValueError("--scheduler-factor must be between 0 and 1")
    if args.scheduler_patience < 1:
        raise ValueError("--scheduler-patience must be positive")
    if args.min_learning_rate <= 0 or args.min_learning_rate > args.learning_rate:
        raise ValueError(
            "--min-learning-rate must be positive and no larger than --learning-rate"
        )
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_split = parse_split(args.train_split)
    valid_split = parse_split(args.valid_split)
    validate_split_pair(train_split, valid_split, (args.tech,))
    excluded_ids = set()
    if args.excluded_sites is not None:
        excluded_frame = pd.read_csv(args.excluded_sites, sep="\t")
        if "site_idx" not in excluded_frame.columns:
            raise ValueError(f"{args.excluded_sites} must contain site_idx")
        excluded_ids = set(
            pd.to_numeric(excluded_frame["site_idx"], errors="raise").astype("int64")
        )
    train_ids = [value for value in train_split[args.tech] if value not in excluded_ids]
    valid_ids = [value for value in valid_split[args.tech] if value not in excluded_ids]
    print(
        f"Excluded unencodable sites: train={len(train_split[args.tech]) - len(train_ids)}, "
        f"valid={len(valid_split[args.tech]) - len(valid_ids)}"
    )
    if args.max_train_sites is not None:
        if args.max_train_sites < 1:
            raise ValueError("--max-train-sites must be positive")
        train_ids = train_ids[: args.max_train_sites]
    if args.max_valid_sites is not None:
        if args.max_valid_sites < 1:
            raise ValueError("--max-valid-sites must be positive")
        valid_ids = valid_ids[: args.max_valid_sites]
    if args.max_train_sites is not None or args.max_valid_sites is not None:
        print(f"SMOKE-TEST CAPS ACTIVE: train={len(train_ids)}, valid={len(valid_ids)}")
    needed_ids = set(train_ids) | set(valid_ids)
    # The shared FASTA is about 24 GB and contains long sequences. Retain only
    # the 41-nt center windows required by IIN to avoid holding many gigabytes
    # of unused sequence text in memory.
    sequences = read_center_windows_by_ids(args.fasta, needed_ids, sequence_length=41)
    ground_truth = read_ground_truth(args.ground_truth)
    train_frame = build_labelled_frame(sequences, ground_truth, train_ids)
    valid_frame = build_labelled_frame(sequences, ground_truth, valid_ids)
    print(
        f"{args.tech}: train={len(train_frame)}, valid={len(valid_frame)}, "
        "training/validation filter=Total>0"
    )

    structures = PrecomputedStructureStore(args.structures)
    elmo_features = PrecomputedELMoStore(args.elmo_features)
    train_dataset = frame_to_dataset(
        train_frame, structures, elmo_features, not args.no_self_loops
    )
    valid_dataset = frame_to_dataset(
        valid_frame, structures, elmo_features, not args.no_self_loops
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, device, args.num_workers
    )
    valid_loader = make_loader(
        valid_dataset, args.batch_size, False, device, args.num_workers
    )

    setup_seed(args.seed)
    config = TrainConfig(
        sequence_length=41,
        folds=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        attention_loss_weight=0.0,
        inverse_loss_weight=0.0,
        seed=args.seed,
    )
    model = M6AIINForwardRegressor(config).to(device)
    optimizer_class = {
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
    }[args.optimizer]
    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_learning_rate,
        )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = args.model_dir / f"{args.tech}_best_model.model"
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.grad_clip,
        )
        valid_loss = validation_loss(model, valid_loader, device)
        valid_predictions = predict_loader(model, valid_loader, device)
        valid_joined = join_predictions(valid_predictions, valid_frame.drop(columns="sequence"))
        metrics = calculate_metrics(valid_joined, args.metric_min_total)
        history.append(
            {
                "epoch": epoch,
                "train_binomial_nll": train_loss,
                "valid_binomial_nll": valid_loss,
                "learning_rate": learning_rate,
                **metrics,
            }
        )
        print(
            f"Epoch {epoch:02d}: valid Binomial NLL={valid_loss:.6f}, "
            f"AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, "
            f"R2={metrics['R2']:.4f}, rho={metrics['rho']:.4f}, cor={metrics['cor']:.4f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, config, args.tech, epoch, valid_loss)
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(valid_loss)
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: no validation Binomial NLL "
                f"improvement for {epochs_without_improvement} epochs"
            )
            break

    pd.DataFrame(history).to_csv(
        args.model_dir / f"{args.tech}_train_valid_log.csv", index=False
    )
    with (args.model_dir / f"{args.tech}_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "technology": args.tech,
                "architecture": (
                    "Author public m6A-IIN architecture at commit "
                    "16a02193855b325538872b52df1331f88e32cea1; scalar-logit adaptation"
                ),
                "input_representation": (
                    "41-nt, author 15-D primary features, RNAfold secondary graph, "
                    "precomputed Flair ELMo-small 256-D features"
                ),
                "output": "scalar logit -> sigmoid(mu)",
                "loss": "Binomial NLL only, Total > 0",
                "best_epoch": "minimum validation Binomial NLL",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "gradient_clip_max_norm": args.grad_clip,
                "scheduler": args.lr_scheduler,
                "scheduler_factor": (
                    args.scheduler_factor if scheduler is not None else None
                ),
                "scheduler_patience": (
                    args.scheduler_patience if scheduler is not None else None
                ),
                "minimum_learning_rate": (
                    args.min_learning_rate if scheduler is not None else None
                ),
                "early_stopping_patience": args.early_stopping_patience,
                "completed_epochs": len(history),
                "cudnn_deterministic": bool(
                    torch.cuda.is_available() and torch.backends.cudnn.deterministic
                ),
                "cudnn_benchmark": bool(
                    torch.cuda.is_available() and torch.backends.cudnn.benchmark
                ),
                "best_epoch_value": best_epoch,
                "best_valid_binomial_nll": best_loss,
            },
            handle,
            indent=2,
        )
    print(f"Best epoch={best_epoch}, validation Binomial NLL={best_loss:.6f}")
    print(f"Best model saved to {best_path}")

    # Save every validation prediction from the single checkpoint selected by
    # validation Binomial NLL. No Total threshold is applied here so metrics
    # can be recomputed later under a different evaluation protocol.
    best_model, _ = load_checkpoint(best_path, device)
    best_valid_predictions = predict_loader(best_model, valid_loader, device)
    best_valid_output = join_predictions(
        best_valid_predictions,
        valid_frame.drop(columns="sequence"),
    )
    best_valid_output.insert(0, "split", "valid")
    best_valid_output.insert(0, "technology", args.tech)
    validation_prediction_path = (
        args.model_dir / f"{args.tech}_best_valid_predictions.csv"
    )
    best_valid_output.to_csv(validation_prediction_path, index=False)
    print(
        f"Saved {len(best_valid_output)} unfiltered validation predictions to "
        f"{validation_prediction_path}"
    )


if __name__ == "__main__":
    main()
