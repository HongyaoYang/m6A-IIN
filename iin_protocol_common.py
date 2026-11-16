"""Shared data and evaluation utilities for the fixed m6A-IIN model.

This module follows the data contract used by the senior student's iM6A
scripts. It intentionally contains no m6A-IIN architecture definitions.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, r2_score, roc_curve
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

# Import only the audited canonical architecture. A silent fallback to an old
# HPC filename could otherwise train the superseded mixed implementation.
from m6a_iin_forward_regression import (
    M6AIINForwardRegressor,
    M6ARegressionDataset,
    RNAfoldPredictor,
    Sample,
    TrainConfig,
    move_batch,
    sequence_features,
    structure_matrices,
)


TECHNOLOGIES = (
    "DART_seq",
    "eTAM_seq",
    "GLORI",
    "Dorado",
    "m6A_REF_seq",
    "m6A_SAC_seq",
    "m6ACE_seq",
    "MAZTER_seq",
    "MeRIP_seq",
    "Oxford_nanopore",
    "scDART_seq",
)

GT_COLUMNS = ("site_idx", "ratio", "m6A", "Total", "class")


def parse_split(path: Path) -> Dict[str, List[int]]:
    """Read the senior script's one-row-per-technology split format."""
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path, dtype={"tech": str, "site_idx": str})
    required = {"tech", "site_idx"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    if frame["tech"].duplicated().any():
        duplicated = frame.loc[frame["tech"].duplicated(), "tech"].tolist()
        raise ValueError(f"Duplicate technologies in {path}: {duplicated}")

    result: Dict[str, List[int]] = {}
    for row in frame.itertuples(index=False):
        tokens = [token.strip() for token in str(row.site_idx).split(",") if token.strip()]
        ids = [int(token) for token in tokens]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate site_idx values for {row.tech} in {path}")
        result[str(row.tech)] = ids
    return result


def validate_split_pair(
    train_split: Mapping[str, Sequence[int]],
    valid_split: Mapping[str, Sequence[int]],
    technologies: Sequence[str] = TECHNOLOGIES,
) -> None:
    for technology in technologies:
        if technology not in train_split or technology not in valid_split:
            raise KeyError(f"Missing train/validation split for technology {technology!r}")
        overlap = set(train_split[technology]) & set(valid_split[technology])
        if overlap:
            raise ValueError(
                f"{technology}: train and validation overlap by {len(overlap)} site_idx values"
            )


def _parse_fasta_id(header: str, path: Path) -> int:
    token = header[1:].strip().split()[0]
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(f"FASTA header must start with integer site_idx in {path}: {header!r}") from error


def read_fasta_ids(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    ids: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                ids.append(_parse_fasta_id(line, path))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate FASTA site_idx headers in {path}")
    return np.asarray(ids, dtype=np.int64)


def read_fasta_by_ids(path: Path, needed_ids: Iterable[int]) -> Dict[int, str]:
    needed = set(map(int, needed_ids))
    sequences: Dict[int, str] = {}
    current_id: Optional[int] = None
    chunks: List[str] = []

    def store_current() -> None:
        if current_id is not None and current_id in needed:
            if current_id in sequences:
                raise ValueError(f"Duplicate FASTA site_idx {current_id} in {path}")
            sequences[current_id] = "".join(chunks).upper()

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                store_current()
                current_id = _parse_fasta_id(line, path)
                chunks = []
            else:
                if current_id is None:
                    raise ValueError(f"Sequence found before first FASTA header in {path}")
                chunks.append(line)
        store_current()
    return sequences


def read_center_windows_by_ids(
    path: Path,
    needed_ids: Iterable[int],
    sequence_length: int = 41,
) -> Dict[int, str]:
    """Read a FASTA once and retain only the fixed IIN window in memory."""
    needed = set(map(int, needed_ids))
    windows: Dict[int, str] = {}
    current_id: Optional[int] = None
    chunks: List[str] = []

    def store_current() -> None:
        if current_id is not None and current_id in needed:
            if current_id in windows:
                raise ValueError(f"Duplicate FASTA site_idx {current_id} in {path}")
            windows[current_id] = center_window("".join(chunks), sequence_length)

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                store_current()
                current_id = _parse_fasta_id(line, path)
                chunks = []
            elif current_id is None:
                raise ValueError(f"Sequence found before first FASTA header in {path}")
            elif current_id in needed:
                chunks.append(line)
        store_current()
    return windows


def center_window(sequence: str, length: int = 41) -> str:
    normalized = sequence.upper().replace("T", "U")
    if len(normalized) < length:
        raise ValueError(f"Sequence length {len(normalized)} is shorter than required window {length}")
    center = len(normalized) // 2
    left = center - length // 2
    window = normalized[left : left + length]
    invalid = sorted(set(window) - set("ACGU"))
    if invalid:
        raise ValueError(f"IIN 15-D input representation does not support bases: {invalid}")
    return window


def read_ground_truth(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = set(GT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[:, list(GT_COLUMNS)].copy()
    for column in GT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("site_idx", "m6A", "Total"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain finite integer values in {path}")
    frame["site_idx"] = frame["site_idx"].astype(np.int64)
    frame["m6A"] = frame["m6A"].astype(np.int64)
    frame["Total"] = frame["Total"].astype(np.int64)
    # The supplied data uses class=NA for some low-coverage sites. Class is not
    # needed by Binomial NLL, so retain it as a nullable integer and validate it
    # only where present. Metric code performs the stricter thresholded check.
    class_present = frame["class"].notna()
    class_values = frame.loc[class_present, "class"].to_numpy(dtype=float)
    if not np.equal(class_values, np.floor(class_values)).all():
        raise ValueError(f"Non-missing class values must be integers in {path}")
    frame["class"] = frame["class"].astype("Int64")
    if frame["site_idx"].duplicated().any():
        raise ValueError(f"Duplicate site_idx values in {path}")
    invalid = (
        (frame["m6A"] < 0)
        | (frame["Total"] < 0)
        | (frame["m6A"] > frame["Total"])
        | (class_present & ~frame["class"].isin([0, 1]))
    )
    if invalid.any():
        raise ValueError(f"Invalid counts or non-binary class values in {path}")
    positive_total = frame["Total"] > 0
    expected_ratio = np.divide(
        frame.loc[positive_total, "m6A"], frame.loc[positive_total, "Total"]
    )
    supplied_ratio = frame.loc[positive_total, "ratio"].to_numpy(dtype=float)
    if not np.allclose(expected_ratio, supplied_ratio, rtol=1e-5, atol=1e-7):
        raise ValueError(f"ratio is inconsistent with m6A/Total in {path}")
    return frame


def build_labelled_frame(
    sequences: Mapping[int, str],
    ground_truth: pd.DataFrame,
    site_ids: Sequence[int],
    sequence_length: int = 41,
) -> pd.DataFrame:
    requested = pd.DataFrame({"site_idx": list(map(int, site_ids))})
    requested["sequence"] = requested["site_idx"].map(sequences)
    missing_sequence = requested["sequence"].isna()
    if missing_sequence.any():
        examples = requested.loc[missing_sequence, "site_idx"].head(10).tolist()
        raise ValueError(f"Missing FASTA sequences for requested site_idx values: {examples}")
    frame = requested.merge(ground_truth, on="site_idx", how="left", validate="one_to_one")
    missing_gt = frame["Total"].isna()
    if missing_gt.any():
        examples = frame.loc[missing_gt, "site_idx"].head(10).tolist()
        raise ValueError(f"Missing ground truth for requested site_idx values: {examples}")
    frame["sequence"] = frame["sequence"].map(lambda value: center_window(value, sequence_length))
    before = len(frame)
    frame = frame.loc[frame["Total"] > 0].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No training/validation sites have Total > 0")
    frame.attrs["n_total_zero_removed"] = before - len(frame)
    return frame


def build_unlabelled_frame(
    sequences: Mapping[int, str],
    site_ids: Sequence[int],
    sequence_length: int = 41,
) -> pd.DataFrame:
    frame = pd.DataFrame({"site_idx": list(map(int, site_ids))})
    frame["sequence"] = frame["site_idx"].map(sequences)
    missing = frame["sequence"].isna()
    if missing.any():
        examples = frame.loc[missing, "site_idx"].head(10).tolist()
        raise ValueError(f"Missing FASTA sequences for OOD site_idx values: {examples}")
    frame["sequence"] = frame["sequence"].map(lambda value: center_window(value, sequence_length))
    return frame


class PrecomputedStructureStore:
    """Read site-indexed dot-bracket encodings from a memory-mapped array."""

    _symbols = np.asarray([ord("."), ord("("), ord(")")], dtype=np.uint8)

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.present_path = self.path.with_name(f"{self.path.stem}.present.npy")
        if not self.path.is_file():
            raise FileNotFoundError(f"Precomputed RNAfold array not found: {self.path}")
        self._array = None

    @property
    def array(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
            if self._array.ndim != 2 or self._array.shape[1] != 41:
                raise ValueError(f"Expected RNAfold array shape (N, 41), got {self._array.shape}")
        return self._array

    def structure(self, site_idx: int) -> str:
        if site_idx < 0 or site_idx >= self.array.shape[0]:
            raise KeyError(f"site_idx {site_idx} is outside the RNAfold array")
        codes = np.asarray(self.array[site_idx])
        if np.any(codes > 2):
            raise KeyError(f"No precomputed RNAfold structure for site_idx {site_idx}")
        return self._symbols[codes].tobytes().decode("ascii")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state


class PrecomputedELMoStore:
    """Read author-style 41x256 Flair ELMo features by site_idx."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.present_path = self.path.with_name(f"{self.path.stem}.present.npy")
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Precomputed author ELMo array not found: {self.path}. "
                "Run precompute_author_elmo_direct.py before training."
            )
        if not self.present_path.is_file():
            raise FileNotFoundError(f"ELMo presence mask not found: {self.present_path}")
        self._array = None
        self._present = None

    @property
    def array(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
            if self._array.ndim != 3 or self._array.shape[1:] != (41, 256):
                raise ValueError(
                    f"Expected ELMo array shape (N, 41, 256), got {self._array.shape}"
                )
        return self._array

    @property
    def present(self):
        if self._present is None:
            self._present = np.load(self.present_path, mmap_mode="r")
            if self._present.ndim != 1 or len(self._present) != len(self.array):
                raise ValueError(
                    f"ELMo presence mask shape {self._present.shape} does not match "
                    f"feature rows {len(self.array)}"
                )
        return self._present

    def features(self, site_idx: int) -> np.ndarray:
        if site_idx < 0 or site_idx >= self.array.shape[0]:
            raise KeyError(f"site_idx {site_idx} is outside the ELMo array")
        if not bool(self.present[site_idx]):
            raise KeyError(f"No precomputed ELMo features for site_idx {site_idx}")
        # The feature cache is opened read-only with mmap_mode="r".  PyTorch
        # warns when torch.from_numpy receives that non-writable view because
        # an accidental in-place tensor operation would have undefined
        # behavior.  Return an owned writable array for each dataset sample.
        values = np.array(self.array[site_idx], dtype=np.float32, copy=True)
        if values.shape != (41, 256) or not np.isfinite(values).all():
            raise KeyError(f"Invalid precomputed ELMo features for site_idx {site_idx}")
        return values

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_present"] = None
        return state


class LazyM6ARegressionDataset(Dataset):
    """Construct IIN input matrices per sample without retaining all matrices."""

    def __init__(
        self,
        samples: Sequence[Sample],
        structures: PrecomputedStructureStore,
        elmo_features: PrecomputedELMoStore,
        add_self_loops: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.structures = structures
        self.elmo_features = elmo_features
        self.add_self_loops = add_self_loops

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        site_idx = int(sample.sample_id)
        structure = self.structures.structure(site_idx)
        adjacency, loop_adjacency = structure_matrices(structure, self.add_self_loops)
        return (
            torch.from_numpy(sequence_features(sample.sequence)),
            torch.from_numpy(adjacency),
            torch.from_numpy(loop_adjacency),
            torch.from_numpy(self.elmo_features.features(site_idx)),
            torch.tensor(sample.modified_count, dtype=torch.float32),
            torch.tensor(sample.total_count, dtype=torch.float32),
            sample.sample_id,
        )


def frame_to_dataset(
    frame: pd.DataFrame,
    structures: PrecomputedStructureStore,
    elmo_features: PrecomputedELMoStore,
    add_self_loops: bool = True,
) -> LazyM6ARegressionDataset:
    labelled = {"m6A", "Total"}.issubset(frame.columns)
    samples = [
        Sample(
            sample_id=str(int(row.site_idx)),
            sequence=str(row.sequence),
            modified_count=int(row.m6A) if labelled else 0,
            total_count=int(row.Total) if labelled else 0,
        )
        for row in frame.itertuples(index=False)
    ]
    return LazyM6ARegressionDataset(
        samples,
        structures,
        elmo_features,
        add_self_loops=add_self_loops,
    )


class MergeSingletonBatchSampler(Sampler[List[int]]):
    """Create full batches without ever emitting a singleton training batch.

    When ``len(dataset) % batch_size == 1``, the last full batch and the final
    sample are merged into one batch of ``batch_size + 1``.  No training site
    is discarded, and BatchNorm always receives at least two samples.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        pending: List[int] = []
        current: List[int] = []
        for index in self.sampler:
            current.append(index)
            if len(current) == self.batch_size:
                if pending:
                    yield pending
                pending = current
                current = []

        if len(current) == 1:
            if not pending:
                raise ValueError("Training dataset must contain at least two sites")
            yield pending + current
        else:
            if pending:
                yield pending
            if current:
                yield current

    def __len__(self) -> int:
        n_samples = len(self.sampler)
        quotient, remainder = divmod(n_samples, self.batch_size)
        if remainder == 0:
            return quotient
        if remainder == 1 and quotient > 0:
            return quotient
        return quotient + 1


def make_loader(
    dataset: M6ARegressionDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    if shuffle:
        # The released model contains BatchNorm1d after site-level pooling and
        # therefore cannot train on a final batch of size one.  Merge that
        # singleton with the preceding batch instead of dropping a site.
        return DataLoader(
            dataset,
            batch_sampler=MergeSingletonBatchSampler(
                RandomSampler(dataset), batch_size
            ),
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def binomial_nll(
    logits: torch.Tensor,
    modified: torch.Tensor,
    total: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable Binomial NLL without the parameter-independent choose term."""
    logits = logits.reshape(-1)
    modified = modified.reshape(-1)
    total = total.reshape(-1)
    if torch.any(total <= 0) or torch.any(modified < 0) or torch.any(modified > total):
        raise ValueError("Binomial NLL requires 0 <= m6A <= Total and Total > 0")
    per_site = total * F.softplus(logits) - modified * logits
    if reduction == "none":
        return per_site
    if reduction == "sum":
        return per_site.sum()
    if reduction == "mean":
        return per_site.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def model_logits(model: M6AIINForwardRegressor, batch, device: torch.device):
    primary, adjacency, loop_adjacency, elmo, modified, total, sample_ids = move_batch(
        batch, device
    )
    logits = model(adjacency, loop_adjacency, primary, elmo)[0]
    return logits, modified, total, sample_ids


@torch.no_grad()
def predict_loader(
    model: M6AIINForwardRegressor,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    ids: List[int] = []
    predictions: List[float] = []
    for batch in loader:
        logits, _, _, sample_ids = model_logits(model, batch, device)
        ids.extend(int(value) for value in sample_ids)
        predictions.extend(torch.sigmoid(logits).cpu().numpy().astype(float).tolist())
    result = pd.DataFrame({"site_idx": ids, "pred": predictions})
    if result["site_idx"].duplicated().any():
        raise RuntimeError("Duplicate site_idx values produced during prediction")
    return result


@torch.no_grad()
def validation_loss(
    model: M6AIINForwardRegressor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_sites = 0
    for batch in loader:
        logits, modified, total, _ = model_logits(model, batch, device)
        total_loss += float(binomial_nll(logits, modified, total, reduction="sum").item())
        n_sites += int(modified.numel())
    if n_sites == 0:
        raise RuntimeError("Validation loader produced no sites")
    return total_loss / n_sites


def _safe_correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    return float(function(observed, predicted).statistic)


def calculate_metrics(joined: pd.DataFrame, min_total: int = 10) -> Dict[str, float]:
    """Calculate only the five agreed metrics after the coverage filter."""
    eligible = joined.loc[joined["Total"] >= min_total].copy()
    result: Dict[str, float] = {
        "n_all": int(len(joined)),
        "n_total_ge_threshold": int(len(eligible)),
        "total_threshold": int(min_total),
        "AUROC": float("nan"),
        "AUPRC": float("nan"),
        "R2": float("nan"),
        "rho": float("nan"),
        "cor": float("nan"),
    }
    if eligible.empty:
        return result
    if eligible["class"].isna().any():
        missing_count = int(eligible["class"].isna().sum())
        raise ValueError(
            f"{missing_count} eligible rows have missing class labels at Total >= {min_total}"
        )
    if not eligible["class"].isin([0, 1]).all():
        raise ValueError("Eligible class labels must be binary (0/1)")
    labels = eligible["class"].to_numpy(dtype=int)
    predictions = eligible["pred"].to_numpy(dtype=float)
    observed = eligible["ratio"].to_numpy(dtype=float)
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if not np.isfinite(observed).all():
        raise ValueError("Observed ratios must be finite when Total passes the threshold")
    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, predictions)
        result["AUROC"] = float(auc(fpr, tpr))
        precision, recall, _ = precision_recall_curve(labels, predictions)
        result["AUPRC"] = float(auc(recall, precision))
    if len(observed) >= 2 and np.unique(observed).size >= 2:
        result["R2"] = float(r2_score(observed, predictions))
    result["rho"] = _safe_correlation(spearmanr, observed, predictions)
    result["cor"] = _safe_correlation(pearsonr, observed, predictions)
    return result


def join_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    return predictions.merge(ground_truth, on="site_idx", how="inner", validate="one_to_one")


def save_checkpoint(
    path: Path,
    model: M6AIINForwardRegressor,
    config: TrainConfig,
    technology: str,
    epoch: int,
    valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": asdict(config),
            "technology": technology,
            "best_epoch": epoch,
            "best_valid_binomial_nll": valid_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> Tuple[M6AIINForwardRegressor, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Checkpoints written by save_checkpoint contain tensors plus primitive
    # metadata only, so the restricted loader is sufficient and avoids unsafe
    # arbitrary pickle execution as well as PyTorch's FutureWarning.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = TrainConfig(**checkpoint["train_config"])
    model = M6AIINForwardRegressor(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
