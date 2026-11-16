#!/usr/bin/env python3
"""Precompute m6A-IIN ELMo-small features without importing Flair.

This follows AllenNLP 0.9.0's ElmoEmbedder computation directly: obtain the
three internal ELMo layers, remove BOS/EOS, and sum the three 256-dimensional
small-model layers exactly as the m6A-IIN preprocessing code does after Flair
concatenates them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch


def fasta_records(path: Path) -> Iterator[Tuple[int, str]]:
    current_id = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(chunks).upper().replace("T", "U")
                current_id = int(line[1:].strip().split()[0])
                chunks = []
            else:
                if current_id is None:
                    raise ValueError("Sequence before first FASTA header")
                chunks.append(line)
        if current_id is not None:
            yield current_id, "".join(chunks).upper().replace("T", "U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--options-file", type=Path, required=True)
    parser.add_argument("--weight-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--flush-every", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.flush_every < 1:
        raise ValueError("batch size and flush interval must be positive")
    for input_path in (args.fasta, args.options_file, args.weight_file):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {args.output}")

    # Import the ELMo implementation only. In particular, do not import
    # allennlp.commands.elmo or Flair; both pull in unrelated legacy packages.
    from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
    from allennlp.nn.util import remove_sentence_boundaries

    records = []
    for site_idx, sequence in fasta_records(args.fasta):
        if site_idx < 0:
            raise ValueError(f"Negative site_idx: {site_idx}")
        if len(sequence) != 41 or set(sequence) - set("ACGU"):
            raise ValueError(f"Invalid 41-nt sequence for site_idx {site_idx}")
        records.append((site_idx, sequence))
    if not records:
        raise ValueError("No FASTA records")

    ids = [site_idx for site_idx, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate FASTA site_idx values")
    rows = max(ids) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.stem}.partial.npy")
    present_output = args.output.with_name(f"{args.output.stem}.present.npy")
    present_partial = args.output.with_name(
        f"{args.output.stem}.present.partial.npy"
    )

    if partial.exists() != present_partial.exists():
        raise RuntimeError("Only one ELMo partial file exists; remove or restore the pair")
    if partial.exists():
        features = np.lib.format.open_memmap(partial, mode="r+")
        present = np.lib.format.open_memmap(present_partial, mode="r+")
        if features.shape != (rows, 41, 256) or present.shape != (rows,):
            raise ValueError("Existing partial ELMo arrays have incompatible shapes")
    else:
        features = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(rows, 41, 256)
        )
        present = np.lib.format.open_memmap(
            present_partial, mode="w+", dtype=np.bool_, shape=(rows,)
        )
        present[:] = False
        present.flush()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # PyTorch 1.8 enables TensorFloat-32 on Ampere GPUs such as the A800.
    # Recurrent ELMo computations can amplify those lower-precision matrix
    # multiplication differences over 41 positions.  Disable TF32 and select
    # deterministic cuDNN behavior so that the cached features remain as close
    # as possible to the author's float32 computation and are reproducible.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(
            "CUDA deterministic mode: TF32 disabled, cuDNN deterministic=True",
            flush=True,
        )

    bilm = _ElmoBiLm(str(args.options_file), str(args.weight_file))
    bilm = bilm.to(device)
    bilm.eval()

    completed = int(np.count_nonzero(present[ids]))
    batches = 0

    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = [item for item in records[start : start + args.batch_size]
                     if not bool(present[item[0]])]
            if not batch:
                continue

            batch_ids = [site_idx for site_idx, _ in batch]
            tokenized = [list(sequence) for _, sequence in batch]
            character_ids = batch_to_ids(tokenized).to(device)
            bilm_output = bilm(character_ids)
            mask_with_bos_eos = bilm_output["mask"]
            without_bos_eos = [
                remove_sentence_boundaries(layer, mask_with_bos_eos)
                for layer in bilm_output["activations"]
            ]
            layers = torch.stack([item[0] for item in without_bos_eos], dim=1)
            mask = without_bos_eos[0][1]

            expected_shape = (len(batch), 3, 41, 256)
            if tuple(layers.shape) != expected_shape:
                raise RuntimeError(
                    f"Expected ELMo-small layers {expected_shape}, got {tuple(layers.shape)}"
                )
            lengths = mask.long().sum(dim=1)
            if not bool(torch.all(lengths == 41)):
                raise RuntimeError(f"Unexpected ELMo token lengths: {lengths.tolist()}")

            # Flair embedding_mode='all' concatenates these layers. The author
            # then adds its three 256-D slices, which is exactly sum(dim=1).
            reduced = layers.sum(dim=1).cpu().numpy().astype(np.float32)
            for row, site_idx in enumerate(batch_ids):
                features[site_idx] = reduced[row]
                present[site_idx] = True
            completed += len(batch)
            batches += 1

            if batches % args.flush_every == 0:
                features.flush()
                present.flush()
                print(f"Encoded {completed}/{len(records)} sites", flush=True)

    features.flush()
    present.flush()
    if int(np.count_nonzero(present[ids])) != len(ids):
        raise RuntimeError("ELMo feature generation ended with missing FASTA records")

    os.replace(partial, args.output)
    os.replace(present_partial, present_output)
    metadata = {
        "source": "ZLSRW/m6A-IIN Pre-Encoding/Preprocess/1.feature_embedding.py",
        "author_commit": "16a02193855b325538872b52df1331f88e32cea1",
        "backend": "AllenNLP 0.9.0 _ElmoBiLm (Flair-free)",
        "equivalence": "three internal 256-D ELMo-small layers summed",
        "shape": [rows, 41, 256],
        "valid_sites": len(ids),
        "dtype": "float32",
        "options_file": str(args.options_file),
        "weight_file": str(args.weight_file),
        "options_sha256": sha256(args.options_file),
        "weights_sha256": sha256(args.weight_file),
        "cuda_tf32_enabled": bool(
            device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_deterministic": bool(
            device.type == "cuda" and torch.backends.cudnn.deterministic
        ),
    }
    with args.output.with_suffix(args.output.suffix + ".json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {len(ids)} direct ELMo-small records to {args.output}")


if __name__ == "__main__":
    main()
