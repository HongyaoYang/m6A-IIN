#!/usr/bin/env python3
"""Batch RNAfold a 41-nt FASTA into a site-indexed NumPy structure array."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


MISSING = 255
STRUCTURE_LUT = np.full(256, MISSING, dtype=np.uint8)
STRUCTURE_LUT[ord(".")] = 0
STRUCTURE_LUT[ord("(")] = 1
STRUCTURE_LUT[ord(")")] = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rnafold", default="RNAfold")
    return parser.parse_args()


def scan_fasta(path: Path) -> tuple[int, int]:
    count = 0
    maximum = -1
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith(">"):
                site_idx = int(raw_line[1:].strip().split()[0])
                count += 1
                maximum = max(maximum, site_idx)
    if count == 0:
        raise ValueError(f"No FASTA records found in {path}")
    return count, maximum


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    fasta = args.fasta.resolve()
    output = args.output.absolute()
    partial = output.with_suffix(output.suffix + ".partial")
    metadata = output.with_suffix(output.suffix + ".json")
    stderr_path = output.with_suffix(output.suffix + ".rnafold.stderr.log")
    for path in (output, partial, metadata, stderr_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    executable = shutil.which(args.rnafold)
    if executable is None:
        raise FileNotFoundError(f"RNAfold executable not found: {args.rnafold}")
    n_records, max_site_idx = scan_fasta(fasta)
    output.parent.mkdir(parents=True, exist_ok=True)
    structures = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.uint8, shape=(max_site_idx + 1, 41)
    )
    structures[:] = MISSING

    command = [executable, "--noPS", f"--jobs={args.jobs}"]
    folded = 0
    current_id = None
    expecting_sequence = False
    with fasta.open("r", encoding="utf-8") as source, stderr_path.open(
        "w", encoding="utf-8"
    ) as error_log:
        process = subprocess.Popen(
            command,
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=error_log,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    raise RuntimeError(f"Incomplete RNAfold record before {line}")
                current_id = int(line[1:].split()[0])
                expecting_sequence = True
                continue
            if current_id is None:
                raise RuntimeError(f"Unexpected RNAfold output before FASTA header: {line}")
            token = line.split()[0]
            if expecting_sequence:
                if len(token) != 41 or set(token.upper()) - set("ACGUT"):
                    raise RuntimeError(f"Invalid RNAfold sequence line for {current_id}: {line}")
                expecting_sequence = False
                continue
            if len(token) != 41 or set(token) - set(".()"):
                raise RuntimeError(f"Invalid RNAfold structure for {current_id}: {line}")
            if structures[current_id, 0] != MISSING:
                raise RuntimeError(f"Duplicate RNAfold output for site_idx {current_id}")
            encoded = STRUCTURE_LUT[np.frombuffer(token.encode("ascii"), dtype=np.uint8)]
            structures[current_id] = encoded
            folded += 1
            current_id = None
            if folded % 100_000 == 0:
                print(f"Folded {folded}/{n_records}", flush=True)
        return_code = process.wait()

    structures.flush()
    if return_code != 0:
        raise RuntimeError(f"RNAfold exited with code {return_code}; see {stderr_path}")
    if current_id is not None:
        raise RuntimeError(f"Incomplete final RNAfold record for site_idx {current_id}")
    if folded != n_records:
        raise RuntimeError(f"Expected {n_records} structures, received {folded}")
    del structures
    os.replace(partial, output)
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata.write_text(
        json.dumps(
            {
                "fasta": str(fasta),
                "records": n_records,
                "max_site_idx": max_site_idx,
                "sequence_length": 41,
                "encoding": {".": 0, "(": 1, ")": 2, "missing": MISSING},
                "rnafold_version": version,
                "jobs": args.jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {folded} structures to {output}")


if __name__ == "__main__":
    main()
