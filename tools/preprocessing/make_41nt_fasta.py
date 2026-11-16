#!/usr/bin/env python3
"""Create a compact 41-nt-center FASTA from the shared long-sequence FASTA."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.absolute()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    excluded_path = (
        args.excluded_output.absolute()
        if args.excluded_output
        else output_path.with_suffix(".excluded.tsv")
    )
    if output_path.exists() or partial_path.exists() or excluded_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}, {partial_path}, or {excluded_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: Set[int] = set()
    excluded_rows: List[tuple[int, str, str]] = []
    current_id: Optional[int] = None
    chunks: List[str] = []
    written = 0

    with input_path.open("r", encoding="utf-8") as source, partial_path.open(
        "w", encoding="utf-8"
    ) as destination:

        def store() -> None:
            nonlocal written
            if current_id is None:
                return
            sequence = "".join(chunks).upper().replace("T", "U")
            if len(sequence) < 41:
                raise ValueError(f"site_idx {current_id}: sequence is shorter than 41 nt")
            center = len(sequence) // 2
            window = sequence[center - 20 : center + 21]
            invalid = sorted(set(window) - set("ACGU"))
            if invalid:
                excluded_rows.append(
                    (current_id, "unsupported_center_base", ",".join(invalid))
                )
                return
            destination.write(f">{current_id}\n{window}\n")
            written += 1

        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                store()
                token = line[1:].split()[0]
                current_id = int(token)
                if current_id in seen:
                    raise ValueError(f"Duplicate FASTA site_idx: {current_id}")
                seen.add(current_id)
                chunks = []
            elif current_id is None:
                raise ValueError("Sequence appears before the first FASTA header")
            else:
                chunks.append(line)
        store()

    if written + len(excluded_rows) != len(seen):
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Read {len(seen)} records but wrote {written} and excluded {len(excluded_rows)}"
        )
    with excluded_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["site_idx", "reason", "unsupported_bases"])
        writer.writerows(excluded_rows)
    os.replace(partial_path, output_path)
    print(f"Read {len(seen)} FASTA records")
    print(f"Wrote {written} center windows to {output_path}")
    print(f"Excluded {len(excluded_rows)} sites; details: {excluded_path}")


if __name__ == "__main__":
    main()
