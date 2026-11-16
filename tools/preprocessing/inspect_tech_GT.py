#!/usr/bin/env python3
"""Inspect the extracted tech_GT data contract required by the IIN pipeline.

The script is read-only. It scans large technology CSVs in chunks and writes a
JSON report, so it is suitable for running on an HPC login or short CPU job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd


TECHNOLOGIES = (
    "DART_seq", "eTAM_seq", "GLORI", "Dorado", "m6A_REF_seq",
    "m6A_SAC_seq", "m6ACE_seq", "MAZTER_seq", "MeRIP_seq",
    "Oxford_nanopore", "scDART_seq",
)
REQUIRED_GT_COLUMNS = {"site_idx", "ratio", "m6A", "Total", "class"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--min-total", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any ERROR")
    return parser.parse_args()


def parse_split(path: Path) -> Dict[str, List[int]]:
    frame = pd.read_csv(path, dtype={"tech": str, "site_idx": str})
    if not {"tech", "site_idx"}.issubset(frame.columns):
        raise ValueError("must contain tech and site_idx columns")
    if frame["tech"].duplicated().any():
        raise ValueError("contains duplicate technology rows")
    result: Dict[str, List[int]] = {}
    for row in frame.itertuples(index=False):
        ids = [int(value.strip()) for value in str(row.site_idx).split(",") if value.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{row.tech} contains duplicate site_idx values")
        result[str(row.tech)] = ids
    return result


def inspect_fasta(path: Path, required_ids: Set[int]) -> Dict[str, object]:
    result: Dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["error"] = "Missing sequences.fa; IIN cannot construct 41-nt inputs."
        return result
    ids: Set[int] = set()
    duplicate_headers = 0
    invalid_sequences = 0
    too_short = 0
    current_id = None
    chunks: List[str] = []

    def store() -> None:
        nonlocal invalid_sequences, too_short
        if current_id is None:
            return
        sequence = "".join(chunks).upper().replace("T", "U")
        too_short += int(len(sequence) < 41)
        invalid_sequences += int(bool(set(sequence) - set("ACGU")))

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                store()
                token = line[1:].split()[0]
                current_id = int(token)
                duplicate_headers += int(current_id in ids)
                ids.add(current_id)
                chunks = []
            elif current_id is None:
                raise ValueError("sequence appears before the first FASTA header")
            else:
                chunks.append(line)
        store()
    missing = required_ids - ids
    result.update({
        "n_sequences": len(ids),
        "duplicate_headers": duplicate_headers,
        "sequences_shorter_than_41": too_short,
        "sequences_with_non_ACGUT": invalid_sequences,
        "required_split_ids_missing": len(missing),
        "missing_examples": sorted(missing)[:10],
    })
    return result


def inspect_gt(path: Path, split_ids: Dict[str, Set[int]], chunk_size: int, min_total: int) -> Dict[str, object]:
    header = pd.read_csv(path, nrows=0)
    missing_columns = sorted(REQUIRED_GT_COLUMNS - set(header.columns))
    result: Dict[str, object] = {
        "path": str(path), "size_bytes": path.stat().st_size,
        "columns": header.columns.tolist(), "missing_required_columns": missing_columns,
    }
    if missing_columns:
        return result

    n_rows = n_positive = n_eligible = n_missing_class = n_missing_class_eligible = 0
    invalid_counts = invalid_ratio = invalid_class = duplicate_ids = 0
    seen_ids: Set[int] = set()
    wanted = set().union(*split_ids.values())
    found_wanted: Set[int] = set()
    for chunk in pd.read_csv(path, usecols=list(REQUIRED_GT_COLUMNS), chunksize=chunk_size):
        for column in ("site_idx", "ratio", "m6A", "Total", "class"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        n_rows += len(chunk)
        site = chunk["site_idx"]
        modified = chunk["m6A"]
        total = chunk["Total"]
        ratio = chunk["ratio"]
        class_value = chunk["class"]
        n_positive += int((total > 0).sum())
        eligible = total >= min_total
        n_eligible += int(eligible.sum())
        n_missing_class += int(class_value.isna().sum())
        n_missing_class_eligible += int(class_value[eligible].isna().sum())
        invalid_counts += int((site.isna() | modified.isna() | total.isna() |
                               (modified < 0) | (total <= 0) | (modified > total)).sum())
        expected = modified / total
        invalid_ratio += int((ratio.isna() | ~np.isclose(ratio, expected, rtol=1e-5, atol=1e-7)).sum())
        present_class = class_value.notna()
        invalid_class += int((present_class & ~class_value.isin([0, 1])).sum())
        integer_ids = site.dropna().astype("int64")
        current = set(integer_ids.tolist())
        duplicate_ids += len(current & seen_ids) + (len(integer_ids) - len(current))
        seen_ids.update(current)
        found_wanted.update(current & wanted)

    result.update({
        "rows": n_rows, "rows_Total_gt_0": n_positive,
        f"rows_Total_ge_{min_total}": n_eligible,
        "missing_class_all_rows": n_missing_class,
        f"missing_class_Total_ge_{min_total}": n_missing_class_eligible,
        "invalid_count_rows": invalid_counts, "ratio_mismatch_rows": invalid_ratio,
        "invalid_nonmissing_class_rows": invalid_class, "duplicate_site_idx": duplicate_ids,
        "split_membership": {
            name: {"requested": len(ids), "missing_from_ground_truth": len(ids - found_wanted)}
            for name, ids in split_ids.items()
        },
    })
    return result


def add_issue(issues: List[Dict[str, str]], severity: str, message: str) -> None:
    issues.append({"severity": severity, "message": message})


def main() -> None:
    args = parse_args()
    base = args.base_dir.resolve()
    report_path = (args.report or base / "IIN_data_contract_report.json").resolve()
    issues: List[Dict[str, str]] = []
    report: Dict[str, object] = {"base_dir": str(base), "technologies": {}, "issues": issues}

    splits: Dict[str, Dict[str, List[int]]] = {}
    for name in ("train", "valid", "test"):
        path = base / f"{name}_site.csv"
        if not path.is_file():
            add_issue(issues, "ERROR", f"Missing {path.name}")
            continue
        try:
            splits[name] = parse_split(path)
        except Exception as error:
            add_issue(issues, "ERROR", f"{path.name}: {error}")
    required_fasta_ids: Set[int] = set()

    if "train" in splits and "valid" in splits:
        for tech in TECHNOLOGIES:
            train_ids = set(splits["train"].get(tech, []))
            valid_ids = set(splits["valid"].get(tech, []))
            test_ids = set(splits.get("test", {}).get(tech, []))
            required_fasta_ids.update(train_ids | valid_ids | test_ids)
            overlap = {
                "train_valid": len(train_ids & valid_ids),
                "train_test": len(train_ids & test_ids),
                "valid_test": len(valid_ids & test_ids),
            }
            if any(overlap.values()):
                add_issue(issues, "ERROR", f"{tech}: split overlap {overlap}")
            gt_path = base / f"{tech}.csv"
            if not gt_path.is_file():
                add_issue(issues, "ERROR", f"Missing {gt_path.name}")
                continue
            try:
                gt_report = inspect_gt(
                    gt_path,
                    {"train": train_ids, "valid": valid_ids, "test": test_ids},
                    args.chunk_size,
                    args.min_total,
                )
                gt_report["split_overlap"] = overlap
                report["technologies"][tech] = gt_report
                if gt_report.get(f"missing_class_Total_ge_{args.min_total}", 0):
                    add_issue(issues, "ERROR", f"{tech}: class is missing at evaluation coverage")
            except Exception as error:
                add_issue(issues, "ERROR", f"{tech}.csv: {error}")

    fasta_path = (args.fasta or base / "sequences.fa").resolve()
    try:
        report["fasta"] = inspect_fasta(fasta_path, required_fasta_ids)
        if not report["fasta"]["exists"]:
            add_issue(issues, "ERROR", "sequences.fa is absent; supply or generate it before IIN training")
    except Exception as error:
        report["fasta"] = {"path": str(fasta_path), "error": str(error)}
        add_issue(issues, "ERROR", f"FASTA validation failed: {error}")

    report["rds_files"] = {
        name: {"exists": (base / name).is_file(), "size_bytes": (base / name).stat().st_size if (base / name).is_file() else None}
        for name in ("grg.rds", "split_sites.rds")
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {report_path}")
    if args.strict and any(item["severity"] == "ERROR" for item in issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
