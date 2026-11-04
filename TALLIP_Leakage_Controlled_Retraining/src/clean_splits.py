#!/usr/bin/env python3
"""Build auditable test-priority splits from the two original TALLIP datasets.

The fixed hold-out test CSV is NEVER modified. The validation split has priority
over training. Known test-near-match candidates are removed from earlier splits.
This is an exact-cross-split guarantee, NOT a proof of all possible near matches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

EXPECTED = {"fakenews": {"train": 75905, "val": 9489, "test": 9489},
            "hatespeech": {"train": 40224, "val": 5028, "test": 5029}}
TEXT_COL = {"fakenews": "text", "hatespeech": "sentence"}
LABEL_COL = {"fakenews": "label", "hatespeech": "hate speech"}
NEAR_THRESHOLD = {"fakenews": 0.80, "hatespeech": 0.85}
NEAR_NGRAMS = {"fakenews": 5, "hatespeech": 2}
SPLITS = ("train", "val", "test")


def normalize_exact(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join("".join(" " if unicodedata.category(c)[0] in ("P", "S", "C")
                            else c for c in text).split())


def shingles(text: str, n: int) -> set[str]:
    words = normalize_exact(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set(words)


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_from_zip(archive: Path, task: str, split: str) -> pd.DataFrame:
    member = ("data/processed/" if task == "fakenews" else "") + split + ".csv"
    with zipfile.ZipFile(archive) as zip_file:
        if member not in zip_file.namelist():
            raise ValueError(f"Expected {member} in {archive}")
        with zip_file.open(member) as fh:
            frame = pd.read_csv(fh, encoding="utf-8-sig", low_memory=False)
    return frame


def evidence_csv(zip_file: zipfile.ZipFile, path: str) -> pd.DataFrame:
    with zip_file.open("EVIDENCE/" + path) as fh:
        return pd.read_csv(fh)


def hashes_of_norms(series: pd.Series) -> str:
    sha = hashlib.sha256()
    for norm in series:
        sha.update(norm.encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()


def check_labels(task: str, df: pd.DataFrame, split: str) -> None:
    if TEXT_COL[task] not in df or LABEL_COL[task] not in df:
        raise AssertionError(f"Missing required columns for {task}/{split}")
    if df[TEXT_COL[task]].isna().any() or df[LABEL_COL[task]].isna().any():
        raise AssertionError(f"Null text/label in {task}/{split}")
    if set(df[LABEL_COL[task]].astype(str).unique()) != {"0", "1"}:
        raise AssertionError(f"Unexpected binary label encoding: {task}/{split}")


def process_task(task: str, source: Path, evidence: zipfile.ZipFile,
                 output_root: Path, write_clean_data: bool) -> dict:
    raw = {split: csv_from_zip(source, task, split) for split in SPLITS}
    for split in SPLITS:
        if len(raw[split]) != EXPECTED[task][split]:
            raise AssertionError(f"Input row count mismatch {task}/{split}: {len(raw[split])}")
        check_labels(task, raw[split], split)

    norm = {split: raw[split][TEXT_COL[task]].map(normalize_exact) for split in SPLITS}
    # A small number of short comments are *only* punctuation/symbols after
    # aggressive normalization. Preserve original TEST rows, remove such rows
    # from earlier splits (no learnable lexical content), and report all counts.
    empty_normalized = {split: int(sum(not item for item in norm[split])) for split in SPLITS}
    norm_sets = {split: {item for item in norm[split] if item} for split in SPLITS}
    original_overlap = {
        "train_val_unique_normalized_texts": len(norm_sets["train"] & norm_sets["val"]),
        "train_test_unique_normalized_texts": len(norm_sets["train"] & norm_sets["test"]),
        "val_test_unique_normalized_texts": len(norm_sets["val"] & norm_sets["test"]),
    }

    drop: dict[str, dict[int, set[str]]] = {
        "train": defaultdict(set), "val": defaultdict(set)
    }
    # Test takes precedence over both earlier partitions; validation over train.
    for row_id, text in enumerate(norm["train"]):
        if not text:
            drop["train"][row_id].add("empty_normalized_text")
        if text in norm_sets["test"]:
            drop["train"][row_id].add("exact_in_test")
        if text in norm_sets["val"]:
            drop["train"][row_id].add("exact_in_validation")
    for row_id, text in enumerate(norm["val"]):
        if not text:
            drop["val"][row_id].add("empty_normalized_text")
        if text in norm_sets["test"]:
            drop["val"][row_id].add("exact_in_test")

    near_records = evidence_csv(evidence, f"{task}_near_candidates.csv")
    required = {"test_row_id", "earlier_split", "earlier_index", "shingle_jaccard"}
    if not required.issubset(near_records):
        raise AssertionError("Near-candidate CSV schema changed")
    score_discrepancies = []
    for match in near_records.itertuples(index=False):
        earlier = str(match.earlier_split)
        older_id, test_id = int(match.earlier_index), int(match.test_row_id)
        # The prior screen indexed train+val as a single concatenated list;
        # val positions therefore start at the ORIGINAL train row count.
        if earlier == "val":
            older_id -= len(raw["train"])
        if earlier not in drop or older_id not in range(len(raw[earlier])) or test_id not in range(len(raw["test"])):
            raise AssertionError(f"Bad near-candidate row references: {match}")
        score = jaccard(shingles(raw[earlier].iloc[older_id][TEXT_COL[task]], NEAR_NGRAMS[task]),
                        shingles(raw["test"].iloc[test_id][TEXT_COL[task]], NEAR_NGRAMS[task]))
        if float(match.shingle_jaccard) < NEAR_THRESHOLD[task] - 1e-9:
            raise AssertionError(f"Prior evidence score below its documented cutoff: {match}")
        if abs(score - float(match.shingle_jaccard)) > 1e-6:
            # Prior audit preprocessed text with a Bangla normalizer; this
            # independent source-CSV rescore may differ. Exclude the prior
            # candidate anyway and report the score discrepancy explicitly.
            score_discrepancies.append({"test_row_id": test_id, "earlier_split": earlier,
                                        "earlier_local_row_id": older_id,
                                        "prior_jaccard": float(match.shingle_jaccard),
                                        "raw_csv_rescore_jaccard": float(score)})
        drop[earlier][older_id].add("known_test_near_candidate")

    # Cross-check prior proposed exact exclusions against independently rebuilt IDs.
    # A few original proposals are *not* reproducible as exact cross-split
    # matches. Preserve them conservatively, but report them honestly.
    prior_discrepancies = {}
    for split in ("train", "val"):
        old = evidence_csv(evidence, f"{task}_proposed_clean_{split}_exclusions.csv")
        id_col = f"{split}_row_id"
        old_ids = set(int(i) for i in old[id_col])
        exact_ids = {i for i, reasons in drop[split].items() if any(x.startswith("exact_") for x in reasons)}
        if exact_ids - old_ids:
            raise AssertionError(f"Previous evidence missed independently detected exact matches: {task}/{split}")
        prior_only = sorted(old_ids - exact_ids)
        for row_id in prior_only:
            if row_id not in range(len(raw[split])):
                raise AssertionError(f"Previous evidence references invalid row: {task}/{split}/{row_id}")
            drop[split][row_id].add("prior_exclusion_not_independently_exact")
        prior_discrepancies[split] = {
            "prior_exclusion_rows": len(old_ids),
            "independent_exact_exclusion_rows": len(exact_ids),
            "prior_only_row_ids_not_independently_exact": prior_only,
        }

    kept = {"test": raw["test"].copy()}
    for split in ("train", "val"):
        kept[split] = raw[split].drop(index=sorted(drop[split])).reset_index(drop=True)
    clean_norm = {"test": norm["test"]}
    for split in ("train", "val"):
        clean_norm[split] = norm[split].drop(index=sorted(drop[split])).reset_index(drop=True)
    clean_sets = {split: {item for item in clean_norm[split] if item} for split in SPLITS}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if clean_sets[first] & clean_sets[second]:
            raise AssertionError(f"Leakage remains after filtering: {task}/{first}-{second}")

    clean_folder = output_root / "cleaned_data" / task
    evidence_folder = output_root / "evidence"
    evidence_folder.mkdir(parents=True, exist_ok=True)
    if write_clean_data:
        clean_folder.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            kept[split].to_csv(clean_folder / f"{split}.csv.gz", index=False,
                               compression={"method": "gzip", "mtime": 0})

    decisions = []
    for split in ("train", "val"):
        for row_id in sorted(drop[split]):
            decisions.append({"task": task, "split": split, "original_zero_based_row_id": row_id,
                              "reasons": "|".join(sorted(drop[split][row_id]))})
    with (evidence_folder / f"{task}_removed_row_ids.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["task", "split", "original_zero_based_row_id", "reasons"])
        writer.writeheader()
        writer.writerows(decisions)

    reason_counts = {split: dict(Counter(reason for reasons in drop[split].values() for reason in reasons))
                     for split in ("train", "val")}
    report = {
        "task": task,
        "source_zip_name": source.name,
        "source_zip_sha256": file_sha256(source),
        "evidence_zip_near_method": "prior SimHash banded candidate screen; not exhaustive",
        "near_jaccard_threshold": NEAR_THRESHOLD[task],
        "near_candidates_test_pairs_validated": int(len(near_records)),
        "near_pair_score_discrepancy_count": len(score_discrepancies),
        "near_pair_score_discrepancies": score_discrepancies,
        "original_rows": {split: len(raw[split]) for split in SPLITS},
        "clean_rows": {split: len(kept[split]) for split in SPLITS},
        "cleaned_csv_gzip_sha256": {split: file_sha256(clean_folder / f"{split}.csv.gz")
                                    for split in SPLITS} if write_clean_data else None,
        "deleted_row_counts": {split: len(drop[split]) for split in ("train", "val")},
        "deleted_reason_counts": reason_counts,
        "original_exact_overlap_distinct_text_counts": original_overlap,
        "empty_normalized_rows_by_split": empty_normalized,
        "prior_evidence_exclusion_crosscheck": prior_discrepancies,
        "post_clean_exact_cross_split_overlap": 0,
        "known_near_test_candidate_training_or_validation_rows_remaining": 0,
        "fixed_original_test_text_digest_sha256": hashes_of_norms(norm["test"]),
        "fixed_test_original_label_distribution": {str(k): int(v) for k, v in raw["test"][LABEL_COL[task]].value_counts().items()},
        "clean_label_distribution": {split: {str(k): int(v) for k, v in kept[split][LABEL_COL[task]].value_counts().items()}
                                     for split in SPLITS},
        "scientific_limits": [
            "Exact disjointness was verified using NFKC/casefold/punctuation-symbol-control removal.",
            "Known near-duplicate pairs to test were removed from prior train and validation rows.",
            "The near-duplicate screen is approximate/non-exhaustive; unobserved near matches may remain.",
            "The original test labels were retained; labels were not independently audited for every item.",
            "New evaluation metrics require fresh base-model training on this cleaned training set."
        ]
    }
    print(f"{task}: original={report['original_rows']}, clean={report['clean_rows']}, "
          f"dropped={report['deleted_row_counts']}, near_pairs={len(near_records)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fakenews-zip", type=Path, required=True)
    parser.add_argument("--hatespeech-zip", type=Path, required=True)
    parser.add_argument("--evidence-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-write-clean-data", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.evidence_zip) as evidence:
        result = {task: process_task(task, source, evidence, args.out_dir,
                                     not args.no_write_clean_data)
                  for task, source in (("fakenews", args.fakenews_zip),
                                       ("hatespeech", args.hatespeech_zip))}
    manifest = {
        "experiment": "TALLIP-26-0521 leakage-controlled base-model retraining",
        "stage": "SPLIT_PREPARATION_VERIFIED__GPU_TRAINING_PENDING",
        "fixed_test_priority": True,
        "original_test_unchanged": True,
        "overlap_evidence_archive_sha256": file_sha256(args.evidence_zip),
        "tasks": result,
    }
    target = args.out_dir / "evidence" / "clean_split_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified manifest written to {target}", flush=True)


if __name__ == "__main__":
    main()
