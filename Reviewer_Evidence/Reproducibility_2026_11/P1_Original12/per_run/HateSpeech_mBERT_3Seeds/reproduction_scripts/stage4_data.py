"""Reproducible loaders and audit helpers for the TALLIP transformer reruns.

Only standard-library modules are imported here so that dataset checks can run
before PyTorch, GPU or Kaggle have been configured.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")
TASK_CONFIG = {
    "fakenews": {"text": "text", "label": "label", "zip_prefix": "data/processed/"},
    "hatespeech": {"text": "sentence", "label": "hate speech", "zip_prefix": ""},
}


def light_clean(text: str, normalizer=None) -> str:
    """Mimic original notebook preprocessing, logging normalizer separately."""
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if normalizer is not None:
        text = normalizer(text)
    return str(text).strip()


def _read_rows(path: Path | None, zip_path: Path | None, member: str):
    if path is not None:
        with (path / Path(member).name).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if zip_path is None:
        raise ValueError("Supply either dataset directory or dataset ZIP")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
            return list(csv.DictReader(text))


def resolve_normalizer(mode: str):
    """Never silently pretend Bangla normalization happened when unavailable."""
    if mode == "none":
        return None
    if mode != "bangla":
        raise ValueError("normalizer must be 'bangla' or 'none'")
    try:
        from normalizer import normalize
    except ImportError as exc:
        raise RuntimeError(
            "Original notebook used csebuetnlp/normalizer; install it or explicitly "
            "select --normalizer none and report the protocol difference."
        ) from exc
    return normalize


def load_task(
    task: str,
    *,
    data_dir: str | None = None,
    data_zip: str | None = None,
    normalizer: str = "bangla",
    drop_cross_split: bool = True,
    limit: int = 0,
):
    """Return row-preserving texts/labels and a data-integrity report.

    Cross-split duplicate removal is *only* done on training, and validation
    collisions with test are excluded from validation. No examples are added
    to training. The test set is always preserved as published.
    """
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task {task}")
    if bool(data_dir) == bool(data_zip):
        raise ValueError("Specify exactly one of data_dir or data_zip")
    normalizer_fn = resolve_normalizer(normalizer)
    config = TASK_CONFIG[task]
    directory = Path(data_dir) if data_dir else None
    zip_path = Path(data_zip) if data_zip else None
    result = {}
    counts_before = {}
    invalid = {}
    for split in SPLITS:
        member = f"{config['zip_prefix']}{split}.csv"
        rows = _read_rows(directory, zip_path, member)
        counts_before[split] = len(rows)
        values, bad = [], 0
        for index, row in enumerate(rows):
            try:
                label = int(row[config["label"]])
            except (ValueError, TypeError, KeyError):
                bad += 1
                continue
            if label not in (0, 1):
                bad += 1
                continue
            text = light_clean(row.get(config["text"], ""), normalizer_fn)
            if not text:
                bad += 1
                continue
            values.append({"text": text, "label": label, "row_id": index})
        result[split] = values
        invalid[split] = bad
    keys = {s: {x["text"] for x in result[s]} for s in SPLITS}
    overlaps = {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    removed = {s: 0 for s in SPLITS}
    if drop_cross_split:
        forbidden = keys["test"] | keys["val"]
        cleaned = [x for x in result["train"] if x["text"] not in forbidden]
        removed["train"] = len(result["train"]) - len(cleaned)
        result["train"] = cleaned
        cleaned = [x for x in result["val"] if x["text"] not in keys["test"]]
        removed["val"] = len(result["val"]) - len(cleaned)
        result["val"] = cleaned
    if limit:
        if limit < 2:
            raise ValueError("limit must be >=2 or 0")
        result = {split: values[:limit] for split, values in result.items()}
    report = {
        "task": task,
        "normalizer": normalizer,
        "data_source": str(directory or zip_path),
        "counts_before": counts_before,
        "invalid_or_empty": invalid,
        "cross_split_unique_text_overlaps": overlaps,
        "rows_removed_to_prevent_overlap": removed,
        "counts_after": {s: len(result[s]) for s in SPLITS},
        "class_counts": {s: dict(Counter(x["label"] for x in result[s])) for s in SPLITS},
        "sha256_first_1000_texts": {
            s: hashlib.sha256("\n".join(x["text"] for x in result[s][:1000]).encode()).hexdigest()
            for s in SPLITS
        },
        "test_preserved": not bool(limit),
    }
    return result, report


def save_report(report: dict, path: str | Path):
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

