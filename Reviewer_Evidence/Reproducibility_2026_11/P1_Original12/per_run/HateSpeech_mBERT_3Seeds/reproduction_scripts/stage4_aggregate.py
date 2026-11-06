"""Summarize independently run model seeds; refuses incomplete/misaligned outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def summarize(root, task, model, seeds):
    runs = []
    aligned_y, aligned_row = None, None
    for seed in seeds:
        directory = Path(root) / "real_runs" / task / model / f"seed_{seed}"
        if not (directory / "COMPLETE.json").exists():
            raise FileNotFoundError(f"Missing FINISHED run: {directory}")
        with np.load(directory / "test_predictions.npz") as data:
            y = data["y_true"].copy()
            row_ids = data["test_row_id"].copy()
            pred = data["y_pred"].copy()
            prob = data["y_prob"].copy()
        if aligned_y is not None and (not np.array_equal(y, aligned_y) or not np.array_equal(row_ids, aligned_row)):
            raise ValueError("SEED ALIGNMENT FAILURE: test row IDs and labels must match across all seeds")
        aligned_y, aligned_row = y, row_ids
        if set(np.unique(y)) - {0, 1} or len(y) != len(pred) or len(prob) != len(y):
            raise ValueError("Invalid prediction arrays")
        score = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if score.get("experimental_only"):
            raise ValueError("Non-publishable smoke run found")
        p, r, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
        runs.append({
            "seed": seed, "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class_precision": p.tolist(), "per_class_recall": r.tolist(),
            "per_class_f1": f1.tolist(), "class_support": support.tolist(),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        })
    macro = np.asarray([run["macro_f1"] for run in runs])
    return {
        "task": task, "model": model, "seeds": list(seeds), "test_count": len(aligned_y),
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_sample_sd": float(macro.std(ddof=1)) if len(macro) > 1 else None,
        "note": "Sample SD reflects model-init/training seeds; not a confidence interval or human annotation agreement.",
        "runs": runs,
    }


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/kaggle/working/tallip_stage4")
    p.add_argument("--task", choices=["fakenews", "hatespeech"], required=True)
    p.add_argument("--model", choices=["banglabert", "mbert"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    p.add_argument("--out")
    args = p.parse_args()
    report = summarize(args.root, args.task, args.model, args.seeds)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    cli()

