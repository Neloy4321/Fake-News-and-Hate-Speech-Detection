#!/usr/bin/env python3
"""Re-train both base Transformer families on VERIFIED cleaned splits.

Does not import Torch until a real GPU run is requested. Run with --plan-only
to validate inputs/configuration on CPU without claiming model results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

MODEL_IDS = {"banglabert": "csebuetnlp/banglabert", "mbert": "bert-base-multilingual-cased"}
TEXT_COL = {"fakenews": "text", "hatespeech": "sentence"}
LABEL_COL = {"fakenews": "label", "hatespeech": "hate speech"}
MAX_LENGTH = {"fakenews": 256, "hatespeech": 128}
SEEDS = (42, 52, 62)
EPOCHS = 3
TRAIN_BATCH = 16
EVAL_BATCH = 32
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def prep_text(text: str, bangla_normalize) -> str:
    # Matches the previous evidence notebook's preprocessing recipe.
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return str(bangla_normalize(text)).strip()


def metrics(true, predicted) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, predicted, labels=[0, 1], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(true, predicted)),
            "macro_precision": float(precision), "macro_recall": float(recall),
            "macro_f1": float(f1),
            "confusion_matrix_labels_0_1": confusion_matrix(true, predicted, labels=[0, 1]).tolist()}


def read_clean_splits(clean_dir: Path, task: str, manifest: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ("train", "val", "test"):
        file = clean_dir / task / (split + ".csv.gz")
        if not file.exists():
            raise FileNotFoundError(f"Expected cleaned file: {file}")
        expected_sha = manifest["tasks"][task].get("cleaned_csv_gzip_sha256", {})
        if not expected_sha or file_sha256(file) != expected_sha.get(split):
            raise AssertionError(f"Cleaned CSV SHA256 differs from audited manifest: {task}/{split}")
        frame = pd.read_csv(file, low_memory=False)
        if len(frame) != manifest["tasks"][task]["clean_rows"][split]:
            raise AssertionError(f"Manifest vs cleaned CSV count mismatch: {task}/{split}")
        if frame[TEXT_COL[task]].isna().any() or frame[LABEL_COL[task]].isna().any():
            raise AssertionError(f"Null text or label in cleaned {task}/{split}")
        labels = sorted(frame[LABEL_COL[task]].astype(int).unique())
        if labels != [0, 1]:
            raise AssertionError(f"Unexpected labels {labels} in cleaned {task}/{split}")
        frames[split] = frame
    return frames


def make_plan(manifest: dict, tasks: list[str], models: list[str], seeds: list[int]) -> list[dict]:
    plan = []
    for task in tasks:
        for model in models:
            for seed in seeds:
                plan.append({
                    "task": task, "model_key": model, "base_model": MODEL_IDS[model],
                    "seed": seed, "max_length": MAX_LENGTH[task], "epochs": EPOCHS,
                    "train_batch_size": TRAIN_BATCH, "eval_batch_size": EVAL_BATCH,
                    "learning_rate": LEARNING_RATE, "warmup_steps": WARMUP_STEPS,
                    "class_weighted": False,
                    "train_rows": manifest["tasks"][task]["clean_rows"]["train"],
                    "val_rows": manifest["tasks"][task]["clean_rows"]["val"],
                    "test_rows": manifest["tasks"][task]["clean_rows"]["test"],
                })
    return plan


def evaluate(model, loader, device, torch, labels: list[int]):
    model.eval()
    preds, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            labels_batch = batch.pop("labels", None)
            args = {k: v.to(device) for k, v in batch.items()}
            logits = model(**args).logits
            prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probabilities.extend(prob.tolist())
            preds.extend(np.argmax(prob, axis=-1).tolist())
            if labels_batch is not None:
                del labels_batch
    result = metrics(labels, preds)
    return result, preds, probabilities


def perform_run(config: dict, clean_dir: Path, out_dir: Path, manifest_path: Path):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from normalizer import normalize as bn_normalize

    if not torch.cuda.is_available():
        raise RuntimeError("GPU absent: refusing to mark Transformer retraining as completed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_digest = file_sha256(manifest_path)
    task, model_key, seed = config["task"], config["model_key"], config["seed"]
    run_dir = out_dir / task / model_key / f"seed_{seed}"
    finished = run_dir / "done.json"
    if finished.exists():
        old = json.loads(finished.read_text(encoding="utf-8"))
        if old.get("manifest_sha256") != manifest_digest or old.get("config") != config:
            raise RuntimeError("Existing completed run has different manifest/config: " + str(run_dir))
        print(f"SKIP verified finished run: {run_dir}", flush=True)
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "config.json", config)

    frames = read_clean_splits(clean_dir, task, manifest)
    texts = {split: [prep_text(x, bn_normalize) for x in frame[TEXT_COL[task]].tolist()]
             for split, frame in frames.items()}
    labels = {split: frame[LABEL_COL[task]].astype(int).tolist() for split, frame in frames.items()}

    class TextDataset(Dataset):
        def __init__(self, texts_, labels_):
            self.texts = texts_
            self.labels = labels_

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.texts[index], self.labels[index]

    # Seed before the model head is initialized from the pretrained BASE.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2, ignore_mismatched_sizes=True)
    model.to("cuda")

    def collate(items):
        strings, ys = zip(*items)
        data = tokenizer(list(strings), max_length=config["max_length"],
                         truncation=True, padding=True, return_tensors="pt")
        data["labels"] = torch.tensor(ys, dtype=torch.long)
        return data

    datasets = {split: TextDataset(texts[split], labels[split]) for split in texts}
    val_loader = DataLoader(datasets["val"], batch_size=EVAL_BATCH,
                            shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=EVAL_BATCH,
                             shuffle=False, collate_fn=collate, num_workers=0)
    train_length = math.ceil(len(datasets["train"]) / TRAIN_BATCH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=train_length * EPOCHS)
    checkpoint = run_dir / "latest_epoch.pt"
    best_dir = run_dir / "best_checkpoint"
    history_file = run_dir / "training_history.json"
    history = []
    best_val = -1.0
    best_epoch = 0
    start_epoch = 0
    if checkpoint.exists():
        old = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if old["config"] != config or old["manifest_sha256"] != manifest_digest:
            raise RuntimeError("Refusing to resume from a different experiment")
        model.load_state_dict(old["model_state"])
        optimizer.load_state_dict(old["optimizer_state"])
        scheduler.load_state_dict(old["scheduler_state"])
        history = old["history"]
        best_val, best_epoch = old["best_val"], old["best_epoch"]
        start_epoch = old["next_epoch"]
        random.setstate(old["python_rng_state"])
        np.random.set_state(old["numpy_rng_state"])
        torch.set_rng_state(old["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(old["cuda_rng_states"])
        print(f"RESUME: {task}/{model_key}/seed_{seed} at epoch {start_epoch+1}", flush=True)

    for epoch in range(start_epoch, EPOCHS):
        # Reproduce shuffle ordering independently of previous DataLoader lives.
        gen = torch.Generator().manual_seed(seed + epoch * 100003)
        train_loader = DataLoader(datasets["train"], batch_size=TRAIN_BATCH,
                                  shuffle=True, generator=gen, collate_fn=collate, num_workers=0)
        model.train()
        total_loss = 0.0
        for idx, batch in enumerate(train_loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            if (idx + 1) % 500 == 0:
                print(f"{task}/{model_key}/seed{seed} epoch{epoch+1}: {idx+1}/{train_length} batches", flush=True)

        val_metrics, _, _ = evaluate(model, val_loader, "cuda", torch, labels["val"])
        history.append({"epoch": epoch + 1, "mean_train_loss": total_loss / train_length,
                        "validation": val_metrics})
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        dump_json(history_file, history)
        state = {
            "config": config, "manifest_sha256": manifest_digest, "next_epoch": epoch + 1,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "history": history,
            "best_val": best_val, "best_epoch": best_epoch,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        tmp = checkpoint.with_name("latest_epoch.tmp.pt")
        torch.save(state, tmp)
        os.replace(tmp, checkpoint)
        print(f"EPOCH {epoch+1}/{EPOCHS}: val macro F1={val_metrics['macro_f1']:.6f}; best={best_val:.6f}", flush=True)

    # Evaluate the held-out TEST only once, after validation-based model selection.
    if not best_dir.exists():
        raise RuntimeError("Best validation model checkpoint is missing")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to("cuda")
    test_metrics, predictions, prob = evaluate(best_model, test_loader, "cuda", torch, labels["test"])
    val_best_metrics, _, _ = evaluate(best_model, val_loader, "cuda", torch, labels["val"])
    predictions_path = run_dir / "fixed_test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_test_row_id", "published_true_label", "predicted_label", "prob_0", "prob_1"])
        for i, (true, predicted, p) in enumerate(zip(labels["test"], predictions, prob)):
            writer.writerow([i, true, predicted, f"{p[0]:.9f}", f"{p[1]:.9f}"])

    safetensors = best_dir / "model.safetensors"
    checkpoint_hash = file_sha256(safetensors) if safetensors.exists() else "NOT_FOUND"
    if checkpoint_hash == "NOT_FOUND":
        raise RuntimeError("Expected best model.safetensors not found")
    result = {
        "status": "GPU_TRAINING_COMPLETED_AND_TEST_EVALUATED",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "manifest_sha256": manifest_digest,
        "best_epoch": best_epoch, "best_validation": val_best_metrics,
        "fixed_original_test": test_metrics,
        "best_checkpoint_safetensors_sha256": checkpoint_hash,
        "history": history,
        "evaluation_scope": "Same original fixed published-label test; new cleaned-split base-model training",
    }
    dump_json(finished, result)
    # A finished run needs only its best checkpoint, not the large optimizer snapshot.
    if checkpoint.exists():
        checkpoint.unlink()
    return result


def aggregate(out_dir: Path, plan: list[dict]) -> dict:
    groups = {}
    for run in plan:
        key = (run["task"], run["model_key"])
        finished = out_dir / run["task"] / run["model_key"] / f"seed_{run['seed']}" / "done.json"
        if finished.exists():
            result = json.loads(finished.read_text(encoding="utf-8"))
            score = result["fixed_original_test"]
            groups.setdefault(key, []).append({"seed": run["seed"], **score})
    summary = {"warning": "Only completed GPU runs count; missing seeds are NOT imputed.", "groups": []}
    for (task, model), items in sorted(groups.items()):
        vals = [item["macro_f1"] for item in items]
        record = {"task": task, "model": model, "completed_seeds": [i["seed"] for i in items],
                  "required_seeds": list(SEEDS), "n": len(items),
                  "test_macro_f1_mean": float(np.mean(vals)),
                  "test_macro_f1_sample_sd": float(np.std(vals, ddof=1)) if len(vals) >= 2 else None,
                  "individual_runs": items}
        if len(items) == 3:
            record["test_macro_f1_95pct_t_CI"] = [
                float(np.mean(vals) - 4.3026527299 * np.std(vals, ddof=1) / math.sqrt(3)),
                float(np.mean(vals) + 4.3026527299 * np.std(vals, ddof=1) / math.sqrt(3))]
        summary["groups"].append(record)
    dump_json(out_dir / "aggregate_completed_only.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TEXT_COL), default=list(TEXT_COL))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_IDS), default=list(MODEL_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        parser.error("Use only original manuscript seeds 42/52/62")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["stage"] != "SPLIT_PREPARATION_VERIFIED__GPU_TRAINING_PENDING":
        raise RuntimeError("Unrecognized/unchecked clean split manifest")
    if any(manifest["tasks"][task]["post_clean_exact_cross_split_overlap"] != 0
           for task in args.tasks):
        raise RuntimeError("Clean-split evidence gate failed")
    for task in args.tasks:
        read_clean_splits(args.clean_dir, task, manifest)
    plan = make_plan(manifest, args.tasks, args.models, args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.out_dir / "run_plan.json", {"status": "PLAN_ONLY" if args.plan_only else "GPU_RUN_REQUESTED",
                                               "source_manifest_sha256": file_sha256(args.manifest), "runs": plan})
    print(f"Validated original CSV and clean split manifest for {len(plan)} runs.", flush=True)
    if args.plan_only:
        print("PLAN ONLY: no training, no test metrics, no completion claim.", flush=True)
        return
    for idx, run in enumerate(plan, 1):
        print(f"START {idx}/{len(plan)}: {run['task']}/{run['model_key']}/seed_{run['seed']}", flush=True)
        perform_run(run, args.clean_dir, args.out_dir, args.manifest)
        aggregate(args.out_dir, plan)
    print("All requested runs finished. Check done.json per run and aggregate_completed_only.json.", flush=True)


if __name__ == "__main__":
    main()
