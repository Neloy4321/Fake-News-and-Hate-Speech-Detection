"""Independent-seed BanglaBERT/mBERT training for the ACM TALLIP revision.

Example: python stage4_multiseed_train.py --task fakenews --model banglabert \
    --seed 42 --fake-zip '/path/to/Multi-Source Bengali Fake News Detection Benchmark.zip'

GPU is required for real experiments. No fabricated scores, synthetic predictions,
or pretrained-only inference are presented as trained-model results.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from stage4_data import load_task, save_report


MODELS = {"banglabert": "csebuetnlp/banglabert", "mbert": "bert-base-multilingual-cased"}
MAX_LENGTH = {"fakenews": 256, "hatespeech": 128}


class EncodedDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.encodings = tokenizer(
            [r["text"] for r in rows], truncation=True, max_length=max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = {k: v[index] for k, v in self.encodings.items()}
        row["labels"] = int(self.rows[index]["label"])
        return row


def metrics(labels, predictions):
    p, r, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f1)}


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


def evaluate(model, loader, device, *, return_scores=False):
    model.eval()
    truth, predictions, logits_all = [], [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu().numpy()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(axis=1).tolist())
            if return_scores:
                logits_all.append(logits)
    score = metrics(truth, predictions)
    if return_scores:
        return score, np.asarray(truth), np.asarray(predictions), np.vstack(logits_all)
    return score


def train(args):
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA GPU. This is a training script, not a source of simulated results. "
            "Run on Kaggle T4/P100 (or use --allow-cpu for very small smoke tests)."
        )
    if args.smoke and not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("CPU smoke tests require --allow-cpu")
    task = args.task
    model_key = args.model
    model_id = MODELS[model_key]
    max_length = MAX_LENGTH[task]
    source = args.fake_dir or args.fake_zip if task == "fakenews" else args.hate_dir or args.hate_zip
    if source is None:
        raise ValueError(f"Missing source path for {task}; supply --fake-dir/zip or --hate-dir/zip")
    out_dir = Path(args.out_root) / ("SMOKE_ONLY" if args.smoke else "real_runs") / task / model_key / f"seed_{args.seed}"
    if (out_dir / "COMPLETE.json").exists() and not args.force:
        print("Already completed; leaving existing results intact:", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    seed_everything(args.seed)
    splits, report = load_task(
        task, data_dir=source if (args.fake_dir if task == "fakenews" else args.hate_dir) else None,
        data_zip=source if (args.fake_zip if task == "fakenews" else args.hate_zip) else None,
        normalizer=args.normalizer,
        drop_cross_split=not args.keep_overlap,
        limit=args.smoke,
    )
    save_report(report, out_dir / "data_audit.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    data = {k: EncodedDataset(v, tokenizer, max_length) for k, v in splits.items()}
    collate = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    rng = torch.Generator()
    rng.manual_seed(args.seed)
    loaders = {
        k: DataLoader(
            d, batch_size=args.batch_size if k == "train" else args.eval_batch_size,
            shuffle=(k == "train"), generator=rng if k == "train" else None,
            num_workers=0, collate_fn=collate,
        ) for k, d in data.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.to(device)
    counts = Counter(x["label"] for x in splits["train"])
    weights = None
    if args.class_weighted:
        n = len(splits["train"])
        weights = torch.tensor([n / (2 * max(counts.get(c, 0), 1)) for c in (0, 1)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(args.warmup_steps, max(total_updates - 1, 0)),
        num_training_steps=total_updates,
    )
    amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_f1, best_epoch, history = -1.0, 0, []
    best_state_file = out_dir / "best_model_weights.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        rolling_loss = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                outputs = model(**batch)
                loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            rolling_loss += float(loss.item())
            if step % 200 == 0:
                print(f"{task}/{model_key}/seed{args.seed} epoch={epoch} step={step}/{len(loaders['train'])} loss={rolling_loss/step:.4f}", flush=True)
        val = evaluate(model, loaders["val"], device)
        val.update({"epoch": epoch, "train_loss": rolling_loss / max(len(loaders["train"]), 1)})
        history.append(val)
        print("VALIDATION", json.dumps(val), flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            # Checkpoint is chosen exclusively on validation, never test.
            torch.save(model.state_dict(), best_state_file)
    try:
        state = torch.load(best_state_file, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_state_file, map_location=device)
    model.load_state_dict(state)
    test, truth, predictions, logits = evaluate(model, loaders["test"], device, return_scores=True)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    prob = exp[:, 1] / exp.sum(axis=1)
    np.savez_compressed(out_dir / "test_predictions.npz", y_true=truth, y_pred=predictions,
                        y_prob=prob, test_row_id=np.array([x["row_id"] for x in splits["test"]]))
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    details = {
        "task": task, "model": model_id, "model_key": model_key, "seed": args.seed,
        "max_length": max_length, "epochs": args.epochs, "train_batch_size": args.batch_size,
        "lr": args.lr, "warmup_steps": args.warmup_steps, "class_weighted": args.class_weighted,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "test": test, "elapsed_seconds": round(time.monotonic() - start, 2),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU_ONLY",
        "torch_version": torch.__version__, "normalizer": args.normalizer,
        "experimental_only": bool(args.smoke),
        "truncation_note": f"Only first {max_length} subword tokens processed; full article not explained.",
        "labels_source": "published dataset CSV labels; not independently human re-annotated",
    }
    (out_dir / "metrics.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_model:
        model.save_pretrained(out_dir / "model")
        tokenizer.save_pretrained(out_dir / "model")
    if not args.keep_weights and not args.save_model:
        best_state_file.unlink(missing_ok=True)
    (out_dir / "COMPLETE.json").write_text(json.dumps({"finished": True, "test_macro_f1": test["macro_f1"]}), encoding="utf-8")
    print("TEST", json.dumps(test), "OUTPUT", out_dir, flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(MAX_LENGTH), required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fake-dir")
    parser.add_argument("--fake-zip")
    parser.add_argument("--hate-dir")
    parser.add_argument("--hate-zip")
    parser.add_argument("--out-root", default="/kaggle/working/tallip_stage4")
    parser.add_argument("--normalizer", choices=["bangla", "none"], default="bangla")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--keep-overlap", action="store_true", help="Legacy reproduction ONLY: allow cross-split duplicates")
    parser.add_argument("--smoke", type=int, default=0, metavar="N", help="Small non-publishable smoke run")
    parser.add_argument("--allow-cpu", action="store_true", help="For smoke runs only")
    parser.add_argument("--force", action="store_true", help="Re-run even if COMPLETE.json exists")
    args = parser.parse_args()
    if args.allow_cpu and not args.smoke:
        parser.error("--allow-cpu is restricted to --smoke mode")
    train(args)


if __name__ == "__main__":
    cli()

