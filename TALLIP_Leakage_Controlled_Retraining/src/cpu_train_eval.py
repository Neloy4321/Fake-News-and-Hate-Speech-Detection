#!/usr/bin/env python3
"""CPU full-fine-tuning for the remaining nine TALLIP clean-split experiments.

Runs on a CPU (not a simulated GPU). Effective batch size remains 16 via
gradient accumulation. Fine-grained checkpoints permit resuming mid-epoch.
This is computationally expensive; running the full nine may take many days.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_eval import (EPOCHS, LEARNING_RATE, TRAIN_BATCH, WARMUP_STEPS,
                        LABEL_COL, TEXT_COL, aggregate, dump_json, evaluate,
                        file_sha256, make_plan, prep_text, read_clean_splits)

NINE = [("fakenews", "mbert", seed) for seed in (42, 52, 62)] + [
    (task, model, seed)
    for task in ("hatespeech",)
    for model in ("banglabert", "mbert")
    for seed in (42, 52, 62)
]


def order_indices(length: int, seed: int, epoch: int) -> np.ndarray:
    """Stable CPU/GPU-independent training order, with explicit resume offset."""
    return np.random.default_rng(seed + epoch * 100003).permutation(length)


def compatible_manifests(old: dict, new: dict) -> bool:
    """Permit re-packed ZIPs only if all cleaned data CSV hashes agree."""
    return all(
        old["tasks"][task]["cleaned_csv_gzip_sha256"] == new["tasks"][task]["cleaned_csv_gzip_sha256"]
        and old["tasks"][task]["fixed_original_test_text_digest_sha256"]
        == new["tasks"][task]["fixed_original_test_text_digest_sha256"]
        for task in ("fakenews", "hatespeech")
    )


def verified_old_experiment(old_digest: str, manifest_path: Path,
                            previous_manifest: Path | None) -> bool:
    current_digest = file_sha256(manifest_path)
    if old_digest == current_digest:
        return True
    if not previous_manifest or not previous_manifest.exists():
        return False
    if file_sha256(previous_manifest) != old_digest:
        return False
    old = json.loads(previous_manifest.read_text(encoding="utf-8"))
    new = json.loads(manifest_path.read_text(encoding="utf-8"))
    return compatible_manifests(old, new)


def cpu_run(config, clean_dir: Path, out_dir: Path, manifest_path: Path,
            previous_manifest: Path | None, micro_batch: int, eval_batch: int,
            checkpoint_every: int, wall_deadline: float) -> bool:
    """Returns False when paused at a durable checkpoint; True when done."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from normalizer import normalize as bn_normalize

    task, model_key, seed = config["task"], config["model_key"], config["seed"]
    name = f"{task}/{model_key}/seed_{seed}"
    run_dir = out_dir / task / model_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = file_sha256(manifest_path)
    done = run_dir / "done.json"
    if done.exists():
        old = json.loads(done.read_text(encoding="utf-8"))
        if old.get("config") != config or not verified_old_experiment(
                old.get("manifest_sha256", ""), manifest_path, previous_manifest):
            raise RuntimeError(f"Existing done.json mismatches this experiment: {name}")
        print(f"SKIP already completed {name}", flush=True)
        return True

    if wall_deadline and time.monotonic() >= wall_deadline:
        return False
    frames = read_clean_splits(clean_dir, task, manifest)
    texts = {part: [prep_text(x, bn_normalize) for x in frame[TEXT_COL[task]].tolist()]
             for part, frame in frames.items()}
    labels = {part: frame[LABEL_COL[task]].astype(int).tolist() for part, frame in frames.items()}
    del frames

    class TextDataset(Dataset):
        def __init__(self, text, y):
            self.text, self.y = text, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.text[i], self.y[i]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2, ignore_mismatched_sizes=True).to("cpu")
    model.train()

    def collate(items):
        ss, ys = zip(*items)
        batch = tokenizer(list(ss), max_length=config["max_length"],
                          truncation=True, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(ys, dtype=torch.long)
        return batch

    ds = {part: TextDataset(texts[part], labels[part]) for part in texts}
    val_loader = DataLoader(ds["val"], batch_size=eval_batch, shuffle=False,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(ds["test"], batch_size=eval_batch, shuffle=False,
                             collate_fn=collate, num_workers=0)
    steps_per_epoch = math.ceil(len(ds["train"]) / TRAIN_BATCH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=steps_per_epoch * EPOCHS)
    checkpoint = run_dir / "latest_epoch.pt"  # compatible name with GPU script
    best_dir = run_dir / "best_checkpoint"
    epoch, next_position, best_epoch, best_f1, global_step = 0, 0, 0, -1.0, 0
    history, old_gpu_state = [], False

    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("config") != config or not verified_old_experiment(
                state.get("manifest_sha256", ""), manifest_path, previous_manifest):
            raise RuntimeError(f"Resume data/config changed for {name}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        best_f1, best_epoch = state["best_val"], state["best_epoch"]
        if "next_position" in state:
            epoch, next_position = state["epoch"], state["next_position"]
            global_step = state["global_step"]
            old_gpu_state = state.get("originally_resumed_from_gpu", False)
        else:
            # Resume a previously completed GPU epoch on CPU. The GPU dropout
            # RNG stream cannot be recreated on CPU; declare the transition.
            epoch, next_position = state["next_epoch"], 0
            global_step = epoch * steps_per_epoch
            old_gpu_state = True
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        print(f"RESUMED {name} at epoch={epoch+1}, data_offset={next_position}", flush=True)

    def save_checkpoint(epoch_index, next_offset):
        """Atomic durable checkpoint, including optimizer, scheduler and RNG."""
        content = {
            "config": config, "manifest_sha256": digest,
            "epoch": epoch_index, "next_position": next_offset,
            "global_step": global_step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history, "best_val": best_f1, "best_epoch": best_epoch,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "originally_resumed_from_gpu": old_gpu_state,
        }
        temporary = run_dir / "latest_cpu_checkpoint.tmp.pt"
        torch.save(content, temporary)
        os.replace(temporary, checkpoint)

    while epoch < EPOCHS:
        order = order_indices(len(ds["train"]), seed, epoch)
        model.train()
        total_loss, steps_this_invocation = 0.0, 0
        started_epoch_segment = time.monotonic()
        for pos in range(next_position, len(order), TRAIN_BATCH):
            indices = order[pos:pos + TRAIN_BATCH].tolist()
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(indices), micro_batch):
                subset = indices[micro_start:micro_start + micro_batch]
                batch = collate([ds["train"][j] for j in subset])
                loss = model(**batch).loss
                # Correct weighting also for a truncated final effective batch.
                (loss * (len(subset) / len(indices))).backward()
                total_loss += float(loss.detach()) * len(subset) / len(indices)
            optimizer.step()
            scheduler.step()
            global_step += 1
            steps_this_invocation += 1
            next_position = min(pos + TRAIN_BATCH, len(order))
            if global_step % checkpoint_every == 0:
                save_checkpoint(epoch, next_position)
                elapsed = max(0.001, time.monotonic() - started_epoch_segment)
                rate = steps_this_invocation / elapsed
                remaining_steps = steps_per_epoch * (EPOCHS - epoch - 1) + steps_per_epoch - math.ceil(next_position / TRAIN_BATCH)
                print(f"CPU CHECKPOINT {name}, epoch {epoch+1}/{EPOCHS}, "
                      f"step {math.ceil(next_position / TRAIN_BATCH)}/{steps_per_epoch}; "
                      f"current rate {rate:.4f} optimizer updates/s; "
                      f"rough time remaining {remaining_steps / max(rate, 1e-9) / 3600:.1f} hours", flush=True)
            if wall_deadline and time.monotonic() >= wall_deadline:
                save_checkpoint(epoch, next_position)
                print(f"PAUSED SAFELY {name}: checkpoint saved at {next_position}/{len(order)} rows", flush=True)
                return False

        val_metrics, _, _ = evaluate(model, val_loader, "cpu", torch, labels["val"])
        history.append({"epoch": epoch + 1, "validation": val_metrics,
                        "session_train_loss_per_completed_step": total_loss / max(1, steps_this_invocation)})
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch + 1
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        dump_json(run_dir / "training_history.json", history)
        epoch += 1
        next_position = 0
        save_checkpoint(epoch, next_position)
        print(f"CPU FINISHED EPOCH {epoch}/{EPOCHS} {name}, best val macro-F1={best_f1:.6f}", flush=True)
        if wall_deadline and time.monotonic() >= wall_deadline:
            return False

    if not best_dir.exists():
        raise RuntimeError(f"No validation-selected best checkpoint for {name}")
    # Avoid holding both the training model+Adam states and a second inference
    # model simultaneously on a memory-limited CPU machine.
    import gc
    del optimizer, scheduler, model
    gc.collect()
    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to("cpu")
    test_metrics, predicted, probabilities = evaluate(best_model, test_loader, "cpu", torch, labels["test"])
    val_best, _, _ = evaluate(best_model, val_loader, "cpu", torch, labels["val"])
    with (run_dir / "fixed_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_test_row_id", "published_true_label", "predicted_label", "prob_0", "prob_1"])
        for i, (true, p, prob) in enumerate(zip(labels["test"], predicted, probabilities)):
            writer.writerow([i, true, p, f"{prob[0]:.9f}", f"{prob[1]:.9f}"])
    model_hash = file_sha256(best_dir / "model.safetensors")
    result = {
        "status": "CPU_FULL_TRAINING_COMPLETED_AND_TEST_EVALUATED",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "execution_device": "cpu", "micro_batch": micro_batch,
        "effective_batch_size": TRAIN_BATCH, "resumed_gpu_epoch_on_cpu": old_gpu_state,
        "config": config, "manifest_sha256": digest,
        "best_epoch": best_epoch, "best_validation": val_best,
        "fixed_original_test": test_metrics, "best_checkpoint_safetensors_sha256": model_hash,
        "history": history,
        "limitations": ["CPU microbatch accumulation is mathematically equivalent to effective batch 16, "
                        "but dropout ordering and floating-point arithmetic may differ from GPU.",
                        "Resuming at a GPU epoch boundary changes the subsequent CPU shuffle order and "
                        "random-number stream; declare mixed-device protocol if used.",
                        "Known near duplicates were filtered; unobserved semantic overlaps not ruled out."],
    }
    dump_json(done, result)
    checkpoint.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--max-wall-hours", type=float, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--runs", nargs="+", default=[f"{t}/{m}/seed_{s}" for t, m, s in NINE])
    args = parser.parse_args()
    for name, x in (("micro-batch", args.micro_batch), ("eval-batch", args.eval_batch),
                    ("threads", args.threads), ("checkpoint-every", args.checkpoint_every)):
        if x <= 0:
            parser.error(f"{name} must be positive")
    if args.micro_batch > TRAIN_BATCH or TRAIN_BATCH % args.micro_batch:
        parser.error("--micro-batch must divide effective training batch 16")
    allowed = {f"{t}/{m}/seed_{s}" for t, m, s in NINE}
    if any(x not in allowed for x in args.runs) or len(set(args.runs)) != len(args.runs):
        parser.error("Only the nine documented remaining runs are allowed; no duplicates")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["stage"] != "SPLIT_PREPARATION_VERIFIED__GPU_TRAINING_PENDING":
        raise RuntimeError("Unexpected cleaned data manifest stage")
    for task in ("fakenews", "hatespeech"):
        read_clean_splits(args.clean_dir, task, manifest)
    all_plan = make_plan(manifest, ["fakenews", "hatespeech"], ["banglabert", "mbert"], [42, 52, 62])
    by_name = {f"{r['task']}/{r['model_key']}/seed_{r['seed']}": r for r in all_plan}
    plan = [by_name[x] for x in args.runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.out_dir / "cpu_run_plan.json", {
        "status": "CPU_PLAN_ONLY" if args.plan_only else "CPU_TRAINING_REQUESTED",
        "runs": plan, "micro_batch": args.micro_batch,
        "effective_batch_size": TRAIN_BATCH, "checkpoint_every_updates": args.checkpoint_every,
        "warning": "CPU full fine-tuning can take many days; no performance metrics are imputed."})
    print(f"Validated cleaned data and CPU plan: {len(plan)} remaining runs.", flush=True)
    if args.plan_only:
        return
    import torch
    torch.set_num_threads(args.threads)
    deadline = time.monotonic() + args.max_wall_hours * 3600 if args.max_wall_hours else 0.0
    for i, run in enumerate(plan, 1):
        print(f"CPU RUN {i}/{len(plan)}: {run['task']}/{run['model_key']}/seed_{run['seed']}", flush=True)
        success = cpu_run(run, args.clean_dir, args.out_dir, args.manifest,
                          args.previous_manifest, args.micro_batch, args.eval_batch,
                          args.checkpoint_every, deadline)
        aggregate(args.out_dir, all_plan)
        if not success:
            print("Paused; rerun the same command to continue from latest_epoch.pt", flush=True)
            return
    print("All selected CPU runs completed; verify 9 done.json files.", flush=True)


if __name__ == "__main__":
    main()
