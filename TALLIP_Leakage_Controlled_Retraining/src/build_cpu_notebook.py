#!/usr/bin/env python3
"""Build a stand-alone Kaggle CPU notebook for only the nine remaining runs."""

import json
from pathlib import Path

import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Reuse the tested original dataset-discovery and split-cleaning cells.
    build_notebook.main()
    base_path = ROOT / "notebooks" / "TALLIP_Leakage_Controlled_12Run_RunAll.ipynb"
    nb = json.loads(base_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0] = build_notebook.markdown("""# ACM TALLIP-26-0521 — Remaining 9 Clean-Split Runs on CPU

**CPU-only full Transformer fine-tuning.** This is **not** a fast substitute for a GPU. The full nine runs may take **many days or longer**, depending on CPU and RAM. No fake metrics, data subset, or reduced epochs are used.

Completed GPU **Fake News + BanglaBERT (seeds 42/52/62)** are intentionally excluded from the nine-run plan. They are **not rerun**. Attach their previous output to combine metrics later if available. The CPU run order prioritizes the shorter Hate Speech experiments first.

## Input requirements

1. Use Kaggle **CPU** (or adapt the same source scripts locally). Attach the **original two datasets**: Multi-Source Bengali Fake News Detection Benchmark and BD-SHS. The notebook rebuilds the *same* cleaned train/validation partitions and retains the original fixed test rows.
2. The previous three completed GPU run folders are **not required** to start these nine, but should be preserved independently. If you have your previous saved `training/` directory, attach that Kaggle notebook output and set `PREVIOUS_RESULTS_DIR` to its full path. To resume a partially trained GPU mBERT epoch on CPU, attach that run's entire folder including `latest_epoch.pt` and `best_checkpoint/`.
3. Click Run All. CPU training writes **atomic optimizer checkpoints** every 200 effective-batch updates and stops gracefully after a configurable wall-time budget (default: **4 hours per notebook execution**). After it stops, *save and retain the output* before another CPU session; do not assume a running notebook's Output tab shows live files.
4. For the next session, attach the previously saved output, set `PREVIOUS_RESULTS_DIR`, and Run All again. It skips completed CPU runs, continues mid-epoch checkpoints, and never starts the three already-completed BanglaBERT/Fake News seeds.

CPU uses `micro_batch=1` and gradient accumulation to keep the **effective train batch size at 16**, with all three epochs, 256/128 subwords, LR 5e-5, warmup 500 and the same two pretrained base models. Actual computational/floating-point results may differ slightly between CPU and GPU.

**Warning:** retaining or merging earlier GPU results requires the real `done.json`/predictions/checkpoints, not just the log's validation scores. Any missing earlier results must be reported as missing.
""")
    cells[2] = build_notebook.code("""from pathlib import Path
import os, sys, json, subprocess, zipfile, shutil, base64
import pandas as pd

ROOT = Path('/kaggle/working/TALLIP_Clean_CPU')
ROOT.mkdir(parents=True, exist_ok=True)
INPUT_ROOT = Path('/kaggle/input')

# Only set explicit ZIP paths when Kaggle auto-discovery cannot find the files.
FAKENEWS_ZIP = ''
HATESPEECH_ZIP = ''
# Example: /kaggle/input/previous-tallip-output/TALLIP_Clean_Retrain/training
PREVIOUS_RESULTS_DIR = ''
CPU_MICRO_BATCH = 1
CPU_EVAL_BATCH = 4
CPU_THREADS = 4
CPU_CHECKPOINT_EVERY_UPDATES = 200
CPU_MAX_WALL_HOURS = 4.0

REMAINING_NINE = [
  f'{task}/{model}/seed_{seed}'
  for task, model in [('hatespeech','banglabert'), ('hatespeech','mbert'), ('fakenews','mbert')]
  for seed in (42, 52, 62)
]
assert INPUT_ROOT.exists(), 'Run on Kaggle with the two original datasets attached.'
""")
    # Change generated source paths to the CPU working directory.
    cells[3] = build_notebook.encoded_file_cell(
        ROOT / "src" / "clean_splits.py", "/kaggle/working/TALLIP_Clean_CPU/src/clean_splits.py")
    cells[4] = build_notebook.encoded_file_cell(
        ROOT / "src" / "train_eval.py", "/kaggle/working/TALLIP_Clean_CPU/src/train_eval.py")
    cells.insert(5, build_notebook.encoded_file_cell(
        ROOT / "src" / "cpu_train_eval.py", "/kaggle/working/TALLIP_Clean_CPU/src/cpu_train_eval.py"))
    # The original cells subsequently resolve ROOT rather than hard-coded paths.
    # [6] embedded near-screen zip, [7] attached input discovery, [8] cleaning.
    # Avoid retaining redundant 315+ MB ZIP copies in Kaggle output: the
    # original attached datasets under /kaggle/input remain intact.
    cells[8]["source"] += [
        "# Remove only locally repacked duplicate ZIPs; originals remain in Kaggle Input.\n",
        "for dataset_zip in (fake_zip, hate_zip):\n",
        "    if dataset_zip.parent == ROOT and dataset_zip.name.endswith('_original_input.zip'):\n",
        "        dataset_zip.unlink(missing_ok=True)\n",
        "        print('Removed redundant working copy:', dataset_zip.name)\n",
    ]
    cells[9] = build_notebook.code("""# Optionally restore the results of this *same cleaned-split* experiment.
training_dir = ROOT/'training'
old_manifest = None
if PREVIOUS_RESULTS_DIR:
    old = Path(PREVIOUS_RESULTS_DIR)
    assert old.is_dir(), f'Previous training folder missing: {old}'
    assert (old/'run_plan.json').exists() or (old/'cpu_run_plan.json').exists(), 'Not a prior TALLIP clean-training folder'
    shutil.copytree(old, training_dir, dirs_exist_ok=True)
    candidate = old.parent/'evidence'/'clean_split_manifest.json'
    if candidate.exists():
        old_manifest = candidate

plan_cmd = [sys.executable, str(ROOT/'src'/'cpu_train_eval.py'),
  '--clean-dir', str(ROOT/'cleaned_data'),
  '--manifest', str(ROOT/'evidence'/'clean_split_manifest.json'),
  '--out-dir', str(training_dir),
  '--micro-batch', str(CPU_MICRO_BATCH), '--eval-batch', str(CPU_EVAL_BATCH),
  '--threads', str(CPU_THREADS),
  '--checkpoint-every', str(CPU_CHECKPOINT_EVERY_UPDATES),
  '--max-wall-hours', str(CPU_MAX_WALL_HOURS),
  '--runs', *REMAINING_NINE]
if old_manifest:
    plan_cmd += ['--previous-manifest', str(old_manifest)]
subprocess.run(plan_cmd + ['--plan-only'], check=True)
print('CPU plan checks passed: exactly NINE remaining runs; existing three are NOT trained again.')
""")
    cells[10] = build_notebook.code("""# Actual CPU full-fine-tuning. No GPU is used or required.
# Stops automatically at the wall-time limit *after saving* a checkpoint.
# Multiple sessions are normally necessary. Do not expect nine runs in a day.
subprocess.run(plan_cmd, check=True)
print('CPU session finished or paused safely. Save Kaggle output before the next session.')
""")
    cells[11] = build_notebook.code("""# Report only legitimately completed runs.
from pathlib import Path
done_files = sorted((training_dir).glob('**/done.json'))
print('COMPLETE TRAINING RUNS PRESENT:', len(done_files))
for path in done_files:
    data = json.loads(path.read_text())
    print(path.relative_to(training_dir), data.get('status'),
          'test macro F1:', data.get('fixed_original_test',{}).get('macro_f1'))
print('Checkpointed/unfinished runs:')
for path in sorted(training_dir.glob('**/latest_epoch.pt')):
    print(path.relative_to(training_dir))
print('OUTPUT ROOT:', ROOT)
""")
    nb["metadata"]["tallip_experiment"] = "nine CPU full-finetuning runs; 3 prior GPU seeds excluded; mid-epoch resume"
    dst = ROOT / "notebooks" / "TALLIP_Remaining9_CPU_Resume_RunAll.ipynb"
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Built:", dst, "size:", dst.stat().st_size)


if __name__ == "__main__":
    main()
