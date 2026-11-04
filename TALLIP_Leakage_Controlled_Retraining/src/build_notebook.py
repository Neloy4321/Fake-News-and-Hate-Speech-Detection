#!/usr/bin/env python3
"""Mechanically package the tested Python sources into a self-contained Kaggle notebook."""

import base64
import json
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def encoded_file_cell(path: Path, destination: str) -> dict:
    compressed = base64.b64encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")
    return code(f"""# Source mirror of {path.name} (full readable script is in GitHub src/).
import base64, zlib
Path({destination!r}).parent.mkdir(parents=True, exist_ok=True)
Path({destination!r}).write_bytes(zlib.decompress(base64.b64decode({compressed!r})))
print('Extracted', {destination!r})
""")


def main() -> None:
    evidence_zip = ROOT / "evidence" / "prior_overlap_screen_minimal.zip"
    evidence_b64 = base64.b64encode(evidence_zip.read_bytes()).decode("ascii")
    cells = [
        markdown("""# ACM TALLIP-26-0521 — Leakage-Controlled Retraining: 12-Run GPU Notebook

**Status before running: no leakage-controlled Transformer metrics exist.** This notebook creates verified clean splits, then fine-tunes **from the original publicly released pretrained base models**: BanglaBERT and mBERT, each on Fake News and Hate Speech, with seeds **42, 52, 62** (12 runs).

The test CSV and its original labels are fixed and unaltered. Validation is used for checkpoint selection; test is evaluated after training. Known near-duplicate test candidates are filtered from earlier splits, but the near screen is **not exhaustive**; do not claim perfect isolation from all possible semantic overlap.

## Before Run All

1. Kaggle Notebook → **GPU accelerator** (prefer T4/P100), Internet **On**.
2. Attach these **original** Kaggle datasets: **Multi-Source Bengali Fake News Detection Benchmark** (processed train/val/test), **BD-SHS** (train/val/test). It supports either ZIP archives or unpacked dataset directories.
3. Run All. The script prints its independent overlap gates and starts 12 GPU runs. This could take **more than one Kaggle 12-hour session**.
4. Every completed epoch writes `latest_epoch.pt` and best model weights. To resume in a *new* session, attach the previous Kaggle output as an Input dataset and set `PREVIOUS_RESULTS_DIR` in the settings cell. **Never resume with earlier contaminated models.** Only use this notebook's previous `training/` directory.
5. After all runs, use `training/aggregate_completed_only.json`; each `done.json` and `fixed_test_predictions.csv` provides its own audit trail. Incomplete seeds are explicitly excluded from the aggregate.

Read `README.md` and `SUBMISSION_GATE.md` in the GitHub project before citing metrics. Scientific caveat: fine-tuning an off-the-shelf BanglaBERT base is **not** itself domain-specific *continued pretraining*.
"""),
        code("""# Environment dependencies (torch is preinstalled by Kaggle GPU images).\n%pip -q install 'transformers>=4.45,<5' 'scikit-learn>=1.3,<2' 'pandas>=2,<3' 'numpy>=1.24,<3' safetensors\n%pip -q install git+https://github.com/csebuetnlp/normalizer.git\n"""),
        code("""from pathlib import Path\nimport os, sys, json, subprocess, zipfile, shutil, base64\nimport pandas as pd\n\nROOT = Path('/kaggle/working/TALLIP_Clean_Retrain')\nROOT.mkdir(parents=True, exist_ok=True)\nINPUT_ROOT = Path('/kaggle/input')\n\n# Optional exact paths to raw ZIPs, if auto-discovery is ambiguous.\nFAKENEWS_ZIP = ''\nHATESPEECH_ZIP = ''\n# Optional path to the training directory from a previous output dataset (for resuming).\nPREVIOUS_RESULTS_DIR = ''\n# Optional run selection to fit GPU session limits. Default runs all 12 sequentially.\nTASKS = ['fakenews', 'hatespeech']\nMODELS = ['banglabert', 'mbert']\nSEEDS = [42, 52, 62]\n\nassert INPUT_ROOT.exists(), 'Run this notebook on Kaggle with the two input datasets attached.'\n"""),
        encoded_file_cell(ROOT / "src" / "clean_splits.py", "/kaggle/working/TALLIP_Clean_Retrain/src/clean_splits.py"),
        encoded_file_cell(ROOT / "src" / "train_eval.py", "/kaggle/working/TALLIP_Clean_Retrain/src/train_eval.py"),
        code(f"""# Pin original overlap evidence inside the notebook (non-exhaustive near screen).\nevidence_path = ROOT / 'prior_overlap_screen_minimal.zip'\nevidence_path.write_bytes(base64.b64decode({evidence_b64!r}))\nprint('Prior evidence:', evidence_path, evidence_path.stat().st_size, 'bytes')\n"""),
        code(textwrap.dedent("""
            # Resolve the two attached input datasets; verify expected schema before use.
            def find_zip(kind, explicit):
                if explicit:
                    target = Path(explicit)
                    assert target.is_file(), f'ZIP not found: {target}'
                    return target
                zips = sorted(INPUT_ROOT.rglob('*.zip'))
                if kind == 'fakenews':
                    matches = [p for p in zips if 'multi-source' in p.name.lower() and 'fake' in p.name.lower()]
                else:
                    matches = [p for p in zips if 'bd-shs' in p.name.lower() or 'bdshs' in p.name.lower()]
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    raise RuntimeError(f'Ambiguous {kind} ZIPs: {matches}; set the explicit ZIP path')

                matches = []
                for train in INPUT_ROOT.rglob('train.csv'):
                    folder = train.parent
                    if not (folder / 'val.csv').exists() or not (folder / 'test.csv').exists():
                        continue
                    header = pd.read_csv(train, nrows=1).columns
                    is_kind = ('text' in header and 'label' in header) if kind == 'fakenews' else ('sentence' in header and 'hate speech' in header)
                    if is_kind:
                        matches.append(folder)
                if len(matches) != 1:
                    raise RuntimeError(f'Expected one {kind} CSV directory, found: {matches}; specify path manually')
                folder = matches[0]
                archive = ROOT / (kind + '_original_input.zip')
                if not archive.exists():
                    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
                        for split in ('train', 'val', 'test'):
                            arc = ('data/processed/' if kind == 'fakenews' else '') + split + '.csv'
                            z.write(folder / (split + '.csv'), arc)
                return archive

            fake_zip = find_zip('fakenews', FAKENEWS_ZIP)
            hate_zip = find_zip('hatespeech', HATESPEECH_ZIP)
            print('Fake News source:', fake_zip)
            print('Hate Speech source:', hate_zip)
        """)),
        code("""# (Re)build clean splits from the original CSVs, independently verify overlap evidence.\nsubprocess.run([sys.executable, str(ROOT/'src'/'clean_splits.py'),\n    '--fakenews-zip', str(fake_zip), '--hatespeech-zip', str(hate_zip),\n    '--evidence-zip', str(evidence_path), '--out-dir', str(ROOT)], check=True)\nmanifest_path = ROOT/'evidence'/'clean_split_manifest.json'\nmanifest = json.loads(manifest_path.read_text())\nfor task, details in manifest['tasks'].items():\n    print(task, 'original', details['original_rows'], 'CLEAN', details['clean_rows'],\n          'exact overlap after cleaning:', details['post_clean_exact_cross_split_overlap'])\n"""),
        code("""# Optionally load ONLY prior outputs of THIS SAME cleaned-base training pipeline.\ntraining_dir = ROOT/'training'\nif PREVIOUS_RESULTS_DIR:\n    old = Path(PREVIOUS_RESULTS_DIR)\n    assert old.is_dir(), f'Prior training directory missing: {old}'\n    assert (old/'run_plan.json').exists(), 'Not a previous clean-training output; refusing to import'\n    shutil.copytree(old, training_dir, dirs_exist_ok=True)\n\nplan_cmd = [sys.executable, str(ROOT/'src'/'train_eval.py'),\n    '--clean-dir', str(ROOT/'cleaned_data'), '--manifest', str(manifest_path),\n    '--out-dir', str(training_dir), '--tasks', *TASKS, '--models', *MODELS,\n    '--seeds', *map(str, SEEDS)]\nsubprocess.run(plan_cmd + ['--plan-only'], check=True)\nprint('CPU validation gates passed. No training claimed yet.')\n"""),
        code("""# ACTUAL GPU RUNS — safe to rerun: verified done.json is skipped; latest_epoch.pt resumes.\n# This may exceed a single Kaggle session; save outputs after each session.\nimport torch\nassert torch.cuda.is_available(), 'GPU required; enable a Kaggle accelerator before starting.'\nsubprocess.run(plan_cmd, check=True)\nprint('GPU runs complete — inspect each done.json before using metrics in the manuscript.')\n"""),
        code("""# Audit/export after *completed* runs; absent runs have no invented scores.\ncompleted = training_dir/'aggregate_completed_only.json'\nif completed.exists():\n    aggregate = json.loads(completed.read_text())\n    print(json.dumps(aggregate, indent=2, ensure_ascii=False)[:10000])\nelse:\n    print('No completed GPU runs yet.')\nprint('RESULT ROOT:', ROOT)\n"""),
    ]

    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "tallip_experiment": "fresh base-model retraining on exact-disjoint known-near-filtered splits"},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = ROOT / "notebooks" / "TALLIP_Leakage_Controlled_12Run_RunAll.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Built:", destination, "size:", destination.stat().st_size)


if __name__ == "__main__":
    main()
