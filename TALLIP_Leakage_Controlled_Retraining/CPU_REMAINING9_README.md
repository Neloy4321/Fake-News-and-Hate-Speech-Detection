# Remaining 9 TALLIP clean-split runs — CPU pathway

**This is genuine full Transformer fine-tuning, not a rapid CPU approximation.** The 75,711-item Fake News cleaned training set and 40,149-item Hate Speech cleaned training set are retained. No samples, epochs or sequence lengths are secretly reduced. Training all nine on an ordinary laptop CPU can take **many days or weeks**; the exact wall time is unknown until real hardware benchmarks are observed.

## What is and is not run

The original three completed **Fake News + BanglaBERT (42, 52, 62)** are **excluded**. The CPU code runs exactly:

| Task | Backbone | Seeds | Number of runs |
|---|---|---|---:|
| Hate Speech | BanglaBERT | 42, 52, 62 | 3 |
| Hate Speech | mBERT | 42, 52, 62 | 3 |
| Fake News | mBERT | 42, 52, 62 | 3 |
| **Total** | | | **9** |

The notebook runs Hate Speech before Fake News because the Hate Speech sequence length is 128 and the dataset is smaller. No third-party GPU access is required.

## Files

- `notebooks/TALLIP_Remaining9_CPU_Resume_RunAll.ipynb` — self-contained Kaggle CPU notebook. Requires original two datasets to be attached. Restores previous clean-model outputs from a previous Kaggle version only if `PREVIOUS_RESULTS_DIR` is explicitly set.
- `src/cpu_train_eval.py` — standalone CPU runner and safer resumable trainer. Uses actual `torch.device('cpu')`/CPU tensors. Full effective batch size 16 is reproduced using gradient accumulation from micro-batch 1. Epoch count 3, original tokenizer lengths 256/128, LR 5e-5, 500 warmup steps and identical published labels are preserved.
- The original `src/clean_splits.py` and `src/train_eval.py` are embedded in the CPU notebook and the clean input fingerprints are validated.
- `tests/test_cpu_planner.py` verifies that nine runs (not 12) are selected, deterministic sample order works, and repacked inputs cannot silently change cleaned data.

## How the pause/resume works

1. The CPU script saves model, AdamW optimizer, LR scheduler, epoch, the **next original-row offset**, RNG states, best checkpoint and full config in `latest_epoch.pt` (same name as the GPU run's epoch checkpoint). It uses atomic replacement and checkpoints every **200 optimizer updates** by default.
2. `CPU_MAX_WALL_HOURS = 4.0` in the notebook settings causes a graceful pause after approximately four hours of **CPU training execution**, writing a checkpoint. The notebook then exits the running cell normally, so an output version may be saved. A Kaggle notebook output version is still **not guaranteed** until you verify the files and save it.
3. A later CPU session must have the *prior saved `training/` directory attached* as input; then set `PREVIOUS_RESULTS_DIR` in the notebook settings. The code copies it into working storage before resuming.
4. Existing done files are skipped and incomplete CPU epochs restart at the checkpointed update, not at the beginning of the epoch. It can also resume a previously finished GPU epoch on CPU when the full `latest_epoch.pt` plus `best_checkpoint/` and the compatible prior manifest are supplied. **GPU-to-CPU dropout and shuffle sequences will differ**; that mixed-device run is no longer an exact continuation of an identical training trajectory, and the output declares the limitation.

## If using a Windows PC directly

The same script can be launched from a terminal, after preparing the cleaned CSVs with `src/clean_splits.py` and installing PyTorch CPU, Transformers, scikit-learn, pandas and the Bengali normalizer. Example:

```bash
python src/cpu_train_eval.py \
  --clean-dir ./outputs/cleaned_data \
  --manifest ./outputs/evidence/clean_split_manifest.json \
  --out-dir ./outputs/training \
  --micro-batch 1 \
  --threads 4 \
  --checkpoint-every 200 \
  --max-wall-hours 4
```

**Do not use 8 GB of RAM as a guarantee that a full mBERT training run will fit.** The model, optimizer state, gradients, CPU activation buffers and simultaneous CSVs can exhaust memory. Prefer a well-provisioned CPU instance, and begin with one Hate Speech run to measure real throughput.

## Reviewer-reporting rules

- Do not combine the old overlap-risk original runs with new clean-split performance as though they were all the same protocol.
- A three-seed group is complete only after all three new clean runs produce `done.json` and fixed-test prediction CSVs.
- Use `fixed_original_test` from actual completed run artifacts only. Validation macro-F1 from console logs is **not** a test result.
- `CPU` and `GPU` can lead to small non-identical floating-point/dropout trajectories even with the same reported hyperparameters; declare the execution platform and cross-device resume in the paper's reproducibility notes.
- If the prior three GPU results cannot be recovered, **do not invent or infer their test metrics** from training logs.
