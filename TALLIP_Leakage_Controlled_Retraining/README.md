# TALLIP-26-0521 — Leakage-Controlled Retraining & Re-evaluation

**Stage:** ✅ Clean split preparation and CPU data-integrity gates; ⏳ 12 GPU transformer training/evaluation runs **NOT yet executed**.

This is a reproducible follow-up to the test-overlap findings from the prior reviewer-evidence audit. It provides **new, independently checked train/validation/test filtering**, a 12-run base-model retraining protocol, and an honest completion gate. It **does not** report new fake news or hate speech model scores before GPU execution.

## Repository contents

| Location | What it provides |
|---|---|
| `notebooks/TALLIP_Leakage_Controlled_12Run_RunAll.ipynb` | Self-contained Kaggle GPU notebook; original datasets + one Run All; sequential, epoch-resumable |
| `src/clean_splits.py` | Exact cross-split normalization and test-priority exclusion; known near candidates; input and clean CSV hashes; assertive audit |
| `src/train_eval.py` | From-base BanglaBERT/mBERT × Fake News/Hate Speech × seeds 42/52/62; best validation macro-F1 selection; fixed test evaluation and prediction CSVs |
| `src/build_notebook.py` | Regenerate the standalone notebook from the reviewed source code and embedded minimal overlap evidence |
| `evidence/clean_split_manifest.json` | Actual CPU-verified source hashes, counts, deletion reasons, zero exact cross-split overlap and residual limitations |
| `evidence/fakenews_removed_row_ids.csv`, `evidence/hatespeech_removed_row_ids.csv` | Every removed original zero-based source row and reason; no raw article/comment text |
| `evidence/prior_overlap_screen_minimal.zip` | Prior overlap evidence subset (non-exhaustive screening); small and text-free except IDs/metadata |
| `SUBMISSION_GATE.md` | What may and may not be stated to reviewers before new GPU outputs exist |
| `tests/test_cleaning.py` | Standalone Unicode/near-score unit tests |

**Original raw datasets and transformer weights are intentionally not published in this public repository.** Original data ZIPs and previous large checkpoints should be retained independently, subject to their license and research-data handling rules.

## Dataset and cleaning method

- Fake News: original processed split `text,label,source,word_count,char_count,bengali_ratio`.
- Hate Speech: original BD-SHS split `sentence,target,type,hate speech`.
- Match keys: Unicode NFKC → casefold → replace Unicode punctuation/symbol/control characters with spaces → collapse whitespace. Preserve the original **test** rows and labels exactly.
- Exclude earlier training entries overlapping test or validation; exclude validation entries overlapping test.
- Exclude the previously flagged **known** near-test candidate rows from earlier partitions (source screen used approximate SimHash candidates, Jaccard threshold 0.80 for Fake News, 0.85 for Hate Speech). Re-score against the original CSV; preserve and disclose candidate-score discrepancies rather than concealing them.
- Conservatively exclude prior flagged rows that are not independently reproducible as exact matches; list these IDs explicitly in the manifest.
- Remove punctuation-only/empty-normalized rows from training and validation, retain original test rows and record how many.
- Independently assert **zero exact normalized cross-split collisions** in the resulting data. This does **not** establish absence of all possible semantic or near-duplicate contamination.

## Exact training settings from the earlier archive metadata

| Parameter | Fake News | Hate Speech |
|---|---:|---:|
| Backbone | `csebuetnlp/banglabert` or `bert-base-multilingual-cased` | Same two backbones |
| Max subword length | 256 | 128 |
| Epochs | 3 | 3 |
| Train batch size | 16 | 16 |
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-5 | 5e-5 |
| Warmup steps | 500 | 500 |
| Class weights | None | None |
| Seeds | 42, 52, 62 | 42, 52, 62 |

This re-implementation uses the same **reported hyperparameters**, not a byte-for-byte re-execution of the old training notebook. It selects the best epoch **on validation macro-F1**, evaluates the unchanged original published-label test split only after training, and saves predictions, confusion matrices, accuracy, macro precision/recall/F1, SHA-256 checkpoint provenance, and mean ± sample SD (with a 3-seed t confidence interval only if all three finish).

## Fastest execution — Kaggle

1. Open `notebooks/TALLIP_Leakage_Controlled_12Run_RunAll.ipynb` in Kaggle, enable **GPU** + **Internet**, and attach the original two datasets. The near-overlap evidence and all runnable source code are embedded in the notebook; the original datasets are not.
2. Run All. It automatically verifies the data/split manifest, then starts the 12 requested experiments.
3. GPU time may exceed a Kaggle session. Each epoch saves `training/<task>/<model>/seed_<seed>/latest_epoch.pt`. At the next session, attach this notebook's previous `training/` directory and configure `PREVIOUS_RESULTS_DIR` in the settings cell. Do **not** use the old seed-42/52/62 contaminated-data checkpoints.
4. Collect all `done.json` and `fixed_test_predictions.csv`; inspect `training/aggregate_completed_only.json` only after **all 12** runs succeed.

## CPU-only split reproduction

From the project root, with the original three ZIP files available:

```bash
python src/clean_splits.py \
  --fakenews-zip '/path/to/Multi-Source Bengali Fake News Detection Benchmark.zip' \
  --hatespeech-zip '/path/to/BD-SHS.zip' \
  --evidence-zip evidence/prior_overlap_screen_minimal.zip \
  --out-dir outputs

python src/train_eval.py \
  --clean-dir outputs/cleaned_data \
  --manifest outputs/evidence/clean_split_manifest.json \
  --out-dir outputs/training \
  --plan-only
```

The clean `.csv.gz` outputs (large, raw text) are kept local to the execution environment, and **not** committed. The published GitHub evidence contains the verified aggregate manifest plus row-ID-only exclusion logs.

## Scientific interpretation

- Prior reported F1 values from checkpoints trained on the original potentially overlapping splits must **not** be relabeled as clean-split results.
- Excluding near matches found against the fixed test split is retrospective and may be criticized as a form of test-aware design. We disclose this explicitly; a separately collected external hold-out or new independently constructed split would be stronger.
- The approximate near-match screening may miss pairs. Document **exact-disjoint + known-near-filtered** instead of asserting “all leakage removed.”
- Fine-tuning a released BanglaBERT base model is **not evidence of new domain-specific continued pretraining**. The manuscript title/claims must be supported by separate actual pretraining experiments or revised.

See [`SUBMISSION_GATE.md`](SUBMISSION_GATE.md) for the reviewer-facing checklist before making numerical claims.
