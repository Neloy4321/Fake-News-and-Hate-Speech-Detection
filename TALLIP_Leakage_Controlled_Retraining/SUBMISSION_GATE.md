# ACM TALLIP-26-0521 — Leakage-controlled evaluation submission gate

This directory was prepared for reviewer revision of **“Domain-Specific Pretraining Helps Where Context Is Long: An Explainable Transformer Framework for Bengali Fake News and Hate Speech Detection.”**

## Evidence classification

| Claim | Current evidence | Status |
|---|---|---|
| Original train/validation/test CSVs reproduced | Local original data ZIPs; SHA-256 in `clean_split_manifest.json` | Verified |
| Exact normalized text absent across cleaned partitions | Independently recomputed from original CSV; hard assertion, per-row exclusion CSV | Verified |
| Prior test-near-candidate pairs removed from earlier partitions | Prior SimHash candidate list, rescored independently from raw CSV. Any score discrepancies are explicitly reported in the manifest. | Verified for **listed row pairs** only |
| All possible near duplicates removed | Prior candidate screen is non-exhaustive | **Not established** |
| Original test text, labels, and row order unchanged | Source test DataFrame retained without dropping or relabeling rows; normalized-test SHA-256 and label counts in manifest. CSV serialization may reformat auxiliary numeric fields. | Verified |
| New BanglaBERT/mBERT model weights trained on clean data | GPU required; do not reuse prior overlapping-data checkpoints | **Pending GPU training** |
| New macro-F1, accuracy, confidence intervals, confusion matrices | Created only from `done.json` files for completed base-model runs | **Pending GPU training** |
| Exact match of all original training library versions | Hyperparameters copied from seed-42 archives, but original training script and environment differ | **Not established** |
| No contamination from model pretraining corpora | Original pretrained models' corpora not audited | **Not established** |
| Independent human correction of all test labels | Audited samples only, not the full test set | **Not established** |

## Conditions before any reviewer letter claims fresh leakage-controlled results

1. Run the self-contained GPU notebook on both original datasets with seeds **42, 52, 62**, both models, both tasks. This means 12 newly trained model runs from their **original released base checkpoints**, not the previous 12 old fine-tuned weights.
2. Ensure **12 `done.json`** exist; check each includes `best_checkpoint_safetensors_sha256`, a `manifest_sha256` matching the final clean manifest, the best validation epoch, and fixed-test predictions.
3. Compare `run_plan.json` to the original manuscript hyperparameters: Fake News max 256 subwords, Hate Speech max 128; 3 epochs, batch 16, LR 5e-5, warmup 500, no class weighting. Record any environment-specific differences.
4. Independently recompute a confusion matrix and Accuracy/Macro-F1 from the individual `fixed_test_predictions.csv` files; verify the group aggregates (n=3 per task/model) before producing any manuscript table.
5. Update the reviewer response with the precise limitations: **exact cross-split isolation, known-near-candidate exclusion, non-exhaustive near screening, fixed published test labels**. A blanket claim “no data leakage” is stronger than the current evidence.
6. If the paper claims domain-specific *continued pretraining*, provide an actual pretraining-stage checkpoint and logs. Standard supervised fine-tuning of `csebuetnlp/banglabert` alone does not establish that claim.
7. Only after these gates pass should the revised manuscript's headline scores be replaced with the clean-split run results. The old scores remain **contamination-risk baseline only**.

## Extra precautions

- Do not publish the original datasets or large model weights in this **public** GitHub repository without reviewing their licenses and any sensitive content.
- The status string `SPLIT_PREPARATION_VERIFIED__GPU_TRAINING_PENDING` is deliberate. No metric was synthesized or substituted from older runs.
- Reviewers may question whether removing training examples matching a fixed test set constitutes post-hoc test-informed cleaning. State this transparently; a **new independently constructed split and an external hold-out set** would give a stronger estimate of generalization.
- If a GPU session expires, bring forward only the **same cleaned experiment's** `training/` results and resume the saved epochs. Never mix old checkpoint artifacts with this fresh protocol.
