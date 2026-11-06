# TALLIP-26-0521 — Archived experiment evidence and provenance

**Scope:** Evidence exported from the authors' saved experiments and revision audits, organized on 7 November 2026. This index describes what is present and what is **not** independently verified. Do not interpret an archive name, completion marker, hash, or numerical score as standalone proof of the historical training environment.

## 1. Original held-out evaluation: P1, 12 model/seed/task runs

- [Per-seed results and aggregate analysis](P1_Original12/aggregate/results/) include `per_seed_metrics.csv`, model means/SD, paired tests, figures, and `analysis_manifest.json`.
- [Individual readable run records](P1_Original12/per_run/) include **12** `training_history.json` files, corresponding metrics and data audits, and the historical `COMPLETE.json` markers.
- [Ten original archives](P1_Original12/archived_run_bundles/) cover the twelve runs; one archive contains all three Hate Speech mBERT seeds. The individual archives retain numerical test predictions (`test_predictions.npz`).
- [Statistical analysis code](P1_Original12/reproduction_code/src/) is preserved from the same archived package. Rerunning aggregate analysis requires checking its local paths and package versions; it does not recreate missing model checkpoints.

P1 uses the **original source labels and partition**. The historical single-seed Fake News mBERT **87.765%** result and the later P1 seed-42 **84.108%** result have different, unresolved run provenance. The same integer seed alone does not establish checkpoint, code or dataset identity.

## 2. Cleaned-split sensitivity: P2 V1, 12 reported runs, seven detailed CUDA archives

- The separately recorded P2 V1 outcomes comprise twelve task/model/seed Accuracy values, documented in Supplementary Table S6; do **not** merge them statistically with P1.
- [Seven individual CUDA run records](P2_V1_Cleaned12/seven_cuda_run_records/) expose `done.json` with configuration, declared checkpoint hash, device and validation information, and `training_history.json`.
- [Seven downloadable run bundles](P2_V1_Cleaned12/archived_run_bundles/) additionally contain the original **numeric-only** fixed-test prediction CSVs (row identifier, source label, predicted label and class probabilities). No raw article/comment text is included in those CSVs.
- [P2 manifest and audit](P2_V1_Cleaned12/) provides cleaned-split counts, original archive checksum declaration, GPU plan and the completed-account summary.

The seven run-level records present here are:

| Task | Encoder | Seed(s) | Detailed record availability |
| --- | --- | --- | --- |
| Fake News | mBERT | 42, 52, 62 | Three archived CUDA logs + predictions |
| Hate Speech | mBERT | 42, 52, 62 | Three archived CUDA logs + predictions |
| Hate Speech | BanglaBERT | 62 | One archived CUDA log + predictions |
| Fake News | BanglaBERT | 42, 52, 62 | Reported results; **no equally complete run archive here** |
| Hate Speech | BanglaBERT | 42, 52 | Reported results; **no equally complete run archive here** |

The other five P2 outcomes were supplied in separate evidence forms during the revision process, without a comparable complete checkpoint/run-history bundle in this available archive. **Do not claim twelve independently verified run logs.** Declared checkpoint digests were not verified by recomputing them from all original `.safetensors` binaries.

[Prediction verification](P2_V1_Cleaned12/prediction_verification.csv) records independently recomputed Accuracy, Macro-F1 and confusion matrices from all seven saved prediction CSVs. All matched their archived completion JSON. This verifies internal agreement on saved source labels.

## 3. Dataset overlap / label checks

[Leakage_Audit](Leakage_Audit/) contains selected recorded exact-overlap and lexical near-match audits. P2 recorded zero normalized exact cross-partition matches. Nevertheless, the Fake News audit found **2,542** train–test near-match pairs at five-word-shingle Jaccard ≥ 0.80, involving **1,981 of 9,489** held-out items (20.88%). Semantic paraphrases, common source/event shortcuts, and complete source-held-out generalization remain untested.

The annotation and corrected-label evidence described in the [supplementary tables](Manuscript_Assets/Tables/Supplementary/) is limited to reviewed cohorts; it is not a full corrected test-label gold standard. Human post-discussion agreement must not be represented as independent blind initial-pass reliability.

## 4. Explanation checks

[Explainability](Explainability/) includes four-case IG convergence, sampled faithfulness bootstrap summary, checkpoint-selection metadata and the 12-article full-article auxiliary summary. These are **sample-restricted** analyses. The Option B sliding-window system is not the original 256-subword classifier running natively on entire documents. P1 prediction checkpoints and separate XAI checkpoints/tokenizers have not been proven byte-identical.

## 5. Manuscript assets

- [Main figure PDFs](Manuscript_Assets/Main_Figures/) include the revised workflow and Figures 7–8.
- [Supplementary figure PDFs](Manuscript_Assets/Supplementary_Figures/) include S2, S3, and S5.
- [Main table sources](Manuscript_Assets/Tables/Main/) and [supplementary table sources](Manuscript_Assets/Tables/Supplementary/) accompany the evidence.

## 6. How to inspect / verify

1. Download an archive from [P1 original run bundles](P1_Original12/archived_run_bundles/) or [P2 seven-run bundles](P2_V1_Cleaned12/archived_run_bundles/).
2. Compare its SHA-256 against [SHA256SUMS.txt](SHA256SUMS.txt), using `sha256sum` (Linux/macOS) or `Get-FileHash -Algorithm SHA256` (PowerShell).
3. Read the run's `metrics.json` / `done.json` and `training_history.json`. For P2, match each `original_test_row_id` and probability columns to the frozen test partition before computing metrics; labels use the **published source labels**.
4. Inspect `P1_Original12/aggregate/results/per_seed_metrics.csv` separately from the P2 summary. Verify claimed model comparisons from their correct series, sample sizes and label definitions.
5. Do not infer independent replay, code-version identity, checkpoint byte identity, full semantic leakage elimination or a full corrected-label gold standard from these documents alone.

## 7. Disclosure and limits

Files were screened for obvious credential patterns before inclusion. Raw training texts, Kaggle session secrets, private reviewer correspondence, and large model checkpoints are intentionally excluded. This is a **public** GitHub repository; users should independently confirm dataset redistribution permissions before distributing any underlying raw corpora. The repository shows reported runs and preserves underlying numerical records; it does not guarantee that every experiment can be rerun exactly with the same environment and weights.
