# TALLIP-26-0521 — Corrected-label evaluation and leakage limitations

**Status:** Frozen-prediction, matched audit-sample sensitivity completed. **NOT** a leakage-free, corrected-ground-truth full-test benchmark. No model retraining or new per-case fact check was performed.

## Alignment and cohort

- Original Bengali Fake News test split: **9,489** rows, with frozen predictions for six runs (BanglaBERT and mBERT, seeds 42/52/62).
- All **360/360** annotation IDs matched by original zero-based test-row ID, raw-text SHA-256 and original source label. All six saved prediction vectors agreed with the full-test frozen file for these IDs.
- Strict author-approved binary label gate: **317** rows (**223 REAL / 94 FAKE**). Remaining **43**: 5 conditional central-claim cases, 13 hold/unresolved, 25 not-news. Conditional suggestions are not strict full-article labels.
- Eligible split: **266 random-stratified** (259 two-human agreements + 7 author-approved adjudications), **51 targeted** error cases (44 consensus + 7 author-approved adjudications). Total original two-human agreements **303**, eligible author-approved adjudications **14**.
- Corrected vs original source label flips on the 317 eligible: **73**.

## Main reviewer-facing result — matched, weighted random panel only

Original and corrected metrics use the **same 266 frozen prediction rows**. Weights reflect the original label-stratified random sampling. Accuracy and macro-F1 are conditional on being binary eligible; this is an exploratory sensitivity estimate, not a population-wide corrected test score.

| Model | Seed | Original acc % | Corrected acc % | Δ acc (pp) | Original macro-F1 | Corrected macro-F1 | Stratified bootstrap Δ acc 95% interval (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| BanglaBERT | 42 | 93.14 | 87.77 | -5.36 | 0.924 | 0.848 | [-8.95, -1.79] |
| BanglaBERT | 52 | 91.87 | 86.51 | -5.36 | 0.910 | 0.834 | [-8.94, -1.79] |
| BanglaBERT | 62 | 92.39 | 84.47 | -7.92 | 0.917 | 0.811 | [-11.48, -4.58] |
| MBERT | 42 | 85.76 | 82.95 | -2.80 | 0.839 | 0.782 | [-6.62, +1.02] |
| MBERT | 52 | 84.49 | 85.25 | +0.76 | 0.824 | 0.811 | [-2.81, +4.57] |
| MBERT | 62 | 85.00 | 84.23 | -0.77 | 0.831 | 0.799 | [-4.60, +2.82] |

Bootstrap: 2,000 resamples within original-label strata, fixed random seeds, design-weighted accuracy change on the eligible random sample. Interval excludes uncertainty from adjudication quality, label ambiguity, train-seed sampling and 43 exclusions.

## Leakage and overlap — critical limits

- Previous audit flagged **48** original test rows with exact train/validation overlap; **1** are in the 317 strict binary-eligible rows.
- Non-exhaustive SimHash screen reported **174** test-row near-duplicate candidates (unconfirmed); **4** are in the 317.
- These are **post-hoc** audit flags. Reusing originally trained checkpoints does **not** remove train/test contamination; drop-overlap sensitivity without retraining is **not** leakage-free generalization.
- For a clean benchmark, deduplicate/exclude before training, re-train the intended model checkpoints and re-evaluate a disjoint held-out test set.

## Methodology and disclosures

1. Verify 360 audited IDs against the held-out test rows and full-test SHA-256 fingerprints. Recheck original labels and all frozen model predictions.
2. Carry over original annotator labels, independent confidence ratings and the final 360-row author-approved disposition; never create a label for the 43 held/excluded cases.
3. Compute paired original-label and corrected-label Accuracy, precision, recall, macro-F1, weighted-F1 and 2×2 confusion matrices on identical row sets.
4. Report the 317-case pooled **descriptive** results and the targeted 51-case results separately; targeted false-positive/false-negative enrichment precludes population inference from a pooled, unweighted result.
5. Report the weighted random 266-case comparison as a **conditional sensitivity** with stratified bootstrap intervals. Exclusion of uncertain items may bias it.
6. Distinguish the 9,489-case original-label full-test benchmark from any human-label sensitivity; retain the benchmark unchanged.

**Adjudication provenance:** 57 disagreements were handled by an AI-assisted review and author-approved dispositions. Fourteen yielded eligible binary decisions. Author acceptance is not a third independent blind human fact-check. The original two annotators' independent scores are preserved in the separate 360-row reviewer annotation audit.

**Mapping:** REAL=0, FAKE=1 for this dataset. Prediction columns ending `Class1_Prob` predict `P(FAKE)`.

**Suggested manuscript-safe claim:** “Frozen predictions were aligned to all 360 independently annotated Bengali fake-news test items through test-row IDs and text fingerprints. In a strict subset of 317 author-approved binary-eligible items, we evaluated paired label sensitivity; an inverse-inclusion-weighted random-panel estimate used 266 eligible cases. We keep this analysis separate from originally reported full-test metrics because adjudication exclusions, targeted sampling and detected train/test overlap prevent claims of corrected full-test or leakage-free performance.”

## Files

- `TALLIP_FN_Corrected_Label_Evaluation_Reviewer_Audit_2026-11-04.xlsx`: readable workbook with analysis scope, full per-run results, all 317 IDs, 43 exceptions, leakage flags, and integrity evidence.
- `all_scope_metrics.csv`: detailed metrics across all cohorts and runs.
- `paired_original_vs_corrected.csv`: original versus corrected paired deltas.
- `audit_317_test_alignment_predictions.csv`: ID/hash-verified case-level frozen predictions (article text omitted).
- `excluded_43_test_alignment.csv`: transparent holds and exclusions.
- `integrity_and_scope.json`: provenance, checksums, leakage summary and source methods.

**Submission gate:** Authors should verify the full original manuscript's evaluation-table numbers, model checkpoint hashes, reviewer requests, public redistribution rights, and exact/near-duplicate retraining implications before inserting the above as final journal claims.
