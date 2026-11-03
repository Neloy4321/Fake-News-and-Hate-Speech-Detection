# Evidence provenance and methodological cautions

## Provenance map

| Tables | Source archive / underlying evidence | Calculation |
| --- | --- | --- |
| Main 1–3 | `Fake_News(3).zip` → submitted `main.tex`; submitted proof `draft_Proof_hi(2).pdf` | Bibliographic condensation and source partition/configuration extraction; manuscript references must be reconciled |
| Main 4, Supplementary S1/S2 | `TALLIP_All_Reviewer_Figures_Complete_RunAll_Package(3).zip` → nested `TALLIP_Revision_Figure_Inputs.zip`, `per_seed_metrics.csv`, `model_mean_sd.csv`, `fulltest_predictions.csv` | 4 task–encoder cells × 3 seeds; sample SD; class metrics checked from TN/FP/FN/TP |
| Main 5 | Same `fulltest_predictions.csv` | Exact paired two-sided McNemar (binomial version) within task+seed; Holm adjustment of all 6 comparisons; paired row-bootstrap CI with 10,000 replicates from discordance categories |
| Main 6–7 | `TALLIP_26-0521_FourIssue_Evidence_RESULTS(1).zip` → human-consensus sensitivity and overlap summaries | Strict eligible consensus, 2-/3-rater agreement, inverse-inclusion-weighted audited-subset sensitivity and uncertainty |
| Main 8 | `length_comparisons.csv` inside nested revision input archive | Three-seed accuracy gaps by raw whitespace-word tercile |
| Main 9, Supplementary S3 | `TALLIP_26-0521_FourIssue_Evidence_RESULTS(1).zip` → `EVIDENCE/faithfulness_bootstrap_summary.csv` | Measured SHAP/LIME top-k deletion, sufficiency, matched random baseline, bootstrap intervals |
| Main 10 | `TALLIP_OptionB_FullArticle_RESULTS(1).zip` → `bootstrap_summary.csv`, `article_prediction_comparison.csv` | Bounded 12-document, 27-window mean-probability aggregation experiment |
| Supplementary S4 | `TALLIP_2Tasks_LongArticle_XAI_RandomBaseline_Sufficiency_Evidence(3).zip` → `summary.json` | Separate 20+20 IG top-k versus five matched random sets |

Sources were made available by the author in the revision workflow. This table folder does not redistribute source corpus text, patient data, individual annotations, or model binaries.

## What each comparison does and does not establish

**Main 4 — test-set performance:** Uses original dataset labels for 9,489 Fake News and 5,029 Hate Speech test rows. Accuracy, macro P/R/F1, and AUC are from archived predictions and runs for seeds 42/52/62, not an independently replicated multi-site study.

**Main 5 — matched comparison:** BanglaBERT and mBERT predictions are compared on the same test items and same seed. Two-sided exact McNemar is appropriate for paired binary correctness; multiple tests corrected using Holm. The row-bootstrap CI concerns held-out row variability, not variability across training seeds. A lower p value alone is not an effect-size or practical-significance claim.

**Main 6 — annotation reliability:** Hate Speech first-two-rater final post-discussion agreement is not a clean pre-adjudication independent IAA; show the independent three-rater snapshot separately. Source-label disagreement is *candidate review*, not ground-truth correction. For Fake News, 360 annotations were attempted, but the strictly paired binary subset is 315.

**Main 7 — label sensitivity:** Random-panel eligible case weighting and conditional evaluation only; targeted error-enriched records are excluded. No 14,518-row relabeled full-test file has been proven. Do not relabel a source-label confusion matrix as consensus-gold confusion matrix.

**Main 8 — length effect:** The longest Fake News tercile does **not** show an increasing advantage over the short/medium terciles. Do not use this table to prove the manuscript title's context-length causal hypothesis.

**Main 9 and S3/S4 — perturbation measures:** Comprehensiveness, sufficiency and randomized token masks probe a model's reliance under perturbed inputs; they do not establish explanations are human-grounded or causal. n=20 selected examples/task. Some long articles exceed the encoder's visible 256-token window.

**Main 10 — full-article Option B:** Only a 12-article auxiliary mean-probability sliding-window aggregation experiment. 27 windows and 1 original-vs-auxiliary prediction change. SHAP k=5 comprehensiveness-minus-random 95% interval crosses zero, so no general positive-significance claim for SHAP.

## Deliberately not invented

- Multiseed scores for TF-IDF+SVM, XGBoost and other classical baselines (the old manuscript has different single-seed outputs).
- Fully audited corrected-label full-test predictions, confusion matrices and corrected-label ROC curves (none was established by these bounded audits).
- Comparative GPU latency, peak memory or carbon outputs without parallel timing/hardware measurements.
- Unverified full split de-duplication, training reproducibility beyond archived logs, or new external fact-checking of every source.

## Revision integration checklist

- [ ] Resolve class-imbalance RQ: produce correctly controlled revised ablation or remove the unsupported claim.
- [ ] Validate hardware, optimizer, warmup and mixed precision against original training logs.
- [ ] Re-check all 8 literature descriptors, citation keys and updated numbering.
- [ ] Replace original table cross-references and cite S1–S4 as appropriate.
- [ ] Discuss selection, sample size, confidence intervals and noncausality accurately.
- [ ] Align figure/table captions, reviewer response letter and revised title/abstract.
