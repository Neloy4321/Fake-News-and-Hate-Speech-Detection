# ACM TALLIP revision: manuscript and supplementary tables

Paper **TALLIP-26-0521** — *Domain-Specific Pretraining Helps Where Context Is Long: An Explainable Transformer Framework for Bengali Fake News and Hate Speech Detection.*

**Delivery:** 10 proposed main-text tables and 4 supplementary tables (14 tables; each has a LaTeX source and a GitHub-readable Markdown preview). Created from the archived submitted manuscript, original-label predictions, multiple completed training seeds, and reviewer-requested bounded new experiments. These are **revision drafts**, not a claim that the journal has accepted the revised analysis.

## Main manuscript tables (10)

| ID | Caption / scope | Files |
| --- | --- | --- |
| Table 1 | Selected research relevant to Bengali language models, detection tasks, and explanation methods (condensed from 30 studies in the submitted manuscript). | [LaTeX](Main/Table01_Related_Work_Condensed.tex) · [Preview](Main/Table01_Related_Work_Condensed.md) |
| Table 2 | Original released source-partition statistics for the fake-news and BD-SHS corpora. | [LaTeX](Main/Table02_Dataset_Partitions.tex) · [Preview](Main/Table02_Dataset_Partitions.md) |
| Table 3 | Reported transformer fine-tuning configuration and verified training-seed scope. | [LaTeX](Main/Table03_Fine_Tuning_Configuration.tex) · [Preview](Main/Table03_Fine_Tuning_Configuration.md) |
| Table 4 | Original-label held-out test performance of two encoders across three training seeds (mean ± sample standard deviation). | [LaTeX](Main/Table04_Multiseed_Performance.tex) · [Preview](Main/Table04_Multiseed_Performance.md) |
| Table 5 | Paired BanglaBERT versus mBERT test differences for each seed, based on the same original-label held-out rows. | [LaTeX](Main/Table05_Paired_McNemar.tex) · [Preview](Main/Table05_Paired_McNemar.md) |
| Table 6 | Annotation consistency and candidate source-label disagreements on independently selected audit items. | [LaTeX](Main/Table06_Human_Label_Audit.tex) · [Preview](Main/Table06_Human_Label_Audit.md) |
| Table 7 | BanglaBERT seed-42 agreement-restricted random-panel performance against original labels and strict two-annotator consensus. | [LaTeX](Main/Table07_Audited_Label_Sensitivity.tex) · [Preview](Main/Table07_Audited_Label_Sensitivity.md) |
| Table 8 | BanglaBERT minus mBERT test accuracy differences by original text length, averaged across seeds. | [LaTeX](Main/Table08_Length_Strata.tex) · [Preview](Main/Table08_Length_Strata.md) |
| Table 9 | Predicted-class SHAP and LIME perturbation results at k=5 for selected BanglaBERT seed-42 examples. | [LaTeX](Main/Table09_SHAP_LIME_Faithfulness.tex) · [Preview](Main/Table09_SHAP_LIME_Faithfulness.md) |
| Table 10 | Selected twelve-article auxiliary sliding-window experiment (mean-probability aggregation) compared with random equal-budget token selection. | [LaTeX](Main/Table10_OptionB_Full_Article.tex) · [Preview](Main/Table10_OptionB_Full_Article.md) |

## Supplementary tables (4)

| ID | Caption / scope | Files |
| --- | --- | --- |
| Table S1 | Complete per-seed source-label test metrics for the four task–encoder conditions. | [LaTeX](Supplementary/TableS01_All_Seed_Results.tex) · [Preview](Supplementary/TableS01_All_Seed_Results.md) |
| Table S2 | Per-class original-label precision, recall, F1 and class support for seed-42 transformer predictions. | [LaTeX](Supplementary/TableS02_Seed42_Per_Class.tex) · [Preview](Supplementary/TableS02_Seed42_Per_Class.md) |
| Table S3 | Sensitivity of SHAP/LIME comprehensiveness advantage over random at varying token budgets. | [LaTeX](Supplementary/TableS03_Faithfulness_K_Sweep.tex) · [Preview](Supplementary/TableS03_Faithfulness_K_Sweep.md) |
| Table S4 | Integrated Gradients selected top-k token deletion compared with equal-budget random masking on model-visible windows. | [LaTeX](Supplementary/TableS04_IG_Random_Baseline.tex) · [Preview](Supplementary/TableS04_IG_Random_Baseline.md) |

## How to use

1. Keep **Main** tables in revised manuscript order (Table 1 through Table 10), and cite supplementary tables as **Table S1** through **Table S4**.
2. The TeX tables use `booktabs` and `adjustbox`. Ensure the preamble includes `\usepackage{booktabs}` and `\usepackage{adjustbox}`.
3. Include a table with `\input{Tables/Main/Table04_Multiseed_Performance.tex}` (adjust relative path to the manuscript layout). Place one `\input{}` at the relevant part of the text; do not add a second surrounding `table` environment.
4. Reconcile legacy `\label{}` references before replacement, because table numbering and which legacy analyses remain have changed.
5. Read [PROVENANCE_AND_CAVEATS.md](PROVENANCE_AND_CAVEATS.md) before final submission.

## Original manuscript mapping and editorial decisions

The submitted `main.tex` contains **9** explicit `table` environments; the journal proof PDF contains **10** (including a computational-efficiency table absent from the current LaTeX export). These should not be interpreted as 10 scientifically valid revised numeric tables. The present 10 are a *new proposed revision arrangement*:

| Old submitted/proof table | Revised placement |
| --- | --- |
| 1 — 30-study related-work review | Main Table 1; 8 selected studies, no unsupported cross-study score comparisons |
| 2 — Dataset breakdown | Main Table 2 |
| 3 — Fine-tuning hyperparameters | Main Table 3; training parameters flagged for log verification |
| 4 — Original model performance | Main Table 4 (new 3-seed BanglaBERT/mBERT); legacy classical baselines are **not** mislabeled as 3-seed experiments |
| 5 — Per-class performance | Supplementary Table S2 (archived seed-42 checks) |
| 6 — Original statistical testing | Main Table 5 (6 new paired tests, Holm-adjusted) |
| 7 — Class-imbalance ablation | **Not certified from revised source evidence**; omitted pending verification |
| 8 — Computational efficiency (proof only) | **Not certified from comparable timing logs**; omitted pending verification |
| 9 — Original XAI quantitative analysis | Main Table 9, Main Table 10, Supplementary S3–S4, each explicit about measured n and protocol |
| 10 — Error examples | Discuss qualitatively only after case/label verification; do not publish user audit text from an unreviewed source |

Additional reviewer-driven material appears in Main Tables 6–8 and 10 and Supplementary Table S1.

## Unresolved submission checks

- A strong claim that pretraining benefits **grow with context length** is **not supported** by the current length-stratum results; revise title/abstract/conclusion or provide a valid additional test.
- The audited-label evaluation is limited to selected **random audited subsets**, **not** a corrected-label full test set. Do not claim corrected full-test performance.
- The historical split is **not independently certified leakage-free** in these table artifacts.
- Paper-method hyperparameters (notably scheduler/precision) are transcribed from submitted documentation and require final training-log validation.
- Main Table 10 concerns a **new auxiliary sliding-window aggregator**, not the old single-window encoder.
- Some explanation probes are based on selected n=20 cases/task, and the full-article prototype has only 12 articles.
- The public repository contains **aggregate tabular metrics only**; no raw potentially sensitive audit text, private news evidence URLs, or model checkpoints were added.

_Last curated: 2026-11-03._
