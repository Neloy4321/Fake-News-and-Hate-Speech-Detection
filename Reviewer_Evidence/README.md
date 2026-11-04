# TALLIP-26-0521 — Reviewer Evidence

**Paper:** *Domain-Specific Pretraining Helps Where Context Is Long: An Explainable Transformer Framework for Bengali Fake News and Hate Speech Detection*.

This folder separates the **independent human annotation audit** from the **paired frozen-prediction sensitivity analysis**. The original model test-set scores must not be silently replaced with corrected-label audit subset scores.

| Folder | Reviewer material | Scope |
|---|---|---|
| [`FakeNews_360_Annotation/`](FakeNews_360_Annotation/) | 360-row XLSX with full article text, two original annotations and confidence ratings, 57 disputed disposition records, original evidence, 317 strict binary-eligible decisions and 43 held/excluded cases | Human-annotation/provenance supplement |
| [`Corrected_Label_Evaluation/`](Corrected_Label_Evaluation/) | Seven-tab XLSX, item-level IDs and six frozen run predictions, per-run paired metrics, bootstrap intervals, leakage flags, and machine-readable source/integrity files | Exploratory matched-test audit sensitivity; **not leakage-free** |

## Essential facts

- Human-annotated Bengali fake-news subset: **360** independent-audit IDs.
- Original annotations: Neloy Pramanik Supto and Sajib Biswas, each with 360 ordinal 1–5 confidence values.
- **303** original binary agreements; **57** disagreements reviewed and author-approved after AI-assisted proposals. This is **not** a claim of an independent third per-case human fact-check.
- **317** cases retained for strict author-approved binary sensitivity (**223 REAL, 94 FAKE**); **43** are conditional/unresolved/not-news and not forced to binary.
- Every audited ID was matched to the original saved test set by row ID, raw-text SHA-256 and original label; the six frozen predictions also matched.
- Reported exact test/train-or-validation overlap exists in the original frozen checkpoint evaluation. Accordingly, post-hoc sensitivity **is not a clean retrained held-out benchmark**.

## Reviewer-facing caution

The workbook with full article texts was furnished as author-supplied audit material. Those texts may be copyrighted or sensitive; confirm permission/ethics requirements for public distribution. Evidence URLs were preserved from the annotation records and were not all independently refreshed during this packaging pass.

For each result, read the scope/denominator notes in the Excel `Reviewer_Summary` and `Agreement_Audit` tabs, and the accompanying methodology report before citing numbers in the revised manuscript.
