# Supplementary figures — scope and cautions

## Included figures

| Figure | Content | Research scope |
|---|---|---|
| S1 | Multi-seed macro-F1 | Archived independent seeds 42, 52, 62, original test labels |
| S4 | Full-test length-stratified performance | Original frozen predictions; observational length strata |
| S6 | IG comprehensiveness versus random-token control | Saved perturbation evaluation; inspect test and selection protocol before making causal claims |
| S7 | IG comprehensiveness and sufficiency | Saved perturbation evaluation, not a ground-truth rationale measurement |
| S8 | New fake-news IG and SHAP token attributions | Three representative seed-42 source-labelled TP, TN, FP cases, 256-step converged IG |
| S9 | New hate-speech IG and SHAP token attributions | Three representative seed-42 source-labelled TP, TN, FP cases, 256-step converged IG |
| S10 | IG sensitivity at k=1,3,5,10 | Only four selected cases per task; **not** full-test aggregate |
| S11 | Long-article XAI by length | Selected 20 examples (4 short, 4 medium, 12 long); **not** population-level estimate |
| S12 | Long-news SHAP/LIME display | Two original long-news cases showing the model-visible 256-token prefix; **not** a full-article explanation |

## Corrected S8/S9 provenance

S8 and S9 were generated anew from archived converged IG numerical attributions and matching SHAP records; unlike earlier exported four-way panels, these selected IG examples each satisfy absolute completeness residual < 0.05. No unverified old S8/S9 panel was uploaded as the updated figures.

- S8 Fake News: seed42 source rows **914 TP**, **4034 TN**, **3845 FP**.
- S9 Hate Speech: seed42 source rows **1395 TP**, **526 TN**, **4759 FP**.
- All six IG plots use 256 integration steps and target the *predicted class*; paired SHAP plots likewise target the predicted class.
- Panels use separate scales: **IG in logit units**, **SHAP in probability units**. Magnitude across panels is not directly comparable.
- No FN item was present in the specific saved 20-example cohorts used for this updated comparison. No fabricated FN example was added.
- TP/TN/FP are against the original dataset annotations, not final human-adjudicated gold labels.
- PDFs for S8/S9 embed high-resolution raster charts (not editable vector paths).

## Deliberate exclusions

- S2 and S3 in the older package reproduce material already covered by main Figure 3 and Figure 4. They are omitted to avoid duplicate visual claims.
- S5 was explicitly labelled *provisional, not gold*. It is **not** appropriate as a visualization of the final adjudicated human-label audit and is omitted.
- The earlier S8 and S9 four-way exports had IG completeness concerns; replaced here with evidence-backed three-case comparisons rather than silently treating them as converged.

This public repository includes presentation figures and documentation only, **not raw annotation records, model weights, privately supplied source files, or reviewer correspondence**. Figure inclusion is not a claim of journal acceptance or full experimental reproducibility.
