# Supplementary figures — provenance and interpretation

S8 and S9 here were freshly rendered from saved converged Integrated Gradients (IG) and matching SHAP attributions, not copied from earlier four-way illustrative exports. They are *selected examples*, not population-level statistics.

- **S8 (Fake News)**: seed-42 BanglaBERT, source test rows 914 (TP), 4034 (TN), 3845 (FP). IG used 256 integration steps and absolute completeness residual < 0.05. SHAP targets the predicted class.
- **S9 (Hate Speech)**: seed-42 BanglaBERT, source test rows 1395 (TP), 526 (TN), 4759 (FP). IG used 256 integration steps and absolute completeness residual < 0.05. SHAP targets the predicted class.
- In both figures, positive/negative scores support/oppose the model's *predicted class*. IG scores are logit-scale and SHAP scores probability-scale; the two panels use independent horizontal scales, so bar lengths must **not** be compared across methods.
- TP/TN/FP are based on original source test labels, not adjudicated corrected gold labels. No false-negative case in the saved 20-case cohort was added or fabricated.
- Figure PDFs contain embedded high-resolution raster plots. Original selected source materials and numerical traces were retained in the local experiment evidence, not published in this public repository.

Other supplementary figures should be interpreted under their individual limitations, and provisional label-audit visualizations are excluded.
