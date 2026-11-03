# Supplementary Table S3: Sensitivity of SHAP/LIME comprehensiveness advantage over random at varying token budgets.

| Task | Method | k | n | Comp.−random mean | Bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| Fake News | SHAP | 1 | 20 | 0.164 | [0.055, 0.298] |
| Fake News | SHAP | 3 | 20 | 0.177 | [0.062, 0.311] |
| Fake News | SHAP | 5 | 20 | 0.239 | [0.096, 0.396] |
| Fake News | SHAP | 10 | 20 | 0.197 | [0.066, 0.352] |
| Fake News | LIME | 1 | 20 | 0.039 | [-0.077, 0.177] |
| Fake News | LIME | 3 | 20 | 0.146 | [-0.032, 0.339] |
| Fake News | LIME | 5 | 20 | 0.207 | [0.042, 0.385] |
| Fake News | LIME | 10 | 20 | 0.189 | [0.059, 0.343] |
| Hate Speech | SHAP | 1 | 20 | 0.272 | [0.124, 0.419] |
| Hate Speech | SHAP | 3 | 20 | 0.273 | [0.124, 0.438] |
| Hate Speech | SHAP | 5 | 20 | 0.215 | [0.077, 0.366] |
| Hate Speech | SHAP | 10 | 20 | 0.078 | [0.013, 0.164] |
| Hate Speech | LIME | 1 | 20 | 0.314 | [0.165, 0.466] |
| Hate Speech | LIME | 3 | 20 | 0.277 | [0.126, 0.445] |
| Hate Speech | LIME | 5 | 20 | 0.256 | [0.112, 0.412] |
| Hate Speech | LIME | 10 | 20 | 0.077 | [0.013, 0.163] |

**Notes.** Token budgets k=1,3,5,10; n=20 cases per task/method. This table supports a sensitivity discussion around k=5, not post-hoc selection of the best k. Random token positions can introduce out-of-distribution perturbations.
