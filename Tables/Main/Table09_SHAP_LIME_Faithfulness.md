# Table 9: Predicted-class SHAP and LIME perturbation results at k=5 for selected BanglaBERT seed-42 examples.

| Task | Method | n | Comp. drop | Suff. gap | Comp.−random | Bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Fake News | SHAP | 20 | 0.466 | 0.118 | 0.239 | [0.096, 0.396] |
| Fake News | LIME | 20 | 0.434 | 0.287 | 0.207 | [0.042, 0.385] |
| Hate Speech | SHAP | 20 | 0.570 | 0.125 | 0.215 | [0.077, 0.366] |
| Hate Speech | LIME | 20 | 0.611 | 0.108 | 0.256 | [0.112, 0.412] |

**Notes.** Comprehensiveness measures the change in predicted-class probability after masking method-selected top-k tokens; the sufficiency gap measures the drop when retaining only them. Positive comp-minus-random is a comparative perturbation probe, not proof of semantic causality. These are 20 selected examples per task from a later measured protocol and must NOT be conflated with original manuscript n=141/200.
