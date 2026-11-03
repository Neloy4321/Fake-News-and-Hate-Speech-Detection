# Supplementary Table S4: Integrated Gradients selected top-k token deletion compared with equal-budget random masking on model-visible windows.

| Task | n | Full k=5 / adjusted | IG mask drop mean | Random drop mean | Difference mean | Difference median |
| --- | --- | --- | --- | --- | --- | --- |
| Fake News | 20 | 20 / 0 | 0.427 | 0.227 | 0.200 | 0.0020 |
| Hate Speech | 20 | 16 / 4 | 0.580 | 0.370 | 0.211 | 0.0001 |

**Notes.** These are IG-selected perturbation probes from the separately validated 20+20 seed-42 XAI cohort. Five deterministic random draw sets per item. Fake News 12/20 samples exceeded the 256-subword window; attribution only covers the visible first window. Strong means but near-zero medians require cautious interpretation.
