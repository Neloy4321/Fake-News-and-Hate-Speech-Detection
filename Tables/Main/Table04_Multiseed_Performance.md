# Table 4: Original-label held-out test performance of two encoders across three training seeds (mean ± sample standard deviation).

| Task | Model | n test | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | ROC AUC (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fake News | BanglaBERT | 9,489 | 91.910 ± 0.122 | 91.649 ± 0.114 | 90.054 ± 0.254 | 90.773 ± 0.159 | 96.885 ± 0.118 |
| Fake News | mBERT | 9,489 | 84.547 ± 1.037 | 83.875 ± 1.389 | 80.700 ± 1.050 | 81.919 ± 1.166 | 89.992 ± 1.815 |
| Hate Speech | BanglaBERT | 5,029 | 92.663 ± 0.259 | 92.686 ± 0.247 | 92.711 ± 0.188 | 92.657 ± 0.251 | 97.459 ± 0.208 |
| Hate Speech | mBERT | 5,029 | 90.873 ± 0.439 | 90.858 ± 0.442 | 90.861 ± 0.435 | 90.859 ± 0.439 | 96.194 ± 0.320 |

**Notes.** These are source-label measurements on archived test partitions; mean ± sample SD is computed across three seeds and is not a row-bootstrap confidence interval. Single-seed classical baselines from the original paper are not treated as new multi-seed runs.
