# Table 8: BanglaBERT minus mBERT test accuracy differences by original text length, averaged across seeds.

| Task | Length stratum (words) | n | Accuracy gap mean ± SD (pp) |
| --- | --- | --- | --- |
| Fake News | short: ≤121 | 3,205 | 7.51 ± 1.30 |
| Fake News | medium: 122–220 | 3,133 | 8.06 ± 1.24 |
| Fake News | long: >220 | 3,151 | 6.52 ± 0.94 |
| Hate Speech | short: ≤6 | 1,900 | 1.81 ± 0.57 |
| Hate Speech | medium: 7–12 | 1,539 | 1.88 ± 0.19 |
| Hate Speech | long: >12 | 1,590 | 1.68 ± 0.92 |

**Notes.** Source-label original full-test predictions, whitespace words (not transformer tokens); three-seed sample SD. The Fake News advantage is not monotonically increasing toward the longest tercile. This is observational and does not establish domain-pretraining or full-context causality.
