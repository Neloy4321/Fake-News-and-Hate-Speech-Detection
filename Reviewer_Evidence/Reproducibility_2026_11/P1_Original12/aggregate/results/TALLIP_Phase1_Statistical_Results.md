# ACM TALLIP-26-0521 — Phase 1 statistical analysis

All 12 model-seed experiments were **recomputed from genuine archived test predictions**.
No model retraining, data relabeling or invented measurements were performed.

## Verified split populations

| Dataset | Train before/after overlap filtering | Validation | Locked test |
|---|---:|---:|---:|
| Fake News | 75,905 / 75,902 | 9,489 | 9,489 |
| Hate Speech | 40,224 / 40,216 | 5,028 | 5,029 |

Across seeds and models within each dataset, test labels, row IDs and processed-text
audit hashes match. All original test rows were retained.

## Per-model test score (mean ± sample SD in percentage points)

| Dataset | Model | Accuracy % | Macro Precision % | Macro Recall % | Macro F1 % | ROC-AUC % |
|---|---|---:|---:|---:|---:|---:|
| fakenews | banglabert | 91.910 ± 0.122 | 91.649 ± 0.114 | 90.054 ± 0.254 | 90.773 ± 0.159 | 96.885 ± 0.118 |
| fakenews | mbert | 84.547 ± 1.037 | 83.875 ± 1.389 | 80.700 ± 1.050 | 81.919 ± 1.166 | 89.992 ± 1.815 |
| hatespeech | banglabert | 92.663 ± 0.259 | 92.686 ± 0.247 | 92.711 ± 0.188 | 92.657 ± 0.251 | 97.459 ± 0.208 |
| hatespeech | mbert | 90.873 ± 0.439 | 90.858 ± 0.442 | 90.861 ± 0.435 | 90.859 ± 0.439 | 96.194 ± 0.320 |

## Paired BanglaBERT versus mBERT on the same test instances

Positive deltas favor BanglaBERT. McNemar p values are exact two-sided
binomial p values based on discordant paired correctness (an accuracy question).
Holm correction spans all six dataset×seed comparisons.
Macro-F1 and accuracy delta CIs are 95% paired bootstrap intervals (10,000 resamples),
**not** across-seed confidence intervals.

| Dataset | Seed | ΔAccuracy pp [95% CI] | ΔMacro-F1 pp [95% CI] | McNemar p (Holm) |
|---|---:|---|---|---:|
| fakenews | 42 | +7.872 [+7.156, +8.568] | +9.385 [+8.542, +10.221] | 1.05e-102 (5.26e-102) |
| fakenews | 52 | +6.039 [+5.385, +6.702] | +7.346 [+6.557, +8.130] | 1.5e-71 (6e-71) |
| fakenews | 62 | +8.178 [+7.440, +8.926] | +9.832 [+8.993, +10.708] | 2.1e-104 (1.26e-103) |
| hatespeech | 42 | +2.048 [+1.412, +2.684] | +2.046 [+1.408, +2.686] | 3.58e-10 (1.07e-09) |
| hatespeech | 52 | +1.233 [+0.537, +1.929] | +1.246 [+0.548, +1.950] | 0.000603 (0.000603) |
| hatespeech | 62 | +2.088 [+1.332, +2.844] | +2.101 [+1.343, +2.856] | 7.05e-08 (1.41e-07) |

## Methodological cautions

1. The seeds share the same fixed original test data; they do not establish independent-domain generalization.
2. Statistical tests treat test rows as independent. Article or comment clusters can make
   simple row bootstrap and McNemar significance optimistic; cluster robustness was not run.
3. McNemar tests error-rate differences; it does not directly test the macro-F1 gap.
4. Sample SD across 3 training seeds is distinct from a paired bootstrap test-row CI.
5. Source dataset labels are unchanged. Human-adjudicated labels and robustness to
   corrected labels have NOT been analyzed in this phase.
6. Previous training logs warned of potentially nondeterministic attention; bitwise
   reproducibility of fresh GPU training cannot be claimed.
7. These tests do not constitute XAI faithfulness, text-length ablation, or leakage-free
   generalization beyond the checked cross-split exact-text overlaps.

## Provenance

All 12 COMPLETE markers, training histories, validation-checkpoint
selection rules and saved predictions matched the recomputed metrics.
All archives' SHA-256 file digests are recorded in the JSON manifest.
