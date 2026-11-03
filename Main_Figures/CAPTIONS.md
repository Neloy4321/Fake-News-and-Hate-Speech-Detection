# Main-figure captions and interpretation

## Figure 1 — Evidence-aware workflow

Figure 1. Evidence-aware workflow for Bengali fake-news
and hate-speech classification. Original source-labelled
test predictions support the multi-seed statistical analysis
and post-hoc explanation evaluation. Human annotations are
reviewed independently and used for restricted-cohort
sensitivity analysis. The full-article sliding-window
Option B experiment is a derived auxiliary system rather
than the original single-window classifier. Cross-split
overlap flags are based on post-hoc screening and do not
establish leakage-free model training.

## Figure 2 — Source dataset class distributions

Figure 2. Class distributions of the original Bengali fake-news and BD-SHS hate-speech datasets. The fake-news corpus contains 62,935 real and 31,948 fake articles (class ratio 1.97:1), whereas BD-SHS contains 26,125 non-hate and 24,156 hate comments (class ratio 1.08:1). Percentages and counts refer to the released source annotations across all three dataset partitions, not to human-corrected audit labels.

## Figure 3 — Seed-42 test confusion matrices

Figure 3. Seed-42 confusion matrices for mBERT and
BanglaBERT on the original fake-news and BD-SHS held-out
test partitions. Rows indicate original dataset labels and
columns indicate model predictions. Each cell reports the
absolute count and percentage within its true-label row.
The common color scale represents row-normalized percentages,
not absolute counts. These seed-42 examples are accompanied
by three-seed mean and standard-deviation results elsewhere
in the manuscript. No claim of full-test human-corrected
ground truth or leakage-free retraining is made.

## Figure 4 — Three-seed ROC curves

Figure 4. Receiver operating characteristic (ROC) analysis
of BanglaBERT and multilingual BERT (mBERT) on the original
held-out test sets for (A) Bengali fake news detection
(n = 9,489) and (B) Bengali hate speech detection
(n = 5,029). Each model was evaluated across three
independent training seeds (42, 52, and 62). Thin,
semitransparent lines represent individual seed-specific
ROC curves; thick lines represent the pointwise mean true
positive rate interpolated over a common false positive
rate grid. Shaded regions indicate ±1 sample standard
deviation across the three seed-specific true positive
rates. Legend values report mean ± sample standard
deviation of the three independently calculated test ROC
AUC values. Results use the original binary test labels,
not the human-corrected annotation subsets.

## Figure 5 — Fake news, four-way SHAP and LIME examples

Representative SHAP and LIME attributions for BanglaBERT fake-news classifications, showing one instance from each original source-label/prediction outcome (TP, TN, FP, FN). Feature signs indicate contributions toward the displayed predicted class. LIME weights are local-surrogate coefficients and are not numerically interchangeable with SHAP values. Examples are illustrative; explanation faithfulness is evaluated separately.

## Figure 6 — Hate speech, four-way SHAP and LIME examples

Representative SHAP and LIME attributions for BanglaBERT hate-speech classifications, showing one instance from each original source-label/prediction outcome (TP, TN, FP, FN). Positive and negative feature weights support and oppose the displayed predicted class respectively. Original dataset labels are distinguished from model predictions; these examples are not full-test quantitative estimates.

**Research boundary:** Figure 5/6 examples were obtained from separately trained XAI checkpoints, and should not be presented as if they directly explain the archived 12-run multi-seed prediction checkpoints. The original source annotations are not the finalized adjudicated annotation subset. The figures use saved experimental records; uploading them does not constitute an independent checkpoint rerun.
