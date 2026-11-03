# Table 10: Selected twelve-article auxiliary sliding-window experiment (mean-probability aggregation) compared with random equal-budget token selection.

| Method | Articles | k | Comp.−random | 95% CI | Random−suff. gap | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| SHAP | 12 | 5 | 0.102 | [-0.007, 0.234] | 0.298 | [0.174, 0.414] |
| LIME | 12 | 5 | 0.146 | [0.025, 0.268] | 0.150 | [0.052, 0.242] |

**Notes.** The auxiliary system used 12 Fake News articles, 27 windows (256 subwords; stride 128) and 1 prediction flip compared with first-window inference. Results describe this NEW sliding-window aggregator, not the originally submitted single-window classifier. n=12, so conclusions are exploratory; SHAP's k=5 comp-minus-random CI crosses zero.
