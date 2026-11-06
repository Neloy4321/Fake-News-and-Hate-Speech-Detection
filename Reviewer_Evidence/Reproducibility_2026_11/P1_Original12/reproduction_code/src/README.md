# TALLIP-26-0521: Reproducible Phase 1 statistical evidence

This folder contains the analysis source and outputs for the 12 completed GPU training experiments:

* Datasets: Bengali Fake News, Bengali Hate Speech
* Models: BanglaBERT, mBERT
* Training seeds: 42, 52, 62

No training or XAI is performed by this script. Results are derived **only from saved original predictions**. The unchanged original test labels were used; the later human annotation audit has not yet been used to correct evaluation labels.

## Run the analysis

For the self-contained ZIP package supplied with this report:

1. Extract the outer ZIP. Keep its `original_evidence/` and `src/` folders together.
2. Install dependencies: `python -m pip install -r src/requirements.txt`.
3. Run from the extracted package root:

   `python src/analyze_all_12_runs.py --input-dir original_evidence --output-dir reproduced_results --bootstrap-replicates 10000`

4. Optionally run the tests: `python -m unittest discover -s src -p 'test_*.py' -v`.

The script validates archive ZIP integrity, completed-run flags, dataset audit counts, validation-chosen checkpoints, all 12 recomputed saved test metrics, and test-row/test-label alignment. It raises an error rather than silently proceeding on mismatched evidence.

## Outputs

* `per_seed_metrics.csv`: accuracy, macro precision/recall/F1, AUC, confusion counts, best validation epoch for all 12 runs.
* `model_mean_sd.csv`: mean and sample SD over 3 seeds for each dataset/model.
* `paired_model_tests.csv`: six matched-seed model comparisons, paired 95% bootstrap CIs on accuracy and macro-F1 differences, exact two-sided McNemar p, Holm-adjusted p.
* `model_comparison_mean_sd.png`: test accuracy and macro-F1 comparison.
* `confusion_matrices_seed42.png`: representative frozen-test confusion matrices.
* `roc_curves_seed42.png`: representative frozen-test ROC curves.
* `TALLIP_Phase1_Statistical_Results.md`: research-ready explanation and limitations.
* `analysis_manifest.json`: input archive SHA-256 hashes, environment versions, test alignment and full numerical evidence.

**Interpretation:** McNemar is a test of paired accuracy/error correctness, **not** a macro-F1 significance test. The paired bootstrap intervals describe sampling uncertainty over test rows conditional on those models/seeds, while 3-seed SD describes training variation. The six Holm-adjusted McNemar tests are all different from a six-run cross-dataset independent validation. Test-row dependence (e.g. news-article clusters), annotation uncertainty, and model faithfulness are not addressed here.

## Source evidence

The `original_evidence/` subfolder contains the ten source ZIPs for the twelve completed runs (one archive contains three mBERT Hate Speech seeds). The SHA-256 digest of each is preserved in `analysis_manifest.json`. Avoid replacing them or changing frozen test labels if you wish to reproduce these exact numbers.

## Paper wording constraint

Do not report that reviewer-adjudicated test labels improved accuracy, that SHAP/LIME/IG was newly completed, or that independent held-out-domain performance was tested based on this package. Those are separate pending analyses.
