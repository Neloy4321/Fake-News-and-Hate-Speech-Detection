"""Unit tests for numerical pieces; toy arrays are not experimental evidence."""
import unittest

import numpy as np
from sklearn.metrics import f1_score

from analyze_all_12_runs import exact_mcnemar, holm, macro_f1_from_cm, paired_bootstrap


class StatisticalAnalysisTests(unittest.TestCase):
    def test_macro_f1_confusion_math(self):
        y = np.array([0, 0, 0, 1, 1, 1, 1])
        p = np.array([0, 1, 0, 1, 1, 0, 1])
        result = macro_f1_from_cm(np.array([2]), np.array([1]),
                                  np.array([1]), np.array([3]))
        self.assertAlmostEqual(float(result[0]), f1_score(y, p, average="macro"))

    def test_identical_predictions(self):
        y = np.array([0, 1, 0, 1] * 20)
        a = np.array([0, 1, 1, 0] * 20)
        stats = exact_mcnemar(y, a, a)
        self.assertEqual(stats["discordant_pairs"], 0)
        self.assertEqual(stats["mcnemar_exact_two_sided_p"], 1.0)
        ci = paired_bootstrap(y, a, a, replicates=100, seed=3)
        self.assertEqual(ci["delta_accuracy_ci95"], [0, 0])
        self.assertEqual(ci["delta_macro_f1_ci95"], [0, 0])

    def test_holm_stepdown(self):
        self.assertEqual(holm([0.04, 0.001, 0.5]), [0.08, 0.003, 0.5])

    def test_fully_different_errors(self):
        y = np.array([0, 0, 1, 1, 1, 1])
        a = np.array([0, 0, 1, 1, 1, 1])
        b = np.array([1, 1, 0, 0, 0, 0])
        result = exact_mcnemar(y, a, b)
        self.assertEqual(result["banglabert_correct_mbert_wrong"], 6)
        self.assertEqual(result["mbert_correct_banglabert_wrong"], 0)
        self.assertAlmostEqual(result["mcnemar_exact_two_sided_p"], 0.03125)


if __name__ == "__main__":
    unittest.main()
