import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clean_splits", ROOT / "src" / "clean_splits.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


class NormalizerTests(unittest.TestCase):
    def test_unicode_compatibility(self):
        self.assertEqual(clean.normalize_exact("ＦＯＯ, bar!"), "foo bar")

    def test_bengali_punctuation(self):
        self.assertEqual(clean.normalize_exact("বাংলা—খবর!"), "বাংলা খবর")

    def test_symbols_and_whitespace(self):
        self.assertEqual(clean.normalize_exact(" A\u200b  ♥ B "), "a b")

    def test_empty_punctuation_normalization(self):
        self.assertEqual(clean.normalize_exact("...!!!"), "")

    def test_shingles_and_jaccard(self):
        first = clean.shingles("a b c d e f", 5)
        second = clean.shingles("a b c d e f", 5)
        self.assertEqual(clean.jaccard(first, second), 1.0)
        self.assertGreaterEqual(clean.jaccard(first, {"different"}), 0.0)
        self.assertLessEqual(clean.jaccard(first, {"different"}), 1.0)


if __name__ == "__main__":
    unittest.main()
