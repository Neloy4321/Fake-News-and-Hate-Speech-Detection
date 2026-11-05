import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import cpu_train_eval as cpu


class CPUPlanTests(unittest.TestCase):
    def test_nine_runs_only(self):
        self.assertEqual(len(cpu.NINE), 9)
        self.assertEqual(len(set(cpu.NINE)), 9)
        self.assertTrue(all((task, model) != ("fakenews", "banglabert")
                            for task, model, _ in cpu.NINE))

    def test_deterministic_permutation(self):
        self.assertEqual(cpu.order_indices(20, 42, 0).tolist(),
                         cpu.order_indices(20, 42, 0).tolist())
        self.assertNotEqual(cpu.order_indices(20, 42, 0).tolist(),
                            cpu.order_indices(20, 42, 1).tolist())
        self.assertEqual(sorted(cpu.order_indices(20, 42, 0).tolist()), list(range(20)))

    def test_mid_epoch_resume_offset_covers_each_item_once(self):
        order = cpu.order_indices(37, 52, 1)
        processed = order[:32].tolist()   # two complete effective batches of 16
        resumed = order[32:].tolist()
        self.assertEqual(len(processed + resumed), 37)
        self.assertEqual(sorted(processed + resumed), list(range(37)))

    def test_repack_compatibility_requires_identical_cleaned_data(self):
        tasks = {task: {"cleaned_csv_gzip_sha256": {"train": "abc", "val": "def", "test": "ghi"},
                        "fixed_original_test_text_digest_sha256": "123"}
                 for task in ("fakenews", "hatespeech")}
        old = {"tasks": tasks}
        new = {"tasks": {task: value.copy() for task, value in tasks.items()}}
        self.assertTrue(cpu.compatible_manifests(old, new))
        new["tasks"]["fakenews"]["fixed_original_test_text_digest_sha256"] = "DIFFERENT"
        self.assertFalse(cpu.compatible_manifests(old, new))


if __name__ == "__main__":
    unittest.main()
