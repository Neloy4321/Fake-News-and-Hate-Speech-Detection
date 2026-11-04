import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvidenceStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "evidence" / "clean_split_manifest.json").read_text())
        cls.plan = json.loads((ROOT / "evidence" / "12_run_plan_verified.json").read_text())

    def test_stage_is_truthful(self):
        self.assertEqual(self.manifest["stage"], "SPLIT_PREPARATION_VERIFIED__GPU_TRAINING_PENDING")
        self.assertEqual(self.plan["status"], "PLAN_ONLY")

    def test_counts_and_fixed_test(self):
        for task, detail in self.manifest["tasks"].items():
            self.assertEqual(detail["clean_rows"]["test"], detail["original_rows"]["test"])
            self.assertEqual(detail["post_clean_exact_cross_split_overlap"], 0)
            for split in ("train", "val"):
                self.assertEqual(detail["original_rows"][split],
                                 detail["clean_rows"][split] + detail["deleted_row_counts"][split])
                filename = ROOT / "evidence" / f"{task}_removed_row_ids.csv"
                with filename.open(encoding="utf-8", newline="") as fh:
                    rows = [row for row in csv.DictReader(fh) if row["split"] == split]
                self.assertEqual(len(rows), detail["deleted_row_counts"][split])

    def test_12_unique_predeclared_runs(self):
        runs = self.plan["runs"]
        self.assertEqual(len(runs), 12)
        keys = {(r["task"], r["model_key"], r["seed"]) for r in runs}
        self.assertEqual(len(keys), 12)
        self.assertEqual({r["seed"] for r in runs}, {42, 52, 62})
        self.assertEqual({r["task"] for r in runs}, {"fakenews", "hatespeech"})
        self.assertEqual({r["model_key"] for r in runs}, {"banglabert", "mbert"})


if __name__ == "__main__":
    unittest.main()
