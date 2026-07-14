from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reproducibility" / "report" / "freuid_technical_report.tex"
BIBLIOGRAPHY = ROOT / "reproducibility" / "report" / "references.bib"


class TechnicalReportTests(unittest.TestCase):
    def test_report_contains_frozen_submission_facts(self):
        source = REPORT.read_text(encoding="utf-8")
        required = [
            "convnextv2_large.fcmae_ft_in22k_in1k_384",
            "69,352",
            "40,005",
            "29,347",
            "672",
            "Google Colab",
            "G4",
            "96~GB",
            "0.00283",
            "submission_convnextv2_epoch03_newest_a100.csv",
            "94a6ae437bf44848ef51490bdc90475d8454445f488b97cca124cf3a62d48b9b",
            "unavailable when this report was prepared",
            "bb3870c09ec7fa21255df1cff40b6cd2ff5a7290904562d8e43a284e4ba5c41a",
            "/submissions/submission.csv",
            "--network none",
        ]
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, source)

        self.assertNotIn("TO" + "DO", source)
        self.assertNotIn("T" + "BD", source)
        self.assertNotIn("score of \\textbf{0.00267}", source)

    def test_bibliography_contains_supplied_competition_citation(self):
        source = BIBLIOGRAPHY.read_text(encoding="utf-8")
        self.assertIn("@misc{the-freuid-challenge-2026-ijcai-ecai", source)
        self.assertIn("Ivan Reli", source)
        self.assertIn("The FREUID Challenge 2026 - IJCAI-ECAI", source)
        self.assertIn("the-freuid-challenge-2026-ijcai-ecai", source)


if __name__ == "__main__":
    unittest.main()
