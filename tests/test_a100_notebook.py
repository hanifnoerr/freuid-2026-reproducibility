import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "freuid_a100_newest_dataset_inference_test.ipynb"


class A100NotebookTests(unittest.TestCase):
    def test_uses_pinned_kagglehub_bulk_download(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn('"--upgrade"', source)
        self.assertIn('"kagglehub==1.0.2"', source)
        self.assertIn("import kagglehub", source)
        self.assertIn("kagglehub.competition_download(", source)
        self.assertIn("output_dir=str(EXTRACT_DIR)", source)
        self.assertIn("force_download=True", source)
        self.assertNotIn('"kaggle", "competitions", "download"', source)
        self.assertNotIn("path=file_name", source)

    def test_reads_colab_secret_before_prompting_and_verifies_kaggle(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("from google.colab import userdata", source)
        self.assertIn('userdata.get("KAGGLE_API_TOKEN")', source)
        self.assertIn('os.environ["KAGGLE_API_TOKEN"] = kaggle_token', source)
        self.assertIn("Kaggle token configured for KaggleHub", source)

    def test_download_failure_reports_kaggle_output_and_free_disk(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn('shutil.disk_usage("/content")', source)
        self.assertIn("private-test bulk archive is not ready", source)


if __name__ == "__main__":
    unittest.main()
