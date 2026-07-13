import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "freuid_a100_newest_dataset_inference_test.ipynb"


class A100NotebookTests(unittest.TestCase):
    def test_uses_known_working_kaggle_cli_download(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn('"--upgrade"', source)
        self.assertIn('"kaggle==2.2.2"', source)
        self.assertIn('"kaggle", "competitions", "download"', source)
        self.assertIn('ZIP_PATH = WORK_DIR / f"{COMPETITION}.zip"', source)
        self.assertIn('subprocess.run(["unzip", "-q", "-o"', source)
        self.assertIn("competition_path = DATA_DIR", source)
        self.assertNotIn("kagglehub", source.lower())

    def test_does_not_create_or_upload_a_kaggle_dataset(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertNotIn("kaggle datasets", source.lower())
        self.assertNotIn("datasets create", source.lower())
        self.assertNotIn("dataset-metadata.json", source)
        self.assertNotIn("drive.mount", source)

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
        self.assertIn("Kaggle token configured for the CLI", source)

    def test_download_failure_reports_kaggle_output_and_free_disk(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn('shutil.disk_usage("/content")', source)
        self.assertIn("Kaggle competition download failed", source)
        self.assertIn("download_output", source)

    def test_preserves_and_enforces_frozen_checkpoint_contract(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn(
            'MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"', source
        )
        self.assertIn("IMAGE_SIZE = 672", source)
        self.assertIn("EXPECTED_EPOCH = 3", source)
        self.assertIn(
            'EXPECTED_CHECKPOINT_SHA256 = "bb3870c09ec7fa21255df1cff40b6cd2ff5a7290904562d8e43a284e4ba5c41a"',
            source,
        )
        self.assertIn(
            "if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:", source
        )


if __name__ == "__main__":
    unittest.main()
