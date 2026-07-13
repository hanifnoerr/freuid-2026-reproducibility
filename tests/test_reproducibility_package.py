import unittest
from pathlib import Path


class ReproducibilityPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (cls.repo_root / "reproducibility" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        cls.inference = (
            cls.repo_root / "reproducibility" / "prepare_submission.py"
        ).read_text(encoding="utf-8")

    def test_docker_image_pins_checkpoint_and_a100_defaults(self):
        self.assertIn("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime", self.dockerfile)
        self.assertIn(
            'ARG MODEL_SHA256="bb3870c09ec7fa21255df1cff40b6cd2ff5a7290904562d8e43a284e4ba5c41a"',
            self.dockerfile,
        )
        self.assertIn("ENV BATCH_SIZE=16", self.dockerfile)
        self.assertIn("NUM_WORKERS=8", self.dockerfile)

    def test_inference_uses_a100_optimized_execution(self):
        self.assertIn("torch.cuda.is_bf16_supported()", self.inference)
        self.assertIn("memory_format=torch.channels_last", self.inference)
        self.assertIn("persistent_workers=(args.num_workers > 0)", self.inference)
        self.assertIn("prefetch_factor=4 if args.num_workers > 0 else None", self.inference)


if __name__ == "__main__":
    unittest.main()
