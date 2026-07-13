from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_MODEL_DRIVE_URL = (
    "https://drive.google.com/uc?id=1PGhqwLlZHwfebCOoNJO-XqsjYkwD7Ip1"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def choose_checkpoint(root: Path, preferred_name: str | None) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found after Drive download in {root}")

    if preferred_name:
        matches = [path for path in candidates if path.name == preferred_name]
        if matches:
            matches.sort(key=lambda p: p.stat().st_size, reverse=True)
            return matches[0]
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            f"Requested checkpoint {preferred_name!r} was not found. "
            f"Downloaded checkpoint candidates: {names}"
        )

    for name in ("model.pt", "best.pt", "last.pt"):
        matches = [path for path in candidates if path.name == name]
        if matches:
            matches.sort(key=lambda p: p.stat().st_size, reverse=True)
            return matches[0]

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def download_checkpoint(model_url: str, tmp_dir: Path, checkpoint_name: str | None) -> Path:
    if "/folders/" in model_url:
        cmd = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            model_url,
            "-O",
            str(tmp_dir),
        ]
        print("Downloading model artifact from Google Drive folder...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        return choose_checkpoint(tmp_dir, checkpoint_name)

    output_name = checkpoint_name or "model.pt"
    output_path = tmp_dir / output_name
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        model_url,
        "-O",
        str(output_path),
    ]
    print("Downloading model artifact from Google Drive file...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def main() -> None:
    model_url = os.environ.get("MODEL_DRIVE_URL", DEFAULT_MODEL_DRIVE_URL).strip()
    checkpoint_name = os.environ.get("MODEL_CHECKPOINT_NAME", "").strip() or None
    expected_sha256 = os.environ.get("MODEL_SHA256", "").strip().lower()
    output_path = Path(os.environ.get("MODEL_OUTPUT_PATH", "/models/model.pt"))

    if not model_url:
        raise SystemExit("MODEL_DRIVE_URL is empty and /models/model.pt does not exist.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="freuid-drive-model-") as tmp_name:
        tmp_dir = Path(tmp_name)
        checkpoint_path = download_checkpoint(model_url, tmp_dir, checkpoint_name)
        print(f"Selected checkpoint: {checkpoint_path}")
        shutil.copy2(checkpoint_path, output_path)

    actual_sha256 = sha256_file(output_path)
    print(f"Checkpoint copied to: {output_path}")
    print(f"Checkpoint SHA256: {actual_sha256}")

    if expected_sha256 and expected_sha256 != actual_sha256.lower():
        output_path.unlink(missing_ok=True)
        raise SystemExit(
            "Checkpoint SHA256 mismatch. "
            f"Expected {expected_sha256}, got {actual_sha256}."
        )


if __name__ == "__main__":
    main()
