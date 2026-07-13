from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import timm
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL = "convnextv2_large.fcmae_ft_in22k_in1k_384"
DEFAULT_IMAGE_SIZE = 672


class FlatImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return path.stem, tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FREUID submission.csv for the Docker no-network sandbox."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--model-path", type=Path, default=Path("/models/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("/submissions/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    return parser.parse_args()


def list_flat_images(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {data_dir}")
    paths = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: p.stem)
    if not paths:
        raise ValueError(f"No supported image files found in {data_dir}")
    return paths


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {path}. "
            "Place the final checkpoint at reproducibility/model/model.pt before building, "
            "or mount it read-only to /models/model.pt."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def checkpoint_config(checkpoint: dict) -> tuple[str, int]:
    config = checkpoint.get("config", {})
    model_name = checkpoint.get("model_name") or config.get("model_name") or DEFAULT_MODEL
    image_size = int(checkpoint.get("image_size") or config.get("image_size") or DEFAULT_IMAGE_SIZE)
    return model_name, image_size


def checkpoint_state_dict(checkpoint: dict) -> dict:
    state = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state_dict")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model_name, _ = checkpoint_config(checkpoint)
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def finite_probability(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite prediction: {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Prediction outside [0, 1]: {value}")
    return f"{value:.8g}"


@torch.no_grad()
def predict_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for ids, images in loader:
        if device.type == "cuda":
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits.float()).cpu().tolist()
        for row_id, score in zip(ids, probs):
            rows.append({"id": row_id, "label": finite_probability(float(score))})
    return rows


def write_submission(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids in output rows")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    model_path = args.model_path.resolve()

    image_paths = list_flat_images(data_dir)
    checkpoint = load_checkpoint(model_path)
    model_name, image_size = checkpoint_config(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform(image_size)
    loader = DataLoader(
        FlatImageDataset(image_paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    model = build_model(checkpoint, device)
    rows = predict_rows(model, loader, device)
    if len(rows) != len(image_paths):
        raise RuntimeError(f"Expected {len(image_paths)} rows, got {len(rows)}")
    write_submission(rows, output_path)

    summary = {
        "model_name": model_name,
        "image_size": image_size,
        "device": str(device),
        "input_dir": str(data_dir),
        "model_path": str(model_path),
        "output_path": str(output_path),
        "rows": len(rows),
    }
    summary_path = output_path.parent / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
