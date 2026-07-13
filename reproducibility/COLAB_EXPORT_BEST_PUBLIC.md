# Colab Export For Best Public Submission

Run this in Colab after mounting Drive. It does not train. It only finds the saved Drive run, copies the selected `best.pt` and submission CSV into a final reproducibility folder, validates the CSV shape, and prints hashes for `FREEZE_MANIFEST.md`.

Set `SELECTED_INDEX` to the run that produced Kaggle public score `0.00267`.

```python
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import shutil

from google.colab import drive

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/FREUID_2026")
RUN_ROOT = DRIVE_ROOT / "outputs" / "convnextv2_g4_672_alltrain"
EXPORT_ROOT = DRIVE_ROOT / "final_reproducibility" / "best_public_convnextv2_672"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 142_818

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def file_mtime(path):
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")

def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"read_error": repr(exc)}

def validate_submission(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"Bad columns: {reader.fieldnames}")
        n = 0
        ids = set()
        for row in reader:
            n += 1
            ids.add(row["id"])
            float(row["label"])
    if n != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {n}")
    if len(ids) != n:
        raise ValueError(f"Duplicate ids: rows={n}, unique={len(ids)}")
    return n

if not RUN_ROOT.exists():
    raise FileNotFoundError(RUN_ROOT)

runs = [p for p in RUN_ROOT.iterdir() if p.is_dir()]
runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

print("Candidate runs under:", RUN_ROOT)
for i, run in enumerate(runs[:30]):
    best = run / "best.pt"
    sub = run / "submission_convnextv2_g4_672_alltrain.csv"
    summary = read_json(run / "submission_summary.json")
    print(
        f"{i:02d}",
        run.name,
        "mtime=" + file_mtime(run),
        "best_pt=" + str(best.exists()),
        "submission=" + str(sub.exists()),
        "rows=" + str(summary.get("rows", "?")),
    )

# Change this to the exact run that produced Kaggle score 0.00267.
SELECTED_INDEX = 0
SELECTED_RUN = runs[SELECTED_INDEX]

checkpoint_src = SELECTED_RUN / "best.pt"
submission_src = SELECTED_RUN / "submission_convnextv2_g4_672_alltrain.csv"
if not checkpoint_src.exists():
    raise FileNotFoundError(checkpoint_src)
if not submission_src.exists():
    raise FileNotFoundError(submission_src)

row_count = validate_submission(submission_src)

checkpoint_dst = EXPORT_ROOT / "model.pt"
submission_dst = EXPORT_ROOT / "submission.csv"
submission_named_dst = EXPORT_ROOT / "submission_convnextv2_g4_672_alltrain.csv"

shutil.copy2(checkpoint_src, checkpoint_dst)
shutil.copy2(submission_src, submission_dst)
shutil.copy2(submission_src, submission_named_dst)

manifest = {
    "selected_run": str(SELECTED_RUN),
    "public_score_user_reported": "0.00267",
    "model_name": "convnextv2_large.fcmae_ft_in22k_in1k_384",
    "image_size": 672,
    "batch_size": 24,
    "epochs": 3,
    "lr": 1e-5,
    "weight_decay": 0.05,
    "submission_rows": row_count,
    "checkpoint_path": str(checkpoint_dst),
    "checkpoint_sha256": sha256_file(checkpoint_dst),
    "submission_path": str(submission_dst),
    "submission_sha256": sha256_file(submission_dst),
}

(EXPORT_ROOT / "final_artifacts.json").write_text(json.dumps(manifest, indent=2))

print(json.dumps(manifest, indent=2))
print("\nCopy these into reproducibility/FREEZE_MANIFEST.md:")
print("Final checkpoint source path:", checkpoint_src)
print("Checkpoint SHA256:", manifest["checkpoint_sha256"])
print("Selected CSV path:", submission_dst)
print("Selected CSV SHA256:", manifest["submission_sha256"])
```

After this, either:

- download/copy `model.pt` into `reproducibility/model/model.pt` before building Docker, or
- keep it in Drive and mount it read-only as `/models/model.pt` during Docker testing.
