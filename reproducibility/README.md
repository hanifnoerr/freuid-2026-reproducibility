# FREUID 2026 Reproducibility Package

This folder implements the official FREUID 2026 Docker sandbox contract:

https://freuid2026.microblink.com/reproducibility.html

Organizer contract summary:

- Runtime must work with `docker run --network none`.
- Test images are mounted as a flat read-only directory at `/data`.
- Output must be `/submissions/submission.csv`.
- Output columns must be exactly `id,label`.
- The `id` is the image filename without extension.
- `label` is a finite fraud score where higher means more likely fraudulent.
- There must be exactly one output row per mounted image file.
- Model weights must be inside the image or mounted read-only; no runtime downloads.

This package downloads the 2.2 GB checkpoint during `docker build` when
`model/model.pt` is not already present. The finished image contains
`/models/model.pt`, so inference still runs with `docker run --network none`.

## Files

```text
Dockerfile
requirements.txt
prepare_submission.py
download_model_from_drive.py
model/README.md
report/freuid_technical_report.tex
REPLY_TEMPLATE.txt
FREEZE_MANIFEST.md
BEST_PUBLIC_SUBMISSION.md
COLAB_EXPORT_BEST_PUBLIC.md
hash_artifacts.py
```

## Prepare Final Checkpoint

First export the exact best-public checkpoint and CSV from Drive using:

```text
COLAB_EXPORT_BEST_PUBLIC.md
```

Copy the final selected checkpoint to:

```text
reproducibility/model/model.pt
```

or mount it read-only at runtime:

```text
/models/model.pt
```

The checkpoint must be a PyTorch dict containing the ConvNeXtV2 state dict under `model`, plus `model_name` and `image_size`, as saved by the training notebooks.

## Build

Default build, using the public Google Drive `best.pt` file:

```bash
cd reproducibility
docker build -t freuid-repro:local .
```

Strict build with the expected checkpoint filename and SHA256:

```bash
cd reproducibility
docker build \
  --build-arg MODEL_CHECKPOINT_NAME=best.pt \
  --build-arg MODEL_SHA256=TODO_CHECKPOINT_SHA256 \
  -t freuid-repro:local .
```

Default checkpoint file:

```text
https://drive.google.com/uc?id=1PGhqwLlZHwfebCOoNJO-XqsjYkwD7Ip1
```

The containing public folder is:

```text
https://drive.google.com/drive/folders/15pj--qzgwVzq-KjH7kJUcxnTS3ZA5c2P?usp=sharing
```

Keep that folder public with "Anyone with the link can view" until organizer
verification is complete. If Drive access fails, copy the checkpoint to
`model/model.pt` and rebuild.

## Run With Weights Inside Image

```bash
mkdir -p out
docker run --rm --network none \
  --gpus all \
  -v /path/to/flat/test/images:/data:ro \
  -v "$PWD/out:/submissions" \
  freuid-repro:local
```

## Run With Read-Only Mounted Weights

```bash
mkdir -p out
docker run --rm --network none \
  --gpus all \
  -v /path/to/flat/test/images:/data:ro \
  -v /path/to/final_checkpoint.pt:/models/model.pt:ro \
  -v "$PWD/out:/submissions" \
  freuid-repro:local
```

## Outputs

```text
out/submission.csv
out/run_summary.json
```

`submission.csv` contains one row for each image in `/data`.

## Fill Before Final Reply

Update:

- `FREEZE_MANIFEST.md`
- `report/freuid_technical_report.tex`, then compile to PDF
- `REPLY_TEMPLATE.txt`

Post exactly one reply on the pinned Kaggle discussion thread by July 15, 2026, 23:59 AoE.
