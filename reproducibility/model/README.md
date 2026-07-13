# Model Checkpoint

Place the selected final checkpoint here as:

```text
model.pt
```

The Docker image expects:

```text
/models/model.pt
```

Two valid ways to satisfy the official requirement:

1. Copy the final checkpoint into this directory before `docker build`; the Dockerfile copies it into the image.
2. Keep the checkpoint outside the image and mount it read-only at runtime:

```bash
docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v /path/to/final_checkpoint.pt:/models/model.pt:ro \
  -v "$PWD/out:/submissions" \
  freuid-repro:local
```

Record the checkpoint SHA256 in `../FREEZE_MANIFEST.md`.
