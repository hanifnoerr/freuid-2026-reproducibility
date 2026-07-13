# FREUID 2026 submission

Code and Docker package for my FREUID Challenge 2026 submission.

Model: `convnextv2_large.fcmae_ft_in22k_in1k_384` at 672 px. The frozen checkpoint is downloaded during `docker build` and verified by SHA-256. Inference runs without network access.

```bash
cd reproducibility
docker build -t freuid-repro .
mkdir -p out
docker run --rm --gpus all --network none \
  -v /path/to/test/images:/data:ro \
  -v "$PWD/out:/submissions" \
  freuid-repro
```

Output: `out/submission.csv`

Training code: `notebooks/freuid_convnextv2_g4_672_alltrain_best_public.ipynb`

Full instructions: `reproducibility/README.md`
