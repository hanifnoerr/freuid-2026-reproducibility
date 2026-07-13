# FREUID 2026 Freeze Manifest

Fill this before posting the Kaggle reproducibility reply.

## Code Freeze

- Private test release date: July 13, 2026
- Final Kaggle deadline: July 15, 2026, 23:59 AoE
- Public repository URL: https://github.com/hanifnoerr/freuid-2026-reproducibility
- Frozen commit SHA: TODO
- Branch/tag: `main` / `freuid-code-freeze-2026-07-13`
- OSI license: MIT

After July 13, 2026, do not change model weights, architecture, or training code. Allowed changes are inference on private images, documentation, Docker packaging, and other packaging work that does not alter weights.

## Final Kaggle Submission

- Kaggle team name: TODO
- Kaggle usernames: TODO
- Final submission label/time as shown on Kaggle: `submission_convnextv2_g4_672_alltrain.csv` / TODO time
- Public leaderboard score: `0.00267` user-reported
- Private leaderboard score: TODO
- Selected CSV path: `/content/drive/MyDrive/FREUID_2026/final_reproducibility/best_public_convnextv2_672/submission.csv` after export
- Selected CSV SHA256: TODO

## Final Model

- Model architecture: `convnextv2_large.fcmae_ft_in22k_in1k_384`
- Image size: `672`
- Training mode: all labeled rows, no type holdout validation split
- Batch size: `24`
- Epochs: `3`
- Learning rate: `1e-5`
- Weight decay: `0.05`
- Final checkpoint source path: `/content/drive/MyDrive/FREUID_2026/outputs/convnextv2_g4_672_alltrain/<selected-run>/best.pt`
- Checkpoint copied/mounted as: `reproducibility/model/model.pt` or `/models/model.pt`
- Checkpoint public storage: `https://drive.google.com/drive/folders/15pj--qzgwVzq-KjH7kJUcxnTS3ZA5c2P?usp=sharing`
- Checkpoint direct build URL: `https://drive.google.com/uc?id=1PGhqwLlZHwfebCOoNJO-XqsjYkwD7Ip1`
- Checkpoint retrieval mode: downloaded during `docker build`, then used from `/models/model.pt` with no runtime network
- Checkpoint SHA256: `bb3870c09ec7fa21255df1cff40b6cd2ff5a7290904562d8e43a284e4ba5c41a`
- Training notebook/script: `notebooks/freuid_convnextv2_g4_672_alltrain_best_public.ipynb`
- Training notebook SHA256: `EEEECD050A11E06BFD4EDE9F930C4807564867C0C30EC7432029C5939E20FA57`
- Resume notebook/script, if used: `notebooks/freuid_convnextv2_g4_672_resume_to6.ipynb`
- Final epoch/checkpoint selected: epoch 3 `best.pt` from the selected best-public run

Compute hashes with:

```bash
python reproducibility/hash_artifacts.py reproducibility/model/model.pt path/to/final_submission.csv
```

## External Resources

- Kaggle FREUID training data and test images.
- Timm pretrained weights for `convnextv2_large.fcmae_ft_in22k_in1k_384`.
- PyTorch/timm/Pillow runtime dependencies listed in `reproducibility/requirements.txt`.
- Final checkpoint hosted in the public Google Drive folder listed above for Docker build-time retrieval.

## Docker Verification

Local no-network test command:

```bash
cd reproducibility
docker build -t freuid-repro:local .
mkdir -p out
docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$PWD/out:/submissions" \
  freuid-repro:local
```

Result:

- `out/submission.csv` exists: yes
- one row per image file: yes (two-image local smoke test)
- no missing/extra ids: yes (two-image local smoke test)
