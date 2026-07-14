# Best Public Submission Lock

Use this as the frozen setting record for the current best public leaderboard submission.

## User-Reported Leaderboard Result

- Kaggle submission file: `submission_convnextv2_g4_672_alltrain.csv`
- Public leaderboard score: `0.00267`
- Selected notebook copied into this repository:
  - `notebooks/freuid_convnextv2_g4_672_alltrain_best_public.ipynb`
- Copied notebook SHA256:
  - `EEEECD050A11E06BFD4EDE9F930C4807564867C0C30EC7432029C5939E20FA57`
- Public checkpoint storage:
  - `https://drive.google.com/drive/folders/15pj--qzgwVzq-KjH7kJUcxnTS3ZA5c2P?usp=sharing`

## Frozen Training Configuration

These values come from the attached best-public notebook.

```text
MODEL_NAME = convnextv2_large.fcmae_ft_in22k_in1k_384
IMAGE_SIZE = 672
BATCH_SIZE = 24
EPOCHS = 3
LR = 1e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 12
VAL_TYPE = None
DUMMY_SCORE = 0.0
MAX_GRAD_NORM = 1.0
```

Training used all labeled rows, not a type holdout validation split. The saved checkpoint to keep is the `best.pt` from the exact Drive run that produced the public `0.00267` submission.

## Frozen Transforms

Training:

```text
RandomResizedCrop(672, scale=(0.88, 1.0), bicubic)
RandomApply(ColorJitter(brightness=0.14, contrast=0.14, saturation=0.05), p=0.5)
RandomApply(GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), p=0.12)
RandomRotation(degrees=1.5)
ToTensor()
ImageNet Normalize
```

Inference:

```text
Resize(672, bicubic)
CenterCrop(672)
ToTensor()
ImageNet Normalize
sigmoid(logit)
```

## What Not To Change

For the final prize/reproducibility package, do not change:

- model architecture
- image size
- trained weights
- preprocessing/inference transform
- output semantics where higher score means more likely fraudulent

Allowed final work is packaging: export the selected checkpoint, compute hashes, build/test Docker, write the PDF report, and post the required Kaggle reproducibility reply.

## Complete-Test Rerun

The unchanged epoch-3 checkpoint was rerun after all test images became
available. This is the selected complete submission for the final package:

- Kaggle submission file: `submission_convnextv2_epoch03_newest_a100.csv`
- Public leaderboard score: `0.00283`
- Rows: `142,818`, all produced from resolved images
- CSV SHA256: `94a6ae437bf44848ef51490bdc90475d8454445f488b97cca124cf3a62d48b9b`
- A100 inference time: `34.28` minutes at batch size `8`

The earlier `0.00267` result remains the historical score used to select the
frozen model. No model, weight, preprocessing, or training change was made for
the complete-test rerun.
