#!/usr/bin/env bash
set -euo pipefail

# Fill these before running, or export them in your shell.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-refusenet}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-/absolute/path/to/sam_vit_b.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
SAVE_EPOCHS="${SAVE_EPOCHS:-50,100,150,200}"

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

"${PYTHON}" tools/train.py \
  --config configs/cityscapes_fcn_resnet50.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --save-epochs "${SAVE_EPOCHS}"

"${PYTHON}" tools/train.py \
  --config configs/cityscapes_segformer_b5.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --save-epochs "${SAVE_EPOCHS}"

for setting in s0 s1 s2 s3 s4 s5; do
  "${PYTHON}" tools/train.py \
    --config "configs/ablation/cityscapes_refusenet_${setting}.yaml" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --save-epochs "${SAVE_EPOCHS}"
done
