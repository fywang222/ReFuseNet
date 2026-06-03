#!/usr/bin/env bash
set -euo pipefail

# Fill this before running, or export it in your shell.
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
  --config configs/camvid_fcn_resnet50.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --save-epochs "${SAVE_EPOCHS}"

"${PYTHON}" tools/train.py \
  --config configs/camvid_segformer_b5.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --save-epochs "${SAVE_EPOCHS}"

"${PYTHON}" tools/train.py \
  --config configs/camvid_refusenet_s0.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --save-epochs "${SAVE_EPOCHS}"
