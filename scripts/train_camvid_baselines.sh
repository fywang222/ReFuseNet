#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-refusenet}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-/absolute/path/to/sam_vit_b.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

mkdir -p outputs/logs

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  2>&1 | tee outputs/logs/camvid_fcn_resnet50.console.log &

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" tools/train.py \
  --config configs/camvid_segformer_b5.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  2>&1 | tee outputs/logs/camvid_segformer_b5.console.log &

CUDA_VISIBLE_DEVICES=2 "${PYTHON}" tools/train.py \
  --config configs/camvid_refusenet_s0.yaml \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  2>&1 | tee outputs/logs/camvid_refusenet_s0.console.log &

wait