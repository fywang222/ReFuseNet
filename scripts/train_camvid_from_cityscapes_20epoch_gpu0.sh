#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-CamVid}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"

mkdir -p outputs/logs

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

run_one() {
  local config="$1"
  local pretrained="$2"
  local name
  name="$(basename "${config}" .yaml)"

  echo "[launch] GPU=${GPU} config=${config} pretrained=${pretrained}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/train.py \
    --config "${config}" \
    --pretrained "${pretrained}" \
    --device "${DEVICE}" \
    2>&1 | tee "outputs/logs/${name}.console.log"
}

run_one \
  "configs/camvid/camvid_fcn_resnet50_cityscapes_pretrained_20epoch.yaml" \
  "outputs/cityscapes_fcn_resnet50/last.pth"

run_one \
  "configs/camvid/camvid_segformer_b5_cityscapes_pretrained_20epoch.yaml" \
  "outputs/cityscapes_segformer_b5/last.pth"

run_one \
  "configs/camvid/camvid_refusenet_s2_cityscapes_pretrained_20epoch.yaml" \
  "outputs/cityscapes_refusenet_s2/last.pth"

echo "Cityscapes-to-CamVid 20-epoch fine-tune jobs finished."
