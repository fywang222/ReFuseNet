#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-CamVid}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
GPUS=(${GPUS:-0 1})

if (( ${#GPUS[@]} < 2 )); then
  echo "Expected two GPUs, for example: GPUS=\"0 1\""
  exit 1
fi

mkdir -p outputs/logs

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

CONFIGS=(
  "configs/camvid/camvid_refusenet_s2_cityscapes_pretrained_50epoch.yaml"
  "configs/camvid/camvid_refusenet_s4_cityscapes_pretrained_50epoch.yaml"
)

PRETRAINED=(
  "outputs/cityscapes_refusenet_s2/last.pth"
  "outputs/cityscapes_refusenet_s4_pseudo_refine/last.pth"
)

for checkpoint in "${PRETRAINED[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing pretrained checkpoint: ${checkpoint}"
    exit 1
  fi
done

run_one() {
  local gpu="$1"
  local config="$2"
  local pretrained="$3"
  local name
  name="$(basename "${config}" .yaml)"

  echo "[launch] GPU=${gpu} config=${config} pretrained=${pretrained}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" tools/train.py \
    --config "${config}" \
    --pretrained "${pretrained}" \
    --device "${DEVICE}" \
    2>&1 | tee "outputs/logs/${name}.console.log"
}

pids=()
for index in 0 1; do
  run_one "${GPUS[$index]}" "${CONFIGS[$index]}" "${PRETRAINED[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "At least one fine-tuning job failed."
  exit 1
fi

echo "S2 and S4 Cityscapes-to-CamVid 50-epoch fine-tuning finished."
