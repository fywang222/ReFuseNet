#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-refusenet}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"

# 默认用物理 GPU 4,5,6,7
GPUS=(${GPUS:-4 5 6 7})

mkdir -p outputs/logs

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

CONFIGS=(
  "configs/cityscapes/cityscapes_fcn_resnet50.yaml"
  "configs/cityscapes/cityscapes_segformer_b5.yaml"
  "configs/cityscapes/cityscapes_refusenet_s0.yaml"
  "configs/cityscapes/cityscapes_refusenet_s1.yaml"
  "configs/cityscapes/cityscapes_refusenet_s2.yaml"
  "configs/cityscapes/cityscapes_refusenet_s3.yaml"
  "configs/cityscapes/cityscapes_refusenet_s4.yaml"
  "configs/cityscapes/cityscapes_refusenet_s5.yaml"
  "configs/cityscapes/cityscapes_refusenet_s6.yaml"
  "configs/cityscapes/cityscapes_refusenet_s7.yaml"
)

run_one() {
  local gpu="$1"
  local config="$2"
  local name
  name="$(basename "${config}" .yaml)"

  echo "[launch] GPU=${gpu} config=${config}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" tools/train.py \
    --config "${config}" \
    --device "${DEVICE}" \
    2>&1 | tee "outputs/logs/${name}.console.log"
}

batch_size="${#GPUS[@]}"
num_configs="${#CONFIGS[@]}"

for ((start=0; start<num_configs; start+=batch_size)); do
  echo "========== launching batch starting at index ${start} =========="

  for ((i=0; i<batch_size; i++)); do
    idx=$((start + i))
    if (( idx >= num_configs )); then
      break
    fi

    run_one "${GPUS[$i]}" "${CONFIGS[$idx]}" &
  done

  wait
  echo "========== batch finished =========="
done

echo "All Cityscapes jobs finished."
