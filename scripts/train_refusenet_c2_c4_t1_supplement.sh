#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-refusenet}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
GPUS=(${GPUS:-2 3 4 5 6 7})

if (( ${#GPUS[@]} != 6 )); then
  echo "Expected exactly 6 GPUs for this supplement run, got: ${GPUS[*]}" >&2
  exit 1
fi

mkdir -p outputs/logs

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

CONFIGS=(
  "configs/camvid/camvid_refusenet_c2.yaml"
  "configs/camvid/camvid_refusenet_c4.yaml"
  "configs/camvid/camvid_refusenet_t1.yaml"
  "configs/cityscapes/cityscapes_refusenet_c2.yaml"
  "configs/cityscapes/cityscapes_refusenet_c4.yaml"
  "configs/cityscapes/cityscapes_refusenet_t1.yaml"
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

for i in "${!CONFIGS[@]}"; do
  run_one "${GPUS[$i]}" "${CONFIGS[$i]}" &
done

wait
echo "All C2/C4/T1 supplement jobs finished."
