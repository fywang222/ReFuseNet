#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-refusenet}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
GPUS=(${GPUS:-5 6 7})

if (( ${#GPUS[@]} != 3 )); then
  echo "Expected exactly 3 GPUs, for example: GPUS=\"5 6 7\"" >&2
  exit 1
fi

mkdir -p outputs/logs

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

CONFIGS=(
  "configs/cityscapes/cityscapes_refusenet_s6.yaml"
  "configs/camvid/camvid_refusenet_s6.yaml"
  "configs/camvid/camvid_refusenet_c4_cityscapes_pretrained_50epoch.yaml"
)

PRETRAINED=(
  ""
  ""
  "outputs/cityscapes_refusenet_c4_cnn_sam_refine/last.pth"
)

for checkpoint in "${PRETRAINED[@]}"; do
  if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
    echo "Missing pretrained checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

run_one() {
  local gpu="$1"
  local config="$2"
  local pretrained="$3"
  local name
  name="$(basename "${config}" .yaml)"

  echo "[launch] GPU=${gpu} config=${config} pretrained=${pretrained:-none}"
  local args=(--config "${config}" --device "${DEVICE}")
  if [[ -n "${pretrained}" ]]; then
    args+=(--pretrained "${pretrained}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" tools/train.py "${args[@]}" \
    2>&1 | tee "outputs/logs/${name}.console.log"
}

pids=()
for index in "${!CONFIGS[@]}"; do
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
  echo "At least one requested job failed." >&2
  exit 1
fi

echo "Boundary S6 and C4 Cityscapes-to-CamVid jobs finished."
