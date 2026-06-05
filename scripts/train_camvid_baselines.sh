#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-refusenet}"
export SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"

if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

GPUS=(${GPUS:-1 2 3})

mkdir -p outputs/logs

CONFIGS=(
  "configs/camvid/camvid_fcn_resnet50.yaml"
  "configs/camvid/camvid_segformer_b5.yaml"
  "configs/camvid/camvid_refusenet_s0.yaml"
  "configs/camvid/camvid_refusenet_s1.yaml"
  "configs/camvid/camvid_refusenet_s2.yaml"
  "configs/camvid/camvid_refusenet_s3.yaml"
  "configs/camvid/camvid_refusenet_s4.yaml"
  "configs/camvid/camvid_refusenet_s5.yaml"
  "configs/camvid/camvid_refusenet_s7.yaml"
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

num_configs="${#CONFIGS[@]}"
next_config=0
failed=0
status_dir="$(mktemp -d outputs/logs/camvid_pool.XXXXXX)"
RUNNING_PIDS=()
RUNNING_GPUS=()
RUNNING_CONFIGS=()
FREE_GPUS=("${GPUS[@]}")

cleanup() {
  rm -rf "${status_dir}"
}
trap cleanup EXIT

launch_on_gpu() {
  local gpu="$1"
  local config="$2"
  local pid

  (
    set +e
    run_one "${gpu}" "${config}"
    status="$?"
    echo "${status}" > "${status_dir}/${BASHPID}.status"
    exit "${status}"
  ) &
  pid="$!"

  RUNNING_PIDS+=("${pid}")
  RUNNING_GPUS+=("${gpu}")
  RUNNING_CONFIGS+=("${config}")
}

launch_next_if_possible() {
  while (( next_config < num_configs && ${#FREE_GPUS[@]} > 0 )); do
    local gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    launch_on_gpu "${gpu}" "${CONFIGS[$next_config]}"
    next_config=$((next_config + 1))
  done
}

collect_finished() {
  local new_pids=()
  local new_gpus=()
  local new_configs=()

  for ((i=0; i<${#RUNNING_PIDS[@]}; i++)); do
    local pid="${RUNNING_PIDS[$i]}"
    local gpu="${RUNNING_GPUS[$i]}"
    local config="${RUNNING_CONFIGS[$i]}"
    local status_file="${status_dir}/${pid}.status"

    if [[ -f "${status_file}" ]]; then
      local status
      status="$(cat "${status_file}")"
      rm -f "${status_file}"
      wait "${pid}" 2>/dev/null || true
      echo "[finish] GPU=${gpu} config=${config} status=${status}"
      if [[ "${status}" != "0" ]]; then
        failed=1
      fi
      FREE_GPUS+=("${gpu}")
    else
      new_pids+=("${pid}")
      new_gpus+=("${gpu}")
      new_configs+=("${config}")
    fi
  done

  RUNNING_PIDS=("${new_pids[@]}")
  RUNNING_GPUS=("${new_gpus[@]}")
  RUNNING_CONFIGS=("${new_configs[@]}")
}

launch_next_if_possible

while (( ${#RUNNING_PIDS[@]} > 0 )); do
  set +e
  wait -n
  set -e
  collect_finished
  launch_next_if_possible
done

if (( failed != 0 )); then
  echo "Some CamVid jobs failed."
  exit 1
fi

echo "All CamVid jobs finished."
