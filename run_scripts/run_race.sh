#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/mnt/data/wangzijian/object_detection_datasets/datasets}"
SHOT="${SHOT:-1}"
SEED="${SEED:-33}"
GPU="${GPU:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/race_dinov2_vitl14_${SHOT}shot_seed${SEED}_${RUN_TIMESTAMP}}"
RACE_SCORE_THRESHOLD="${RACE_SCORE_THRESHOLD:-}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/data/wangzijian/FSOD-VFM-Public/fsod_cache}"
REBUILD_PROTOTYPE_CACHE="${REBUILD_PROTOTYPE_CACHE:-0}"
REBUILD_SUPPORT_FEATURE_CACHE="${REBUILD_SUPPORT_FEATURE_CACHE:-0}"
DISABLE_PROTOTYPE_CACHE="${DISABLE_PROTOTYPE_CACHE:-0}"
DISABLE_SUPPORT_FEATURE_CACHE="${DISABLE_SUPPORT_FEATURE_CACHE:-0}"
SUPPORT_BOX_BATCH_SIZE="${SUPPORT_BOX_BATCH_SIZE:-64}"
SKIP_COCO_EVAL="${SKIP_COCO_EVAL:-1}"

mkdir -p "${OUTPUT_DIR}"
echo "Run results will be saved to: ${OUTPUT_DIR}"

SUPPORT_JSON="./data/race/support/${SHOT}shot_seed${SEED}.json"
if [[ ! -f "${SUPPORT_JSON}" ]]; then
  echo "Support file not found: ${SUPPORT_JSON}" >&2
  echo "Generate it first with tools/prepare_race_dataset.py (SHOT may be a number or 'all')." >&2
  exit 1
fi

RACE_EVAL_ARGS=(--race_eval --race_eval_output_dir "${OUTPUT_DIR}")
if [[ -n "${RACE_SCORE_THRESHOLD}" ]]; then
  RACE_EVAL_ARGS+=(--race_score_threshold "${RACE_SCORE_THRESHOLD}")
fi

CACHE_ARGS=(
  --prototype_cache_dir "${CACHE_ROOT}/prototypes"
  --support_feature_cache_dir "${CACHE_ROOT}/support_features"
  --support_box_batch_size "${SUPPORT_BOX_BATCH_SIZE}"
)
if [[ "${SKIP_COCO_EVAL}" == "1" ]]; then
  CACHE_ARGS+=(--skip_coco_eval)
fi
if [[ "${REBUILD_PROTOTYPE_CACHE}" == "1" ]]; then
  CACHE_ARGS+=(--rebuild_prototype_cache)
fi
if [[ "${REBUILD_SUPPORT_FEATURE_CACHE}" == "1" ]]; then
  CACHE_ARGS+=(--rebuild_support_feature_cache)
fi
if [[ "${DISABLE_PROTOTYPE_CACHE}" == "1" ]]; then
  CACHE_ARGS+=(--disable_prototype_cache)
fi
if [[ "${DISABLE_SUPPORT_FEATURE_CACHE}" == "1" ]]; then
  CACHE_ARGS+=(--disable_support_feature_cache)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python ./main.py \
  --json_path "${SUPPORT_JSON}" \
  --test_json "./data/race/annotations/val_seed${SEED}.json" \
  --test_img_dir "${DATASET_ROOT}/images/train" \
  --data_dir "${DATASET_ROOT}" \
  --pred_json "${OUTPUT_DIR}/race_${SHOT}shot_seed${SEED}_predictions.json" \
  --model_version "${MODEL_VERSION:-dinov2_vitl14}" \
  --feat_extractor_name DINOV2 \
  --repo_or_dir /mnt/data/wangzijian/FSOD-VFM-Public/dinov2 \
  --dinov2_checkpoint_dir /mnt/data/wangzijian/FSOD-VFM-Public/checkpoints \
  --min_threshold 0.01 \
  --diffusion_steps 30 \
  --alp 0.3 \
  --lamb 0.5 \
  "${CACHE_ARGS[@]}" \
  "${RACE_EVAL_ARGS[@]}"
