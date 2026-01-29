#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-tvr}
SPLIT=${SPLIT:-val}
BOUNDARIES_JSON=${BOUNDARIES_JSON:-/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_${DATASET}_${SPLIT}.json}
FRAMES_ROOT=${FRAMES_ROOT:-/dev/hdd2/gjw/datasets/tvr/frames}
VIDEO_ID=${VIDEO_ID:-house_s03e05_seg02_clip_20}
OUTPUT_PATH=${OUTPUT_PATH:-/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/vis/${VIDEO_ID}_boundaries.jpg}
LEVEL=${LEVEL:-levels}
SHOW_COARSE=${SHOW_COARSE:-1}

STRIDE=${STRIDE:-1}
COLS=${COLS:-10}
TILE_WIDTH=${TILE_WIDTH:-160}
TILE_HEIGHT=${TILE_HEIGHT:-120}
DRAW_INDEX=${DRAW_INDEX:-1}

ARGS=(
  --boundaries_json "${BOUNDARIES_JSON}"
  --frames_root "${FRAMES_ROOT}"
  --video_id "${VIDEO_ID}"
  --output_path "${OUTPUT_PATH}"
  --level "${LEVEL}"
  --stride "${STRIDE}"
  --cols "${COLS}"
  --tile_width "${TILE_WIDTH}"
  --tile_height "${TILE_HEIGHT}"
)

if [[ "${SHOW_COARSE}" == "1" ]]; then
  ARGS+=(--show_coarse)
fi

if [[ "${DRAW_INDEX}" == "1" ]]; then
  ARGS+=(--draw_index)
fi

python /dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/visualize_boundaries.py "${ARGS[@]}" "$@"
