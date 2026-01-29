#!/usr/bin/env bash
set -euo pipefail

BOUNDARIES_JSON=${BOUNDARIES_JSON:-/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_val.json}
FRAMES_ROOT=${FRAMES_ROOT:-/dev/hdd2/gjw/datasets/tvr/frames}
VIDEO_ID=${VIDEO_ID:-house_s08e08_seg02_clip_05}
OUTPUT_PATH=${OUTPUT_PATH:-/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/${VIDEO_ID}_boundaries_diffs.jpg}
LEVEL=${LEVEL:-fine}

STRIDE=${STRIDE:-1}
COLS=${COLS:-10}
TILE_WIDTH=${TILE_WIDTH:-160}
TILE_HEIGHT=${TILE_HEIGHT:-120}
DRAW_INDEX=${DRAW_INDEX:-1}
GRAPH_HEIGHT=${GRAPH_HEIGHT:-240}
GRAPH_PADDING=${GRAPH_PADDING:-24}
ROWS_PER_PAGE=${ROWS_PER_PAGE:-5}

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
  --graph_height "${GRAPH_HEIGHT}"
  --graph_padding "${GRAPH_PADDING}"
  --rows_per_page "${ROWS_PER_PAGE}"
)

if [[ "${DRAW_INDEX}" == "1" ]]; then
  ARGS+=(--draw_index)
fi

python /dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/visualize_boundaries_with_diffs.py "${ARGS[@]}"
