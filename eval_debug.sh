#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-act_frames}"
CKPT="${2:-/dev/ssd1/gjw/prvr/semantic-transformer-v2/results/act_frames/SiT-act-frames-gmm-softmil-c7-f3-level-complete-model/best.ckpt}"
VIDEO_ID="${3:-v_9XmzbuByY_E}"
shift $(( $# < 3 ? $# : 3 ))

case "${DATASET}" in
  activitynet)
    DATASET="act_frames"
    ;;
esac

if [[ -z "$VIDEO_ID" ]]; then
  echo "Usage: ./eval_debug.sh <dataset> <ckpt_path> <video_id>"
  echo "Example: ./eval_debug.sh tvr_frames checkpoints/best.ckpt tvr_12345"
  echo "Example: ./eval_debug.sh act_frames checkpoints/best.ckpt v_123ABC"
  exit 1
fi

TOPK="${TOPK:-10}"
SEG_TOPK="${SEG_TOPK:-10}"
GPU="${GPU:-0}"
FRAMES_ROOT="${FRAMES_ROOT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_ROOT="${OUT_ROOT:-debug}"
TS="$(date +%Y%m%d_%H%M%S)"
SAFE_VIDEO_ID="$(printf "%s" "$VIDEO_ID" | tr '/ ' '__')"
OUT_DIR="${OUT_ROOT}/${TS}_${SAFE_VIDEO_ID}"
VIZ_CMD="${VIZ_CMD:-}"
if [[ -z "${FRAMES_ROOT}" ]]; then
  case "${DATASET}" in
    act|activitynet|act_frames)
      FRAMES_ROOT="/dev/hdd2/gjw/datasets/activitynet/frames"
      ;;
    tvr|tvr_frames|tvr_clip)
      FRAMES_ROOT="/dev/hdd2/gjw/datasets/tvr/frames"
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="${2:-0}"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="${2:-$OUT_DIR}"
      shift 2
      ;;
    --viz_cmd)
      VIZ_CMD="${2:-}"
      shift 2
      ;;
    --frames_root)
      FRAMES_ROOT="${2:-}"
      shift 2
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} $*"
      fi
      break
      ;;
    *)
      EXTRA_ARGS="${EXTRA_ARGS} $1"
      shift
      ;;
  esac
done

CMD=(python src/main.py -d "$DATASET" --eval --gpu "$GPU" \
  --eval_debug_vid "$VIDEO_ID" \
  --eval_debug_topk "$TOPK" \
  --eval_debug_segment_topk "$SEG_TOPK")

if [[ -n "$FRAMES_ROOT" ]]; then
  CMD+=(--eval_debug_frames_root "$FRAMES_ROOT")
fi

if [[ -n "$CKPT" ]]; then
  CMD+=(--resume "$CKPT")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  CMD+=($EXTRA_ARGS)
fi

mkdir -p "$OUT_DIR"
printf "%s\n" "${CMD[*]}" > "${OUT_DIR}/command.txt"
echo "Running: ${CMD[*]}"
echo "Output: ${OUT_DIR}/log.txt"
if [[ -n "$VIZ_CMD" ]]; then
  echo "Viz: ${VIZ_CMD}"
fi

(
  "${CMD[@]}"
  if [[ -n "$VIZ_CMD" ]]; then
    echo "[eval_debug] running viz_cmd..."
    bash -lc "$VIZ_CMD"
  fi
) 2>&1 | tee "${OUT_DIR}/log.txt"
