#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../temp:${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TS}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[run.sh] Log file: ${LOG_FILE}"

DATASET="charades"

H5_PATH=${H5_PATH:-}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-json}
SEQ_LEN=${SEQ_LEN:-3}
FEATURE_DIM=${FEATURE_DIM:-512}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-20}
NUM_LAYERS=${NUM_LAYERS:-2}
NUM_HEADS=${NUM_HEADS:-8}
FF_DIM=${FF_DIM:-20248}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./checkpoints}
USE_LOCAL_MAX=${USE_LOCAL_MAX:-0}
ALPHA=${ALPHA:-3}

FINE_CHECKPOINT_DIR=${FINE_CHECKPOINT_DIR:-${CHECKPOINT_DIR}/fine}
LEVEL_CHECKPOINT_DIR=${LEVEL_CHECKPOINT_DIR:-${CHECKPOINT_DIR}/level}
COARSE_CHECKPOINT_DIR=${COARSE_CHECKPOINT_DIR:-${LEVEL_CHECKPOINT_DIR}}
RUN_COARSE=${RUN_COARSE:-1}
COARSE_SEQ_LEN=${COARSE_SEQ_LEN:-3}
COARSE_EPOCHS=${COARSE_EPOCHS:-20}
COARSE_LR=${COARSE_LR:-1e-4}
COARSE_NUM_LAYERS=${COARSE_NUM_LAYERS:-2}
COARSE_NUM_HEADS=${COARSE_NUM_HEADS:-8}
COARSE_FF_DIM=${COARSE_FF_DIM:-2048}
COARSE_USE_LOCAL_MAX=${COARSE_USE_LOCAL_MAX:-0}
COARSE_ALPHA=${COARSE_ALPHA:-2}
COARSE_THRESHOLD_STD=${COARSE_THRESHOLD_STD:-0}
COARSE_THRESHOLD_MODE=${COARSE_THRESHOLD_MODE:-mad}
COARSE_MODE=${COARSE_MODE:-peaks}
COARSE_SIM_THRESHOLD=${COARSE_SIM_THRESHOLD:-0.65}
SEGMENT_POOLING=${SEGMENT_POOLING:-self_attn}

RECURSIVE_LEVELS=${RECURSIVE_LEVELS:-5}
RECURSIVE_SEQ_LEN=${RECURSIVE_SEQ_LEN:-3}
RECURSIVE_ALPHA=${RECURSIVE_ALPHA:-2}
RECURSIVE_A=${RECURSIVE_A:-0}
RECURSIVE_USE_LOCAL_MAX=${RECURSIVE_USE_LOCAL_MAX:-0}

RECURSIVE_SEQ_LEN_LIST=${RECURSIVE_SEQ_LEN_LIST:-}
RECURSIVE_ALPHA_LIST=${RECURSIVE_ALPHA_LIST:-}
RECURSIVE_A_LIST=${RECURSIVE_A_LIST:-}

RECURSIVE_CHECKPOINT=${RECURSIVE_CHECKPOINT:-}

RECURSIVE_UNTIL_ONE=${RECURSIVE_UNTIL_ONE:-1}

DEFAULT_H5_PATH="./output/charades_frame_embeds.h5"
DEFAULT_NPY_DIR="/dev/hdd2/gjw/datasets/charades/features"
DEFAULT_TRAIN_JSONL_PATHS="/dev/hdd2/gjw/datasets/charades/charades_train.jsonl,/dev/hdd2/gjw/datasets/charades/charades_val.jsonl"
DEFAULT_INFER_JSONL_PATHS="/dev/hdd2/gjw/datasets/charades/charades_train.jsonl,/dev/hdd2/gjw/datasets/charades/charades_val.jsonl"
DEFAULT_INFER_OUTPUT_JSONS="./output/boundaries_cha_train.${OUTPUT_FORMAT},./output/boundaries_cha_val.${OUTPUT_FORMAT}"
DEFAULT_MERGE_TRAIN_VAL=1

if [[ -z "${H5_PATH}" ]]; then
  H5_PATH="${DEFAULT_H5_PATH}"
fi

if [[ -n "${JSONL_PATH:-}" ]]; then
  TRAIN_JSONL_PATH="${JSONL_PATH}"
  INFER_JSONL_PATHS="${JSONL_PATH}"
  INFER_OUTPUT_JSONS="${OUTPUT_JSON:-./output/boundaries.${OUTPUT_FORMAT}}"
  MERGE_TRAIN_VAL=${MERGE_TRAIN_VAL:-0}
else
  TRAIN_JSONL_PATHS=${TRAIN_JSONL_PATHS:-${DEFAULT_TRAIN_JSONL_PATHS}}
  INFER_JSONL_PATHS=${INFER_JSONL_PATHS:-${DEFAULT_INFER_JSONL_PATHS}}
  INFER_OUTPUT_JSONS=${INFER_OUTPUT_JSONS:-${DEFAULT_INFER_OUTPUT_JSONS}}
  MERGE_TRAIN_VAL=${MERGE_TRAIN_VAL:-${DEFAULT_MERGE_TRAIN_VAL}}
fi

NPY_DIR=${NPY_DIR:-${DEFAULT_NPY_DIR:-}}
BUILD_H5_IF_MISSING=${BUILD_H5_IF_MISSING:-1}
H5_JSONL_PATHS=${H5_JSONL_PATHS:-${TRAIN_JSONL_PATHS}}

if [[ ! -f "${H5_PATH}" ]]; then
  if [[ "${BUILD_H5_IF_MISSING}" == "1" ]]; then
    if [[ -z "${NPY_DIR}" ]]; then
      echo "H5_PATH not found and NPY_DIR is empty. Set NPY_DIR to build H5." >&2
      exit 1
    fi
    echo "[run.sh] H5 not found. Building from NPY: ${H5_PATH}"
    BUILD_ARGS=()
    if [[ "${DATASET}" == "tvr" ]]; then
      BUILD_ARGS+=(--tvr_style)
    fi
    python "${SCRIPT_DIR}/build_h5_from_npy.py"       --jsonl_paths "${H5_JSONL_PATHS}"       --npy_dir "${NPY_DIR}"       --h5_out "${H5_PATH}"       "${BUILD_ARGS[@]}"       --allow_missing
  else
    echo "H5_PATH not found: ${H5_PATH}" >&2
    exit 1
  fi
fi

if [[ "${MERGE_TRAIN_VAL}" == "1" ]]; then
  MERGED_JSONL=${MERGED_JSONL:-./output/${DATASET}_trainval_merged.jsonl}
  mkdir -p "$(dirname "${MERGED_JSONL}")"
  IFS=',' read -r -a TRAIN_LIST <<< "${TRAIN_JSONL_PATHS}"
  : > "${MERGED_JSONL}"
  for fpath in "${TRAIN_LIST[@]}"; do
    if [[ -s "${MERGED_JSONL}" ]]; then
      printf '
' >> "${MERGED_JSONL}"
    fi
    cat "${fpath}" >> "${MERGED_JSONL}"
  done
  TRAIN_JSONL_PATH="${MERGED_JSONL}"
  TRAIN_SPLIT=${TRAIN_SPLIT:-all}
  echo "[run.sh] Merged train/val JSONL: ${TRAIN_JSONL_PATH}"
else
  TRAIN_SPLIT=${TRAIN_SPLIT:-train}
  if [[ -z "${TRAIN_JSONL_PATH:-}" ]]; then
    TRAIN_JSONL_PATH="${TRAIN_JSONL_PATHS%%,*}"
  fi
  echo "[run.sh] Train JSONL: ${TRAIN_JSONL_PATH}"
fi

INFER_SPLIT=${INFER_SPLIT:-all}
echo "[run.sh] Inference JSONLs: ${INFER_JSONL_PATHS}"
echo "[run.sh] Inference outputs: ${INFER_OUTPUT_JSONS}"
echo "[run.sh] Coarse mode: ${COARSE_MODE} (sim_threshold=${COARSE_SIM_THRESHOLD})"
echo "[run.sh] Fine params: seq_len=${SEQ_LEN} feature_dim=${FEATURE_DIM} batch_size=${BATCH_SIZE} lr=${LR} epochs=${EPOCHS} layers=${NUM_LAYERS} heads=${NUM_HEADS} ff_dim=${FF_DIM} use_local_max=${USE_LOCAL_MAX} alpha=${ALPHA:-5} threshold_std=${THRESHOLD_STD:-1.0} segment_pooling=${SEGMENT_POOLING}"
echo "[run.sh] Level params: seq_len=${COARSE_SEQ_LEN} batch_size=${BATCH_SIZE} lr=${COARSE_LR} epochs=${COARSE_EPOCHS} layers=${COARSE_NUM_LAYERS} heads=${COARSE_NUM_HEADS} ff_dim=${COARSE_FF_DIM} use_local_max=${COARSE_USE_LOCAL_MAX} alpha=${COARSE_ALPHA} threshold_std=${COARSE_THRESHOLD_STD} threshold_mode=${COARSE_THRESHOLD_MODE} ckpt_dir=${COARSE_CHECKPOINT_DIR}"
echo "[run.sh] Recursive params: levels=${RECURSIVE_LEVELS} until_one=${RECURSIVE_UNTIL_ONE} seq_len=${RECURSIVE_SEQ_LEN} seq_len_list=${RECURSIVE_SEQ_LEN_LIST} alpha=${RECURSIVE_ALPHA} alpha_list=${RECURSIVE_ALPHA_LIST} a=${RECURSIVE_A} a_list=${RECURSIVE_A_LIST} checkpoint=${RECURSIVE_CHECKPOINT} use_local_max=${RECURSIVE_USE_LOCAL_MAX}"
mode="fine"
if [[ "${RECURSIVE_UNTIL_ONE}" == "1" ]] || [[ "${RECURSIVE_LEVELS}" != "0" ]]; then
  mode+="+recursive"
fi
if [[ "${RUN_COARSE}" == "1" ]]; then
  mode+="+coarse"
fi
echo "[run.sh] Mode: ${mode}"

train_cmd=(
  python "${SCRIPT_DIR}/../temp/train.py"
  --jsonl_path "${TRAIN_JSONL_PATH}"
  --h5_path "${H5_PATH}"
  --seq_len "${SEQ_LEN}"
  --feature_dim "${FEATURE_DIM}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --num_layers "${NUM_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --ff_dim "${FF_DIM}"
  --split "${TRAIN_SPLIT}"
  --checkpoint_dir "${FINE_CHECKPOINT_DIR}"
)
"${train_cmd[@]}"

FINE_CKPT="${FINE_CHECKPOINT_DIR}/model_epoch_${EPOCHS}.pt"

if [[ "${RUN_COARSE}" == "1" ]]; then
  # Coarse model trains on pooled segment embeddings produced from the merged train/val JSONL.
  if [[ "${MERGE_TRAIN_VAL}" != "1" ]]; then
    echo "RUN_COARSE=1 requires MERGE_TRAIN_VAL=1 to build a single trainval segment H5." >&2
    exit 1
  fi

  TRAINVAL_SEGMENT_H5=${TRAINVAL_SEGMENT_H5:-./output/segments_trainval.h5}
  TRAINVAL_BOUNDARIES=${TRAINVAL_BOUNDARIES:-./output/boundaries_trainval_fine.${OUTPUT_FORMAT}}
  echo "[run.sh] Coarse trainval segment H5: ${TRAINVAL_SEGMENT_H5}"
  echo "[run.sh] Coarse trainval boundaries: ${TRAINVAL_BOUNDARIES}"

  EXTRA_ARGS=()
  if [[ "${USE_LOCAL_MAX}" != "1" ]]; then
    EXTRA_ARGS+=(--no_local_max)
  fi
    EXTRA_ARGS+=(--output_format "${OUTPUT_FORMAT}")
    EXTRA_ARGS+=(--alpha "${ALPHA}")
  EXTRA_ARGS+=(--segment_h5_out "${TRAINVAL_SEGMENT_H5}")
  EXTRA_ARGS+=(--coarse_alpha "${COARSE_ALPHA}")
  EXTRA_ARGS+=(--coarse_threshold_std "${COARSE_THRESHOLD_STD}")
  EXTRA_ARGS+=(--coarse_threshold_mode "${COARSE_THRESHOLD_MODE}")
  EXTRA_ARGS+=(--coarse_mode "${COARSE_MODE}")
  EXTRA_ARGS+=(--coarse_sim_threshold "${COARSE_SIM_THRESHOLD}")
  EXTRA_ARGS+=(--segment_pooling "${SEGMENT_POOLING}")

  EXTRA_ARGS+=(--recursive_levels "${RECURSIVE_LEVELS}")
  EXTRA_ARGS+=(--recursive_seq_len "${RECURSIVE_SEQ_LEN}")
  EXTRA_ARGS+=(--recursive_seq_len_list "${RECURSIVE_SEQ_LEN_LIST}")
  EXTRA_ARGS+=(--recursive_alpha "${RECURSIVE_ALPHA}")
  EXTRA_ARGS+=(--recursive_alpha_list "${RECURSIVE_ALPHA_LIST}")
  EXTRA_ARGS+=(--recursive_a "${RECURSIVE_A}")
  EXTRA_ARGS+=(--recursive_a_list "${RECURSIVE_A_LIST}")
  if [[ -n "${RECURSIVE_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--recursive_checkpoint "${RECURSIVE_CHECKPOINT}")
  fi
  if [[ "${RECURSIVE_USE_LOCAL_MAX}" != "1" ]]; then
    EXTRA_ARGS+=(--recursive_no_local_max)
  fi
  if [[ "${RECURSIVE_UNTIL_ONE}" == "1" ]]; then
    EXTRA_ARGS+=(--recursive_until_one)
  fi

  infer_cmd=(
    python "${SCRIPT_DIR}/inference_boundary.py"
    --jsonl_path "${TRAIN_JSONL_PATH}"
    --h5_path "${H5_PATH}"
    --seq_len "${SEQ_LEN}"
    --feature_dim "${FEATURE_DIM}"
    --batch_size "${BATCH_SIZE}"
    --checkpoint "${FINE_CKPT}"
    --num_layers "${NUM_LAYERS}"
    --num_heads "${NUM_HEADS}"
    --ff_dim "${FF_DIM}"
    --split "${INFER_SPLIT}"
    --output_json "${TRAINVAL_BOUNDARIES}"
    "${EXTRA_ARGS[@]}"
  )
  "${infer_cmd[@]}"

  coarse_train_cmd=(
    python "${SCRIPT_DIR}/../temp/train.py"
    --jsonl_path "${TRAIN_JSONL_PATH}"
    --h5_path "${TRAINVAL_SEGMENT_H5}"
    --seq_len "${COARSE_SEQ_LEN}"
    --feature_dim "${FEATURE_DIM}"
    --batch_size "${BATCH_SIZE}"
    --lr "${COARSE_LR}"
    --epochs "${COARSE_EPOCHS}"
    --num_layers "${COARSE_NUM_LAYERS}"
    --num_heads "${COARSE_NUM_HEADS}"
    --ff_dim "${COARSE_FF_DIM}"
    --split "${TRAIN_SPLIT}"
    --checkpoint_dir "${COARSE_CHECKPOINT_DIR}"
  )
  "${coarse_train_cmd[@]}"
fi

JSONL_LIST=(${INFER_JSONL_PATHS//,/ })
OUTPUT_LIST=(${INFER_OUTPUT_JSONS//,/ })

if [[ "${#JSONL_LIST[@]}" -ne "${#OUTPUT_LIST[@]}" ]]; then
  echo "INFER_JSONL_PATHS and INFER_OUTPUT_JSONS must have the same number of entries." >&2
  exit 1
fi

for idx in "${!JSONL_LIST[@]}"; do
  EXTRA_ARGS=()
  if [[ "${USE_LOCAL_MAX}" != "1" ]]; then
    EXTRA_ARGS+=(--no_local_max)
  fi
  EXTRA_ARGS+=(--output_format "${OUTPUT_FORMAT}")
  EXTRA_ARGS+=(--alpha "${ALPHA}")
  EXTRA_ARGS+=(--segment_pooling "${SEGMENT_POOLING}")

  EXTRA_ARGS+=(--recursive_levels "${RECURSIVE_LEVELS}")
  EXTRA_ARGS+=(--recursive_seq_len "${RECURSIVE_SEQ_LEN}")
  EXTRA_ARGS+=(--recursive_seq_len_list "${RECURSIVE_SEQ_LEN_LIST}")
  EXTRA_ARGS+=(--recursive_alpha "${RECURSIVE_ALPHA}")
  EXTRA_ARGS+=(--recursive_alpha_list "${RECURSIVE_ALPHA_LIST}")
  EXTRA_ARGS+=(--recursive_a "${RECURSIVE_A}")
  EXTRA_ARGS+=(--recursive_a_list "${RECURSIVE_A_LIST}")
  if [[ -n "${RECURSIVE_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--recursive_checkpoint "${RECURSIVE_CHECKPOINT}")
  fi
  if [[ "${RECURSIVE_USE_LOCAL_MAX}" != "1" ]]; then
    EXTRA_ARGS+=(--recursive_no_local_max)
  fi
  if [[ "${RECURSIVE_UNTIL_ONE}" == "1" ]]; then
    EXTRA_ARGS+=(--recursive_until_one)
  fi

  SEGMENT_H5_OUT="./output/segments_$(basename "${OUTPUT_LIST[$idx]}")"
  SEGMENT_H5_OUT="${SEGMENT_H5_OUT%.*}.h5"
  EXTRA_ARGS+=(--segment_h5_out "${SEGMENT_H5_OUT}")

  if [[ "${RUN_COARSE}" == "1" ]]; then
    COARSE_CKPT="${COARSE_CHECKPOINT_DIR}/model_epoch_${COARSE_EPOCHS}.pt"
    if [[ -z "${RECURSIVE_CHECKPOINT}" ]]; then
      RECURSIVE_CHECKPOINT="${COARSE_CKPT}"
    fi
    EXTRA_ARGS+=(--coarse_checkpoint "${COARSE_CKPT}")
    EXTRA_ARGS+=(--coarse_seq_len "${COARSE_SEQ_LEN}")
    EXTRA_ARGS+=(--coarse_alpha "${COARSE_ALPHA}")
    EXTRA_ARGS+=(--coarse_threshold_std "${COARSE_THRESHOLD_STD}")
    EXTRA_ARGS+=(--coarse_threshold_mode "${COARSE_THRESHOLD_MODE}")
    EXTRA_ARGS+=(--coarse_mode "${COARSE_MODE}")
    EXTRA_ARGS+=(--coarse_sim_threshold "${COARSE_SIM_THRESHOLD}")
    EXTRA_ARGS+=(--coarse_num_layers "${COARSE_NUM_LAYERS}")
    EXTRA_ARGS+=(--coarse_num_heads "${COARSE_NUM_HEADS}")
    EXTRA_ARGS+=(--coarse_ff_dim "${COARSE_FF_DIM}")
    if [[ "${COARSE_USE_LOCAL_MAX}" != "1" ]]; then
      EXTRA_ARGS+=(--coarse_no_local_max)
    fi
  fi
  infer_cmd=(
    python "${SCRIPT_DIR}/inference_boundary.py"
    --jsonl_path "${JSONL_LIST[$idx]}"
    --h5_path "${H5_PATH}"
    --seq_len "${SEQ_LEN}"
    --feature_dim "${FEATURE_DIM}"
    --batch_size "${BATCH_SIZE}"
    --checkpoint "${FINE_CKPT}"
    --num_layers "${NUM_LAYERS}"
    --num_heads "${NUM_HEADS}"
    --ff_dim "${FF_DIM}"
    --split "${INFER_SPLIT}"
    --output_json "${OUTPUT_LIST[$idx]}"
    "${EXTRA_ARGS[@]}"
  )
  "${infer_cmd[@]}"
done
