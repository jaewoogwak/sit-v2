#!/usr/bin/env bash
set -euo pipefail

# Helper script to sweep --search_limit values with/without --timing_on_gpu.
# Usage:
#   ./scripts/run_timing_sweep.sh "<base command without search_limit/timing flags>" [search_limit ...]
# Example:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_timing_sweep.sh "python src/main.py -d webvid_dummy_18 --gpu 0 --eval --eval_query_limit 1000 --search"

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"<base command>\" [search_limit ...]"
    exit 1
fi

BASE_CMD="$1"
shift

if [ $# -gt 0 ]; then
    SEARCH_LIMITS=("$@")
else
    SEARCH_LIMITS=(10000 30000 50000 100000 200000 300000)
fi

LOG_DIR="results/timing_sweep_logs"
mkdir -p "$LOG_DIR"

echo "Base command: $BASE_CMD"
echo "Search limits: ${SEARCH_LIMITS[*]}"
echo "Logs will be stored under $LOG_DIR"

for mode in on; do
    if [ "$mode" = "on" ]; then
        timing_flag="--timing_on_gpu"
    else
        timing_flag=""
    fi

    echo "\n=== timing_on_gpu=${mode} ==="
    for limit in "${SEARCH_LIMITS[@]}"; do
        label="timing_${mode}_limit_${limit}"
        log_file="$LOG_DIR/${label}.log"
        echo "--- search_limit=${limit} ---"
        cmd="$BASE_CMD $timing_flag --search_limit $limit"
        echo "Executing: $cmd"
        # shellcheck disable=SC2086
        eval "$cmd" | tee "$log_file"
    done
done

echo "\nSummary:"
grep -H "Similarity (end-to-end, incl argsort" "$LOG_DIR"/*.log | sed "s|$LOG_DIR/||"
