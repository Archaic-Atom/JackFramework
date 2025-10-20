#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for the minimal JackFramework smoke test.
#
# Usage examples:
#   # CPU single-process
#   bash Scripts/run_minimal.sh
#
#   # Single node, 2 GPUs with torchrun if available (else mp.spawn)
#   GPUS=2 DIST=true bash Scripts/run_minimal.sh
#
#   # nohup with coloured progress to file
#   NOHUP=true LOG=run.log bash Scripts/run_minimal.sh
#
# Tunables via env (defaults shown):
#   PYTHON=python
#   DIST=false         # true to enable DDP
#   GPUS=0             # number of GPUs for DDP
#   EPOCHS=1 IMGS=32 VAL_IMGS=0 BATCH=8 DEBUG=false
#   NOHUP=false LOG=Result/run-minimal.log
#   TRAIN_LIST=/dev/null VAL_LIST=/dev/null   # override list paths if needed
#   JF_PROGRESS_COLOR=1  # coloured progress (set 0 or NO_COLOR=1 to disable)
#   PASSTHRU="--your extra --flags here"     # appended to the Python cmd

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ENTRY="$ROOT_DIR/Example/Minimal/main.py"

export PYTHONPATH="$ROOT_DIR/Source:${PYTHONPATH:-}"

# Defaults
DIST="${DIST:-false}"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-1}"
IMGS="${IMGS:-32}"
VAL_IMGS="${VAL_IMGS:-0}"
BATCH="${BATCH:-8}"
DEBUG="${DEBUG:-false}"
NOHUP_RUN="${NOHUP:-false}"
LOG_FILE="${LOG:-$ROOT_DIR/Result/run-minimal.log}"
TRAIN_LIST="${TRAIN_LIST:-/dev/null}"
VAL_LIST="${VAL_LIST:-/dev/null}"
PASSTHRU="${PASSTHRU:-}"

# Coloured progress by default; honour NO_COLOR/JF_PROGRESS_COLOR if already set
export JF_PROGRESS_COLOR="${JF_PROGRESS_COLOR:-1}"

declare -a CMD
if [[ "$DIST" == "true" && "$GPUS" -gt 0 ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    CMD=(torchrun --nproc_per_node="$GPUS" "$ENTRY" --dist true --gpu "$GPUS")
  else
    CMD=("$PYTHON" "$ENTRY" --dist true --gpu "$GPUS")
  fi
else
  CMD=("$PYTHON" "$ENTRY" --dist false --gpu 0)
fi

# Common training knobs
CMD+=(--imgNum "$IMGS" --valImgNum "$VAL_IMGS" --batchSize "$BATCH" --maxEpochs "$EPOCHS" --debug "$DEBUG")
# Minimal interface ignores list contents, but init checks path existence.
CMD+=(--trainListPath "$TRAIN_LIST" --valListPath "$VAL_LIST")

# Append any extra user-provided flags
if [[ -n "${PASSTHRU}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${PASSTHRU} )
  CMD+=("${EXTRA[@]}")
fi

echo "[run_minimal] PYTHONPATH=$PYTHONPATH"
echo "[run_minimal] TRAIN_LIST=$TRAIN_LIST VAL_LIST=$VAL_LIST"
echo "[run_minimal] CMD: ${CMD[*]}"

if [[ "$NOHUP_RUN" == "true" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  # Use nohup for background; preserve colours in the log when JF_PROGRESS_COLOR=1
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo "[run_minimal] Started in background (PID $!). Log: $LOG_FILE"
else
  exec "${CMD[@]}"
fi
