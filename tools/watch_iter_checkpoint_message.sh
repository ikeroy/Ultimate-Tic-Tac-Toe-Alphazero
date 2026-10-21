#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ikidy/Projects/alphazero-pybind11"
RUN_NAME="${1:?run name required}"
ITER="${2:?iter required}"
PANE="${3:-0:0.0}"
INTERVAL="${4:-60}"

checkpoint=$(printf "%04d-%s.pt" "$ITER" "$RUN_NAME")
checkpoint_path="$ROOT/data/uttt/$RUN_NAME/checkpoint/$checkpoint"
log_path="$ROOT/logs/train/$RUN_NAME.log"

send_codex() {
  local message="$1"
  tmux send-keys -t "$PANE" "$message"
  sleep 1
  tmux send-keys -t "$PANE" Enter
}

while true; do
  if [[ -f "$checkpoint_path" ]]; then
    send_codex "Run ${RUN_NAME} reached checkpoint $(printf "%04d" "$ITER"). Inspect data/uttt/${RUN_NAME} and ${log_path}, then decide whether to continue or start a different branch."
    exit 0
  fi
  if ! pgrep -u "$USER" -f "src/train.py --resume data/uttt/${RUN_NAME}" >/dev/null; then
    send_codex "Run ${RUN_NAME} stopped before checkpoint $(printf "%04d" "$ITER"). Inspect ${log_path}."
    exit 0
  fi
  sleep "$INTERVAL"
done
