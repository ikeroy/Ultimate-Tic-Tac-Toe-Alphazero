#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ikidy/Projects/alphazero-pybind11"
PANE="0:0.0"
RUNS=(
  "collapse77-trainbest"
  "collapse77-noprune"
  "collapse77-replayanchor"
  "collapse77-explore"
)

send_codex() {
  local message="$1"
  tmux send-keys -t "$PANE" "$message"
  sleep 1
  tmux send-keys -t "$PANE" Enter
}

cd "$ROOT"
mkdir -p logs/train

send_codex "UTTT collapse branch runner started. Running ${RUNS[*]} sequentially from iter 77 to 95."

for run in "${RUNS[@]}"; do
  log="logs/train/${run}.log"
  attempt=1
  while true; do
    send_codex "Starting ${run} attempt ${attempt}. I will message again when it finishes."
    set +e
    uv run python src/train.py --resume "data/uttt/${run}" > "$log" 2>&1
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      break
    fi
    if [[ "$status" -eq 130 && "$attempt" -lt 2 ]]; then
      send_codex "${run} was interrupted with exit code 130 during attempt ${attempt}; retrying once from the same checkpoint."
      attempt=$((attempt + 1))
      continue
    fi
    send_codex "${run} failed with exit code ${status}. Inspect ${log} before continuing."
    exit "$status"
  done
  send_codex "${run} finished. Inspect data/uttt/${run} and ${log}; next branch will start now."
done

send_codex "All four UTTT collapse branch tests finished. Please inspect collapse77-* Aim/log data and summarize the result."
