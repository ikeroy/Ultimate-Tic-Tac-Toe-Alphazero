#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ikidy/Projects/alphazero-pybind11"
cd "$ROOT"

OUT="logs/debug_collapse"
mkdir -p "$OUT"

PANE="${CODEX_TMUX_PANE:-0:0.0}"
EXP="data/uttt/micromacro-6d24c-g4096-m64-lr00737-from77-to120"
MODEL="models/uttt_ai/policy_value_net_stage2.onnx"
ITERS=(77 88 95 100 103)

msg() {
  local text="$1"
  tmux send-keys -t "$PANE" "$text"
  sleep 1
  tmux send-keys -t "$PANE" Enter
}

pause_for_steering() {
  local step="$1"
  local control="$OUT/control.txt"
  msg "Codex diagnostic ${step} finished. Inspect logs/debug_collapse and decide whether to continue, adjust, or stop. I will continue automatically in 10 minutes unless control.txt says stop."
  for _ in $(seq 1 60); do
    if [[ -f "$control" ]] && grep -qi '^stop' "$control"; then
      msg "Codex diagnostic runner stopping because logs/debug_collapse/control.txt requested stop after ${step}."
      exit 0
    fi
    sleep 10
  done
}

run_uttt_ai_eval() {
  local iter="$1"
  local ckpt="$EXP/checkpoint/$(printf '%04d' "$iter")-micromacro-6d24c-g4096-m64-lr00737-from77-to120.pt"
  uv run python src/tournament.py \
    --game uttt \
    --format roundrobin \
    --uttt-ai \
    --uttt-ai-model "$MODEL" \
    --uttt-ai-require-cuda \
    --networks "$ckpt" \
    --mcts_visits 256 \
    --batch_size 256 \
    --games 256 \
    --cache_size 200000 \
    > "$OUT/step1_uttt_ai_iter${iter}_256g256v.log" 2>&1
}

run_pair_eval() {
  local a="$1"
  local b="$2"
  local ca="$EXP/checkpoint/$(printf '%04d' "$a")-micromacro-6d24c-g4096-m64-lr00737-from77-to120.pt"
  local cb="$EXP/checkpoint/$(printf '%04d' "$b")-micromacro-6d24c-g4096-m64-lr00737-from77-to120.pt"
  uv run python src/tournament.py \
    --game uttt \
    --format roundrobin \
    --networks "$ca" \
    --networks "$cb" \
    --mcts_visits 128 \
    --batch_size 128 \
    --games 512 \
    --cache_size 200000 \
    > "$OUT/step4_pair_${a}_vs_${b}_512g128v.log" 2>&1
}

{
  echo "Starting collapse diagnostics at $(date -Is)"
  echo "Experiment: $EXP"
  echo "Iters: ${ITERS[*]}"
} > "$OUT/runner.log"

msg "Codex diagnostic runner started for UTTT collapse debugging. Step 1 is fixed UTTT.ai tournaments for iters ${ITERS[*]}."

for iter in "${ITERS[@]}"; do
  echo "Step 1 eval iter $iter at $(date -Is)" >> "$OUT/runner.log"
  run_uttt_ai_eval "$iter"
done
echo "Step 1 finished at $(date -Is)" >> "$OUT/runner.log"
pause_for_steering "step 1 fixed UTTT.ai tournaments"

echo "Step 2 fixed validation loss at $(date -Is)" >> "$OUT/runner.log"
uv run python tools/diagnose_uttt_collapse.py loss-policy \
  --exp-dir "$EXP" \
  --iters "${ITERS[@]}" \
  --sample-max-iter 77 \
  --sample-size 4096 \
  --batch-size 512 \
  --seed 0 \
  --out "$OUT/step2_fixed_loss_policy.json" \
  > "$OUT/step2_fixed_loss_policy.log" 2>&1
echo "Step 2 finished at $(date -Is)" >> "$OUT/runner.log"
pause_for_steering "step 2 fixed validation loss"

echo "Step 3 policy drift already computed with step 2 at $(date -Is)" >> "$OUT/runner.log"
pause_for_steering "step 3 policy drift and entropy"

echo "Step 4 gate reliability at $(date -Is)" >> "$OUT/runner.log"
uv run python tools/diagnose_uttt_collapse.py gate-summary \
  --exp-dir "$EXP" \
  --iters "${ITERS[@]}" \
  --out "$OUT/step4_gate_summary.json" \
  > "$OUT/step4_gate_summary.log" 2>&1
run_pair_eval 77 88
run_pair_eval 88 100
run_pair_eval 77 100
echo "Step 4 finished at $(date -Is)" >> "$OUT/runner.log"
msg "Codex diagnostic step 4 finished. All collapse diagnostics are complete. Please inspect logs/debug_collapse/step*.md and tournament logs, then summarize cause and recommended fix."
