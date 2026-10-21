#!/usr/bin/env python3
"""Run one UTTT.ai DAgger-style imitation trial.

The rollout policy is a student checkpoint, but every reached position is
labelled by UTTT.ai's raw policy distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import alphazero
import neural_net
from game_runner import play_raw_policy_match
from tools.uttt_ai_adapter import UtttAiPolicyValueNet


def send_codex(message: str, pane: str) -> None:
    try:
        subprocess.run(["tmux", "send-keys", "-t", pane, message], check=False)
        time.sleep(1)
        subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], check=False)
    except Exception:
        pass


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        return int(subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode)


def latest_checkpoint(exp_dir: Path) -> Path:
    checkpoints = sorted((exp_dir / "checkpoint").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {exp_dir / 'checkpoint'}")
    return checkpoints[-1]


def eval_raw_policy(checkpoint: Path, teacher_model: Path, games: int, seed: int) -> dict:
    student = neural_net.NNWrapper.load_checkpoint(
        alphazero.UltimateTicTacToeGS, str(checkpoint.parent), checkpoint.name
    )
    teacher = UtttAiPolicyValueNet(teacher_model, require_cuda=True)
    win_rate, draw_rate, avg_len = play_raw_policy_match(
        alphazero.UltimateTicTacToeGS,
        [student, teacher],
        games,
        temperature=0.0,
        seed=seed,
        opening_plies=6,
    )
    return {
        "win_rate": float(win_rate),
        "draw_rate": float(draw_rate),
        "loss_rate": float(max(0.0, 1.0 - win_rate - draw_rate)),
        "score": float(win_rate + 0.5 * draw_rate),
        "average_game_length": float(avg_len),
    }


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uttt_micromacro_6d24c_g4096_m64_f128_utttai_pretrain.yaml")
    parser.add_argument("--teacher-model", default="models/uttt_ai/policy_value_net_stage2.onnx")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--name", default="utttai-dagger-s12000-adamw003-e120")
    parser.add_argument("--games", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="adamw")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--pane", default=os.environ.get("CODEX_PANE", "0:0"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    teacher_model = root / args.teacher_model
    student_checkpoint = Path(args.student_checkpoint)
    if not student_checkpoint.is_absolute():
        student_checkpoint = root / student_checkpoint
    dataset_dir = root / "data" / "uttt_ai_imitation" / args.name
    exp_name = f"{args.name}-pretrain"
    exp_dir = root / "data" / "uttt" / exp_name
    results_path = root / "logs" / "train" / f"{args.name}-results.csv"
    gen_log = root / "logs" / "train" / f"{args.name}-generate.log"
    train_log = root / "logs" / "train" / f"{args.name}-pretrain.log"

    send_codex(
        f"UTTT.ai DAgger trial {args.name} started: {args.games} student-rollout games, "
        f"{args.epochs} epochs, {args.optimizer} lr={args.lr}.",
        args.pane,
    )

    gen_cmd = [
        "uv", "run", "python", "tools/generate_uttt_ai_imitation_dataset.py",
        "--output-dir", str(dataset_dir),
        "--model", str(teacher_model),
        "--games", str(args.games),
        "--target", "raw",
        "--mcts-sims", "0",
        "--move-policy", "student",
        "--student-checkpoint", str(student_checkpoint),
        "--random-opening-plies", "6",
        "--move-temperature", "1.0",
        "--shard-size", "60000",
        "--seed", str(args.seed),
        "--overwrite",
    ]
    status = run_logged(gen_cmd, gen_log)
    if status != 0:
        send_codex(f"UTTT.ai DAgger dataset generation failed with exit code {status}. Inspect {gen_log}.", args.pane)
        raise SystemExit(status)

    train_cmd = [
        "uv", "run", "python", "src/train_offline_dataset.py",
        "--config", args.config,
        "--dataset-dir", str(dataset_dir),
        "--experiment", exp_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--policy-only",
        "--optimizer", args.optimizer,
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--init-checkpoint", str(student_checkpoint),
        "--overwrite",
    ]
    status = run_logged(train_cmd, train_log)
    if status != 0:
        send_codex(f"UTTT.ai DAgger pretraining failed with exit code {status}. Inspect {train_log}.", args.pane)
        raise SystemExit(status)

    checkpoint = latest_checkpoint(exp_dir)
    eval_result = eval_raw_policy(checkpoint, teacher_model, args.eval_games, args.seed + 100000)
    metadata_path = dataset_dir / "metadata.json"
    positions = 0
    if metadata_path.exists():
        positions = int(json.loads(metadata_path.read_text()).get("positions", 0))
    row = {
        "name": args.name,
        "games": args.games,
        "positions": positions,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "student_checkpoint": str(student_checkpoint),
        "checkpoint": str(checkpoint),
        **eval_result,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    append_result(results_path, row)
    send_codex(
        f"UTTT.ai DAgger trial {args.name} complete: score={row['score']:.3f}, "
        f"W/D/L={row['win_rate']:.3f}/{row['draw_rate']:.3f}/{row['loss_rate']:.3f}, "
        f"positions={positions}. Inspect {results_path}.",
        args.pane,
    )


if __name__ == "__main__":
    main()
