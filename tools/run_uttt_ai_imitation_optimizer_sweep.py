#!/usr/bin/env python3
"""Optimizer sweep for UTTT.ai raw-policy distillation."""

from __future__ import annotations

import argparse
import csv
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


TRIALS = [
    {"name": "sgd-lr011-e120", "optimizer": "sgd", "lr": 0.011, "epochs": 120, "weight_decay": 1e-4},
    {"name": "sgd-lr044-e120", "optimizer": "sgd", "lr": 0.044, "epochs": 120, "weight_decay": 1e-4},
    {"name": "adamw-lr0003-e120", "optimizer": "adamw", "lr": 0.0003, "epochs": 120, "weight_decay": 1e-4},
    {"name": "adamw-lr001-e120", "optimizer": "adamw", "lr": 0.001, "epochs": 120, "weight_decay": 1e-4},
    {"name": "adamw-lr003-e120", "optimizer": "adamw", "lr": 0.003, "epochs": 120, "weight_decay": 1e-4},
]


def send_codex(message: str, pane: str = "0:0") -> None:
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
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def latest_checkpoint(exp_dir: Path) -> Path:
    checkpoints = sorted((exp_dir / "checkpoint").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {exp_dir / 'checkpoint'}")
    return checkpoints[-1]


def eval_raw_policy(checkpoint: Path, model_path: Path, games: int, seed: int) -> dict:
    student = neural_net.NNWrapper.load_checkpoint(
        alphazero.UltimateTicTacToeGS, str(checkpoint.parent), checkpoint.name
    )
    teacher = UtttAiPolicyValueNet(model_path, require_cuda=True)
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
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uttt_micromacro_6d24c_g4096_m64_f128_utttai_pretrain.yaml")
    parser.add_argument("--dataset-dir", default="data/uttt_ai_imitation/utttai-raw-policyonly-sweep-20260708-s04-12000g-raw-e160")
    parser.add_argument("--model", default="models/uttt_ai/policy_value_net_stage2.onnx")
    parser.add_argument("--prefix", default="utttai-optimizer-sweep-20260708")
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--pane", default=os.environ.get("CODEX_PANE", "0:0"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    model_path = root / args.model
    dataset_dir = root / args.dataset_dir
    results_path = root / "logs" / "train" / f"{args.prefix}-results.csv"
    send_codex(
        f"UTTT.ai optimizer sweep started on {dataset_dir}. {len(TRIALS)} trials, {args.eval_games} eval games each.",
        pane=args.pane,
    )

    for idx, trial in enumerate(TRIALS, start=1):
        exp_name = f"{args.prefix}-{idx:02d}-{trial['name']}"
        exp_dir = root / "data" / "uttt" / exp_name
        log_path = root / "logs" / "train" / f"{exp_name}.log"
        send_codex(
            f"Starting optimizer trial {idx}/{len(TRIALS)}: {trial['name']}.",
            pane=args.pane,
        )
        cmd = [
            "uv", "run", "python", "src/train_offline_dataset.py",
            "--config", args.config,
            "--dataset-dir", str(dataset_dir),
            "--experiment", exp_name,
            "--epochs", str(trial["epochs"]),
            "--batch-size", str(args.batch_size),
            "--optimizer", trial["optimizer"],
            "--lr", str(trial["lr"]),
            "--weight-decay", str(trial["weight_decay"]),
            "--policy-only",
            "--overwrite",
        ]
        status = run_logged(cmd, log_path)
        if status != 0:
            send_codex(
                f"Optimizer trial {trial['name']} failed with exit code {status}. Inspect {log_path}.",
                pane=args.pane,
            )
            return
        checkpoint = latest_checkpoint(exp_dir)
        eval_result = eval_raw_policy(
            checkpoint,
            model_path,
            games=args.eval_games,
            seed=20260708 + idx,
        )
        row = {
            "trial": idx,
            "name": trial["name"],
            "optimizer": trial["optimizer"],
            "lr": trial["lr"],
            "weight_decay": trial["weight_decay"],
            "epochs": trial["epochs"],
            "checkpoint": str(checkpoint),
            **eval_result,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        append_result(results_path, row)
        send_codex(
            f"Optimizer trial {trial['name']} complete: score={row['score']:.3f}, "
            f"W/D/L={row['win_rate']:.3f}/{row['draw_rate']:.3f}/{row['loss_rate']:.3f}. "
            f"Inspect {results_path}.",
            pane=args.pane,
        )

    send_codex(f"UTTT.ai optimizer sweep complete. Inspect {results_path}.", pane=args.pane)


if __name__ == "__main__":
    main()
