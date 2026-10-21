#!/usr/bin/env python3
"""Increase UTTT.ai imitation data/training until raw policy reaches parity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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


DEFAULT_STAGES = [
    {"games": 1000, "sims": 0, "epochs": 60, "batch": 8192},
    {"games": 3000, "sims": 0, "epochs": 90, "batch": 8192},
    {"games": 6000, "sims": 0, "epochs": 120, "batch": 8192},
    {"games": 12000, "sims": 0, "epochs": 160, "batch": 8192},
    {"games": 24000, "sims": 0, "epochs": 200, "batch": 8192},
]


def send_codex(message: str, pane: str = "0:0") -> None:
    """Send a short message to the Codex tmux pane, best-effort."""
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


def read_dataset_positions(dataset_dir: Path) -> int:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return 0
    with metadata_path.open() as f:
        return int(json.load(f).get("positions", 0))


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
    parser.add_argument("--model", default="models/uttt_ai/policy_value_net_stage2.onnx")
    parser.add_argument("--prefix", default="utttai-imitation-sweep-20260708")
    parser.add_argument("--target", choices=("raw", "mcts"), default="raw")
    parser.add_argument("--one-hot", action="store_true", help="Use teacher argmax one-hot targets instead of soft distributions.")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--pane", default=os.environ.get("CODEX_PANE", "0:0"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    model_path = root / args.model
    results_path = root / "logs" / "train" / f"{args.prefix}-results.csv"
    target_desc = f"one-hot {args.target}" if args.one_hot else args.target
    send_codex(
        f"UTTT.ai imitation sweep started with {target_desc} targets. Stop threshold is raw-policy score >= {args.threshold:.2f} over {args.eval_games} games."
    )

    for idx, stage in enumerate(DEFAULT_STAGES, start=1):
        if idx < args.start_index:
            continue
        target_tag = "raw" if args.target == "raw" else f"mcts{stage['sims']}"
        if args.one_hot:
            target_tag = f"onehot-{target_tag}"
        tag = f"{args.prefix}-s{idx:02d}-{stage['games']}g-{target_tag}-e{stage['epochs']}"
        dataset_dir = root / "data" / "uttt_ai_imitation" / tag
        exp_name = f"{tag}-pretrain"
        exp_dir = root / "data" / "uttt" / exp_name
        gen_log = root / "logs" / "train" / f"{tag}-generate.log"
        train_log = root / "logs" / "train" / f"{tag}-pretrain.log"

        if args.overwrite:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(exp_dir, ignore_errors=True)

        send_codex(
            f"Starting imitation stage {idx}: {stage['games']} games, {stage['sims']} UTTT.ai MCTS sims, {stage['epochs']} epochs."
        )

        gen_cmd = [
            "uv", "run", "python", "tools/generate_uttt_ai_imitation_dataset.py",
            "--output-dir", str(dataset_dir),
            "--model", str(model_path),
            "--games", str(stage["games"]),
            "--target", args.target,
            "--mcts-sims", str(stage["sims"]),
            "--random-opening-plies", "6",
            "--move-temperature", "1.0",
            "--shard-size", "60000",
            "--overwrite",
        ]
        if args.one_hot:
            gen_cmd.append("--one-hot")
        status = run_logged(gen_cmd, gen_log)
        if status != 0:
            send_codex(f"Imitation stage {idx} dataset generation failed with exit code {status}. Inspect {gen_log}.")
            return

        train_cmd = [
            "uv", "run", "python", "src/train_offline_dataset.py",
            "--config", args.config,
            "--dataset-dir", str(dataset_dir),
            "--experiment", exp_name,
            "--epochs", str(stage["epochs"]),
            "--batch-size", str(stage["batch"]),
            "--policy-only",
            "--overwrite",
        ]
        status = run_logged(train_cmd, train_log)
        if status != 0:
            send_codex(f"Imitation stage {idx} pretraining failed with exit code {status}. Inspect {train_log}.")
            return

        checkpoint = latest_checkpoint(exp_dir)
        eval_result = eval_raw_policy(
            checkpoint,
            model_path,
            games=args.eval_games,
            seed=20260708 + idx,
        )
        row = {
            "stage": idx,
            "tag": tag,
            "games": stage["games"],
            "mcts_sims": stage["sims"],
            "epochs": stage["epochs"],
            "batch_size": stage["batch"],
            "positions": read_dataset_positions(dataset_dir),
            "checkpoint": str(checkpoint),
            **eval_result,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        append_result(results_path, row)
        send_codex(
            "Imitation stage "
            f"{idx} train+compare complete: score={row['score']:.3f}, "
            f"W/D/L={row['win_rate']:.3f}/{row['draw_rate']:.3f}/{row['loss_rate']:.3f}, "
            f"positions={row['positions']}. Inspect {results_path}."
        )
        if row["score"] >= args.threshold:
            send_codex(
                f"UTTT.ai imitation sweep reached comparability at stage {idx}: {tag}. I will stop increasing data."
            )
            return

    send_codex(
        f"UTTT.ai imitation sweep finished all stages without reaching score {args.threshold:.2f}. Inspect {results_path}."
    )


if __name__ == "__main__":
    main()
