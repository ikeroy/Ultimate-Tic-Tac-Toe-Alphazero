#!/usr/bin/env python3
"""Build a UTTT.ai search anchor and launch one anchored self-play trial."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {log_path}")


def _clear_or_require_empty(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        shutil.rmtree(path)
        return
    if any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/uttt_teacher_anchor_g512_m32_m256_50.yaml",
    )
    parser.add_argument(
        "--experiment",
        default="teacher-anchor-g512m32-m256-s15-50",
    )
    parser.add_argument(
        "--student-checkpoint",
        default=(
            "data/uttt/fresh-g512-m32-lr022-gentledrop-sr12-20260701/checkpoint/"
            "0300-fresh-g512-m32-lr022-gentledrop-sr12-20260701.pt"
        ),
    )
    parser.add_argument(
        "--anchor-name",
        default="utttai-anchor-g512m32-student-mcts256-1000g",
    )
    parser.add_argument("--anchor-games", type=int, default=1000)
    parser.add_argument("--anchor-sims", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = ROOT / args.config
    student_checkpoint = ROOT / args.student_checkpoint
    raw_anchor = ROOT / "data" / "uttt_ai_imitation" / args.anchor_name
    sym_anchor = ROOT / "data" / "uttt_ai_imitation" / f"{args.anchor_name}-sym"
    experiment_dir = ROOT / "data" / "uttt" / args.experiment
    log_dir = ROOT / "logs" / "train"

    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if not student_checkpoint.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {student_checkpoint}")

    for path in (raw_anchor, sym_anchor, experiment_dir):
        _clear_or_require_empty(path, overwrite=args.overwrite)

    metadata_path = ROOT / "logs" / "train" / f"{args.experiment}-trial.json"
    metadata = {
        "status": "running",
        "experiment": args.experiment,
        "student_checkpoint": str(student_checkpoint),
        "raw_anchor": str(raw_anchor),
        "sym_anchor": str(sym_anchor),
        "anchor_games": args.anchor_games,
        "anchor_sims": args.anchor_sims,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    python = sys.executable
    run_logged(
        [
            python,
            "tools/generate_uttt_ai_imitation_dataset.py",
            "--output-dir", str(raw_anchor),
            "--games", str(args.anchor_games),
            "--target", "mcts",
            "--mcts-sims", str(args.anchor_sims),
            "--move-policy", "student",
            "--student-checkpoint", str(student_checkpoint),
            "--random-opening-plies", "6",
            "--move-temperature", "1.0",
            "--shard-size", "60000",
            "--seed", str(args.seed),
            "--require-cuda",
            "--overwrite",
        ],
        log_dir / f"{args.experiment}-anchor-generate.log",
    )
    run_logged(
        [
            python,
            "tools/expand_uttt_imitation_symmetries.py",
            "--input-dir", str(raw_anchor),
            "--output-dir", str(sym_anchor),
            "--config", str(config),
            "--overwrite",
        ],
        log_dir / f"{args.experiment}-anchor-symmetry.log",
    )
    run_logged(
        [
            python,
            "src/train.py",
            "--config", str(config),
            "--experiment", args.experiment,
            "--init-checkpoint", str(student_checkpoint),
        ],
        log_dir / f"{args.experiment}.log",
    )

    metadata.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
