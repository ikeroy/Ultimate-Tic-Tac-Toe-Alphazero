#!/usr/bin/env python3
"""Launch a detached UTTT raw-policy model-size sweep.

The sweep is intentionally sequential so per-run wall time is meaningful and
large models are not penalized by GPU contention from sibling trials.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TRIALS = [
    # label, depth, channels, head_channels, transformer_layers, transformer_heads
    ("rawsize-01-4d16c-t1", 4, 16, 32, 1, 4),
    ("rawsize-02-6d24c-t2", 6, 24, 48, 2, 4),
    ("rawsize-03-8d32c-t2", 8, 32, 64, 2, 4),
    ("rawsize-04-10d48c-t4", 10, 48, 96, 4, 6),
    ("rawsize-05-14d64c-t4", 14, 64, 128, 4, 8),
    ("rawsize-06-18d96c-t6", 18, 96, 192, 6, 8),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def latest_checkpoint(exp_dir: Path) -> int:
    ckpt_dir = exp_dir / "checkpoint"
    if not ckpt_dir.exists():
        return 0
    best = 0
    for path in ckpt_dir.glob("*.pt"):
        prefix = path.name.split("-", 1)[0]
        if prefix.isdigit():
            best = max(best, int(prefix))
    return best


def train_cmd(
    *,
    config: str,
    base_dir: str,
    experiment: str,
    iterations: int,
    depth: int,
    channels: int,
    head_channels: int,
    transformer_layers: int,
    transformer_heads: int,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "src/train.py",
        "--config",
        config,
        "--experiment",
        experiment,
        "--base-dir",
        base_dir,
        "--iterations",
        str(iterations),
        "--depth",
        str(depth),
        "--channels",
        str(channels),
        "--head-channels",
        str(head_channels),
        "--uttt-macro-transformer-layers",
        str(transformer_layers),
        "--uttt-macro-transformer-heads",
        str(transformer_heads),
        "--selfplay-mcts-visits",
        "1",
        "--fast-mcts-visits",
        "1",
        "--compare-mcts-visits",
        "1",
        "--playout-cap-percent",
        "0.0",
        "--gumbel-enabled",
        "false",
        "--fast-search-algo",
        "puct",
        "--policy-target-pruning",
        "false",
        "--uttt-ai-eval-interval",
        "10",
        "--uttt-ai-eval-games",
        "128",
        "--train-sample-rate",
        "2",
        "--hist-size",
        "20000",
        "--max-cache-size",
        "50000",
        "--streaming-active-files",
        "4",
        "--data-workers",
        "4",
        "--loader-threads",
        "4",
        "--torch-compile",
        "false",
    ]


def run_sweep(args: argparse.Namespace) -> int:
    root = repo_root()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    sweep_prefix = args.prefix
    game_dir = root / args.base_dir / "uttt"

    for label, depth, channels, head_channels, layers, heads in TRIALS:
        experiment = f"{sweep_prefix}-{label}"
        exp_dir = game_dir / experiment
        done_iter = latest_checkpoint(exp_dir)
        if done_iter >= args.iterations:
            print(f"[skip] {experiment}: checkpoint {done_iter} >= {args.iterations}", flush=True)
            continue

        if done_iter > 0:
            cmd = [
                "uv",
                "run",
                "python",
                "src/train.py",
                "--resume",
                str(exp_dir),
                "--iterations",
                str(args.iterations),
            ]
        else:
            cmd = train_cmd(
                config=args.config,
                base_dir=args.base_dir,
                experiment=experiment,
                iterations=args.iterations,
                depth=depth,
                channels=channels,
                head_channels=head_channels,
                transformer_layers=layers,
                transformer_heads=heads,
            )

        train_log = log_dir / f"{experiment}.log"
        print(f"[start] {experiment}: {' '.join(cmd)}", flush=True)
        with train_log.open("ab") as out:
            proc = subprocess.run(cmd, cwd=root, stdout=out, stderr=subprocess.STDOUT)
        print(f"[done] {experiment}: exit={proc.returncode}", flush=True)
        if proc.returncode != 0:
            return proc.returncode

    print("[done] raw size sweep complete", flush=True)
    return 0


def detach(args: argparse.Namespace) -> None:
    root = repo_root()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    supervisor_log = log_dir / f"{args.prefix}-supervisor.log"
    cmd = [sys.executable, __file__, "--no-detach"]
    for key in ("config", "base_dir", "prefix", "iterations"):
        cmd.extend([f"--{key.replace('_', '-')}", str(getattr(args, key))])
    with supervisor_log.open("ab") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"Started detached UTTT size sweep supervisor PID {proc.pid}")
    print(f"Supervisor log: {supervisor_log}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uttt_transformer_g512_m32.yaml")
    parser.add_argument("--base-dir", default="data")
    parser.add_argument("--prefix", default="size-sweep-rawpolicy-lite-20260702")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-detach", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_detach:
        return run_sweep(args)
    detach(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
