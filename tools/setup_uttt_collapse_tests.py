#!/usr/bin/env python3
"""Create isolated UTTT collapse-debug branches from a source checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TESTS = {
    "trainbest": {
        "description": "train rejected candidates from gated best",
        "overrides": {
            "train_from_best": True,
            "iterations": 95,
        },
    },
    "noprune": {
        "description": "disable policy target pruning",
        "overrides": {
            "train_from_best": True,
            "policy_target_pruning": False,
            "iterations": 95,
        },
    },
    "replayanchor": {
        "description": "reduce replay overfitting to fresh data",
        "overrides": {
            "train_from_best": True,
            "train_sample_rate": 6,
            "reservoir_recency_decay": 1.0,
            "iterations": 95,
        },
    },
    "explore": {
        "description": "increase root/search trajectory diversity",
        "overrides": {
            "train_from_best": True,
            "mcts_root_temp": 1.75,
            "temp_decay_half_life": 20,
            "iterations": 95,
        },
    },
    "noprune-explore150": {
        "description": "train from best with no pruning and higher root exploration",
        "overrides": {
            "train_from_best": True,
            "policy_target_pruning": False,
            "mcts_root_temp": 1.75,
            "temp_decay_half_life": 20,
            "iterations": 150,
        },
    },
    "noprune150": {
        "description": "train from best with no policy target pruning",
        "overrides": {
            "train_from_best": True,
            "policy_target_pruning": False,
            "iterations": 150,
        },
    },
}


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_checkpoints(src_dir: Path, dst_dir: Path, dst_name: str, through_iter: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.pt")):
        match = re.match(r"(\d+)-", src.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if iteration > through_iter:
            continue
        shutil.copy2(src, dst_dir / f"{iteration:04d}-{dst_name}.pt")


def copy_top_level_files(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def make_branch(source: Path, dest: Path, through_iter: int, overrides: dict) -> None:
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    dest.mkdir(parents=True)

    copy_top_level_files(source, dest)
    for dirname in ("history", "reservoir"):
        copy_tree(source / dirname, dest / dirname)
    for dirname in ("checkpoint", "tmp_history"):
        (dest / dirname).mkdir(parents=True, exist_ok=True)

    copy_checkpoints(source / "checkpoint", dest / "checkpoint", dest.name, through_iter)

    config_path = dest / "config.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f) or {}
    config.update(overrides)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    source_lr_path = source / "lr_state.json"
    lr_state = {"current_lr": config.get("lr", 0.01), "num_drops": 0,
                "last_drop_iter": -999, "last_best_iter": through_iter}
    if source_lr_path.exists():
        with source_lr_path.open() as f:
            lr_state = json.load(f)
    lr_state["last_best_iter"] = through_iter
    with (dest / "lr_state.json").open("w") as f:
        json.dump(lr_state, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/uttt/micromacro-6d24c-g4096-m64-f128-feasible")
    parser.add_argument("--iter", type=int, default=77)
    parser.add_argument("--prefix", default="collapse77")
    parser.add_argument("--only", choices=sorted(TESTS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = (PROJECT_ROOT / args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    selected = {args.only: TESTS[args.only]} if args.only else TESTS
    for key, spec in selected.items():
        dest = PROJECT_ROOT / "data" / "uttt" / f"{args.prefix}-{key}"
        if dest.exists() and args.force:
            shutil.rmtree(dest)
        make_branch(source, dest, args.iter, spec["overrides"])
        print(f"{dest.relative_to(PROJECT_ROOT)}: {spec['description']}")


if __name__ == "__main__":
    main()
