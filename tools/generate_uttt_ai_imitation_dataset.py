#!/usr/bin/env python3
"""Generate supervised UTTT history shards from the local UTTT.ai model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import alphazero
import neural_net
from game_runner import save_compressed
from tools.uttt_ai_adapter import (
    UtttAiMCTS,
    UtttAiPolicyValueNet,
    UtttAiState,
    default_model_path,
)


def scalar_value_to_wld(value: float) -> np.ndarray:
    """Map UTTT.ai's scalar current-player value to this repo's W/L/D target."""
    scalar = float(np.clip(value, -1.0, 1.0))
    win = (scalar + 1.0) * 0.5
    return np.array([win, 1.0 - win, 0.0], dtype=np.float32)


def policy_dict_to_vector(policy: dict[int, float], *, fallback_action: int | None = None) -> np.ndarray:
    """Convert an action-probability dict to a normalized 81-action vector."""
    out = np.zeros(81, dtype=np.float32)
    for action, prob in policy.items():
        if 0 <= int(action) < 81 and np.isfinite(prob) and prob > 0:
            out[int(action)] = float(prob)
    total = float(out.sum())
    if total > 0:
        out /= total
    elif fallback_action is not None:
        out[int(fallback_action)] = 1.0
    return out


def choose_action(policy: np.ndarray, legal: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
    legal = np.asarray(legal, dtype=bool)
    probs = np.asarray(policy, dtype=np.float64).copy()
    probs[~legal] = 0.0
    if probs.sum() <= 0:
        legal_actions = np.flatnonzero(legal)
        if legal_actions.size == 0:
            raise ValueError("no legal actions to choose from")
        return int(rng.choice(legal_actions))
    if temperature <= 0:
        return int(np.argmax(probs))
    if temperature != 1.0:
        probs = np.power(probs + 1e-12, 1.0 / temperature)
        probs[~legal] = 0.0
    probs /= probs.sum()
    return int(rng.choice(np.arange(81), p=probs))


def student_distribution(student, canonical: np.ndarray) -> np.ndarray:
    """Return a normalized 81-action policy from this repo's NNWrapper."""
    _, policy = student.process(torch.tensor(canonical[None, ...], dtype=torch.float32))
    if hasattr(policy, "detach"):
        policy = policy.detach().cpu().numpy()
    out = np.asarray(policy, dtype=np.float32).reshape(-1)
    if out.shape[0] != 81:
        raise ValueError(f"expected 81-action student policy, got {out.shape}")
    total = float(out.sum())
    if total > 0 and np.isfinite(total):
        out = out / total
    return out


def teacher_distribution(
    net: UtttAiPolicyValueNet,
    state: UtttAiState,
    *,
    target: str,
    mcts_sims: int,
    exploration_strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (value_target, policy_target, metadata) for one state."""
    if target == "mcts" and mcts_sims > 0:
        root = UtttAiMCTS(
            net,
            exploration_strength=exploration_strength,
            seed=seed,
        ).search(state, simulations=mcts_sims)
        display_policy = UtttAiMCTS.display_policy(root)
        policy = policy_dict_to_vector(display_policy)
        if policy.sum() > 0:
            action = int(np.argmax(policy))
            one_hot = policy_dict_to_vector({}, fallback_action=action)
        else:
            _, raw = net.evaluate(state)
            raw_policy = policy_dict_to_vector(raw)
            action = int(np.argmax(raw_policy))
            policy = raw_policy
            one_hot = policy_dict_to_vector({}, fallback_action=action)
        if target == "mcts":
            return (
                scalar_value_to_wld(root.state_value_mean),
                policy,
                {
                    "source": "mcts",
                    "root_visits": int(root.visit_count),
                    "argmax": action,
                    "one_hot": one_hot,
                },
            )

    value, raw = net.evaluate(state)
    policy = policy_dict_to_vector(raw)
    action = int(np.argmax(policy))
    return (
        scalar_value_to_wld(value),
        policy,
        {
            "source": "raw",
            "argmax": action,
            "one_hot": policy_dict_to_vector({}, fallback_action=action),
        },
    )


def _save_shard(history_dir: Path, iteration: int, batch: int, c, v, pi, zstd_level: int) -> None:
    n = len(c)
    if n == 0:
        return
    prefix = history_dir / f"{iteration:04d}-{batch:04d}"
    save_compressed(torch.tensor(np.stack(c), dtype=torch.float32), f"{prefix}-canonical-{n}.ptz", zstd_level=zstd_level)
    save_compressed(torch.tensor(np.stack(v), dtype=torch.float32), f"{prefix}-v-{n}.ptz", zstd_level=zstd_level)
    save_compressed(torch.tensor(np.stack(pi), dtype=torch.float32), f"{prefix}-pi-{n}.ptz", zstd_level=zstd_level)


def generate_dataset(
    *,
    output_dir: Path,
    model_path: Path,
    games: int,
    target: str,
    mcts_sims: int,
    one_hot: bool,
    move_temperature: float,
    move_policy: str,
    student_checkpoint: Path | None,
    random_opening_plies: int,
    max_positions_per_game: int,
    shard_size: int,
    seed: int,
    require_cuda: bool,
    overwrite: bool,
    zstd_level: int,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"{output_dir} already exists and is not empty; pass --overwrite")
        shutil.rmtree(output_dir)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    net = UtttAiPolicyValueNet(model_path, require_cuda=require_cuda)
    student = None
    if move_policy == "student":
        if student_checkpoint is None:
            raise ValueError("--student-checkpoint is required when --move-policy=student")
        student = neural_net.NNWrapper.load_checkpoint(
            alphazero.UltimateTicTacToeGS,
            str(student_checkpoint.parent),
            student_checkpoint.name,
        )
        student.enable_inference_optimizations()
    metadata = {
        "status": "running",
        "model": str(model_path.resolve()),
        "providers": getattr(net, "providers", []),
        "games": games,
        "target": "one_hot_" + target if one_hot else target,
        "mcts_sims": mcts_sims,
        "move_temperature": move_temperature,
        "move_policy": move_policy,
        "student_checkpoint": str(student_checkpoint.resolve()) if student_checkpoint else "",
        "random_opening_plies": random_opening_plies,
        "max_positions_per_game": max_positions_per_game,
        "seed": seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    c_buf: list[np.ndarray] = []
    v_buf: list[np.ndarray] = []
    pi_buf: list[np.ndarray] = []
    batch = 0
    positions = 0
    game_lengths = []
    target_counts = {"mcts": 0, "raw": 0}
    start = time.time()

    for game_idx in range(games):
        game = alphazero.UltimateTicTacToeGS()
        state = UtttAiState.initial()
        ply = 0
        recorded = 0
        while game.scores() is None:
            legal = np.asarray(game.valid_moves(), dtype=bool)
            legal_actions = np.flatnonzero(legal)
            teacher_legal = state.legal_indexes()
            if legal_actions.size == 0 or not teacher_legal:
                break

            if ply < random_opening_plies:
                action = int(rng.choice(legal_actions))
            else:
                canonical = np.asarray(game.canonicalized(), dtype=np.float32)
                value_target, policy_target, info = teacher_distribution(
                    net,
                    state,
                    target=target,
                    mcts_sims=mcts_sims,
                    exploration_strength=2.0,
                    seed=seed + game_idx * 1000 + ply,
                )
                target_counts[info["source"]] = target_counts.get(info["source"], 0) + 1
                chosen_target = info["one_hot"] if one_hot else policy_target
                c_buf.append(canonical)
                v_buf.append(value_target)
                pi_buf.append(chosen_target.astype(np.float32))
                positions += 1
                recorded += 1
                if len(c_buf) >= shard_size:
                    _save_shard(history_dir, 0, batch, c_buf, v_buf, pi_buf, zstd_level)
                    c_buf.clear()
                    v_buf.clear()
                    pi_buf.clear()
                    batch += 1
                if move_policy == "student":
                    action_policy = student_distribution(student, canonical)
                else:
                    action_policy = policy_target
                action = choose_action(action_policy, legal, rng, move_temperature)
                if max_positions_per_game > 0 and recorded >= max_positions_per_game:
                    break

            game.play_move(action)
            state = state.execute(action)
            ply += 1

        game_lengths.append(ply)
        if (game_idx + 1) % 10 == 0:
            metadata.update(
                {
                    "completed_games": game_idx + 1,
                    "positions": positions,
                    "elapsed_seconds": time.time() - start,
                    "target_counts": target_counts,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if c_buf:
        _save_shard(history_dir, 0, batch, c_buf, v_buf, pi_buf, zstd_level)

    metadata.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": time.time() - start,
            "completed_games": games,
            "positions": positions,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "target_counts": target_counts,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=str(default_model_path()))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--target", choices=("mcts", "raw"), default="mcts")
    parser.add_argument("--mcts-sims", type=int, default=32)
    parser.add_argument("--one-hot", action="store_true")
    parser.add_argument("--move-temperature", type=float, default=1.0)
    parser.add_argument(
        "--move-policy",
        choices=("teacher", "student"),
        default="teacher",
        help="Policy used to advance games. Targets still come from UTTT.ai.",
    )
    parser.add_argument("--student-checkpoint", default="", help="Checkpoint used when --move-policy=student.")
    parser.add_argument("--random-opening-plies", type=int, default=6)
    parser.add_argument("--max-positions-per-game", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zstd-level", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"UTTT.ai model not found: {model_path}")

    metadata = generate_dataset(
        output_dir=Path(args.output_dir),
        model_path=model_path,
        games=args.games,
        target=args.target,
        mcts_sims=args.mcts_sims,
        one_hot=args.one_hot,
        move_temperature=args.move_temperature,
        move_policy=args.move_policy,
        student_checkpoint=Path(args.student_checkpoint) if args.student_checkpoint else None,
        random_opening_plies=args.random_opening_plies,
        max_positions_per_game=args.max_positions_per_game,
        shard_size=args.shard_size,
        seed=args.seed,
        require_cuda=args.require_cuda,
        overwrite=args.overwrite,
        zstd_level=args.zstd_level,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
