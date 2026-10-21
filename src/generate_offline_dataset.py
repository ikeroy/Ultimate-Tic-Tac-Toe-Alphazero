#!/usr/bin/env python3
"""Generate offline AlphaZero history shards from a fixed teacher model."""

import argparse
import json
import shutil
import time
from pathlib import Path

import alphazero
import neural_net
from config import load_config
from game_runner import GRArgs, GameRunner, base_params, prepare_inference_model
from game_runner import set_eval_types, set_model_groups


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _checkpoint_parts(path: Path):
    return str(path.parent), path.name


def _count_history_shards(history_dir: Path):
    return {
        "canonical": len(list(history_dir.glob("*-canonical-*.ptz"))),
        "v": len(list(history_dir.glob("*-v-*.ptz"))),
        "pi": len(list(history_dir.glob("*-pi-*.ptz"))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate an offline UTTT dataset using a checkpoint as the search teacher."
    )
    parser.add_argument("--config", required=True, help="Training config YAML to reuse game/search settings from.")
    parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory where metadata and history shards are written.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--visits", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--concurrent-games", type=int, default=128)
    parser.add_argument("--data-save-size", type=int, default=60_000)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number embedded in history shard names.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for teacher inference.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP inference.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.visits <= 0:
        raise ValueError("--visits must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.concurrent_games <= 0:
        raise ValueError("--concurrent-games must be positive")
    if args.data_save_size <= 0:
        raise ValueError("--data-save-size must be positive")

    output_dir = Path(args.output_dir)
    history_dir = output_dir / "history"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{output_dir} already exists and is not empty; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, {})
    Game = config.Game
    ckpt = Path(args.checkpoint)
    folder, filename = _checkpoint_parts(ckpt)
    teacher = neural_net.NNWrapper.load_checkpoint(Game, folder, filename)
    prepare_inference_model(teacher, config, compile=args.compile)
    if args.no_amp:
        teacher._amp_enabled = False

    params = base_params(config, config.self_play_temp, args.batch_size, 1)
    params.games_to_play = args.games
    params.concurrent_games = args.concurrent_games
    params.max_batch_size = args.batch_size
    params.mcts_visits = [args.visits] * Game.NUM_PLAYERS()
    params.history_enabled = True
    params.history_value_target = alphazero.HistoryValueTarget.FINAL
    params.epsilon = args.epsilon
    params.playout_cap_randomization = False
    params.playout_cap_depth = args.visits
    params.playout_cap_percent = 0.0
    params.resign_percent = 0.0
    params.resign_playthrough_percent = 0.0
    params.mcts_root_temp = config.mcts_root_temp
    params.root_fpu_zero = config.root_fpu_zero
    params.shaped_dirichlet = config.shaped_dirichlet
    params.policy_target_pruning = config.policy_target_pruning

    players = [teacher] * Game.NUM_PLAYERS()
    set_model_groups(params, players)
    set_eval_types(params, players)

    metadata = {
        "status": "running",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt.resolve()),
        "game": config.game,
        "teacher_input_channels": getattr(teacher, "_input_channels", None),
        "saved_input_channels": int(Game.CANONICAL_SHAPE()[0]),
        "games": args.games,
        "visits": args.visits,
        "batch_size": args.batch_size,
        "concurrent_games": args.concurrent_games,
        "epsilon": args.epsilon,
        "history_value_target": "final",
        "playout_cap_percent": 0.0,
        "resign_percent": 0.0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default) + "\n")

    start = time.time()
    pm = alphazero.PlayManager(Game(), params)
    grargs = GRArgs(
        title=f"Offline Dataset {args.games}x{args.visits}",
        game=Game,
        iteration=args.iteration,
        max_batch_size=args.batch_size,
        data_save_size=args.data_save_size,
        record_batch_metrics=True,
        data_folder=str(history_dir),
        progress_path=str(progress_path),
    )
    runner = GameRunner(players, pm, grargs)
    runner.run()

    scores = pm.scores()
    elapsed = time.time() - start
    metadata.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": elapsed,
            "scores": list(scores),
            "avg_game_length": pm.avg_game_length(),
            "avg_leaf_depth": pm.avg_leaf_depth(),
            "avg_search_entropy": pm.avg_search_entropy(),
            "avg_valid_moves": pm.avg_valid_moves(),
            "cache_hits": pm.cache_hits(),
            "cache_misses": pm.cache_misses(),
            "history_shards": _count_history_shards(history_dir),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default) + "\n")


if __name__ == "__main__":
    main()
