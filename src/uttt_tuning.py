#!/usr/bin/env python3
"""UTTT training-run parser, Aim exporter, and sweep launcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HPARAM_KEYS = [
    "selfplay_mcts_visits",
    "fast_mcts_visits",
    "compare_mcts_visits",
    "playout_cap_percent",
    "self_play_batch_size",
    "train_batch_size",
    "train_sample_rate",
    "optimizer",
    "weight_decay",
    "adamw_beta1",
    "adamw_beta2",
    "adamw_eps",
    "lr",
    "lr_schedule",
    "lr_cosine_min_factor",
    "lr_cosine_start_iter",
    "lr_cosine_end_iter",
    "lr_patience",
    "lr_min_between_drops",
    "gating_panel_size",
    "gating_panel_win_rate",
    "gating_best_win_rate",
    "cpuct",
    "fpu_reduction",
    "kernel_size",
]


@dataclass
class RunData:
    name: str
    path: Path
    config: dict[str, Any]
    latest_iter: int
    checkpoint_count: int
    elo: np.ndarray
    whr: np.ndarray
    train_steps: np.ndarray
    wins: np.ndarray
    draws: np.ndarray
    pair_games: np.ndarray
    pair_scores: np.ndarray
    per_iter_games: np.ndarray
    per_iter_score: np.ndarray
    per_iter_draw_rate: np.ndarray
    wall_times: np.ndarray

    @property
    def best_elo(self) -> float:
        return _best_rating_prefix(self.elo, self.per_iter_games, self.latest_iter)

    @property
    def final_elo(self) -> float:
        return _last_observed_rating(self.elo, self.per_iter_games, self.latest_iter)

    @property
    def best_whr(self) -> float:
        return _best_rating_prefix(self.whr, self.per_iter_games, self.latest_iter)

    @property
    def final_whr(self) -> float:
        return _last_observed_rating(self.whr, self.per_iter_games, self.latest_iter)

    @property
    def total_pair_games(self) -> int:
        if self.pair_games.size == 0:
            return 0
        return int(np.triu(self.pair_games, 1).sum())

    @property
    def total_draws(self) -> int:
        if self.draws.size == 0:
            return 0
        return int(np.triu(self.draws, 1).sum())

    @property
    def total_draw_rate(self) -> float:
        games = self.total_pair_games
        return self.total_draws / games if games > 0 else float("nan")


def _load_array(path: Path, ndmin: int = 1, fill: float = 0.0) -> np.ndarray:
    if not path.exists():
        return np.zeros((0,) * ndmin, dtype=float)
    arr = np.genfromtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim < ndmin:
        arr = np.reshape(arr, (1,) * (ndmin - arr.ndim) + arr.shape)
    if fill is not None:
        arr = np.nan_to_num(arr, nan=fill)
    return arr


def _load_config(path: Path) -> dict[str, Any]:
    cfg = path / "config.yaml"
    if not cfg.exists():
        return {}
    with cfg.open() as f:
        return yaml.safe_load(f) or {}


def _checkpoint_iters(path: Path) -> list[int]:
    ckpt_dir = path / "checkpoint"
    if not ckpt_dir.exists():
        return []
    out = []
    for p in ckpt_dir.glob("*.pt"):
        prefix = p.name.split("-", 1)[0]
        if prefix.isdigit():
            out.append(int(prefix))
    return sorted(out)


def _checkpoint_wall_times(path: Path) -> np.ndarray:
    ckpt_dir = path / "checkpoint"
    times: dict[int, float] = {}
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("*.pt"):
            prefix = p.name.split("-", 1)[0]
            if prefix.isdigit():
                iteration = int(prefix)
                times[iteration] = max(times.get(iteration, 0.0), p.stat().st_mtime)

    hist_dir = path / "history"
    if hist_dir.exists():
        for p in hist_dir.glob("*-canonical-*"):
            prefix = p.name.split("-", 1)[0]
            if prefix.isdigit():
                iteration = int(prefix)
                times[iteration] = max(times.get(iteration, 0.0), p.stat().st_mtime)

    if not times:
        return np.zeros(0, dtype=float)
    out = np.zeros(max(times) + 1, dtype=float)
    for iteration, wall_time in times.items():
        out[iteration] = wall_time
    return out


def _active_latest(path: Path, elo: np.ndarray, wins: np.ndarray, draws: np.ndarray) -> int:
    ckpts = _checkpoint_iters(path)
    latest = max(ckpts) if ckpts else 0
    if wins.size:
        pair_games = wins + wins.T + draws
        nz = np.argwhere(pair_games > 0)
        if nz.size:
            latest = max(latest, int(nz.max()))
    if elo.size:
        nonzero = np.flatnonzero(np.abs(np.nan_to_num(elo)) > 1e-12)
        if nonzero.size:
            latest = max(latest, int(nonzero.max()))
    return latest


def _compute_pair_stats(wins: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if wins.size == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    n = max(wins.shape[0], wins.shape[1], draws.shape[0], draws.shape[1])
    w = np.zeros((n, n), dtype=float)
    d = np.zeros((n, n), dtype=float)
    w[: wins.shape[0], : wins.shape[1]] = wins
    d[: draws.shape[0], : draws.shape[1]] = draws
    games = w + w.T + d
    scores = np.full((n, n), np.nan, dtype=float)
    np.divide(w + 0.5 * d, games, out=scores, where=games > 0)
    return games, scores


def _compute_per_iter(wins: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    games, scores = _compute_pair_stats(wins, draws)
    if games.size == 0:
        return np.zeros(0), np.full(0, np.nan), np.full(0, np.nan)
    own_score = wins + 0.5 * draws
    with np.errstate(invalid="ignore", divide="ignore"):
        score = own_score.sum(axis=1) / games.sum(axis=1)
        draw_rate = draws.sum(axis=1) / games.sum(axis=1)
    return games.sum(axis=1), score, draw_rate


def parse_run(path: Path) -> RunData:
    cfg = _load_config(path)
    elo = _load_array(path / "elo.csv")
    whr = _load_array(path / "whr.csv")
    train_steps = _load_array(path / "train_steps_per_iter.csv")
    wins = _load_array(path / "wins.csv", ndmin=2)
    draws = _load_array(path / "draws.csv", ndmin=2)
    pair_games, pair_scores = _compute_pair_stats(wins, draws)
    per_games, per_score, per_draw_rate = _compute_per_iter(wins, draws)
    ckpts = _checkpoint_iters(path)
    return RunData(
        name=path.name,
        path=path,
        config=cfg,
        latest_iter=_active_latest(path, elo, wins, draws),
        checkpoint_count=len(ckpts),
        elo=elo,
        whr=whr,
        train_steps=train_steps,
        wins=wins,
        draws=draws,
        pair_games=pair_games,
        pair_scores=pair_scores,
        per_iter_games=per_games,
        per_iter_score=per_score,
        per_iter_draw_rate=per_draw_rate,
        wall_times=_checkpoint_wall_times(path),
    )


def discover_runs(base: Path, game: str = "uttt", prefix: str = "") -> list[RunData]:
    root = base / game
    if not root.exists():
        return []
    runs = []
    for p in sorted(root.iterdir()):
        if (
            p.is_dir()
            and (not prefix or p.name.startswith(prefix))
            and (p / "config.yaml").exists()
        ):
            runs.append(parse_run(p))
    return runs


def _export_signature(run: RunData) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    metric_files = (
        "elo.csv",
        "whr.csv",
        "win_rate.csv",
        "wins.csv",
        "draws.csv",
        "train_steps_per_iter.csv",
        "lr_state.json",
    )
    stats = []
    for name in metric_files:
        path = run.path / name
        if path.exists():
            stat = path.stat()
            stats.append((name, stat.st_mtime_ns, stat.st_size))
    return run.latest_iter, tuple(stats)


def _finite_prefix(arr: np.ndarray, latest: int) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(0)
    end = min(latest + 1, arr.shape[0])
    vals = np.asarray(arr[:end], dtype=float)
    return vals[np.isfinite(vals)]


def _best_prefix(arr: np.ndarray, latest: int) -> float:
    vals = _finite_prefix(arr, latest)
    return float(vals.max()) if vals.size else float("nan")


def _observed_rating_indices(arr: np.ndarray, games: np.ndarray, latest: int) -> list[int]:
    if arr.size == 0:
        return []
    end = min(latest, arr.shape[0] - 1)
    indices = []
    for i in range(end + 1):
        val = float(arr[i])
        if not math.isfinite(val):
            continue
        if i == 0 or abs(val) > 1e-12:
            indices.append(i)
    return indices


def _best_rating_prefix(arr: np.ndarray, games: np.ndarray, latest: int) -> float:
    indices = _observed_rating_indices(arr, games, latest)
    if not indices:
        return float("nan")
    return float(max(arr[i] for i in indices))


def _best_rating_derivative(
    arr: np.ndarray,
    games: np.ndarray,
    latest: int,
    window: int = 1,
) -> float:
    """Per-iteration slope of best-rating-so-far ending at latest."""
    indices = _observed_rating_indices(arr, games, latest)
    if len(indices) < 2:
        return 0.0
    current_iter = indices[-1]
    current_best = _best_rating_prefix(arr, games, current_iter)
    if not math.isfinite(current_best):
        return float("nan")
    if window <= 1:
        prev_iter = indices[-2]
    else:
        cutoff = max(0, current_iter - window)
        candidates = [idx for idx in indices if idx <= cutoff]
        prev_iter = candidates[-1] if candidates else indices[0]
    if prev_iter == current_iter:
        return 0.0
    prev_best = _best_rating_prefix(arr, games, prev_iter)
    if not math.isfinite(prev_best):
        return float("nan")
    return float((current_best - prev_best) / max(1, current_iter - prev_iter))


def _at_iter(arr: np.ndarray, iteration: int) -> float:
    if arr.size == 0 or iteration >= arr.shape[0]:
        return float("nan")
    val = float(arr[iteration])
    return val if math.isfinite(val) else float("nan")


def _last_observed_rating(arr: np.ndarray, games: np.ndarray, latest: int) -> float:
    indices = _observed_rating_indices(arr, games, latest)
    if not indices:
        return float("nan")
    return _at_iter(arr, max(indices))


def _hparams(run: RunData) -> dict[str, Any]:
    return {k: run.config.get(k) for k in HPARAM_KEYS if k in run.config}


def _hparam_metrics(run: RunData, objective_iteration: int = 20) -> dict[str, float]:
    objective = milestone_elo(run, objective_iteration)
    return {
        f"hparam/objective_elo_iter_{objective_iteration}": (
            0.0 if math.isnan(objective) else objective
        ),
        "hparam/best_elo": 0.0 if math.isnan(run.best_elo) else run.best_elo,
        "hparam/final_elo": 0.0 if math.isnan(run.final_elo) else run.final_elo,
        "hparam/best_whr": 0.0 if math.isnan(run.best_whr) else run.best_whr,
        "hparam/pair_games": float(run.total_pair_games),
    }


def _walltime_for_iter(run: RunData, iteration: int) -> float | None:
    if iteration < run.wall_times.shape[0] and run.wall_times[iteration] > 0:
        return float(run.wall_times[iteration])
    previous = run.wall_times[: min(iteration + 1, run.wall_times.shape[0])]
    previous = previous[previous > 0]
    if previous.size:
        return float(previous[-1])
    return None


def _elapsed_for_iter(run: RunData, iteration: int) -> float:
    walltime = _walltime_for_iter(run, iteration)
    if walltime is None:
        return 0.0
    starts: list[float] = []
    if run.wall_times.size:
        starts.extend(float(t) for t in run.wall_times if t > 0)
    config_path = run.path / "config.yaml"
    if config_path.exists():
        starts.append(float(config_path.stat().st_mtime))
    start_time = min(starts) if starts else walltime
    return max(0.0, walltime - start_time)


def _run_walltime_anchor(run: RunData) -> float:
    starts: list[float] = []
    if run.wall_times.size:
        starts.extend(float(t) for t in run.wall_times if t > 0)
    config_path = run.path / "config.yaml"
    if config_path.exists():
        starts.append(float(config_path.stat().st_mtime))
    return min(starts) if starts else time.time()


def _score_from_counts(run: RunData, newer: int, older: int) -> tuple[float, float]:
    if (
        run.wins.size == 0
        or newer >= run.wins.shape[0]
        or older >= run.wins.shape[1]
        or newer >= run.draws.shape[0]
        or older >= run.draws.shape[1]
    ):
        return float("nan"), 0.0
    games = float(run.wins[newer, older] + run.wins[older, newer] + run.draws[newer, older])
    if games <= 0:
        return float("nan"), 0.0
    score = float((run.wins[newer, older] + 0.5 * run.draws[newer, older]) / games)
    return score, games


def _gating_series(run: RunData) -> dict[str, np.ndarray]:
    n = run.latest_iter + 1
    opponent = np.full(n, np.nan)
    score = np.full(n, np.nan)
    games = np.zeros(n, dtype=float)
    accepted = np.zeros(n, dtype=float)
    best_iter = np.zeros(n, dtype=float)

    threshold = float(run.config.get("gating_best_win_rate", run.config.get("gating_panel_win_rate", 0.52)))
    current_best = 0
    for i in range(n):
        if i > 0 and i < run.pair_games.shape[0]:
            candidates = [int(j) for j in np.flatnonzero(run.pair_games[i, :i] > 0)]
            if candidates:
                # With panel size 1 the largest prior opponent is the then-current best.
                # Other comparisons are retained from earlier/resumed runs and are older anchors.
                gate_opponent = max(candidates)
                gate_score, gate_games = _score_from_counts(run, i, gate_opponent)
                opponent[i] = gate_opponent
                score[i] = gate_score
                games[i] = gate_games
                if math.isfinite(gate_score) and gate_score > threshold:
                    current_best = i
                    accepted[i] = 1.0
        best_iter[i] = current_best

    return {
        "opponent": opponent,
        "score": score,
        "games": games,
        "accepted": accepted,
        "best_iter": best_iter,
    }


def _project_lr_series(run: RunData, gating: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = run.latest_iter + 1
    lr = np.zeros(n, dtype=float)
    drops = np.zeros(n, dtype=float)
    stale_iters = np.zeros(n, dtype=float)

    schedule = str(run.config.get("lr_schedule", "constant"))
    base_lr = float(run.config.get("lr", 0.0))
    drop_factor = float(run.config.get("lr_drop_factor", 0.3))
    patience = int(run.config.get("lr_patience", 8))
    min_iter = int(run.config.get("lr_min_iter", 50))
    min_between = int(run.config.get("lr_min_between_drops", 30))
    max_drops = int(run.config.get("lr_max_drops", 3))
    warmup_target = int(run.config.get("lr_warmup_target", 0))
    warmup_floor = float(run.config.get("lr_warmup_floor", 0.2))
    cosine_min_factor = float(run.config.get("lr_cosine_min_factor", 0.1))
    cosine_start_config = int(run.config.get("lr_cosine_start_iter", -1))
    cosine_end_config = int(run.config.get("lr_cosine_end_iter", -1))
    exponential_final_factor = float(run.config.get("lr_exponential_final_factor", 0.1))
    exponential_start_config = int(run.config.get("lr_exponential_start_iter", -1))
    exponential_end_config = int(run.config.get("lr_exponential_end_iter", -1))
    total_iterations = int(run.config.get("iterations", max(n, 1)))

    current_lr = base_lr
    num_drops = 0
    last_drop_iter = -min_between
    last_best_iter = 0

    step_lrs = run.config.get("lr_steps") or []
    for i in range(n):
        if i < gating["accepted"].shape[0] and gating["accepted"][i] > 0:
            last_best_iter = i

        if schedule == "step":
            current_lr = step_lrs[0][1] if step_lrs else base_lr
            for step_iter, step_lr in step_lrs:
                if i >= int(step_iter):
                    current_lr = float(step_lr)
        elif schedule == "adaptive":
            can_drop = (
                i >= min_iter
                and i - last_drop_iter >= min_between
                and (max_drops == 0 or num_drops < max_drops)
                and i - last_best_iter >= patience
            )
            if can_drop:
                current_lr *= drop_factor
                num_drops += 1
                last_drop_iter = i
        elif schedule == "cosine":
            cosine_start = (
                cosine_start_config
                if cosine_start_config >= 0
                else max(0, warmup_target)
            )
            cosine_end = (
                cosine_end_config
                if cosine_end_config >= 0
                else total_iterations - 1
            )
            cosine_length = max(1, cosine_end - cosine_start)
            progress = min(1.0, max(0.0, (i - cosine_start) / cosine_length))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = base_lr * (
                cosine_min_factor + (1.0 - cosine_min_factor) * cosine_factor
            )
        elif schedule == "exponential":
            exponential_start = (
                exponential_start_config
                if exponential_start_config >= 0
                else max(0, warmup_target)
            )
            exponential_end = (
                exponential_end_config
                if exponential_end_config >= 0
                else total_iterations - 1
            )
            exponential_length = max(1, exponential_end - exponential_start)
            progress = min(1.0, max(0.0, (i - exponential_start) / exponential_length))
            current_lr = base_lr * exponential_final_factor ** progress

        iter_lr = current_lr
        if warmup_target > 0 and i < warmup_target:
            warmup_factor = warmup_floor + (1.0 - warmup_floor) * (i + 1) / warmup_target
            iter_lr *= warmup_factor

        lr[i] = iter_lr
        drops[i] = num_drops
        stale_iters[i] = i - last_best_iter

    return {"learning_rate": lr, "drops": drops, "stale_iters": stale_iters}


def summarize_runs(runs: list[RunData]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        row = {
            "run": run.name,
            "latest_iter": run.latest_iter,
            "checkpoints": run.checkpoint_count,
            "best_elo": run.best_elo,
            "final_elo": run.final_elo,
            "best_whr": run.best_whr,
            "final_whr": run.final_whr,
            "pair_games": run.total_pair_games,
            "draw_rate": run.total_draw_rate,
        }
        row.update(_hparams(run))
        rows.append(row)
    rows.sort(key=lambda r: (float("-inf") if math.isnan(r["best_elo"]) else r["best_elo"]), reverse=True)
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    preferred = [
        "run", "latest_iter", "checkpoints", "best_elo", "final_elo",
        "best_whr", "final_whr", "pair_games", "draw_rate",
    ]
    fieldnames = preferred + [k for k in keys if k not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], limit: int) -> None:
    for row in rows[:limit]:
        print(
            f"{row['run']:38s} iter={row['latest_iter']:4d} "
            f"best_elo={row['best_elo']:8.1f} final_elo={row['final_elo']:8.1f} "
            f"pairs={row['pair_games']:6d} draws={row['draw_rate']:.3f}"
        )


def export_aim(
    runs: list[RunData],
    repo: str | None = None,
    include_plot: bool = True,
    include_hparams: bool = True,
    writers: dict[str, Any] | None = None,
    start_iters: dict[str, int] | None = None,
    walltime_anchors: dict[str, float] | None = None,
) -> None:
    import aim

    repo = str(repo) if repo is not None else None
    for run in runs:
        aim_hash = run.path / ".aim_run_hash"
        if aim_hash.exists():
            aim_run = aim.Run(run_hash=aim_hash.read_text().strip(), repo=repo, force_resume=True)
        else:
            aim_run = aim.Run(experiment=run.name, repo=repo)
            aim_run.name = run.name
            aim_hash.write_text(aim_run.hash)
        if walltime_anchors is None:
            walltime_anchor = _run_walltime_anchor(run)
        else:
            walltime_anchor = walltime_anchors.setdefault(run.name, _run_walltime_anchor(run))
        start_iter = max(0, (start_iters or {}).get(run.name, 0))
        hp = _hparams(run)
        if include_hparams and hp:
            aim_run["hparams"] = hp
            aim_run["summary"] = summarize_runs([run])[0]
        observed_elo = set(_observed_rating_indices(run.elo, run.per_iter_games, run.latest_iter))
        observed_whr = set(_observed_rating_indices(run.whr, run.per_iter_games, run.latest_iter))
        gating = _gating_series(run)
        lr_series = _project_lr_series(run, gating)
        try:
            for i in range(start_iter, run.latest_iter + 1):
                step = int(run.train_steps[i]) if i < run.train_steps.shape[0] and math.isfinite(float(run.train_steps[i])) else i
                if i in observed_elo:
                    aim_run.track(float(run.elo[i]), name="elo", epoch=i, step=step, context={"type": "current", "source": "csv_export"})
                    aim_run.track(_best_rating_prefix(run.elo, run.per_iter_games, i), name="elo", epoch=i, step=step, context={"type": "best_so_far", "source": "csv_export"})
                    aim_run.track(_best_rating_derivative(run.elo, run.per_iter_games, i), name="elo_derivative", epoch=i, step=step, context={"window": "instant", "source": "csv_export"})
                    aim_run.track(_best_rating_derivative(run.elo, run.per_iter_games, i, window=10), name="elo_derivative", epoch=i, step=step, context={"window": "10iter", "source": "csv_export"})
                if i in observed_whr:
                    aim_run.track(float(run.whr[i]), name="whr", epoch=i, step=step, context={"type": "current", "source": "csv_export"})
                    aim_run.track(_best_rating_prefix(run.whr, run.per_iter_games, i), name="whr", epoch=i, step=step, context={"type": "best_so_far", "source": "csv_export"})
                if i < run.train_steps.shape[0] and math.isfinite(float(run.train_steps[i])):
                    aim_run.track(float(run.train_steps[i]), name="cumulative_steps", epoch=i, step=step, context={"source": "csv_export"})
                if i < run.per_iter_games.shape[0] and run.per_iter_games[i] > 0:
                    aim_run.track(float(run.per_iter_games[i]), name="pair_eval", epoch=i, step=step, context={"stat": "games", "source": "csv_export"})
                    aim_run.track(float(run.per_iter_score[i]), name="pair_eval", epoch=i, step=step, context={"stat": "weighted_score", "source": "csv_export"})
                    aim_run.track(float(run.per_iter_draw_rate[i]), name="pair_eval", epoch=i, step=step, context={"stat": "draw_rate", "source": "csv_export"})
                if i < gating["score"].shape[0] and math.isfinite(float(gating["score"][i])):
                    aim_run.track(float(gating["score"][i]), name="gating", epoch=i, step=step, context={"stat": "vs_best_score", "source": "csv_export"})
                    aim_run.track(float(gating["games"][i]), name="gating", epoch=i, step=step, context={"stat": "vs_best_games", "source": "csv_export"})
                    aim_run.track(float(gating["opponent"][i]), name="gating", epoch=i, step=step, context={"stat": "vs_best_opponent_iter", "source": "csv_export"})
                    aim_run.track(float(gating["accepted"][i]), name="gating", epoch=i, step=step, context={"stat": "accepted_best", "source": "csv_export"})
                if i < gating["best_iter"].shape[0]:
                    aim_run.track(float(gating["best_iter"][i]), name="gating", epoch=i, step=step, context={"stat": "current_best_iter", "source": "csv_export"})
                if i < lr_series["learning_rate"].shape[0]:
                    aim_run.track(float(lr_series["learning_rate"][i]), name="learning_rate", epoch=i, step=step, context={"source": "csv_export"})
                    aim_run.track(float(lr_series["drops"][i]), name="lr_drops", epoch=i, step=step, context={"source": "csv_export"})
                    aim_run.track(float(lr_series["stale_iters"][i]), name="iters_since_best", epoch=i, step=step, context={"source": "csv_export"})
            if include_hparams and hp:
                for tag, value in _hparam_metrics(run).items():
                    aim_run.track(float(value), name="hparam_metric", epoch=run.latest_iter, step=run.latest_iter, context={"metric": tag})
        finally:
            aim_run.close()


def write_hparam_plot(runs: list[RunData], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = summarize_runs(runs)
    rows = [r for r in rows if not math.isnan(r["best_elo"])]
    if not rows:
        return
    interesting = [
        "selfplay_mcts_visits",
        "fast_mcts_visits",
        "self_play_batch_size",
        "train_sample_rate",
        "gating_panel_size",
        "gating_best_win_rate",
        "lr",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for ax, key in zip(axes, interesting):
        xs = [r.get(key) for r in rows]
        ys = [r["best_elo"] for r in rows]
        labels = [r["run"] for r in rows]
        numeric = []
        y_numeric = []
        label_numeric = []
        for x, y, label in zip(xs, ys, labels):
            if isinstance(x, (int, float)) and x is not None:
                numeric.append(float(x))
                y_numeric.append(y)
                label_numeric.append(label)
        if numeric:
            ax.scatter(numeric, y_numeric)
            for x, y, label in zip(numeric, y_numeric, label_numeric):
                ax.annotate(label[-8:], (x, y), fontsize=7, alpha=0.7)
        ax.set_title(key)
        ax.set_ylabel("best Elo")
    axes[-1].axis("off")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _trial_configs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("panel1_sr6_96v", {
            "selfplay_mcts_visits": 96,
            "fast_mcts_visits": 24,
            "compare_mcts_visits": 96,
            "gating_panel_size": 1,
            "gating_panel_win_rate": 0.52,
            "gating_best_win_rate": 0.52,
            "train_sample_rate": 6,
        }),
        ("panel1_sr8_128v", {
            "selfplay_mcts_visits": 128,
            "fast_mcts_visits": 32,
            "compare_mcts_visits": 128,
            "gating_panel_size": 1,
            "gating_panel_win_rate": 0.52,
            "gating_best_win_rate": 0.52,
            "train_sample_rate": 8,
        }),
        ("panel2_sr6_128v", {
            "selfplay_mcts_visits": 128,
            "fast_mcts_visits": 32,
            "compare_mcts_visits": 128,
            "gating_panel_size": 2,
            "gating_panel_win_rate": 0.52,
            "gating_best_win_rate": 0.52,
            "train_sample_rate": 6,
        }),
        ("panel1_lowgate_128v", {
            "selfplay_mcts_visits": 128,
            "fast_mcts_visits": 32,
            "compare_mcts_visits": 128,
            "gating_panel_size": 1,
            "gating_panel_win_rate": 0.50,
            "gating_best_win_rate": 0.50,
            "train_sample_rate": 6,
        }),
    ]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def launch_sweep(args: argparse.Namespace) -> None:
    base_cfg = load_yaml(Path(args.config))
    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    trials = _trial_configs()[: args.trials]
    started = time.strftime("%Y%m%d-%H%M%S")
    for idx, (name, overrides) in enumerate(trials, start=1):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        cfg["iterations"] = int(args.iterations)
        cfg["torch_compile"] = bool(args.torch_compile)
        cfg.setdefault("game", "uttt")
        exp_name = f"tune-{started}-{idx:02d}-{name}"
        cfg_path = cfg_dir / f"{exp_name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        cmd = [
            sys.executable,
            "src/train.py",
            "--config",
            str(cfg_path),
            "--experiment",
            exp_name,
            "--base-dir",
            args.base_dir,
        ]
        print("Running:", " ".join(cmd), flush=True)
        with (log_dir / f"{exp_name}.log").open("w") as lf:
            lf.write(" ".join(cmd) + "\n")
            lf.flush()
            result = subprocess.run(cmd, cwd=Path.cwd(), stdout=lf, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(f"Trial failed: {exp_name}; see {log_dir / (exp_name + '.log')}")
        runs = discover_runs(Path(args.base_dir), "uttt")
        export_aim([r for r in runs if r.name == exp_name], args.aim_repo, include_plot=False)
    runs = discover_runs(Path(args.base_dir), "uttt")
    selected = [r for r in runs if r.name.startswith(f"tune-{started}-")]
    write_summary_csv(summarize_runs(selected), out_dir / f"summary-{started}.csv")
    export_aim(selected, args.aim_repo, include_plot=True)


def _write_trial_config(
    base_cfg: dict[str, Any],
    cfg_dir: Path,
    exp_name: str,
    overrides: dict[str, Any],
    iterations: int,
    torch_compile: bool,
) -> Path:
    cfg = dict(base_cfg)
    cfg.update(overrides)
    cfg["iterations"] = int(iterations)
    cfg["torch_compile"] = bool(torch_compile)
    cfg.setdefault("game", "uttt")
    cfg_path = cfg_dir / f"{exp_name}.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def _run_trial(
    exp_name: str,
    cfg_path: Path,
    base_dir: str,
    log_dir: Path,
    aim_repo: Path,
) -> None:
    cmd = [
        sys.executable,
        "src/train.py",
        "--config",
        str(cfg_path),
        "--experiment",
        exp_name,
        "--base-dir",
        base_dir,
    ]
    print("Running:", " ".join(cmd), flush=True)
    with (log_dir / f"{exp_name}.log").open("w") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        result = subprocess.run(cmd, cwd=Path.cwd(), stdout=lf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"Trial failed: {exp_name}; see {log_dir / (exp_name + '.log')}")
    runs = discover_runs(Path(base_dir), "uttt")
    export_aim([r for r in runs if r.name == exp_name], aim_repo, include_plot=False)


def _run_resume_trial(
    exp_name: str,
    experiment_dir: Path,
    log_dir: Path,
    aim_repo: Path,
    base_dir: str,
) -> None:
    cmd = [
        sys.executable,
        "src/train.py",
        "--resume",
        str(experiment_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    with (log_dir / f"{exp_name}.log").open("w") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        result = subprocess.run(cmd, cwd=Path.cwd(), stdout=lf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"Trial failed: {exp_name}; see {log_dir / (exp_name + '.log')}")
    runs = discover_runs(Path(base_dir), "uttt")
    export_aim([r for r in runs if r.name == exp_name], aim_repo, include_plot=False)


def _train_process_running(exp_name: str) -> bool:
    pattern = f"src/train.py.*--experiment {exp_name}"
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _process_running(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _score_for_selection(run: RunData) -> tuple[float, float, int]:
    final = run.final_elo if math.isfinite(run.final_elo) else float("-inf")
    best = run.best_elo if math.isfinite(run.best_elo) else float("-inf")
    return final, best, run.latest_iter


def _select_best_run(runs: list[RunData]) -> RunData:
    viable = [r for r in runs if r.total_pair_games > 0 and math.isfinite(r.best_elo)]
    if not viable:
        raise SystemExit("No viable runs with rating data to select from")
    return max(viable, key=_score_for_selection)


def _copy_hparams_from_run(base_cfg: dict[str, Any], run: RunData, iterations: int, torch_compile: bool) -> dict[str, Any]:
    cfg = dict(base_cfg)
    for key in HPARAM_KEYS:
        if key in run.config:
            cfg[key] = run.config[key]
    cfg["iterations"] = int(iterations)
    cfg["torch_compile"] = bool(torch_compile)
    cfg.setdefault("game", "uttt")
    return cfg


def _write_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _coerce_override(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _parse_set_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --set override {item!r}; expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --set override {item!r}; key is empty")
        overrides[key] = _coerce_override(value)
    return overrides


def _override_arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _path_iter_prefix(path: Path) -> int | None:
    prefix = path.name.split("-", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def _copy_tree_through_iter(src: Path, dest: Path, through_iter: int | None) -> None:
    if through_iter is None:
        shutil.copytree(src, dest)
        return

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            iteration = _path_iter_prefix(Path(name))
            if iteration is not None and iteration > through_iter:
                ignored.add(name)
        return ignored

    shutil.copytree(src, dest, ignore=ignore)


def _trim_copied_metrics(dest: Path, through_iter: int) -> None:
    matrix_files = ["win_rate.csv", "wins.csv", "draws.csv"]
    matrix_files.extend(path.name for path in dest.glob("win_rate_*.csv"))
    matrix_files.extend(path.name for path in dest.glob("wins_*.csv"))
    matrix_files.extend(path.name for path in dest.glob("draws_*.csv"))
    for filename in matrix_files:
        path = dest / filename
        if not path.exists():
            continue
        arr = np.genfromtxt(path, delimiter=",")
        if arr.ndim != 2:
            continue
        if arr.shape[0] <= through_iter + 1 and arr.shape[1] <= through_iter + 1:
            continue
        if filename.startswith("win_rate"):
            arr[through_iter + 1:, :] = np.nan
            arr[:, through_iter + 1:] = np.nan
        else:
            arr[through_iter + 1:, :] = 0
            arr[:, through_iter + 1:] = 0
        np.savetxt(path, arr, delimiter=",")

    vector_files = ["elo.csv", "whr.csv"]
    vector_files.extend(path.name for path in dest.glob("elo_*.csv"))
    vector_files.extend(path.name for path in dest.glob("whr_*.csv"))
    for filename in vector_files:
        path = dest / filename
        if not path.exists():
            continue
        arr = np.genfromtxt(path, delimiter=",")
        if arr.ndim != 1 or arr.shape[0] <= through_iter + 1:
            continue
        arr[through_iter + 1:] = 0
        np.savetxt(path, arr, delimiter=",")


def branch_run(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"Source run does not exist: {source}")
    if not (source / "config.yaml").exists():
        raise SystemExit(f"Source run has no config.yaml: {source}")

    game = args.game
    dest = (Path(args.base_dir) / game / args.experiment).resolve()
    if dest.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination already exists: {dest}; pass --overwrite to replace it")
        shutil.rmtree(dest)

    dest.mkdir(parents=True)
    through_iter = getattr(args, "through_iter", None)

    for filename in (
        "config.yaml",
        "elo.csv",
        "whr.csv",
        "win_rate.csv",
        "wins.csv",
        "draws.csv",
        "total_train_steps.txt",
        "train_steps_per_iter.csv",
        "lr_state.json",
        "compare_past_state.json",
        "bootstrap_meta.json",
    ):
        src = source / filename
        if src.exists():
            shutil.copy2(src, dest / filename)

    for dirname in ("history", "reservoir"):
        src = source / dirname
        if src.exists():
            _copy_tree_through_iter(src, dest / dirname, through_iter)

    ckpt_src = source / "checkpoint"
    ckpt_dst = dest / "checkpoint"
    ckpt_dst.mkdir(exist_ok=True)
    for src in sorted(ckpt_src.glob("*.pt")):
        prefix = src.name.split("-", 1)[0]
        if prefix.isdigit():
            if through_iter is not None and int(prefix) > through_iter:
                continue
            dst_name = f"{prefix}-{args.experiment}.pt"
        else:
            dst_name = src.name
        shutil.copy2(src, ckpt_dst / dst_name)

    (dest / "tmp_history").mkdir(exist_ok=True)
    if through_iter is not None:
        _trim_copied_metrics(dest, int(through_iter))

    cfg = load_yaml(dest / "config.yaml")
    cfg["game"] = game
    if args.iterations is not None:
        cfg["iterations"] = int(args.iterations)
    cfg.update(_parse_set_overrides(args.set))
    _write_config(dest / "config.yaml", cfg)

    lr_overrides = _parse_set_overrides(args.set_lr_state)
    if lr_overrides:
        lr_state_path = dest / "lr_state.json"
        lr_state = {}
        if lr_state_path.exists():
            with lr_state_path.open() as f:
                lr_state = json.load(f)
        lr_state.update(lr_overrides)
        with lr_state_path.open("w") as f:
            json.dump(lr_state, f)

    print(f"Branched {source.name} -> {dest}")
    print(f"Resume with: {sys.executable} src/train.py --resume {dest}")


def _branch_run(
    source: Path,
    experiment: str,
    base_dir: Path,
    game: str,
    iterations: int | None,
    config_overrides: dict[str, Any],
    lr_state_overrides: dict[str, Any],
    overwrite: bool,
) -> Path:
    dest = (base_dir / game / experiment).resolve()
    if dest.exists() and not overwrite:
        cfg_path = dest / "config.yaml"
        if iterations is not None and cfg_path.exists():
            cfg = load_yaml(cfg_path)
            if int(cfg.get("iterations", 0)) < int(iterations):
                cfg["iterations"] = int(iterations)
                _write_config(cfg_path, cfg)
        print(f"Using existing branch {dest}", flush=True)
        return dest

    args = argparse.Namespace(
        source=str(source),
        experiment=experiment,
        base_dir=str(base_dir),
        game=game,
        iterations=iterations,
        set=[f"{key}={_override_arg_value(value)}" for key, value in config_overrides.items()],
        set_lr_state=[f"{key}={_override_arg_value(value)}" for key, value in lr_state_overrides.items()],
        overwrite=overwrite,
        through_iter=None,
    )
    branch_run(args)
    return (base_dir / game / experiment).resolve()


def _plateau_trial_configs(source: Path) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    cfg = load_yaml(source / "config.yaml")
    lr_state_path = source / "lr_state.json"
    lr_state = {}
    if lr_state_path.exists():
        with lr_state_path.open() as f:
            lr_state = json.load(f)
    current_lr = float(lr_state.get("current_lr", cfg.get("lr", 0.0075)))
    latest = max(_checkpoint_iters(source) or [0])
    return [
        ("baseline", {}, {}),
        ("lrdrop-now", {}, {
            "current_lr": current_lr * float(cfg.get("lr_drop_factor", 0.3)),
            "num_drops": int(lr_state.get("num_drops", 0)) + 1,
            "last_drop_iter": latest,
        }),
        ("lrslow", {
            "lr_patience": 12,
            "lr_min_between_drops": 20,
        }, {}),
        ("lrfast", {
            "lr_min_iter": min(int(cfg.get("lr_min_iter", 50)), latest),
            "lr_patience": 3,
            "lr_min_between_drops": 6,
        }, {}),
        ("sr10", {
            "train_sample_rate": 10,
        }, {}),
        ("v160", {
            "selfplay_mcts_visits": 160,
            "fast_mcts_visits": 40,
            "compare_mcts_visits": 160,
        }, {}),
    ]


def plateau_sweep(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    trials = _plateau_trial_configs(source)[: args.trials]
    created: list[str] = []
    for idx, (name, cfg_overrides, lr_overrides) in enumerate(trials, start=1):
        exp_name = f"{args.prefix}-{idx:02d}-{name}"
        dest = _branch_run(
            source,
            exp_name,
            Path(args.base_dir),
            args.game,
            args.iterations,
            cfg_overrides,
            lr_overrides,
            args.overwrite,
        )
        created.append(exp_name)
        latest = max(_checkpoint_iters(dest) or [0])
        if latest >= int(args.iterations):
            print(f"{exp_name} already complete at iter {latest}; skipping", flush=True)
            continue
        _run_resume_trial(exp_name, dest, log_dir, args.aim_repo, args.base_dir)
    runs = discover_runs(Path(args.base_dir), args.game)
    selected = [r for r in runs if r.name in set(created)]
    rows = summarize_runs(selected)
    write_summary_csv(rows, out_dir / f"{args.prefix}-summary.csv")
    export_aim(selected, args.aim_repo, include_plot=True)


def _fine_trial_configs(best: RunData, max_trials: int) -> list[tuple[str, dict[str, Any]]]:
    cfg = best.config
    visits = int(cfg.get("selfplay_mcts_visits", 128))
    sample_rate = int(cfg.get("train_sample_rate", 8))
    kernel = int(cfg.get("kernel_size", 3))
    gate = float(cfg.get("gating_best_win_rate", cfg.get("gating_panel_win_rate", 0.52)))
    base = {
        "selfplay_mcts_visits": visits,
        "fast_mcts_visits": max(16, visits // 4),
        "compare_mcts_visits": visits,
        "gating_panel_size": int(cfg.get("gating_panel_size", 1)),
        "gating_panel_win_rate": gate,
        "gating_best_win_rate": gate,
        "train_sample_rate": sample_rate,
        "kernel_size": kernel,
    }
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("center", {}),
        (f"k{5 if kernel == 3 else 3}", {"kernel_size": 5 if kernel == 3 else 3}),
        (f"sr{max(4, sample_rate - 2)}", {"train_sample_rate": max(4, sample_rate - 2)}),
        (f"sr{sample_rate + 2}", {"train_sample_rate": sample_rate + 2}),
        (f"v{max(64, visits - 32)}", {
            "selfplay_mcts_visits": max(64, visits - 32),
            "fast_mcts_visits": max(16, max(64, visits - 32) // 4),
            "compare_mcts_visits": max(64, visits - 32),
        }),
        (f"v{visits + 32}", {
            "selfplay_mcts_visits": visits + 32,
            "fast_mcts_visits": max(16, (visits + 32) // 4),
            "compare_mcts_visits": visits + 32,
        }),
        ("gate50", {"gating_panel_win_rate": 0.50, "gating_best_win_rate": 0.50}),
        ("gate54", {"gating_panel_win_rate": 0.54, "gating_best_win_rate": 0.54}),
    ]
    out = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for name, overrides in candidates:
        merged = dict(base)
        merged.update(overrides)
        key = tuple(sorted(merged.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, merged))
        if len(out) >= max_trials:
            break
    return out


def nightly_autotune(args: argparse.Namespace) -> None:
    base_cfg = load_yaml(Path(args.config))
    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    current_pattern = f"uttt_tuning.py continue-sweep .*--started {args.wait_prefix}"
    print(f"{time.strftime('%H:%M:%S')} waiting for current sweep {args.wait_prefix}", flush=True)
    while _process_running(current_pattern):
        runs = discover_runs(Path(args.base_dir), "uttt")
        selected = [r for r in runs if r.name.startswith(f"tune-{args.wait_prefix}-")]
        rows = summarize_runs(selected)
        write_summary_csv(rows, Path(args.summary_csv))
        if rows:
            best_row = rows[0]
            print(
                f"{time.strftime('%H:%M:%S')} current sweep best {best_row['run']} "
                f"iter={best_row['latest_iter']} final_elo={best_row['final_elo']:.1f} "
                f"best_elo={best_row['best_elo']:.1f}",
                flush=True,
            )
        time.sleep(args.interval)

    runs = discover_runs(Path(args.base_dir), "uttt")
    coarse = [r for r in runs if r.name.startswith(f"tune-{args.wait_prefix}-")]
    if len(coarse) < args.coarse_trials:
        raise SystemExit(
            f"Only found {len(coarse)} coarse trials for tune-{args.wait_prefix}; "
            f"expected {args.coarse_trials}. Not starting fine sweep."
        )
    coarse_best = _select_best_run(coarse)
    print(
        f"{time.strftime('%H:%M:%S')} coarse best {coarse_best.name}: "
        f"final_elo={coarse_best.final_elo:.1f} best_elo={coarse_best.best_elo:.1f}",
        flush=True,
    )

    fine_started = time.strftime("%Y%m%d-%H%M%S")
    fine_prefix = f"fine-{fine_started}"
    for idx, (name, overrides) in enumerate(_fine_trial_configs(coarse_best, args.fine_trials), start=1):
        exp_name = f"{fine_prefix}-{idx:02d}-{name}"
        cfg = _copy_hparams_from_run(base_cfg, coarse_best, args.fine_iterations, args.torch_compile)
        cfg.update(overrides)
        cfg_path = cfg_dir / f"{exp_name}.yaml"
        _write_config(cfg_path, cfg)
        print(f"{time.strftime('%H:%M:%S')} fine trial {idx}/{args.fine_trials}: {exp_name}", flush=True)
        _run_trial(exp_name, cfg_path, args.base_dir, log_dir, args.aim_repo)

    runs = discover_runs(Path(args.base_dir), "uttt")
    fine = [r for r in runs if r.name.startswith(fine_prefix)]
    combined = coarse + fine
    rows = summarize_runs(combined)
    write_summary_csv(rows, out_dir / f"nightly-summary-{fine_started}.csv")
    export_aim(combined, args.aim_repo, include_plot=True)
    best = _select_best_run(combined)
    print(
        f"{time.strftime('%H:%M:%S')} selected {best.name}: "
        f"final_elo={best.final_elo:.1f} best_elo={best.best_elo:.1f}",
        flush=True,
    )

    final_name = f"final-best-{fine_started}"
    final_cfg = _copy_hparams_from_run(base_cfg, best, args.final_iterations, args.torch_compile)
    final_cfg_path = cfg_dir / f"{final_name}.yaml"
    _write_config(final_cfg_path, final_cfg)
    print(f"{time.strftime('%H:%M:%S')} starting final training {final_name}", flush=True)
    _run_trial(final_name, final_cfg_path, args.base_dir, log_dir, args.aim_repo)


def continue_sweep(args: argparse.Namespace) -> None:
    base_cfg = load_yaml(Path(args.config))
    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    trials = _trial_configs()[: args.trials]
    watch_exp = args.watch_experiment
    if watch_exp:
        while _train_process_running(watch_exp):
            runs = discover_runs(Path(args.base_dir), "uttt")
            selected = [r for r in runs if r.name.startswith(f"tune-{args.started}-")]
            rows = summarize_runs(selected)
            write_summary_csv(rows, Path(args.summary_csv))
            row = next((r for r in rows if r["run"] == watch_exp), None)
            if row:
                print(
                    f"{time.strftime('%H:%M:%S')} watching {watch_exp}: "
                    f"iter={row['latest_iter']} best_elo={row['best_elo']:.1f} "
                    f"draw_rate={row['draw_rate']:.3f}",
                    flush=True,
                )
            else:
                print(f"{time.strftime('%H:%M:%S')} watching {watch_exp}: no parsed data yet", flush=True)
            time.sleep(args.interval)
        print(f"{time.strftime('%H:%M:%S')} {watch_exp} is no longer running", flush=True)

    for idx, (name, overrides) in enumerate(trials, start=1):
        if idx < args.start_index:
            continue
        exp_name = f"tune-{args.started}-{idx:02d}-{name}"
        cfg_path = cfg_dir / f"{exp_name}.yaml"
        if not cfg_path.exists():
            cfg_path = _write_trial_config(
                base_cfg,
                cfg_dir,
                exp_name,
                overrides,
                args.iterations,
                args.torch_compile,
            )
        if _train_process_running(exp_name):
            print(f"{time.strftime('%H:%M:%S')} {exp_name} is already running; waiting", flush=True)
            while _train_process_running(exp_name):
                time.sleep(args.interval)
            continue
        _run_trial(exp_name, cfg_path, args.base_dir, log_dir, args.aim_repo)

    runs = discover_runs(Path(args.base_dir), "uttt")
    selected = [r for r in runs if r.name.startswith(f"tune-{args.started}-")]
    write_summary_csv(summarize_runs(selected), out_dir / f"summary-{args.started}.csv")
    export_aim(selected, args.aim_repo, include_plot=True)


def watch_aim(args: argparse.Namespace) -> None:
    last_seen: dict[str, int] = {}
    last_signatures: dict[str, tuple[int, tuple[tuple[str, int, int], ...]]] = {}
    hparams_written: set[str] = set()
    walltime_anchors: dict[str, float] = {}
    print(
        f"Watching {args.base_dir}/{args.game}; writing parsed metrics to Aim",
        flush=True,
    )
    while True:
        try:
            runs = discover_runs(Path(args.base_dir), args.game, prefix=args.prefix)
            completed = [run for run in runs if run.latest_iter >= args.best_iteration]
            tuning_complete = args.complete_count > 0 and len(completed) >= args.complete_count
            if tuning_complete:
                selected = list(runs)
                for prefix in args.final_prefix:
                    selected.extend(discover_runs(Path(args.base_dir), args.game, prefix=prefix))
            elif args.latest_runs > 0:
                latest = sorted(runs, key=lambda run: run.name)[-args.latest_runs:]
                rated = [
                    run for run in runs
                    if math.isfinite(milestone_elo(run, args.best_iteration))
                ]
                best = max(
                    rated,
                    key=lambda run: milestone_elo(run, args.best_iteration),
                    default=None,
                )
                selected = latest + ([best] if best is not None else [])
            else:
                selected = list(runs)
            root = Path(args.base_dir) / args.game
            for name in args.baseline_run:
                path = root / name
                if path.is_dir() and (path / "config.yaml").exists():
                    selected.append(parse_run(path))
            runs = list({run.name: run for run in selected}.values())
            changed = [r for r in runs if last_signatures.get(r.name) != _export_signature(r)]
            if changed:
                first_exports = [r for r in changed if r.name not in hparams_written]
                repeat_exports = [r for r in changed if r.name in hparams_written]
                if first_exports:
                    export_aim(
                        first_exports,
                        repo=args.aim_repo,
                        include_plot=False,
                        include_hparams=True,
                        walltime_anchors=walltime_anchors,
                    )
                    hparams_written.update(r.name for r in first_exports)
                if repeat_exports:
                    start_iters = {
                        # Ratings and match results are written after the
                        # checkpoint, so refresh the trailing completed steps.
                        run.name: max(0, min(last_seen.get(run.name, -1) + 1, run.latest_iter - 2))
                        for run in repeat_exports
                    }
                    export_aim(
                        repeat_exports,
                        repo=args.aim_repo,
                        include_plot=False,
                        include_hparams=False,
                        start_iters=start_iters,
                        walltime_anchors=walltime_anchors,
                    )
                for run in changed:
                    last_seen[run.name] = run.latest_iter
                    last_signatures[run.name] = _export_signature(run)
            rows = summarize_runs(runs)
            write_summary_csv(rows, Path(args.summary_csv))
            print(
                f"{time.strftime('%H:%M:%S')} exported {len(changed)}/{len(runs)} changed runs to Aim",
                flush=True,
            )
            if tuning_complete:
                print(
                    f"{time.strftime('%H:%M:%S')} tuning complete; exported final Aim view",
                    flush=True,
                )
                return
        except Exception as exc:
            print(f"{time.strftime('%H:%M:%S')} export failed: {exc}", flush=True)
        time.sleep(args.interval)


GENETIC_KEYS = (
    "optimizer",
    "lr",
    "weight_decay",
    "adamw_beta2",
    "train_sample_rate",
    "gating_panel_win_rate",
    "gating_best_win_rate",
    "playout_cap_percent",
)

TPE_SGD_LR_RANGE = (0.005, 0.15)
TPE_ADAMW_LR_RANGE = (0.0001, 0.005)
TPE_WEIGHT_DECAY_RANGE = (1e-6, 1e-3)
TPE_BETA2_RANGE = (0.95, 0.9999)
TPE_SAMPLE_RATE_RANGE = (6, 14)
TPE_GATE_RANGE = (0.50, 0.55)
TPE_PLAYOUT_CAP_RANGE = (0.50, 0.90)

FOCUSED_SGD_RANGES = {
    "lr": (0.04, 0.09),
    "gating_threshold": (0.515, 0.548),
}
FOCUSED_ADAMW_RANGES = {
    "playout_cap_percent": (0.68, 0.77),
    "adamw_beta2": (0.988, 0.9999),
    "train_sample_rate": (9, 14),
}
VISIT_TPE_DEFAULT_SEED_RUNS = (
    "nineplane-hierarchy-6d24c-gumbel128-lr022-50",
    "nineplane-hierarchy-6d24c-gumbel256-m24-from0-50",
    "nineplane-hierarchy-6d24c-gumbel512-m32-from0-50",
)


def genetic_fitness(run: RunData, window: int = 15) -> float:
    """Rate a run using current strength, recent progress, and stability."""
    indices = _observed_rating_indices(run.elo, run.per_iter_games, run.latest_iter)
    if not indices:
        return float("-inf")
    indices = indices[-window:]
    values = np.asarray([run.elo[i] for i in indices], dtype=float)
    slope = 0.0
    if values.size >= 3:
        slope = float(np.polyfit(np.asarray(indices, dtype=float), values, 1)[0])
    volatility = float(np.std(np.diff(values))) if values.size >= 3 else 0.0
    return float(values[-1] + 5.0 * slope - 0.15 * volatility)


def milestone_elo(run: RunData, iteration: int = 20, window: int = 4) -> float:
    indices = [
        i for i in _observed_rating_indices(run.elo, run.per_iter_games, run.latest_iter)
        if i <= iteration
    ][-window:]
    if not indices:
        return float("nan")
    return float(np.mean([run.elo[i] for i in indices]))


def tpe_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    optimizer = str(config.get("optimizer", "sgd"))
    params: dict[str, Any] = {"optimizer": optimizer}
    if optimizer == "adamw":
        params["adamw_lr"] = _clamp(float(config.get("lr", 0.001)), *TPE_ADAMW_LR_RANGE)
        params["adamw_beta2"] = _clamp(float(config.get("adamw_beta2", 0.999)), *TPE_BETA2_RANGE)
    else:
        params["sgd_lr"] = _clamp(float(config.get("lr", 0.05)), *TPE_SGD_LR_RANGE)
    params["weight_decay"] = _clamp(
        max(float(config.get("weight_decay", 1e-4)), TPE_WEIGHT_DECAY_RANGE[0]),
        *TPE_WEIGHT_DECAY_RANGE,
    )
    params["train_sample_rate"] = int(_clamp(
        int(config.get("train_sample_rate", 10)), *TPE_SAMPLE_RATE_RANGE
    ))
    params["gating_threshold"] = _clamp(
        float(config.get("gating_best_win_rate", config.get("gating_panel_win_rate", 0.52))),
        *TPE_GATE_RANGE,
    )
    params["playout_cap_percent"] = _clamp(
        float(config.get("playout_cap_percent", 0.75)), *TPE_PLAYOUT_CAP_RANGE
    )
    return params


def tpe_config_overrides(params: dict[str, Any]) -> dict[str, Any]:
    optimizer = str(params["optimizer"])
    gate = float(params["gating_threshold"])
    overrides = {
        "optimizer": optimizer,
        "lr": float(params["adamw_lr"] if optimizer == "adamw" else params["sgd_lr"]),
        "weight_decay": float(params["weight_decay"]),
        "train_sample_rate": int(params["train_sample_rate"]),
        "gating_panel_win_rate": gate,
        "gating_best_win_rate": gate,
        "playout_cap_percent": float(params["playout_cap_percent"]),
        "lr_schedule": "adaptive",
    }
    if optimizer == "adamw":
        overrides["adamw_beta2"] = float(params["adamw_beta2"])
    return overrides


def _suggest_tpe_params(trial: Any) -> dict[str, Any]:
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adamw"])
    params: dict[str, Any] = {"optimizer": optimizer}
    if optimizer == "adamw":
        params["adamw_lr"] = trial.suggest_float("adamw_lr", *TPE_ADAMW_LR_RANGE, log=True)
        params["adamw_beta2"] = trial.suggest_float("adamw_beta2", *TPE_BETA2_RANGE)
    else:
        params["sgd_lr"] = trial.suggest_float("sgd_lr", *TPE_SGD_LR_RANGE, log=True)
    params["weight_decay"] = trial.suggest_float(
        "weight_decay", *TPE_WEIGHT_DECAY_RANGE, log=True
    )
    params["train_sample_rate"] = trial.suggest_int(
        "train_sample_rate", *TPE_SAMPLE_RATE_RANGE
    )
    params["gating_threshold"] = trial.suggest_float("gating_threshold", *TPE_GATE_RANGE)
    params["playout_cap_percent"] = trial.suggest_float(
        "playout_cap_percent", *TPE_PLAYOUT_CAP_RANGE
    )
    return params


def _tpe_distributions(params: dict[str, Any]) -> dict[str, Any]:
    import optuna

    distributions: dict[str, Any] = {
        "optimizer": optuna.distributions.CategoricalDistribution(["sgd", "adamw"]),
        "weight_decay": optuna.distributions.FloatDistribution(*TPE_WEIGHT_DECAY_RANGE, log=True),
        "train_sample_rate": optuna.distributions.IntDistribution(*TPE_SAMPLE_RATE_RANGE),
        "gating_threshold": optuna.distributions.FloatDistribution(*TPE_GATE_RANGE),
        "playout_cap_percent": optuna.distributions.FloatDistribution(*TPE_PLAYOUT_CAP_RANGE),
    }
    if params["optimizer"] == "adamw":
        distributions["adamw_lr"] = optuna.distributions.FloatDistribution(*TPE_ADAMW_LR_RANGE, log=True)
        distributions["adamw_beta2"] = optuna.distributions.FloatDistribution(*TPE_BETA2_RANGE)
    else:
        distributions["sgd_lr"] = optuna.distributions.FloatDistribution(*TPE_SGD_LR_RANGE, log=True)
    return distributions


def seed_tpe_study(
    study: Any,
    base_dir: Path,
    game: str,
    iteration: int,
    exclude_prefix: str = "",
    include_prefixes: tuple[str, ...] = (),
) -> int:
    import optuna

    imported = {
        trial.user_attrs.get("source_run")
        for trial in study.trials
        if trial.user_attrs.get("source_run")
    }
    added = 0
    for run in discover_runs(base_dir, game):
        if (
            run.latest_iter < iteration
            or run.name in imported
            or (exclude_prefix and run.name.startswith(exclude_prefix))
            or (include_prefixes and not run.name.startswith(include_prefixes))
        ):
            continue
        value = milestone_elo(run, iteration)
        if not math.isfinite(value):
            continue
        params = tpe_params_from_config(run.config)
        trial = optuna.trial.create_trial(
            params=params,
            distributions=_tpe_distributions(params),
            value=value,
            user_attrs={
                "source_run": run.name,
                "historical": True,
                "objective_iteration": iteration,
            },
        )
        study.add_trial(trial)
        added += 1
    return added


def seed_lr_only_study(
    study: Any,
    run_dir: Path,
    iteration: int,
    lr_range: tuple[float, float],
) -> int:
    """Seed an LR-only study from one compatible completed baseline run."""
    import optuna

    if not (run_dir / "config.yaml").exists():
        return 0
    source_name = run_dir.name
    if any(trial.user_attrs.get("source_run") == source_name for trial in study.trials):
        return 0
    run = parse_run(run_dir)
    if run.latest_iter < iteration:
        return 0
    value = milestone_elo(run, iteration)
    lr = float(run.config.get("lr", float("nan")))
    low, high = lr_range
    if not math.isfinite(value) or not math.isfinite(lr) or not low <= lr <= high:
        return 0
    study.add_trial(optuna.trial.create_trial(
        params={"lr": lr},
        distributions={
            "lr": optuna.distributions.FloatDistribution(low, high, log=True),
        },
        value=value,
        user_attrs={
            "source_run": source_name,
            "historical": True,
            "objective_iteration": iteration,
        },
    ))
    return 1


def tpe_tune(args: argparse.Namespace) -> None:
    import optuna

    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / (args.prefix + '.db')).resolve()}"
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        study_name=args.prefix,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    base_config_path = Path(args.config)
    lr_range = (float(args.lr_min), float(args.lr_max))
    if args.lr_only:
        added = seed_lr_only_study(
            study, base_config_path.parent, args.iterations, lr_range,
        )
    else:
        added = seed_tpe_study(
            study,
            Path(args.base_dir),
            args.game,
            args.iterations,
            exclude_prefix=args.prefix,
            include_prefixes=tuple(args.historical_prefix),
        )
    print(f"Imported {added} new historical observations; study has {len(study.trials)} trials", flush=True)

    if not args.lr_only and not any(trial.user_attrs.get("lr050_start") for trial in study.trials):
        start_params = tpe_params_from_config(load_yaml(Path(args.config)))
        study.enqueue_trial(start_params, user_attrs={"lr050_start": True})

    base_cfg = load_yaml(Path(args.config))

    def objective(trial: Any) -> float:
        if args.lr_only:
            params = {"lr": trial.suggest_float("lr", *lr_range, log=True)}
            overrides = {"optimizer": "sgd", "lr": params["lr"]}
        else:
            params = _suggest_tpe_params(trial)
            overrides = tpe_config_overrides(params)
        name = f"{args.prefix}-t{trial.number:04d}"
        trial.set_user_attr("experiment", name)
        cfg = dict(base_cfg)
        cfg.update(overrides)
        cfg.update({
            "game": args.game,
            "iterations": args.iterations,
            "torch_compile": args.torch_compile,
        })
        cfg_path = cfg_dir / f"{name}.yaml"
        _write_config(cfg_path, cfg)
        run_dir = Path(args.base_dir) / args.game / name
        if run_dir.exists() and max(_checkpoint_iters(run_dir) or [0]) >= args.iterations:
            run = parse_run(run_dir)
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            cmd = [
                sys.executable,
                "src/train.py",
                "--config",
                str(cfg_path),
                "--experiment",
                name,
                "--base-dir",
                args.base_dir,
            ]
            with (log_dir / f"{name}.log").open("a") as lf:
                lf.write(" ".join(cmd) + "\n")
                lf.flush()
                result = subprocess.run(cmd, cwd=Path.cwd(), stdout=lf, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(f"Trial failed: {name}; see {log_dir / (name + '.log')}")
            run = parse_run(run_dir)
        value = milestone_elo(run, args.iterations)
        trial.set_user_attr("best_elo", run.best_elo)
        trial.set_user_attr("final_elo", run.final_elo)
        trial.set_user_attr("objective_iteration", args.iterations)
        if not math.isfinite(value):
            raise RuntimeError(f"No rating data for {name}")
        return value

    generated = sum(
        1 for trial in study.trials
        if trial.user_attrs.get("experiment")
    )
    remaining = max(0, args.trials - generated)
    print(f"Generated trials: {generated}/{args.trials}; scheduling {remaining}", flush=True)
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    print(f"Best TPE trial: {study.best_trial.number} value={study.best_value:.1f}", flush=True)


def _focused_trial_overrides(kind: str, trial: Any) -> dict[str, Any]:
    if kind == "sgd":
        gate = trial.suggest_float("gating_threshold", *FOCUSED_SGD_RANGES["gating_threshold"])
        return {
            "lr": trial.suggest_float("lr", *FOCUSED_SGD_RANGES["lr"], log=True),
            "gating_panel_win_rate": gate,
            "gating_best_win_rate": gate,
        }
    return {
        "playout_cap_percent": trial.suggest_float(
            "playout_cap_percent", *FOCUSED_ADAMW_RANGES["playout_cap_percent"]
        ),
        "adamw_beta2": trial.suggest_float("adamw_beta2", *FOCUSED_ADAMW_RANGES["adamw_beta2"]),
        "train_sample_rate": trial.suggest_int(
            "train_sample_rate", *FOCUSED_ADAMW_RANGES["train_sample_rate"]
        ),
    }


def _focused_base_params(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    if kind == "sgd":
        return {
            "lr": float(config["lr"]),
            "gating_threshold": float(config["gating_best_win_rate"]),
        }
    return {
        "playout_cap_percent": float(config["playout_cap_percent"]),
        "adamw_beta2": float(config["adamw_beta2"]),
        "train_sample_rate": int(config["train_sample_rate"]),
    }


def _focused_startup_trials(kind: str, args: argparse.Namespace) -> int:
    return args.sgd_startup_trials if kind == "sgd" else args.adamw_startup_trials


def _run_focused_training(
    name: str,
    config_path: Path,
    base_dir: Path,
    game: str,
    log_dir: Path,
    resume: bool = False,
) -> None:
    cmd = [sys.executable, "src/train.py"]
    if resume:
        cmd.extend(["--resume", str(base_dir / game / name)])
    else:
        cmd.extend(["--config", str(config_path), "--experiment", name, "--base-dir", str(base_dir)])
    with (log_dir / f"{name}.log").open("a") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Focused trial failed: {name}; see {log_dir / (name + '.log')}")


def focused_tune(args: argparse.Namespace) -> None:
    import optuna

    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    state_path = out_dir / "state.json"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state = {"phase": "screening", "current": "", "promoted": [], "complete": False}
    state_path.write_text(json.dumps(state, indent=2))
    studies: dict[str, Any] = {}

    for kind, base_path, label in (
        ("sgd", Path(args.sgd_config), "SGD"),
        ("adamw", Path(args.adamw_config), "AdamW"),
    ):
        base_cfg = load_yaml(base_path)
        storage = f"sqlite:///{(out_dir / (kind + '.db')).resolve()}"
        study = optuna.create_study(
            study_name=f"focused-{kind}",
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed + (0 if kind == "sgd" else 1),
                n_startup_trials=_focused_startup_trials(kind, args),
            ),
            load_if_exists=True,
        )
        studies[kind] = study
        if not study.trials:
            study.enqueue_trial(_focused_base_params(kind, base_cfg))
        for stale in study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
        ):
            name = str(stale.user_attrs.get("experiment", f"{label}-{stale.number + 1:02d}"))
            run_dir = Path(args.base_dir) / args.game / name
            if not run_dir.exists():
                study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
                continue
            cfg_path = run_dir / "config.yaml"
            cfg = load_yaml(cfg_path)
            cfg["iterations"] = args.iterations
            _write_config(cfg_path, cfg)
            state.update({"phase": f"screening-{kind}", "current": name})
            state_path.write_text(json.dumps(state, indent=2))
            if max(_checkpoint_iters(run_dir) or [0]) < args.iterations:
                _run_focused_training(
                    name, cfg_path, Path(args.base_dir), args.game, log_dir, resume=True
                )
            run = parse_run(run_dir)
            value = milestone_elo(run, args.iterations)
            if math.isfinite(value):
                study.tell(stale.number, value)
            else:
                study.tell(stale.number, state=optuna.trial.TrialState.FAIL)

        def objective(trial: Any, *, kind: str = kind, label: str = label, base_cfg: dict[str, Any] = base_cfg) -> float:
            name = f"{label}-{trial.number + 1:02d}"
            trial.set_user_attr("experiment", name)
            cfg = dict(base_cfg)
            cfg.update(_focused_trial_overrides(kind, trial))
            cfg.update({"iterations": args.iterations, "torch_compile": args.torch_compile, "game": args.game})
            cfg_path = cfg_dir / f"{name}.yaml"
            _write_config(cfg_path, cfg)
            run_dir = Path(args.base_dir) / args.game / name
            state.update({"phase": f"screening-{kind}", "current": name})
            state_path.write_text(json.dumps(state, indent=2))
            if not run_dir.exists() or max(_checkpoint_iters(run_dir) or [0]) < args.iterations:
                if run_dir.exists():
                    _write_config(run_dir / "config.yaml", cfg)
                    _run_focused_training(
                        name, run_dir / "config.yaml", Path(args.base_dir), args.game, log_dir,
                        resume=True,
                    )
                else:
                    _run_focused_training(name, cfg_path, Path(args.base_dir), args.game, log_dir)
            run = parse_run(run_dir)
            value = milestone_elo(run, args.iterations)
            trial.set_user_attr("objective_iteration", args.iterations)
            trial.set_user_attr("best_elo", run.best_elo)
            if not math.isfinite(value):
                raise RuntimeError(f"No rating data for {name}")
            return value

        remaining = max(0, args.trials - len(study.trials))
        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    state.update({"phase": "promotion", "current": ""})
    state_path.write_text(json.dumps(state, indent=2))
    promoted: list[str] = []
    for kind in ("sgd", "adamw"):
        top = sorted(studies[kind].trials, key=lambda trial: trial.value or float("-inf"), reverse=True)[:args.promote]
        for trial in top:
            name = str(trial.user_attrs["experiment"])
            run_dir = Path(args.base_dir) / args.game / name
            cfg_path = run_dir / "config.yaml"
            cfg = load_yaml(cfg_path)
            cfg["iterations"] = args.promote_iterations
            _write_config(cfg_path, cfg)
            state.update({"current": name, "promoted": promoted + [name]})
            state_path.write_text(json.dumps(state, indent=2))
            if max(_checkpoint_iters(run_dir) or [0]) < args.promote_iterations:
                _run_focused_training(
                    name, cfg_path, Path(args.base_dir), args.game, log_dir, resume=True
                )
            promoted.append(name)

    state.update({"phase": "complete", "current": "", "promoted": promoted, "complete": True})
    state_path.write_text(json.dumps(state, indent=2))


def _visit_params_from_config(config: dict[str, Any]) -> dict[str, int]:
    return {
        "selfplay_mcts_visits": int(config.get("selfplay_mcts_visits", 128)),
        "gumbel_m": int(config.get("gumbel_m", 16)),
    }


def _visit_distributions(args: argparse.Namespace) -> dict[str, Any]:
    import optuna

    return {
        "selfplay_mcts_visits": optuna.distributions.IntDistribution(
            args.visits_min, args.visits_max, step=args.visits_step
        ),
        "gumbel_m": optuna.distributions.IntDistribution(
            args.gumbel_m_min, args.gumbel_m_max, step=args.gumbel_m_step
        ),
    }


def seed_visit_study(study: Any, args: argparse.Namespace) -> int:
    import optuna

    seed_names = list(args.seed_run or VISIT_TPE_DEFAULT_SEED_RUNS)
    already = {
        trial.user_attrs.get("source_run")
        for trial in study.trials
        if trial.user_attrs.get("source_run")
    }
    distributions = _visit_distributions(args)
    added = 0
    for name in seed_names:
        if name in already:
            continue
        run_dir = Path(args.base_dir) / args.game / name
        if not run_dir.exists():
            continue
        run = parse_run(run_dir)
        if run.latest_iter < args.iterations:
            continue
        params = _visit_params_from_config(run.config)
        visits = params["selfplay_mcts_visits"]
        gumbel_m = params["gumbel_m"]
        if not (
            args.visits_min <= visits <= args.visits_max
            and (visits - args.visits_min) % args.visits_step == 0
            and args.gumbel_m_min <= gumbel_m <= args.gumbel_m_max
            and (gumbel_m - args.gumbel_m_min) % args.gumbel_m_step == 0
        ):
            continue
        value = milestone_elo(run, args.iterations)
        if not math.isfinite(value):
            continue
        study.add_trial(optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=value,
            user_attrs={
                "source_run": name,
                "historical": True,
                "objective_iteration": args.iterations,
                "best_elo": run.best_elo,
                "final_elo": run.final_elo,
            },
        ))
        added += 1
    return added


def visit_tune(args: argparse.Namespace) -> None:
    import optuna

    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / (args.prefix + '.db')).resolve()}"
    study = optuna.create_study(
        study_name=args.prefix,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=args.startup_trials,
            multivariate=True,
            group=True,
        ),
        load_if_exists=True,
    )
    added = seed_visit_study(study, args)
    print(f"Imported {added} visit observations; study has {len(study.trials)} trials", flush=True)

    base_cfg = load_yaml(Path(args.config))
    has_512 = any(
        trial.params.get("selfplay_mcts_visits") == 512
        and trial.params.get("gumbel_m") == 32
        for trial in study.trials
    )
    if not has_512 and not any(trial.user_attrs.get("baseline_512") for trial in study.trials):
        study.enqueue_trial(
            {
                "selfplay_mcts_visits": min(max(512, args.visits_min), args.visits_max),
                "gumbel_m": min(max(32, args.gumbel_m_min), args.gumbel_m_max),
            },
            user_attrs={"baseline_512": True},
        )

    def objective(trial: Any) -> float:
        visits = trial.suggest_int(
            "selfplay_mcts_visits", args.visits_min, args.visits_max, step=args.visits_step
        )
        gumbel_m = trial.suggest_int(
            "gumbel_m", args.gumbel_m_min, args.gumbel_m_max, step=args.gumbel_m_step
        )
        name = f"{args.prefix}-t{trial.number:04d}-v{visits}-m{gumbel_m}"
        trial.set_user_attr("experiment", name)
        cfg = dict(base_cfg)
        cfg.update({
            "game": args.game,
            "iterations": args.iterations,
            "selfplay_mcts_visits": int(visits),
            "gumbel_m": int(gumbel_m),
            "gumbel_enabled": True,
            "fast_search_algo": "puct",
            "torch_compile": args.torch_compile,
        })
        cfg_path = cfg_dir / f"{name}.yaml"
        _write_config(cfg_path, cfg)
        run_dir = Path(args.base_dir) / args.game / name
        if run_dir.exists() and max(_checkpoint_iters(run_dir) or [0]) >= args.iterations:
            run = parse_run(run_dir)
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            cmd = [
                sys.executable,
                "src/train.py",
                "--config",
                str(cfg_path),
                "--experiment",
                name,
                "--base-dir",
                args.base_dir,
            ]
            with (log_dir / f"{name}.log").open("a") as log:
                log.write(" ".join(cmd) + "\n")
                log.flush()
                result = subprocess.run(cmd, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(f"Visit trial failed: {name}; see {log_dir / (name + '.log')}")
            run = parse_run(run_dir)
        value = milestone_elo(run, args.iterations)
        trial.set_user_attr("best_elo", run.best_elo)
        trial.set_user_attr("final_elo", run.final_elo)
        trial.set_user_attr("objective_iteration", args.iterations)
        if not math.isfinite(value):
            raise RuntimeError(f"No rating data for {name}")
        return value

    generated = sum(1 for trial in study.trials if trial.user_attrs.get("experiment"))
    remaining = max(0, args.trials - generated)
    print(f"Generated visit trials: {generated}/{args.trials}; scheduling {remaining}", flush=True)
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    best = study.best_trial
    print(
        f"Best visit trial: {best.number} value={study.best_value:.1f} "
        f"params={best.params}",
        flush=True,
    )


def watch_focused_aim(args: argparse.Namespace) -> None:
    state_path = Path(args.state)
    base_root = Path(args.base_dir) / args.game
    last_signatures: dict[str, tuple[int, tuple[tuple[str, int, int], ...]]] = {}
    while True:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        runs = []
        for prefix in ("SGD-", "AdamW-"):
            group = discover_runs(Path(args.base_dir), args.game, prefix=prefix)
            if state.get("complete"):
                runs.extend(group)
            elif group:
                completed = [run for run in group if run.latest_iter >= args.objective_iteration]
                if completed:
                    runs.append(max(completed, key=lambda run: milestone_elo(run, args.objective_iteration)))
                current = next((run for run in group if run.name == state.get("current")), None)
                if current is not None:
                    runs.append(current)
        baseline = base_root / args.baseline_run
        if baseline.exists():
            runs.append(parse_run(baseline))
        runs = list({run.name: run for run in runs}.values())
        keep = {run.name for run in runs}
        changed = [run for run in runs if last_signatures.get(run.name) != _export_signature(run)]
        if changed:
            export_aim(changed, repo=args.aim_repo, include_plot=False)
            for run in changed:
                last_signatures[run.name] = _export_signature(run)
        if state.get("complete"):
            print(f"Focused tuning complete; exported {len(runs)} runs", flush=True)
            return
        print(
            f"{time.strftime('%H:%M:%S')} exported {len(changed)}/{len(runs)} focused runs: "
            f"{sorted(keep)}",
            flush=True,
        )
        time.sleep(args.interval)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def mutate_genetic_hparams(parent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Create one bounded mutation while preserving paired gating settings."""
    child = {key: parent[key] for key in GENETIC_KEYS if key in parent}
    child.setdefault("optimizer", "sgd")
    child.setdefault("lr", 0.05 if child["optimizer"] == "sgd" else 0.001)
    child.setdefault("weight_decay", 1e-4)
    child.setdefault("adamw_beta2", 0.999)
    child.setdefault("train_sample_rate", 10)
    child.setdefault("gating_panel_win_rate", 0.52)
    child.setdefault("gating_best_win_rate", child["gating_panel_win_rate"])
    child.setdefault("playout_cap_percent", 0.75)

    changed = False
    if rng.random() < 0.15:
        child["optimizer"] = "adamw" if child["optimizer"] == "sgd" else "sgd"
        child["lr"] = 0.001 if child["optimizer"] == "adamw" else 0.05
        changed = True
    if rng.random() < 0.8:
        low, high = ((1e-4, 1e-2) if child["optimizer"] == "adamw" else (5e-3, 0.2))
        child["lr"] = _clamp(float(child["lr"]) * math.exp(rng.gauss(0.0, 0.45)), low, high)
        changed = True
    if rng.random() < 0.35:
        wd = float(child["weight_decay"])
        if rng.random() < 0.1:
            wd = 0.0
        else:
            wd = max(wd, 1e-6) * math.exp(rng.gauss(0.0, 0.7))
            wd = _clamp(wd, 1e-6, 1e-3)
        child["weight_decay"] = wd
        changed = True
    if child["optimizer"] == "adamw" and rng.random() < 0.45:
        child["adamw_beta2"] = _clamp(float(child["adamw_beta2"]) + rng.gauss(0.0, 0.015), 0.9, 0.9999)
        changed = True
    if rng.random() < 0.3:
        child["train_sample_rate"] = int(_clamp(int(child["train_sample_rate"]) + rng.choice([-2, -1, 1, 2]), 4, 20))
        changed = True
    if rng.random() < 0.3:
        gate = _clamp(float(child["gating_panel_win_rate"]) + rng.gauss(0.0, 0.01), 0.50, 0.56)
        child["gating_panel_win_rate"] = gate
        child["gating_best_win_rate"] = gate
        changed = True
    if rng.random() < 0.3:
        child["playout_cap_percent"] = _clamp(float(child["playout_cap_percent"]) + rng.gauss(0.0, 0.08), 0.4, 0.9)
        changed = True
    if not changed:
        low, high = ((1e-4, 1e-2) if child["optimizer"] == "adamw" else (5e-3, 0.2))
        child["lr"] = _clamp(float(child["lr"]) * 1.1, low, high)
    return child


def initial_genetic_population(base_cfg: dict[str, Any], population: int, seed: int) -> list[dict[str, Any]]:
    base = {key: base_cfg[key] for key in GENETIC_KEYS if key in base_cfg}
    base.setdefault("optimizer", "sgd")
    base.setdefault("adamw_beta2", 0.999)
    adamw = dict(base)
    adamw.update({"optimizer": "adamw", "lr": 0.001, "adamw_beta2": 0.999})
    out = [base, adamw]
    rng = random.Random(seed)
    while len(out) < population:
        out.append(mutate_genetic_hparams(out[len(out) % 2], rng))
    return out[:population]


def _genetic_name(prefix: str, generation: int, index: int) -> str:
    return f"{prefix}-g{generation:02d}-t{index:02d}"


def _export_genetic_scores(
    candidates: list[dict[str, Any]],
    rung: int,
    repo: str | None,
    survivors: set[str],
) -> None:
    import aim

    for candidate in candidates:
        run_dir = Path("data") / "uttt" / candidate["name"]
        aim_hash = run_dir / ".aim_run_hash"
        if aim_hash.exists():
            run = aim.Run(run_hash=aim_hash.read_text().strip(), repo=repo, force_resume=True)
        else:
            run = aim.Run(experiment=candidate["name"], repo=repo)
            run.name = candidate["name"]
        try:
            generation = float(candidate["name"].split("-g")[-1].split("-")[0])
            run.track(float(candidate["fitness"]), name="genetic", epoch=rung, step=rung, context={"stat": "fitness"})
            run.track(float(candidate["name"] in survivors), name="genetic", epoch=rung, step=rung, context={"stat": "survived_rung"})
            run.track(generation, name="genetic", epoch=rung, step=rung, context={"stat": "generation"})
        finally:
            run.close()


def _run_genetic_rung(
    candidates: list[dict[str, Any]],
    rung: int,
    args: argparse.Namespace,
    cfg_dir: Path,
    log_dir: Path,
) -> None:
    pending: list[tuple[str, Path]] = []
    for candidate in candidates:
        name = candidate["name"]
        run_dir = Path(args.base_dir) / args.game / name
        latest = max(_checkpoint_iters(run_dir) or [0])
        if latest >= rung:
            continue
        if run_dir.exists() and latest > 0:
            cfg = load_yaml(run_dir / "config.yaml")
            cfg["iterations"] = rung
            _write_config(run_dir / "config.yaml", cfg)
            pending.append((name, run_dir))
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            cfg = load_yaml(Path(args.config))
            cfg.update(candidate["hparams"])
            cfg.update({"iterations": rung, "torch_compile": args.torch_compile, "game": args.game})
            cfg_path = cfg_dir / f"{name}.yaml"
            _write_config(cfg_path, cfg)
            pending.append((name, cfg_path))

    for offset in range(0, len(pending), args.concurrency):
        processes = []
        for name, path in pending[offset:offset + args.concurrency]:
            if path.is_dir():
                cmd = [sys.executable, "src/train.py", "--resume", str(path)]
            else:
                cmd = [sys.executable, "src/train.py", "--config", str(path), "--experiment", name, "--base-dir", args.base_dir]
            log = (log_dir / f"{name}.log").open("a")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            processes.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        for name, process, log in processes:
            returncode = process.wait()
            log.close()
            if returncode != 0:
                raise SystemExit(f"Genetic trial failed: {name}; see {log_dir / (name + '.log')}")


def genetic_tune(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    state_path = out_dir / f"{args.prefix}-state.json"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rungs = sorted(set(args.rungs))
    if not rungs or rungs[-1] != args.iterations:
        raise SystemExit("--rungs must end at --iterations")

    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        population = initial_genetic_population(load_yaml(Path(args.config)), args.population, args.seed)
        state = {"generation": 0, "rung_index": 0, "candidates": [], "history": []}
        state["candidates"] = [
            {"name": _genetic_name(args.prefix, 0, i + 1), "hparams": hp}
            for i, hp in enumerate(population)
        ]
        state_path.write_text(json.dumps(state, indent=2))

    while state["generation"] < args.generations:
        candidates = state["candidates"]
        for rung_index in range(state["rung_index"], len(rungs)):
            rung = rungs[rung_index]
            print(f"Generation {state['generation']} rung {rung}: {len(candidates)} candidates", flush=True)
            _run_genetic_rung(candidates, rung, args, cfg_dir, log_dir)
            names = {candidate["name"] for candidate in candidates}
            runs = [run for run in discover_runs(Path(args.base_dir), args.game) if run.name in names]
            by_name = {run.name: run for run in runs}
            for candidate in candidates:
                candidate["fitness"] = genetic_fitness(by_name[candidate["name"]])
                candidate["latest_iter"] = by_name[candidate["name"]].latest_iter
            candidates.sort(key=lambda item: item["fitness"], reverse=True)
            write_summary_csv(summarize_runs(runs), out_dir / f"{args.prefix}-summary.csv")
            state["rung_index"] = rung_index + 1
            state["candidates"] = candidates
            if rung_index < len(rungs) - 1:
                keep = min(len(candidates), max(2, math.ceil(len(candidates) / 2)))
                _export_genetic_scores(candidates, rung, args.aim_repo, {
                    candidate["name"] for candidate in candidates[:keep]
                })
                candidates = candidates[:keep]
                state["candidates"] = candidates
            else:
                _export_genetic_scores(candidates, rung, args.aim_repo, {
                    candidate["name"] for candidate in candidates
                })
            state_path.write_text(json.dumps(state, indent=2))

        state["history"].extend(state["candidates"])
        parents = sorted(state["history"], key=lambda item: item["fitness"], reverse=True)[:2]
        state["generation"] += 1
        state["rung_index"] = 0
        if state["generation"] >= args.generations:
            state_path.write_text(json.dumps(state, indent=2))
            break
        rng = random.Random(args.seed + state["generation"])
        next_candidates = []
        for index in range(args.population):
            parent = parents[index % len(parents)]["hparams"]
            hp = mutate_genetic_hparams(parent, rng)
            next_candidates.append({
                "name": _genetic_name(args.prefix, state["generation"], index + 1),
                "hparams": hp,
            })
        state["candidates"] = next_candidates
        state_path.write_text(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--base-dir", default="data")
    p_sum.add_argument("--game", default="uttt")
    p_sum.add_argument("--out", default="data/uttt_run_summary.csv")
    p_sum.add_argument("--limit", type=int, default=20)

    p_tb = sub.add_parser("aim-export")
    p_tb.add_argument("--base-dir", default="data")
    p_tb.add_argument("--game", default="uttt")
    p_tb.add_argument("--aim-repo", default=None)

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--config", default="configs/uttt.yaml")
    p_sweep.add_argument("--base-dir", default="data")
    p_sweep.add_argument("--out-dir", default="data/uttt_tuning")
    p_sweep.add_argument("--aim-repo", default=None)
    p_sweep.add_argument("--iterations", type=int, default=40)
    p_sweep.add_argument("--trials", type=int, default=4)
    p_sweep.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)

    p_continue = sub.add_parser("continue-sweep")
    p_continue.add_argument("--config", default="configs/uttt.yaml")
    p_continue.add_argument("--base-dir", default="data")
    p_continue.add_argument("--out-dir", default="data/uttt_tuning")
    p_continue.add_argument("--aim-repo", default=None)
    p_continue.add_argument("--iterations", type=int, default=40)
    p_continue.add_argument("--trials", type=int, default=4)
    p_continue.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    p_continue.add_argument("--started", required=True)
    p_continue.add_argument("--start-index", type=int, default=2)
    p_continue.add_argument("--watch-experiment", default="")
    p_continue.add_argument("--interval", type=float, default=300.0)
    p_continue.add_argument("--summary-csv", default="data/uttt_tuning/current_sweep_summary.csv")

    p_nightly = sub.add_parser("nightly-autotune")
    p_nightly.add_argument("--config", default="configs/uttt.yaml")
    p_nightly.add_argument("--base-dir", default="data")
    p_nightly.add_argument("--out-dir", default="data/uttt_tuning")
    p_nightly.add_argument("--aim-repo", default=None)
    p_nightly.add_argument("--wait-prefix", required=True)
    p_nightly.add_argument("--coarse-trials", type=int, default=4)
    p_nightly.add_argument("--fine-trials", type=int, default=6)
    p_nightly.add_argument("--fine-iterations", type=int, default=30)
    p_nightly.add_argument("--final-iterations", type=int, default=200)
    p_nightly.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    p_nightly.add_argument("--interval", type=float, default=300.0)
    p_nightly.add_argument("--summary-csv", default="data/uttt_tuning/current_sweep_summary.csv")

    p_watch = sub.add_parser("watch-aim")
    p_watch.add_argument("--base-dir", default="data")
    p_watch.add_argument("--game", default="uttt")
    p_watch.add_argument("--aim-repo", default=None)
    p_watch.add_argument("--interval", type=float, default=30.0)
    p_watch.add_argument("--prefix", default="")
    p_watch.add_argument("--summary-csv", default="data/uttt_run_summary.csv")
    p_watch.add_argument("--baseline-run", action="append", default=[])
    p_watch.add_argument("--latest-runs", type=int, default=0)
    p_watch.add_argument("--best-iteration", type=int, default=20)
    p_watch.add_argument("--complete-count", type=int, default=0)
    p_watch.add_argument("--final-prefix", action="append", default=[])

    p_branch = sub.add_parser("branch-run")
    p_branch.add_argument("--source", required=True)
    p_branch.add_argument("--experiment", required=True)
    p_branch.add_argument("--base-dir", default="data")
    p_branch.add_argument("--game", default="uttt")
    p_branch.add_argument("--iterations", type=int, default=None)
    p_branch.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p_branch.add_argument("--set-lr-state", action="append", default=[], metavar="KEY=VALUE")
    p_branch.add_argument("--through-iter", type=int, default=None)
    p_branch.add_argument("--overwrite", action="store_true")

    p_plateau = sub.add_parser("plateau-sweep")
    p_plateau.add_argument("--source", required=True)
    p_plateau.add_argument("--prefix", default="plateau-k5")
    p_plateau.add_argument("--base-dir", default="data")
    p_plateau.add_argument("--game", default="uttt")
    p_plateau.add_argument("--out-dir", default="data/uttt_tuning")
    p_plateau.add_argument("--aim-repo", default=None)
    p_plateau.add_argument("--iterations", type=int, default=80)
    p_plateau.add_argument("--trials", type=int, default=6)
    p_plateau.add_argument("--overwrite", action="store_true")

    p_genetic = sub.add_parser("genetic-tune")
    p_genetic.add_argument("--config", default="data/uttt/lrscan-20260605-08-lr050/config.yaml")
    p_genetic.add_argument("--base-dir", default="data")
    p_genetic.add_argument("--game", default="uttt")
    p_genetic.add_argument("--out-dir", default="data/uttt_genetic")
    p_genetic.add_argument("--aim-repo", default=None)
    p_genetic.add_argument("--prefix", default="genetic-uttt")
    p_genetic.add_argument("--population", type=int, default=4)
    p_genetic.add_argument("--generations", type=int, default=4)
    p_genetic.add_argument("--iterations", type=int, default=60)
    p_genetic.add_argument("--rungs", type=int, nargs="+", default=[20, 40, 60])
    p_genetic.add_argument("--concurrency", type=int, default=2)
    p_genetic.add_argument("--seed", type=int, default=20260607)
    p_genetic.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)

    p_tpe = sub.add_parser("tpe-tune")
    p_tpe.add_argument("--config", default="data/uttt/lrscan-20260605-08-lr050/config.yaml")
    p_tpe.add_argument("--base-dir", default="data")
    p_tpe.add_argument("--game", default="uttt")
    p_tpe.add_argument("--out-dir", default="data/uttt_tpe")
    p_tpe.add_argument("--aim-repo", default=None)
    p_tpe.add_argument("--prefix", default="tpe-uttt")
    p_tpe.add_argument("--iterations", type=int, default=20)
    p_tpe.add_argument("--trials", type=int, default=30)
    p_tpe.add_argument("--seed", type=int, default=20260609)
    p_tpe.add_argument("--startup-trials", type=int, default=5)
    p_tpe.add_argument("--lr-only", action="store_true")
    p_tpe.add_argument("--lr-min", type=float, default=0.005)
    p_tpe.add_argument("--lr-max", type=float, default=0.05)
    p_tpe.add_argument(
        "--historical-prefix",
        action="append",
        default=[],
        help="Only seed historical observations whose run name starts with this prefix; repeatable",
    )
    p_tpe.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)

    p_focused = sub.add_parser("focused-tune")
    p_focused.add_argument("--sgd-config", default="data/uttt/genetic-uttt-g01-t02/config.yaml")
    p_focused.add_argument("--adamw-config", default="data/uttt/genetic-uttt-g00-t04/config.yaml")
    p_focused.add_argument("--base-dir", default="data")
    p_focused.add_argument("--game", default="uttt")
    p_focused.add_argument("--out-dir", default="data/uttt_focused")
    p_focused.add_argument("--iterations", type=int, default=20)
    p_focused.add_argument("--trials", type=int, default=15)
    p_focused.add_argument("--promote", type=int, default=2)
    p_focused.add_argument("--promote-iterations", type=int, default=60)
    p_focused.add_argument("--seed", type=int, default=20260611)
    p_focused.add_argument("--sgd-startup-trials", type=int, default=4)
    p_focused.add_argument("--adamw-startup-trials", type=int, default=8)
    p_focused.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)

    p_visit = sub.add_parser("visit-tune")
    p_visit.add_argument("--config", default="data/uttt/nineplane-hierarchy-6d24c-gumbel512-m32-from0-50/config.yaml")
    p_visit.add_argument("--base-dir", default="data")
    p_visit.add_argument("--game", default="uttt")
    p_visit.add_argument("--out-dir", default="data/uttt_visit_tpe")
    p_visit.add_argument("--prefix", default="visit-tpe")
    p_visit.add_argument("--iterations", type=int, default=50)
    p_visit.add_argument("--trials", type=int, default=8)
    p_visit.add_argument("--seed", type=int, default=20260623)
    p_visit.add_argument("--startup-trials", type=int, default=4)
    p_visit.add_argument("--visits-min", type=int, default=128)
    p_visit.add_argument("--visits-max", type=int, default=2048)
    p_visit.add_argument("--visits-step", type=int, default=128)
    p_visit.add_argument("--gumbel-m-min", type=int, default=16)
    p_visit.add_argument("--gumbel-m-max", type=int, default=96)
    p_visit.add_argument("--gumbel-m-step", type=int, default=8)
    p_visit.add_argument("--seed-run", action="append", default=[])
    p_visit.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)

    p_focused_watch = sub.add_parser("watch-focused-aim")
    p_focused_watch.add_argument("--base-dir", default="data")
    p_focused_watch.add_argument("--game", default="uttt")
    p_focused_watch.add_argument("--aim-repo", default=None)
    p_focused_watch.add_argument("--state", default="data/uttt_focused/state.json")
    p_focused_watch.add_argument("--baseline-run", default="lrscan-20260605-08-lr050")
    p_focused_watch.add_argument("--objective-iteration", type=int, default=20)
    p_focused_watch.add_argument("--interval", type=float, default=120.0)

    args = parser.parse_args()
    if args.cmd == "summarize":
        runs = discover_runs(Path(args.base_dir), args.game)
        rows = summarize_runs(runs)
        write_summary_csv(rows, Path(args.out))
        print_summary(rows, args.limit)
        print(f"Wrote {args.out}")
    elif args.cmd == "aim-export":
        runs = discover_runs(Path(args.base_dir), args.game)
        export_aim(runs, args.aim_repo)
        print("Wrote parsed metrics to Aim")
    elif args.cmd == "sweep":
        launch_sweep(args)
    elif args.cmd == "continue-sweep":
        continue_sweep(args)
    elif args.cmd == "nightly-autotune":
        nightly_autotune(args)
    elif args.cmd == "watch-aim":
        watch_aim(args)
    elif args.cmd == "branch-run":
        branch_run(args)
    elif args.cmd == "plateau-sweep":
        plateau_sweep(args)
    elif args.cmd == "genetic-tune":
        genetic_tune(args)
    elif args.cmd == "tpe-tune":
        tpe_tune(args)
    elif args.cmd == "focused-tune":
        focused_tune(args)
    elif args.cmd == "visit-tune":
        visit_tune(args)
    elif args.cmd == "watch-focused-aim":
        watch_focused_aim(args)


if __name__ == "__main__":
    main()
