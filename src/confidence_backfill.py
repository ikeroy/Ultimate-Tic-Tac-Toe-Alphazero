#!/usr/bin/env python3
"""Backfill policy/value confidence diagnostics for completed UTTT runs."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import alphazero
import neural_net
from game_runner import glob_file_triples, load_compressed


LEGAL_MOVES_PLANE = 3
EPS = 1e-12


def action_coords(action: int) -> tuple[int, int]:
    """Map UTTT action id board*9+cell to canonical 9x9 row/col."""
    group, cell = divmod(action, 9)
    return (group // 3) * 3 + cell // 3, (group % 3) * 3 + cell % 3


def spatial_legal_plane_to_action_mask(plane: np.ndarray) -> np.ndarray:
    """Convert a spatial legal-move plane to action-vector order.

    Canonical observations are laid out as a physical 9x9 board. Policy
    targets are laid out as action ids: local_board * 9 + local_cell.
    """
    arr = np.asarray(plane)
    if arr.shape[-2:] != (9, 9):
        raise ValueError(f"expected trailing shape (9, 9), got {arr.shape}")
    flat_shape = arr.shape[:-2] + (81,)
    out = np.zeros(flat_shape, dtype=np.float32)
    for action in range(81):
        row, col = action_coords(action)
        out[..., action] = arr[..., row, col]
    return out > 0.5


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    denom = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)


def distribution_stats(probs: np.ndarray, prefix: str) -> dict[str, float]:
    p = normalize_rows(probs)
    entropy = -(p * np.log(p + EPS)).sum(axis=1)
    top1 = p.max(axis=1)
    nonzero = (p > 0).sum(axis=1)
    return {
        f"{prefix}_entropy_mean": float(entropy.mean()),
        f"{prefix}_effective_moves_mean": float(np.exp(entropy).mean()),
        f"{prefix}_top1_mean": float(top1.mean()),
        f"{prefix}_top1_p95": float(np.quantile(top1, 0.95)),
        f"{prefix}_top1_p99": float(np.quantile(top1, 0.99)),
        f"{prefix}_gt90_frac": float((top1 > 0.90).mean()),
        f"{prefix}_gt97_frac": float((top1 > 0.97).mean()),
        f"{prefix}_nonzero_mean": float(nonzero.mean()),
    }


def masked_policy(raw_pi: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(raw_pi, dtype=np.float64) * legal_mask
    sums = masked.sum(axis=1, keepdims=True)
    legal_counts = legal_mask.sum(axis=1, keepdims=True)
    uniform = np.divide(
        legal_mask.astype(np.float64),
        legal_counts,
        out=np.zeros_like(masked, dtype=np.float64),
        where=legal_counts > 0,
    )
    return np.where(sums > 0, masked / np.maximum(sums, EPS), uniform)


def _find_checkpoint(checkpoint_dir: Path, iteration: int) -> Path | None:
    matches = sorted(checkpoint_dir.glob(f"{iteration:04d}-*.pt"))
    return matches[0] if matches else None


Sample = tuple[torch.Tensor, np.ndarray, np.ndarray]


def _load_history_iteration_sample(
    history_dir: Path,
    iteration: int,
    max_samples: int,
    rng: np.random.Generator,
) -> Sample | None:
    triples = glob_file_triples(str(history_dir), f"{iteration:04d}-*-canonical-*.ptz")
    if not triples:
        triples = glob_file_triples(str(history_dir), f"{iteration:04d}-*-canonical-*.pt")
    if not triples:
        return None

    total = sum(size for _, _, _, size in triples)
    remaining = min(max_samples, total)
    canonicals: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []

    for idx, (c_path, v_path, pi_path, size) in enumerate(triples):
        if remaining <= 0:
            break
        files_left = len(triples) - idx
        take = min(size, max(1, math.ceil(remaining / files_left)))
        c = load_compressed(c_path).float()
        v = load_compressed(v_path).float()
        pi = load_compressed(pi_path).float()
        if take < size:
            selected = torch.from_numpy(rng.choice(size, size=take, replace=False))
            c = c[selected]
            v = v[selected]
            pi = pi[selected]
        canonicals.append(c)
        values.append(v)
        policies.append(pi)
        remaining -= take

    if not canonicals:
        return None
    return (
        torch.cat(canonicals, dim=0),
        torch.cat(values, dim=0).numpy(),
        torch.cat(policies, dim=0).numpy(),
    )


def _load_reservoir_samples(
    reservoir_dir: Path,
    iterations: set[int],
    max_samples: int,
    seed: int,
) -> dict[int, Sample]:
    """Sample reservoir rows for many iterations while scanning chunks once."""
    if not reservoir_dir.exists() or not iterations:
        return {}

    rng = np.random.default_rng(seed)
    c_parts: dict[int, list[torch.Tensor]] = defaultdict(list)
    v_parts: dict[int, list[torch.Tensor]] = defaultdict(list)
    pi_parts: dict[int, list[torch.Tensor]] = defaultdict(list)
    counts = {iteration: 0 for iteration in iterations}

    iter_paths = sorted(reservoir_dir.glob("chunk_*_iters.ptz"))
    for iter_path in iter_paths:
        prefix = str(iter_path).removesuffix("_iters.ptz")
        c_path = Path(prefix + "_canonical.ptz")
        v_path = Path(prefix + "_v.ptz")
        pi_path = Path(prefix + "_pi.ptz")
        if not (c_path.exists() and v_path.exists() and pi_path.exists()):
            continue

        chunk_iters = load_compressed(str(iter_path)).cpu().numpy().astype(np.int64)
        relevant = [
            iteration for iteration in np.unique(chunk_iters)
            if int(iteration) in iterations and counts[int(iteration)] < max_samples
        ]
        if not relevant:
            continue

        c = load_compressed(str(c_path)).float()
        v = load_compressed(str(v_path)).float()
        pi = load_compressed(str(pi_path)).float()

        for iteration in relevant:
            iteration = int(iteration)
            remaining = max_samples - counts[iteration]
            if remaining <= 0:
                continue
            idx = np.flatnonzero(chunk_iters == iteration)
            if idx.size == 0:
                continue
            take = min(remaining, idx.size)
            if take < idx.size:
                idx = rng.choice(idx, size=take, replace=False)
            idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
            c_parts[iteration].append(c[idx_t])
            v_parts[iteration].append(v[idx_t])
            pi_parts[iteration].append(pi[idx_t])
            counts[iteration] += take

        del c, v, pi, chunk_iters
        gc.collect()

    samples: dict[int, Sample] = {}
    for iteration, parts in c_parts.items():
        if not parts:
            continue
        samples[iteration] = (
            torch.cat(parts, dim=0),
            torch.cat(v_parts[iteration], dim=0).numpy(),
            torch.cat(pi_parts[iteration], dim=0).numpy(),
        )
    return samples


def _predict_in_batches(
    net: neural_net.NNWrapper,
    canonical: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    for start in range(0, canonical.shape[0], batch_size):
        batch = canonical[start:start + batch_size]
        v, pi = net.process(batch)
        values.append(v.cpu().numpy())
        policies.append(pi.cpu().numpy())
    return np.concatenate(values, axis=0), np.concatenate(policies, axis=0)


def compute_iteration_metrics(
    experiment_dir: Path,
    iteration: int,
    max_samples: int,
    batch_size: int,
    seed: int,
    sample: Sample | None = None,
) -> dict[str, float] | None:
    if sample is None:
        sample = _load_history_iteration_sample(
            experiment_dir / "history",
            iteration,
            max_samples,
            np.random.default_rng(seed + iteration),
        )
    checkpoint = _find_checkpoint(experiment_dir / "checkpoint", iteration)
    if sample is None or checkpoint is None:
        return None

    canonical, target_v, target_pi = sample
    if canonical.shape[1] <= LEGAL_MOVES_PLANE:
        raise ValueError(
            f"canonical tensor has {canonical.shape[1]} planes; "
            f"cannot read legal-move plane {LEGAL_MOVES_PLANE}"
        )
    legal_mask = spatial_legal_plane_to_action_mask(
        canonical[:, LEGAL_MOVES_PLANE].numpy()
    )

    net = neural_net.NNWrapper.load_checkpoint(
        alphazero.UltimateTicTacToeGS,
        folder="",
        filename=str(checkpoint),
    )
    raw_v, raw_pi = _predict_in_batches(net, canonical, batch_size)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_pi_legal = masked_policy(raw_pi, legal_mask)

    metrics: dict[str, float] = {
        "iteration": float(iteration),
        "samples": float(canonical.shape[0]),
        "legal_moves_mean": float(legal_mask.sum(axis=1).mean()),
    }
    metrics.update(distribution_stats(target_pi, "target_policy"))
    metrics.update(distribution_stats(raw_pi, "raw_policy_all"))
    metrics.update(distribution_stats(raw_pi_legal, "raw_policy_legal"))
    metrics.update(distribution_stats(raw_v, "raw_value"))
    metrics.update(distribution_stats(target_v, "target_value"))
    metrics.update({
        "raw_value_win_mean": float(raw_v[:, 0].mean()),
        "raw_value_loss_mean": float(raw_v[:, 1].mean()),
        "raw_value_draw_mean": float(raw_v[:, 2].mean()),
        "target_value_win_mean": float(target_v[:, 0].mean()),
        "target_value_loss_mean": float(target_v[:, 1].mean()),
        "target_value_draw_mean": float(target_v[:, 2].mean()),
    })
    return metrics


def export_aim(metrics_by_iter: list[dict[str, float]], experiment_dir: Path, repo: str | None = None) -> None:
    import aim

    aim_hash = experiment_dir / ".aim_run_hash"
    if aim_hash.exists():
        run = aim.Run(run_hash=aim_hash.read_text().strip(), repo=repo, force_resume=True)
    else:
        run = aim.Run(experiment=experiment_dir.name, repo=repo)
        run.name = experiment_dir.name
        aim_hash.write_text(run.hash)
    tag_map = {
        "samples": ("diagnostics_confidence", {"stat": "samples"}),
        "legal_moves_mean": ("diagnostics_confidence", {"stat": "legal_moves_mean"}),
        "target_policy_entropy_mean": ("diagnostics_confidence", {"stat": "target_policy_entropy"}),
        "target_policy_effective_moves_mean": ("diagnostics_confidence", {"stat": "target_policy_effective_moves"}),
        "target_policy_top1_mean": ("diagnostics_confidence", {"stat": "target_policy_top1"}),
        "target_policy_top1_p95": ("diagnostics_confidence", {"stat": "target_policy_top1_p95"}),
        "target_policy_gt90_frac": ("diagnostics_confidence", {"stat": "target_policy_gt90_frac"}),
        "target_policy_gt97_frac": ("diagnostics_confidence", {"stat": "target_policy_gt97_frac"}),
        "raw_policy_legal_entropy_mean": ("diagnostics_confidence", {"stat": "raw_policy_legal_entropy"}),
        "raw_policy_legal_effective_moves_mean": ("diagnostics_confidence", {"stat": "raw_policy_legal_effective_moves"}),
        "raw_policy_legal_top1_mean": ("diagnostics_confidence", {"stat": "raw_policy_legal_top1"}),
        "raw_policy_legal_top1_p95": ("diagnostics_confidence", {"stat": "raw_policy_legal_top1_p95"}),
        "raw_policy_legal_gt90_frac": ("diagnostics_confidence", {"stat": "raw_policy_legal_gt90_frac"}),
        "raw_policy_legal_gt97_frac": ("diagnostics_confidence", {"stat": "raw_policy_legal_gt97_frac"}),
        "raw_policy_all_entropy_mean": ("diagnostics_confidence", {"stat": "raw_policy_all_entropy"}),
        "raw_policy_all_top1_mean": ("diagnostics_confidence", {"stat": "raw_policy_all_top1"}),
        "raw_value_entropy_mean": ("diagnostics_confidence", {"stat": "raw_value_entropy"}),
        "raw_value_effective_moves_mean": ("diagnostics_confidence", {"stat": "raw_value_effective_outcomes"}),
        "raw_value_top1_mean": ("diagnostics_confidence", {"stat": "raw_value_top1"}),
        "raw_value_top1_p95": ("diagnostics_confidence", {"stat": "raw_value_top1_p95"}),
        "raw_value_draw_mean": ("diagnostics_confidence", {"stat": "raw_value_draw"}),
        "target_value_entropy_mean": ("diagnostics_confidence", {"stat": "target_value_entropy"}),
        "target_value_top1_mean": ("diagnostics_confidence", {"stat": "target_value_top1"}),
        "target_value_draw_mean": ("diagnostics_confidence", {"stat": "target_value_draw"}),
    }
    try:
        for row in metrics_by_iter:
            step = int(row["iteration"])
            for key, (name, context) in tag_map.items():
                if key in row and math.isfinite(float(row[key])):
                    run.track(float(row[key]), name=name, epoch=step, step=step, context=context)
    finally:
        run.close()


def write_csv(metrics_by_iter: list[dict[str, float]], path: Path) -> None:
    if not metrics_by_iter:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in metrics_by_iter for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_by_iter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill confidence diagnostics for a completed UTTT run."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("data/uttt/nineplane-hierarchy-6d24c-gumbel512-m32-from0-50"),
    )
    parser.add_argument(
        "--aim-repo",
        type=Path,
        default=None,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260627)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = args.experiment_dir
    if not exp_dir.exists():
        raise SystemExit(f"experiment dir does not exist: {exp_dir}")
    exp_name = exp_dir.name
    checkpoint_dir = exp_dir / "checkpoint"
    available_iters = sorted(
        int(p.name.split("-", 1)[0])
        for p in checkpoint_dir.glob("*.pt")
        if p.name[:4].isdigit()
    )
    if not available_iters:
        raise SystemExit(f"no checkpoints found under {checkpoint_dir}")
    end = args.end if args.end is not None else max(available_iters)
    wanted = range(args.start, end + 1, max(1, args.stride))

    wanted_iters = list(wanted)
    reservoir_needed: set[int] = set()
    for iteration in wanted_iters:
        if _find_checkpoint(checkpoint_dir, iteration) is None:
            continue
        history_triples = glob_file_triples(
            str(exp_dir / "history"), f"{iteration:04d}-*-canonical-*.ptz"
        )
        if not history_triples:
            history_triples = glob_file_triples(
                str(exp_dir / "history"), f"{iteration:04d}-*-canonical-*.pt"
            )
        if not history_triples:
            reservoir_needed.add(iteration)
    reservoir_samples = _load_reservoir_samples(
        exp_dir / "reservoir",
        reservoir_needed,
        args.max_samples,
        args.seed,
    )

    rows: list[dict[str, float]] = []
    for iteration in wanted_iters:
        sample = reservoir_samples.get(iteration)
        metrics = compute_iteration_metrics(
            exp_dir,
            iteration,
            args.max_samples,
            args.batch_size,
            args.seed,
            sample=sample,
        )
        if metrics is None:
            print(f"skip iter {iteration}: missing checkpoint or history")
            continue
        rows.append(metrics)
        print(
            f"iter {iteration:04d}: "
            f"raw_top={metrics['raw_policy_legal_top1_mean']:.3f} "
            f"raw_H={metrics['raw_policy_legal_entropy_mean']:.3f} "
            f"target_top={metrics['target_policy_top1_mean']:.3f} "
            f"value_top={metrics['raw_value_top1_mean']:.3f}"
        )

    csv_path = exp_dir / "confidence_metrics.csv"
    write_csv(rows, csv_path)
    export_aim(rows, exp_dir, repo=str(args.aim_repo) if args.aim_repo else None)
    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote confidence diagnostics to Aim run for {exp_name}")


if __name__ == "__main__":
    main()
