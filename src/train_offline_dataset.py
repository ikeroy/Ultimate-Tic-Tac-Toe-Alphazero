#!/usr/bin/env python3
"""Supervised training from offline AlphaZero history shards."""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neural_net
from config import load_config
from game_runner import StreamingCompressedDataset, glob_file_triples


class AimRun:
    """Small adapter for NNWrapper.train's run.track(...) interface."""

    def __init__(self, experiment: str, repo: str | None = None):
        import aim

        self.run = aim.Run(experiment=experiment, repo=repo)
        self.run.name = experiment

    def track(self, value, name, epoch=None, step=None, context=None):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        self.run.track(value, name=name, epoch=epoch, step=step, context=context)

    def set_hparams(self, hparams):
        self.run["hparams"] = hparams

    def close(self):
        self.run.close()


def _total_samples(file_triples):
    return sum(size for _, _, _, size in file_triples)


def _write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _load_expanded_checkpoint(student, init_checkpoint: Path):
    """Load matching weights from a checkpoint, expanding the input conv if needed."""
    teacher = neural_net.NNWrapper.load_checkpoint(
        student.game, str(init_checkpoint.parent), init_checkpoint.name
    )
    student_state = student.nnet.state_dict()
    teacher_state = teacher.nnet.state_dict()
    copied = []
    expanded = []
    skipped = []
    for key, value in teacher_state.items():
        if key not in student_state:
            skipped.append({"key": key, "reason": "missing_in_student"})
            continue
        target = student_state[key]
        if value.shape == target.shape:
            student_state[key] = value
            copied.append(key)
            continue
        if (
            value.ndim == 4
            and target.ndim == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
            and value.shape[1] < target.shape[1]
        ):
            expanded_value = target.clone()
            expanded_value.zero_()
            expanded_value[:, : value.shape[1], :, :] = value
            student_state[key] = expanded_value
            expanded.append(
                {
                    "key": key,
                    "from_shape": list(value.shape),
                    "to_shape": list(target.shape),
                }
            )
            continue
        skipped.append(
            {
                "key": key,
                "reason": "shape_mismatch",
                "teacher_shape": list(value.shape),
                "student_shape": list(target.shape),
            }
        )
    student.nnet.load_state_dict(student_state)
    return {
        "checkpoint": str(init_checkpoint.resolve()),
        "teacher_input_channels": getattr(teacher, "_input_channels", None),
        "student_input_channels": getattr(student, "_input_channels", None),
        "copied_tensors": len(copied),
        "expanded_tensors": expanded,
        "skipped_tensors": skipped,
    }


def train_policy_only(nn, dataloader, steps_to_train, run, epoch, total_train_steps):
    """Train only the policy head/trunk on imitation targets."""
    nn.nnet.train()
    pi_loss = 0.0
    current_step = 0
    pbar = tqdm.tqdm(
        total=steps_to_train, unit="batches", desc="Training Policy NN", leave=False
    )
    while current_step < steps_to_train:
        for batch in dataloader:
            if current_step == steps_to_train:
                break
            canonical, _target_vs, target_pis = batch
            canonical = canonical.float().contiguous().to(
                nn.device, non_blocking=nn._non_blocking
            )
            target_pis = target_pis.float().contiguous().to(
                nn.device, non_blocking=nn._non_blocking
            )
            canonical = nn._adapt_input_planes(canonical)

            nn.optimizer.zero_grad()
            _out_v, out_pi = nn.nnet(canonical)
            l_pi = nn.loss_pi(target_pis, out_pi)
            l_pi.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nn.nnet.parameters(), float("inf"))
            nn.optimizer.step()

            pi_val = float(l_pi.item())
            pi_loss += pi_val
            with torch.no_grad():
                mask = target_pis > 0
                target_entropy_val = -(
                    target_pis[mask] * torch.log(target_pis[mask])
                ).sum().item() / target_pis.size(0)
            run.track(
                pi_val,
                name="loss",
                epoch=epoch,
                step=total_train_steps + current_step,
                context={"type": "policy"},
            )
            run.track(
                pi_val - target_entropy_val,
                name="loss",
                epoch=epoch,
                step=total_train_steps + current_step,
                context={"type": "policy_kl_gap"},
            )
            run.track(
                grad_norm.item(),
                name="grad_norm",
                epoch=epoch,
                step=total_train_steps + current_step,
            )
            pbar.set_postfix({"pi loss": pi_val})
            pbar.update(1)
            current_step += 1
            if current_step >= steps_to_train:
                break
    pbar.close()
    return pi_loss / max(1, current_step)


def main():
    parser = argparse.ArgumentParser(description="Train a network from an offline dataset.")
    parser.add_argument("--config", required=True, help="Config YAML defining the student architecture and optimizer.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing history shards, or its history/ subdir.")
    parser.add_argument("--experiment", required=True, help="Experiment name under --base-dir/uttt.")
    parser.add_argument("--base-dir", default="data", help="Base output directory.")
    parser.add_argument("--aim-repo", default=None, help="Aim repo directory. Defaults to the project .aim repo.")
    parser.add_argument("--epochs", type=int, default=0, help="Full dataset passes. 0 uses config.train_sample_rate.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override config.train_batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override config.lr.")
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default=None, help="Override config.optimizer.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override config.weight_decay.")
    parser.add_argument("--adamw-beta1", type=float, default=None, help="Override config.adamw_beta1.")
    parser.add_argument("--adamw-beta2", type=float, default=None, help="Override config.adamw_beta2.")
    parser.add_argument("--adamw-eps", type=float, default=None, help="Override config.adamw_eps.")
    parser.add_argument("--active-files", type=int, default=0, help="Loaded shard slots for streaming.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save every N epochs.")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint to initialize from.")
    parser.add_argument("--policy-only", action="store_true", help="Train only policy loss; useful for move imitation.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    config = load_config(args.config, {})
    if args.lr is not None:
        config.lr = args.lr
    if args.optimizer is not None:
        config.optimizer = args.optimizer
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.adamw_beta1 is not None:
        config.adamw_beta1 = args.adamw_beta1
    if args.adamw_beta2 is not None:
        config.adamw_beta2 = args.adamw_beta2
    if args.adamw_eps is not None:
        config.adamw_eps = args.adamw_eps
    if args.batch_size > 0:
        config.train_batch_size = args.batch_size
    if args.active_files > 0:
        config.streaming_active_files = args.active_files

    dataset_dir = Path(args.dataset_dir)
    history_dir = dataset_dir / "history" if (dataset_dir / "history").is_dir() else dataset_dir
    file_triples = glob_file_triples(str(history_dir), "*-canonical-*.ptz")
    if not file_triples:
        file_triples = glob_file_triples(str(history_dir), "*-canonical-*.pt")
    if not file_triples:
        raise SystemExit(f"No history triples found in {history_dir}")

    samples = _total_samples(file_triples)
    batch_size = config.train_batch_size
    steps_per_epoch = math.ceil(samples / batch_size)
    epochs = args.epochs or int(config.train_sample_rate)
    total_steps = steps_per_epoch * epochs

    experiment_dir = Path(args.base_dir) / config.game / args.experiment
    checkpoint_dir = experiment_dir / "checkpoint"
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{experiment_dir} already exists and is not empty; pass --overwrite")
        shutil.rmtree(experiment_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(experiment_dir / "config.yaml"))

    metadata = {
        "status": "running",
        "experiment": args.experiment,
        "config": str(Path(args.config).resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "history_dir": str(history_dir.resolve()),
        "samples": samples,
        "shards": len(file_triples),
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_steps": total_steps,
        "lr": config.lr,
        "policy_only": bool(args.policy_only),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path = experiment_dir / "offline_train_metadata.json"
    _write_json(metadata_path, metadata)

    run = AimRun(args.experiment, repo=args.aim_repo)
    run.set_hparams({
        "lr": config.lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "samples": samples,
        "depth": config.depth,
        "channels": config.channels,
        "kernel_size": config.kernel_size,
        "uttt_hierarchical": int(config.uttt_hierarchical),
        "teacher_visits": 4096 if "4096" in str(dataset_dir) else 0,
    })

    Game = config.Game
    nn = neural_net.NNWrapper(Game, neural_net.nnargs_from_config(config))
    init_report = None
    if args.init_checkpoint:
        init_report = _load_expanded_checkpoint(nn, Path(args.init_checkpoint))
        metadata["init_checkpoint"] = init_report
        _write_json(metadata_path, metadata)
    nn.save_checkpoint(str(checkpoint_dir), f"0000-{args.experiment}.pt", zstd_level=config.zstd_level)

    start = time.time()
    total_train_steps = 0
    last_v_loss = None
    last_pi_loss = None
    try:
        for epoch in range(epochs):
            ds = StreamingCompressedDataset(
                file_triples,
                batch_size=batch_size,
                passes=1,
                active_files=config.streaming_active_files,
                seed=args.seed + epoch,
            )
            dl = DataLoader(ds, batch_size=None, num_workers=0)
            if args.policy_only:
                v_loss = 0.0
                pi_loss = train_policy_only(
                    nn,
                    dl,
                    steps_per_epoch,
                    run,
                    epoch,
                    total_train_steps,
                )
            else:
                v_loss, pi_loss = nn.train(
                    dl,
                    steps_per_epoch,
                    run,
                    epoch,
                    total_train_steps,
                    ema_averaging=config.ema_averaging,
                )
            total_train_steps += steps_per_epoch
            last_v_loss = float(v_loss)
            last_pi_loss = float(pi_loss)
            run.track(config.lr, name="learning_rate", epoch=epoch, step=total_train_steps)
            run.track(last_v_loss + last_pi_loss, name="offline_epoch_loss", epoch=epoch, step=total_train_steps, context={"type": "total"})
            run.track(last_v_loss, name="offline_epoch_loss", epoch=epoch, step=total_train_steps, context={"type": "value"})
            run.track(last_pi_loss, name="offline_epoch_loss", epoch=epoch, step=total_train_steps, context={"type": "policy"})
            metadata.update(
                {
                    "completed_epochs": epoch + 1,
                    "completed_steps": total_train_steps,
                    "last_value_loss": last_v_loss,
                    "last_policy_loss": last_pi_loss,
                    "elapsed_seconds": time.time() - start,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            _write_json(metadata_path, metadata)
            if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == epochs:
                nn.save_checkpoint(
                    str(checkpoint_dir),
                    f"{epoch + 1:04d}-{args.experiment}.pt",
                    zstd_level=config.zstd_level,
                )
    finally:
        run.close()

    metadata.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": time.time() - start,
            "completed_epochs": epochs,
            "completed_steps": total_train_steps,
            "last_value_loss": last_v_loss,
            "last_policy_loss": last_pi_loss,
        }
    )
    _write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
