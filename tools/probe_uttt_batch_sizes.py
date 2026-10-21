#!/usr/bin/env python3
"""Probe CUDA memory limits for UTTT network batch sizes."""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import alphazero  # noqa: E402
from neural_net import NNArgs, NNWrapper  # noqa: E402


def _parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _cuda_mem_mb() -> tuple[float, float]:
    allocated = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    return allocated, reserved


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _make_targets(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    v_idx = torch.randint(0, 3, (batch_size,), device=device)
    target_v = torch.nn.functional.one_hot(v_idx, num_classes=3).float()
    logits = torch.randn(batch_size, 81, device=device)
    target_pi = torch.softmax(logits, dim=1)
    return target_v, target_pi


def _make_wrapper(args: NNArgs) -> NNWrapper:
    wrapper = NNWrapper(alphazero.UltimateTicTacToeGS, args)
    wrapper.nnet.to(wrapper.device)
    return wrapper


def _probe_inference(args: NNArgs, batch_size: int, amp: bool) -> dict:
    _clear_cuda()
    wrapper = _make_wrapper(args)
    wrapper.nnet.eval()
    shape = alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE()
    channels = int(args.input_channels or shape[0])
    batch = torch.randn(batch_size, channels, 9, 9, device=wrapper.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        if amp:
            with torch.amp.autocast(wrapper.device.type, dtype=torch.float16):
                wrapper.nnet(batch)
        else:
            wrapper.nnet(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated, reserved = _cuda_mem_mb()
    params = sum(p.numel() for p in wrapper.nnet.parameters())
    del batch, wrapper
    _clear_cuda()
    return {
        "mode": "inference_amp" if amp else "inference_fp32",
        "batch_size": batch_size,
        "ok": True,
        "seconds": elapsed,
        "samples_per_sec": batch_size / elapsed if elapsed > 0 else 0.0,
        "max_allocated_mb": allocated,
        "max_reserved_mb": reserved,
        "params": params,
    }


def _probe_training(args: NNArgs, batch_size: int) -> dict:
    _clear_cuda()
    wrapper = _make_wrapper(args)
    wrapper.nnet.train()
    shape = alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE()
    channels = int(args.input_channels or shape[0])
    batch = torch.randn(batch_size, channels, 9, 9, device=wrapper.device)
    target_v, target_pi = _make_targets(batch_size, wrapper.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    wrapper.optimizer.zero_grad(set_to_none=True)
    out_v, out_pi = wrapper.nnet(batch)
    loss = wrapper.loss_v(target_v, out_v) + wrapper.loss_pi(target_pi, out_pi)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapper.nnet.parameters(), float("inf"))
    wrapper.optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated, reserved = _cuda_mem_mb()
    params = sum(p.numel() for p in wrapper.nnet.parameters())
    del batch, target_v, target_pi, out_v, out_pi, loss, wrapper
    _clear_cuda()
    return {
        "mode": "training_fp32",
        "batch_size": batch_size,
        "ok": True,
        "seconds": elapsed,
        "samples_per_sec": batch_size / elapsed if elapsed > 0 else 0.0,
        "max_allocated_mb": allocated,
        "max_reserved_mb": reserved,
        "params": params,
    }


def _run_probe(fn, args: NNArgs, batch_size: int) -> dict:
    try:
        return fn(args, batch_size)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        _clear_cuda()
        return {
            "mode": getattr(fn, "__name__", "unknown"),
            "batch_size": batch_size,
            "ok": False,
            "error": "oom",
        }


def _model_args(args: argparse.Namespace, channels: int, depth: int, head_channels: int) -> NNArgs:
    macro_channels = args.macro_channels if args.macro_channels > 0 else -1
    return NNArgs(
        num_channels=channels,
        depth=depth,
        kernel_size=args.kernel_size,
        dense_net=True,
        uttt_hierarchical=True,
        head_channels=head_channels,
        head_pool=True,
        uttt_policy_head="flat",
        uttt_destination_context=False,
        uttt_macro_channels=macro_channels,
        uttt_macro_transformer_layers=args.transformer_layers,
        uttt_macro_transformer_heads=args.transformer_heads,
        uttt_macro_transformer_ff_mult=args.transformer_ff_mult,
        uttt_macro_transformer_dropout=0.0,
        trunk_norm=args.trunk_norm,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        input_channels=alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE()[0],
    )


def _print_row(model_name: str, row: dict) -> None:
    if row.get("ok"):
        print(
            f"{model_name:18s} {row['mode']:14s} bs={row['batch_size']:5d} "
            f"ok  {row['samples_per_sec']:8.1f} samples/s  "
            f"alloc={row['max_allocated_mb']:8.0f}MB  "
            f"reserved={row['max_reserved_mb']:8.0f}MB  "
            f"params={row['params']:,}",
            flush=True,
        )
    else:
        print(
            f"{model_name:18s} {row['mode']:14s} bs={row['batch_size']:5d} OOM",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default="96,128,160", help="comma-separated trunk widths")
    parser.add_argument("--depths", default="12", help="comma-separated depths")
    parser.add_argument("--head-channels", default="256", help="comma-separated head widths")
    parser.add_argument("--macro-channels", type=int, default=-1, help="macro/transformer width; -1 ties to channels")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-mult", type=float, default=4.0)
    parser.add_argument("--trunk-norm", default="batch")
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--inference-batches", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--training-batches", default="256,512,1024,2048,4096")
    parser.add_argument("--no-amp-inference", action="store_true")
    parsed = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this probe is only useful on GPU.")

    channels_list = _parse_ints(parsed.channels)
    depth_list = _parse_ints(parsed.depths)
    head_list = _parse_ints(parsed.head_channels)
    inference_batches = _parse_ints(parsed.inference_batches)
    training_batches = _parse_ints(parsed.training_batches)

    print(torch.cuda.get_device_name())
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print()

    for channels, depth, head_channels in itertools.product(channels_list, depth_list, head_list):
        args = _model_args(parsed, channels, depth, head_channels)
        macro_part = f"-m{parsed.macro_channels}" if parsed.macro_channels > 0 else ""
        model_name = f"{depth}d-{channels}c{macro_part}-h{head_channels}"
        params = sum(p.numel() for p in _make_wrapper(args).nnet.parameters())
        _clear_cuda()
        print(f"== {model_name}: {params:,} params ==")

        for bs in inference_batches:
            fn = (
                (lambda a, b: _probe_inference(a, b, amp=not parsed.no_amp_inference))
            )
            row = _run_probe(fn, args, bs)
            row["mode"] = "inference_amp" if not parsed.no_amp_inference else "inference_fp32"
            _print_row(model_name, row)
            if not row.get("ok"):
                break

        for bs in training_batches:
            row = _run_probe(_probe_training, args, bs)
            row["mode"] = "training_fp32"
            _print_row(model_name, row)
            if not row.get("ok"):
                break
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
