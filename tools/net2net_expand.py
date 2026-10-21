#!/usr/bin/env python3
"""Function-preserving checkpoint expansion for UTTT hierarchical networks."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import alphazero  # noqa: E402
from neural_net import NNArgs, NNWrapper  # noqa: E402


def _local_depth(args: NNArgs) -> int:
    return max(1, args.depth // 2)


def _macro_depth(args: NNArgs) -> int:
    return max(1, args.depth - _local_depth(args))


def _is_norm_key(key: str) -> bool:
    parts = key.split(".")
    return (
        "bn1" in parts
        or "bn2" in parts
        or "bn_ds" in parts
        or "pi_bn" in parts
        or ".norm" in key
        or key.startswith("arch.local_stem.1.")
        or key.startswith("arch.fuse.1.")
    )


def _expanded_tensor(dst: torch.Tensor, src: torch.Tensor, key: str) -> torch.Tensor:
    if dst.shape == src.shape:
        return src.detach().clone().to(dtype=dst.dtype, device=dst.device)

    out = dst.clone()
    if key.endswith("running_var") or (_is_norm_key(key) and key.endswith(".weight")):
        out.fill_(1)
    elif out.is_floating_point():
        out.zero_()
    else:
        out.zero_()

    slices = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, src.shape))
    out[slices] = src.detach().to(dtype=out.dtype, device=out.device)[slices]
    return out


def _replication_map(old_size: int, new_size: int) -> list[int]:
    if new_size < old_size:
        raise ValueError("replication map cannot shrink a dimension")
    return [i if i < old_size else i % old_size for i in range(new_size)]


def _replication_counts(mapping: list[int], old_size: int) -> list[int]:
    return [max(1, mapping.count(i)) for i in range(old_size)]


def _duplicate_1d(dst: torch.Tensor, src: torch.Tensor, mapping: list[int]) -> torch.Tensor:
    out = dst.clone()
    src = src.detach().to(dtype=out.dtype, device=out.device)
    for new_idx, old_idx in enumerate(mapping):
        out[new_idx] = src[old_idx]
    return out


def _duplicate_conv_out(dst: torch.Tensor, src: torch.Tensor, out_map: list[int]) -> torch.Tensor:
    out = dst.clone()
    out.zero_()
    src = src.detach().to(dtype=out.dtype, device=out.device)
    for new_out, old_out in enumerate(out_map):
        out[new_out, : src.shape[1], ...] = src[old_out]
    return out


def _duplicate_conv_in(
    dst: torch.Tensor,
    src: torch.Tensor,
    in_map: list[int],
    in_counts: list[int],
) -> torch.Tensor:
    out = dst.clone()
    out.zero_()
    src = src.detach().to(dtype=out.dtype, device=out.device)
    for new_in, old_in in enumerate(in_map):
        out[: src.shape[0], new_in, ...] = src[:, old_in, ...] / in_counts[old_in]
    return out


def _duplicate_conv_in_out(
    dst: torch.Tensor,
    src: torch.Tensor,
    out_map: list[int],
    in_map: list[int],
    in_counts: list[int],
) -> torch.Tensor:
    out = dst.clone()
    out.zero_()
    src = src.detach().to(dtype=out.dtype, device=out.device)
    for new_out, old_out in enumerate(out_map):
        for new_in, old_in in enumerate(in_map):
            out[new_out, new_in, ...] = src[old_out, old_in, ...] / in_counts[old_in]
    return out


def _duplicate_linear(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    out_map: list[int] | None = None,
    in_map: list[int] | None = None,
    in_counts: list[int] | None = None,
) -> torch.Tensor:
    out = dst.clone()
    out.zero_()
    src = src.detach().to(dtype=out.dtype, device=out.device)
    out_rows = range(out.shape[0])
    in_cols = range(out.shape[1])
    for new_out in out_rows:
        old_out = out_map[new_out] if out_map is not None else new_out
        if old_out >= src.shape[0]:
            continue
        for new_in in in_cols:
            old_in = in_map[new_in] if in_map is not None else new_in
            if old_in >= src.shape[1]:
                continue
            scale = in_counts[old_in] if in_counts is not None else 1
            out[new_out, new_in] = src[old_out, old_in] / scale
    return out


def _duplicate_flat_policy_weight(
    dst: torch.Tensor,
    src: torch.Tensor,
    head_map: list[int],
    head_counts: list[int],
) -> torch.Tensor:
    out = dst.clone()
    out.zero_()
    src = src.detach().to(dtype=out.dtype, device=out.device)
    old_head = len(head_counts)
    old_area = src.shape[1] // old_head
    new_area = out.shape[1] // len(head_map)
    if old_area != new_area:
        raise ValueError("flat policy spatial area changed during head expansion")
    for new_head, old_head_idx in enumerate(head_map):
        dst_slice = slice(new_head * new_area, (new_head + 1) * new_area)
        src_slice = slice(old_head_idx * old_area, (old_head_idx + 1) * old_area)
        out[:, dst_slice] = src[:, src_slice] / head_counts[old_head_idx]
    return out


def _expand_key_net2wider(
    key: str,
    dst: torch.Tensor,
    src: torch.Tensor,
    src_args: NNArgs,
    dst_args: NNArgs,
) -> torch.Tensor | None:
    head_map = _replication_map(src_args.head_channels, dst_args.head_channels)
    head_counts = _replication_counts(head_map, src_args.head_channels)
    v_hidden_map = _replication_map(src_args.v_fc_hidden, dst_args.v_fc_hidden)
    v_hidden_counts = _replication_counts(v_hidden_map, src_args.v_fc_hidden)
    pi_hidden_map = _replication_map(src_args.pi_fc_hidden, dst_args.pi_fc_hidden)
    pi_hidden_counts = _replication_counts(pi_hidden_map, src_args.pi_fc_hidden)

    if key.startswith("arch.fuse.0.") and key.endswith(".weight"):
        return _duplicate_conv_out(dst, src, head_map)
    if key.startswith("arch.fuse.1.") and dst.ndim == 1:
        return _duplicate_1d(dst, src, head_map)
    if key == "arch.pi_conv.weight":
        return _duplicate_conv_in_out(dst, src, head_map, head_map, head_counts)
    if key.startswith("arch.pi_bn.") and dst.ndim == 1:
        return _duplicate_1d(dst, src, head_map)
    if key == "arch.pi_fc1.weight" and src_args.pi_fc_layers == 0:
        return _duplicate_flat_policy_weight(dst, src, head_map, head_counts)
    if key == "arch.pi_fc1.weight":
        old_cols = src.shape[1] // src_args.head_channels
        new_cols = dst.shape[1] // dst_args.head_channels
        if old_cols == new_cols:
            flat_map = []
            flat_counts = []
            for new_head, old_head in enumerate(head_map):
                flat_map.extend(range(old_head * old_cols, (old_head + 1) * old_cols))
                flat_counts.extend([head_counts[old_head]] * old_cols)
            return _duplicate_linear(
                dst,
                src,
                out_map=pi_hidden_map,
                in_map=flat_map,
                in_counts=flat_counts,
            )
    if key == "arch.pi_fc1.bias" and src_args.pi_fc_layers > 0:
        return _duplicate_1d(dst, src, pi_hidden_map)
    if key == "arch.pi_fc_out.weight":
        return _duplicate_linear(dst, src, in_map=pi_hidden_map, in_counts=pi_hidden_counts)
    if key.startswith("arch.pi_fc_extra.") and key.endswith(".weight"):
        return _duplicate_linear(
            dst,
            src,
            out_map=pi_hidden_map,
            in_map=pi_hidden_map,
            in_counts=pi_hidden_counts,
        )
    if key.startswith("arch.pi_fc_extra.") and key.endswith(".bias"):
        return _duplicate_1d(dst, src, pi_hidden_map)
    if key == "arch.pi_cell_conv.weight":
        return _duplicate_conv_in(dst, src, head_map, head_counts)
    if key.startswith("arch.pi_extra_convs.") and key.endswith(".weight") and dst.ndim == 4:
        return _duplicate_conv_in_out(dst, src, head_map, head_map, head_counts)
    if key.startswith("arch.pi_extra_convs.") and dst.ndim == 1:
        return _duplicate_1d(dst, src, head_map)

    if key == "arch.v_fc1.weight":
        return _duplicate_linear(dst, src, out_map=v_hidden_map)
    if key == "arch.v_fc1.bias":
        return _duplicate_1d(dst, src, v_hidden_map)
    if key == "arch.v_fc2.weight":
        return _duplicate_linear(dst, src, in_map=v_hidden_map, in_counts=v_hidden_counts)
    if key.startswith("arch.v_fc_extra.") and key.endswith(".weight"):
        return _duplicate_linear(
            dst,
            src,
            out_map=v_hidden_map,
            in_map=v_hidden_map,
            in_counts=v_hidden_counts,
        )
    if key.startswith("arch.v_fc_extra.") and key.endswith(".bias"):
        return _duplicate_1d(dst, src, v_hidden_map)

    return None


def _init_identity_residual_block(state: dict[str, torch.Tensor], prefix: str) -> None:
    for key, tensor in list(state.items()):
        if not key.startswith(prefix + "."):
            continue
        if key.endswith(".weight") and tensor.ndim >= 2:
            tensor.zero_()
        elif key.endswith(".weight") and _is_norm_key(key):
            tensor.fill_(1)
        elif key.endswith("running_var"):
            tensor.fill_(1)
        elif tensor.is_floating_point():
            tensor.zero_()
        else:
            tensor.zero_()


def _validate_supported_expansion(src_args: NNArgs, dst_args: NNArgs) -> None:
    if not src_args.uttt_hierarchical or not dst_args.uttt_hierarchical:
        raise ValueError("net2net expansion currently supports only uttt_hierarchical networks")
    if dst_args.depth < src_args.depth:
        raise ValueError("target depth must be greater than or equal to source depth")
    if dst_args.head_channels < src_args.head_channels:
        raise ValueError("target head_channels must be greater than or equal to source head_channels")
    if dst_args.num_channels != src_args.num_channels:
        raise ValueError(
            "exact Net2Net widening of num_channels is disabled for the current "
            "UTTT transformer trunk because LayerNorm would change the function"
        )

    same_fields = [
        "kernel_size",
        "dense_net",
        "input_channels",
        "uttt_policy_head",
        "uttt_destination_context",
        "uttt_macro_transformer_layers",
        "uttt_macro_transformer_heads",
        "uttt_macro_transformer_ff_mult",
        "uttt_macro_transformer_dropout",
        "trunk_norm",
        "trunk_act",
        "v_head_convs",
        "pi_head_convs",
        "v_fc_layers",
        "pi_fc_layers",
    ]
    for field in same_fields:
        if getattr(dst_args, field) != getattr(src_args, field):
            raise ValueError(f"target {field} must match source {field}")

    if _local_depth(dst_args) < _local_depth(src_args):
        raise ValueError("target local residual depth must not shrink")
    if _macro_depth(dst_args) < _macro_depth(src_args):
        raise ValueError("target macro residual depth must not shrink")
    if dst_args.v_fc_hidden < src_args.v_fc_hidden:
        raise ValueError("target v_fc_hidden must be greater than or equal to source")
    if dst_args.pi_fc_hidden < src_args.pi_fc_hidden:
        raise ValueError("target pi_fc_hidden must be greater than or equal to source")


def expand_state_dict(
    src_state: dict[str, torch.Tensor],
    dst_state: dict[str, torch.Tensor],
    src_args: NNArgs,
    dst_args: NNArgs,
) -> dict[str, torch.Tensor]:
    """Return a destination state dict initialized from a smaller source model.

    Existing channels/blocks are copied exactly. Added head channels and FC
    hidden units are zero-connected. Added residual blocks are initialized as
    identity blocks by zeroing their convolutions.
    """
    _validate_supported_expansion(src_args, dst_args)

    expanded = {key: value.detach().clone() for key, value in dst_state.items()}
    for key, src_value in src_state.items():
        if key not in expanded:
            continue
        net2wider_value = _expand_key_net2wider(
            key,
            expanded[key],
            src_value,
            src_args,
            dst_args,
        )
        expanded[key] = (
            net2wider_value
            if net2wider_value is not None
            else _expanded_tensor(expanded[key], src_value, key)
        )

    for idx in range(_local_depth(src_args), _local_depth(dst_args)):
        _init_identity_residual_block(expanded, f"arch.local_blocks.{idx}")
    for idx in range(_macro_depth(src_args), _macro_depth(dst_args)):
        _init_identity_residual_block(expanded, f"arch.macro_blocks.{idx}")

    return expanded


def expanded_args(
    src_args: NNArgs,
    *,
    depth: int | None = None,
    head_channels: int | None = None,
    lr: float | None = None,
    v_fc_hidden: int | None = None,
    pi_fc_hidden: int | None = None,
) -> NNArgs:
    args_dict = asdict(src_args)
    old_head = int(src_args.head_channels)
    new_head = int(head_channels if head_channels is not None else src_args.head_channels)
    args_dict["depth"] = int(depth if depth is not None else src_args.depth)
    args_dict["head_channels"] = new_head
    if lr is not None:
        args_dict["lr"] = float(lr)

    old_v_auto = int(src_args.v_fc_hidden) == old_head * 8
    old_pi_auto = int(src_args.pi_fc_hidden) == old_head * 8
    args_dict["v_fc_hidden"] = (
        int(v_fc_hidden)
        if v_fc_hidden is not None
        else (new_head * 8 if old_v_auto else int(src_args.v_fc_hidden))
    )
    args_dict["pi_fc_hidden"] = (
        int(pi_fc_hidden)
        if pi_fc_hidden is not None
        else (new_head * 8 if old_pi_auto else int(src_args.pi_fc_hidden))
    )
    return NNArgs(**args_dict)


def expand_wrapper(source: NNWrapper, target_args: NNArgs) -> NNWrapper:
    target = NNWrapper(source.game, target_args)
    expanded = expand_state_dict(
        source.nnet.state_dict(),
        target.nnet.state_dict(),
        source.args,
        target_args,
    )
    target.nnet.load_state_dict(expanded, strict=True)
    return target


def _load_checkpoint(path: Path) -> NNWrapper:
    return NNWrapper.load_checkpoint(alphazero.UltimateTicTacToeGS, str(path.parent), path.name)


def _verify(source: NNWrapper, target: NNWrapper, samples: int) -> tuple[float, float]:
    if samples <= 0:
        return 0.0, 0.0
    source.nnet.eval()
    target.nnet.eval()
    shape = alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE()
    channels = int(source.args.input_channels or shape[0])
    batch = torch.randn(samples, channels, shape[1], shape[2], device=source.device)
    with torch.no_grad():
        src_v, src_pi = source.nnet(batch)
        dst_v, dst_pi = target.nnet(batch.to(target.device))
    return (
        float((src_v.detach().cpu() - dst_v.detach().cpu()).abs().max().item()),
        float((src_pi.detach().cpu() - dst_pi.detach().cpu()).abs().max().item()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="source checkpoint .pt")
    parser.add_argument("--output", required=True, type=Path, help="expanded checkpoint .pt")
    parser.add_argument("--depth", type=int, help="target depth")
    parser.add_argument("--head-channels", type=int, help="target head channel count")
    parser.add_argument("--lr", type=float, help="target learning rate")
    parser.add_argument("--v-fc-hidden", type=int, help="target value hidden size")
    parser.add_argument("--pi-fc-hidden", type=int, help="target policy hidden size")
    parser.add_argument("--verify-samples", type=int, default=8)
    args = parser.parse_args()

    source = _load_checkpoint(args.source)
    target_args = expanded_args(
        source.args,
        depth=args.depth,
        head_channels=args.head_channels,
        lr=args.lr,
        v_fc_hidden=args.v_fc_hidden,
        pi_fc_hidden=args.pi_fc_hidden,
    )
    target = expand_wrapper(source, target_args)
    value_diff, policy_diff = _verify(source, target, args.verify_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target.save_checkpoint(str(args.output.parent), args.output.name)
    print(
        "expanded checkpoint saved",
        f"source={args.source}",
        f"output={args.output}",
        f"depth={target_args.depth}",
        f"channels={target_args.num_channels}",
        f"head_channels={target_args.head_channels}",
        f"v_fc_hidden={target_args.v_fc_hidden}",
        f"max_value_diff={value_diff:.3g}",
        f"max_policy_diff={policy_diff:.3g}",
        sep="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
