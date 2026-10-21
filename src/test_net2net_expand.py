import sys
from pathlib import Path

import torch

import alphazero
from neural_net import NNArgs, NNWrapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.net2net_expand import (
    _local_depth,
    _macro_depth,
    expanded_args,
    expand_wrapper,
)


def _small_hierarchical_args(**overrides):
    args = {
        "num_channels": 8,
        "depth": 4,
        "kernel_size": 3,
        "dense_net": True,
        "uttt_hierarchical": True,
        "head_channels": 12,
        "uttt_macro_transformer_layers": 1,
        "uttt_macro_transformer_heads": 2,
        "uttt_macro_transformer_ff_mult": 2.0,
        "uttt_macro_transformer_dropout": 0.0,
        "input_channels": alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE()[0],
    }
    args.update(overrides)
    return NNArgs(**args)


def test_net2net_expansion_preserves_hierarchical_outputs():
    torch.manual_seed(123)
    source = NNWrapper(alphazero.UltimateTicTacToeGS, _small_hierarchical_args())
    target_args = expanded_args(source.args, depth=10, head_channels=28)

    expanded = expand_wrapper(source, target_args)
    source.nnet.eval()
    expanded.nnet.eval()

    batch = torch.randn(
        5,
        source._input_channels,
        9,
        9,
        device=source.device,
    )
    with torch.no_grad():
        src_value, src_policy = source.nnet(batch)
        dst_value, dst_policy = expanded.nnet(batch.to(expanded.device))

    assert torch.allclose(src_value.cpu(), dst_value.cpu(), atol=1e-6)
    assert torch.allclose(src_policy.cpu(), dst_policy.cpu(), atol=1e-6)


def test_net2net_expansion_auto_expands_hidden_sizes_with_head_channels():
    source_args = _small_hierarchical_args(head_channels=12)
    target_args = expanded_args(source_args, depth=source_args.depth, head_channels=30)

    assert source_args.v_fc_hidden == 12 * 8
    assert target_args.v_fc_hidden == 30 * 8
    assert target_args.pi_fc_hidden == 30 * 8


def test_net2net_added_residual_blocks_are_identity_initialized():
    torch.manual_seed(123)
    source = NNWrapper(alphazero.UltimateTicTacToeGS, _small_hierarchical_args())
    target_args = expanded_args(source.args, depth=12, head_channels=source.args.head_channels)
    expanded = expand_wrapper(source, target_args)
    state = expanded.nnet.state_dict()

    for idx in range(_local_depth(source.args), _local_depth(target_args)):
        assert torch.count_nonzero(state[f"arch.local_blocks.{idx}.conv1.weight"]) == 0
        assert torch.count_nonzero(state[f"arch.local_blocks.{idx}.conv2.weight"]) == 0
        assert torch.all(state[f"arch.local_blocks.{idx}.bn1.weight"] == 1)
        assert torch.all(state[f"arch.local_blocks.{idx}.bn2.running_var"] == 1)

    for idx in range(_macro_depth(source.args), _macro_depth(target_args)):
        assert torch.count_nonzero(state[f"arch.macro_blocks.{idx}.conv1.weight"]) == 0
        assert torch.count_nonzero(state[f"arch.macro_blocks.{idx}.conv2.weight"]) == 0
        assert torch.all(state[f"arch.macro_blocks.{idx}.bn1.weight"] == 1)
        assert torch.all(state[f"arch.macro_blocks.{idx}.bn2.running_var"] == 1)


def test_net2net_widened_head_channels_are_duplicate_connected():
    torch.manual_seed(123)
    source = NNWrapper(alphazero.UltimateTicTacToeGS, _small_hierarchical_args())
    target_args = expanded_args(source.args, depth=source.args.depth, head_channels=28)
    expanded = expand_wrapper(source, target_args)

    src_state = source.nnet.state_dict()
    dst_state = expanded.nnet.state_dict()
    cells = 9 * 9

    # New head channel 12 is a duplicate of old channel 0, not a zero channel.
    assert torch.allclose(
        dst_state["arch.fuse.0.weight"][12].cpu(),
        src_state["arch.fuse.0.weight"][0].cpu(),
    )

    # Old channel 0 appears at new channels 0, 12, and 24. The downstream flat
    # policy columns are divided across the three copies so their sum preserves
    # the source logits while all copies remain trainable.
    expected = src_state["arch.pi_fc1.weight"][:, :cells].cpu() / 3
    for head_idx in (0, 12, 24):
        cols = slice(head_idx * cells, (head_idx + 1) * cells)
        assert torch.allclose(dst_state["arch.pi_fc1.weight"][:, cols].cpu(), expected)


def test_net2net_rejects_transformer_width_change():
    source = NNWrapper(alphazero.UltimateTicTacToeGS, _small_hierarchical_args())
    target_args = _small_hierarchical_args(num_channels=16, depth=4, head_channels=12)

    try:
        expand_wrapper(source, target_args)
    except ValueError as exc:
        assert "LayerNorm" in str(exc)
    else:
        raise AssertionError("Expected transformer width expansion to be rejected")
