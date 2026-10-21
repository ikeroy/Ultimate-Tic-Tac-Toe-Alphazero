import io
from dataclasses import asdict

import torch
import zstandard as zstd

import alphazero
from neural_net import NNArgs, NNWrapper


def test_legacy_uttt_checkpoint_infers_nine_input_planes(tmp_path):
    args = NNArgs(
        num_channels=4,
        depth=2,
        kernel_size=3,
        dense_net=True,
        uttt_hierarchical=True,
        input_channels=9,
    )
    original = NNWrapper(alphazero.UltimateTicTacToeGS, args)
    legacy_args = asdict(args)
    legacy_args.pop("input_channels")
    legacy_args.pop("uttt_macro_transformer_layers")
    legacy_args.pop("uttt_macro_transformer_heads")
    legacy_args.pop("uttt_macro_transformer_ff_mult")
    legacy_args.pop("uttt_macro_transformer_dropout")

    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": original.nnet.state_dict(),
            "opt_state": original.optimizer.state_dict(),
            "args": legacy_args,
            "game": alphazero.UltimateTicTacToeGS,
            "version": "legacy-test",
        },
        buffer,
    )
    ckpt = tmp_path / "legacy-nine-plane.pt"
    ckpt.write_bytes(zstd.ZstdCompressor(level=1).compress(buffer.getvalue()))

    loaded = NNWrapper.load_checkpoint(
        alphazero.UltimateTicTacToeGS, str(tmp_path), ckpt.name
    )

    assert loaded._input_channels == 9
    canonical = torch.zeros(2, *alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE())
    value, policy = loaded.process(canonical)
    assert value.shape == (2, alphazero.UltimateTicTacToeGS.NUM_PLAYERS() + 1)
    assert policy.shape == (2, alphazero.UltimateTicTacToeGS.NUM_MOVES())


def test_checkpoint_input_adapter_rejects_missing_planes():
    args = NNArgs(
        num_channels=4,
        depth=2,
        kernel_size=3,
        dense_net=True,
        uttt_hierarchical=True,
        input_channels=9,
    )
    nn = NNWrapper(alphazero.UltimateTicTacToeGS, args)

    short_batch = torch.zeros(1, 8, 9, 9)
    try:
        nn.process(short_batch)
    except ValueError as exc:
        assert "expects 9 input planes" in str(exc)
    else:
        raise AssertionError("Expected missing input planes to raise ValueError")
