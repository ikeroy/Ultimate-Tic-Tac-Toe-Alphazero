#!/usr/bin/env python3
"""Export the strongest UTTT checkpoint to browser-friendly ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import alphazero  # noqa: E402
import neural_net  # noqa: E402


class ProbabilityWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, canonical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value_log_probs, policy_log_probs = self.model(canonical)
        return torch.exp(value_log_probs), torch.exp(policy_log_probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "data/uttt/nineplane-hierarchy-6d24c-gumbel512-m32-from0-50/"
            "checkpoint/0300-nineplane-hierarchy-6d24c-gumbel512-m32-from0-50.pt"
        ),
    )
    parser.add_argument("--out", default="export/uttt-model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    wrapper = neural_net.NNWrapper.load_checkpoint(
        alphazero.UltimateTicTacToeGS,
        str(checkpoint.parent),
        checkpoint.name,
    )
    model = ProbabilityWrapper(wrapper.nnet).cpu().eval()
    game_shape = alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE()
    input_channels = int(getattr(wrapper, "_input_channels", game_shape[0]))
    dummy = torch.zeros(
        1,
        input_channels,
        int(game_shape[1]),
        int(game_shape[2]),
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch_value, torch_policy = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["canonical"],
        output_names=["value_probs", "policy_probs"],
        dynamic_axes={
            "canonical": {0: "batch"},
            "value_probs": {0: "batch"},
            "policy_probs": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    if args.validate:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(out))
        onnx.checker.check_model(onnx_model)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        ort_value, ort_policy = sess.run(None, {"canonical": dummy.numpy()})
        value_delta = float(np.max(np.abs(ort_value - torch_value.numpy())))
        policy_delta = float(np.max(np.abs(ort_policy - torch_policy.numpy())))
        print(f"validated {out}")
        print(f"value max_abs_delta={value_delta:.8g}")
        print(f"policy max_abs_delta={policy_delta:.8g}")
    else:
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
