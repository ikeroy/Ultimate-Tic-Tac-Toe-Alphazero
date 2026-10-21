"""Small local adapter for the client-side model used by https://www.uttt.ai/.

The website stores UTTT state as 93 bytes:
  0..80   board cells, board-major, 0 empty / 1 X / 2 O
  81..89  local board results, 0 unfinished / 1 X / 2 O / 3 draw
  90      next symbol, 1 X / 2 O
  91      constraint board, 0..8 or 9 for unconstrained
  92      global result, 0 unfinished / 1 X / 2 O / 3 draw

The ONNX model takes [1, 4, 9, 9]:
  0 current-player stones
  1 opponent stones
  2 side-to-move plane, +1 for X and -1 for O
  3 legal moves
"""

from __future__ import annotations

import random
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np


EMPTY = 0
X = 1
O = 2
DRAW = 3
UNCONSTRAINED = 9
STATE_SIZE = 93

MODEL_URL = "https://www.uttt.ai/policy_value_net_stage2.onnx"

# Map website board-major move ids to row-major positions in the model output.
ZN = np.array(
    [
        0,
        1,
        2,
        9,
        10,
        11,
        18,
        19,
        20,
        3,
        4,
        5,
        12,
        13,
        14,
        21,
        22,
        23,
        6,
        7,
        8,
        15,
        16,
        17,
        24,
        25,
        26,
        27,
        28,
        29,
        36,
        37,
        38,
        45,
        46,
        47,
        30,
        31,
        32,
        39,
        40,
        41,
        48,
        49,
        50,
        33,
        34,
        35,
        42,
        43,
        44,
        51,
        52,
        53,
        54,
        55,
        56,
        63,
        64,
        65,
        72,
        73,
        74,
        57,
        58,
        59,
        66,
        67,
        68,
        75,
        76,
        77,
        60,
        61,
        62,
        69,
        70,
        71,
        78,
        79,
        80,
    ],
    dtype=np.int64,
)
INV_ZN = np.empty_like(ZN)
INV_ZN[ZN] = np.arange(ZN.size, dtype=np.int64)

WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def other(symbol: int) -> int:
    if symbol == X:
        return O
    if symbol == O:
        return X
    raise ValueError(f"bad symbol: {symbol}")


def _winner(cells: Iterable[int]) -> int:
    values = list(cells)
    for a, b, c in WIN_LINES:
        if values[a] and values[a] == values[b] == values[c]:
            return values[a]
    return EMPTY


def canonical_batch_to_model_input(canonical: np.ndarray) -> np.ndarray:
    """Convert this project's UTTT canonical tensor to uttt.ai ONNX input.

    The local game uses relative planes: current-player stones, opponent
    stones, legal moves, etc. uttt.ai's model uses the same current/opponent
    convention, but only four planes and an explicit side-to-move plane.
    """
    arr = np.asarray(canonical, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[1] < 4 or arr.shape[2:] != (9, 9):
        raise ValueError(
            "expected canonical tensor with shape [B, C>=4, 9, 9] "
            f"or [C>=4, 9, 9], got {arr.shape}"
        )

    out = np.zeros((arr.shape[0], 4, 9, 9), dtype=np.float32)
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 1]
    stone_counts = np.rint((arr[:, 0] + arr[:, 1]).sum(axis=(1, 2))).astype(np.int64)
    x_to_move = (stone_counts % 2) == 0
    out[:, 2] = np.where(x_to_move[:, None, None], 1.0, -1.0)
    out[:, 3] = arr[:, 3]
    return out


@dataclass(frozen=True)
class UtttAiState:
    state: np.ndarray

    @classmethod
    def initial(cls) -> "UtttAiState":
        state = np.zeros(STATE_SIZE, dtype=np.uint8)
        state[90] = X
        state[91] = UNCONSTRAINED
        return cls(state)

    @classmethod
    def from_query_state(cls, query_state: str) -> "UtttAiState":
        if len(query_state) != STATE_SIZE or not query_state.isdigit():
            raise ValueError("uttt.ai query state must be 93 decimal digits")
        return cls(np.fromiter((int(ch) for ch in query_state), dtype=np.uint8))

    @property
    def next_symbol(self) -> int:
        return int(self.state[90])

    @property
    def constraint(self) -> int:
        return int(self.state[91])

    @property
    def result(self) -> int:
        return int(self.state[92])

    def legal_indexes(self) -> list[int]:
        if self.result:
            return []
        if self.constraint == UNCONSTRAINED:
            indexes: list[int] = []
            for board in range(9):
                if self.state[81 + board] == EMPTY:
                    indexes.extend(self._empty_indexes(board))
            return indexes
        return self._empty_indexes(self.constraint)

    def execute(self, index: int) -> "UtttAiState":
        if index not in self.legal_indexes():
            raise ValueError(f"illegal move: {index}")
        new_state = self.state.copy()
        symbol = self.next_symbol
        new_state[index] = symbol

        board = index // 9
        local_cells = new_state[board * 9 : board * 9 + 9]
        local_winner = _winner(local_cells)
        if local_winner:
            new_state[81 + board] = local_winner
        elif np.all(local_cells != EMPTY):
            new_state[81 + board] = DRAW

        macro_result = _winner(new_state[81:90])
        if macro_result:
            new_state[92] = macro_result
        elif np.all(new_state[81:90] != EMPTY):
            new_state[92] = DRAW

        new_state[90] = other(symbol)
        target_board = index % 9
        new_state[91] = (
            UNCONSTRAINED if new_state[81 + target_board] else target_board
        )
        return UtttAiState(new_state)

    def query_state(self) -> str:
        return "".join(str(int(v)) for v in self.state)

    def encode_tensor(self) -> np.ndarray:
        data = np.zeros(4 * 81, dtype=np.float32)
        data[162:243] = 1.0 if self.next_symbol == X else -1.0
        x_plane = ZN if self.next_symbol == X else ZN + 81
        o_plane = ZN + 81 if self.next_symbol == X else ZN
        for index, value in enumerate(self.state[:81]):
            if value == X:
                data[x_plane[index]] = 1.0
            elif value == O:
                data[o_plane[index]] = 1.0
        for index in self.legal_indexes():
            data[ZN[index] + 243] = 1.0
        return data.reshape(1, 4, 9, 9)

    def _empty_indexes(self, board: int) -> list[int]:
        start = board * 9
        return [idx for idx in range(start, start + 9) if self.state[idx] == EMPTY]


def _ort_providers(require_cuda: bool = False) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    requested = os.environ.get("UTTT_AI_ORT_PROVIDERS")
    if requested:
        providers = [p.strip() for p in requested.split(",") if p.strip()]
    else:
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
    if require_cuda and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "UTTT.ai CUDA inference requested, but ONNX Runtime does not expose "
            f"CUDAExecutionProvider. Available providers: {available}"
        )
    return providers


class UtttAiPolicyValueNet:
    def __init__(self, model_path: str | Path, require_cuda: bool = False):
        import onnxruntime as ort

        providers = _ort_providers(require_cuda=require_cuda)
        self.session = ort.InferenceSession(
            str(model_path), providers=providers
        )
        self.providers = self.session.get_providers()
        if require_cuda and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(
                "UTTT.ai CUDA inference requested, but the ONNX Runtime session "
                f"fell back to providers: {self.providers}"
            )

    def enable_inference_optimizations(self, *args, **kwargs):
        """GameRunner compatibility hook; ONNX Runtime handles this internally."""

    def warmup_graphs(self, *args, **kwargs):
        """GameRunner compatibility hook; this CPU ONNX model has no CUDA graphs."""

    def evaluate(self, state: UtttAiState) -> tuple[float, dict[int, float]]:
        policy_logits, state_value = self.session.run(
            None, {"input": state.encode_tensor()}
        )
        logits = policy_logits.reshape(-1)
        legal = state.legal_indexes()
        legal_logits = np.array([logits[ZN[index]] for index in legal], dtype=np.float64)
        legal_logits -= np.max(legal_logits)
        probs = np.exp(legal_logits)
        probs /= np.sum(probs)
        return float(state_value[0]), {
            index: float(prob) for index, prob in zip(legal, probs)
        }

    def top_moves(self, state: UtttAiState, limit: int = 10) -> list[tuple[int, float]]:
        _, policy = self.evaluate(state)
        return sorted(policy.items(), key=lambda item: item[1], reverse=True)[:limit]

    def process(self, canonical_batch):
        """NNWrapper-compatible inference for GameRunner/PlayManager.

        Returns torch tensors:
          value: [B, 3] as approximate [win, loss, draw] probabilities
          policy: [B, 81] in this project's action order

        uttt.ai's value head is scalar, so draw probability is not recoverable.
        We map scalar value in [-1, 1] to win/loss mass and leave draw at 0.
        This is mainly intended for raw-policy or low-visit fixed-opponent eval.
        """
        import torch

        if hasattr(canonical_batch, "detach"):
            canonical_np = canonical_batch.detach().cpu().numpy()
        else:
            canonical_np = np.asarray(canonical_batch)
        model_inputs = canonical_batch_to_model_input(canonical_np)
        values = []
        policies = []
        for model_input in model_inputs:
            policy_logits, state_value = self.session.run(
                None, {"input": model_input[None, ...].astype(np.float32)}
            )
            logits = policy_logits.reshape(-1).astype(np.float64)
            legal = np.flatnonzero(model_input[3].reshape(-1) > 0.5)
            legal_actions = INV_ZN[legal].astype(np.int64)
            action_logits = np.zeros(81, dtype=np.float64)
            if legal_actions.size:
                legal_logits = np.array(
                    [logits[ZN[action]] for action in legal_actions],
                    dtype=np.float64,
                )
                legal_logits -= np.max(legal_logits)
                probs = np.exp(legal_logits)
                probs /= np.sum(probs)
                for action, prob in zip(legal_actions, probs):
                    action_logits[action] = prob
            else:
                action_logits.fill(1.0 / 81.0)

            scalar = float(np.clip(state_value.reshape(-1)[0], -1.0, 1.0))
            win = (scalar + 1.0) * 0.5
            values.append([win, 1.0 - win, 0.0])
            policies.append(action_logits)

        return (
            torch.tensor(np.asarray(values), dtype=torch.float32),
            torch.tensor(np.asarray(policies), dtype=torch.float32),
        )


class UtttAiNode:
    def __init__(self, state: UtttAiState, action: int | None = None):
        self.state = state
        self.action = action
        self.action_probability: float | None = None
        self.children: list[UtttAiNode] = []
        self.visit_count = 0
        self.state_value: float | None = None
        self.state_value_sum = 0.0
        self.state_value_mean = 0.0

    def is_leaf(self) -> bool:
        return not self.children

    def is_terminal(self) -> bool:
        return self.state.result != EMPTY

    def expand(self) -> None:
        if not self.is_leaf() or self.is_terminal():
            return
        legal = self.state.legal_indexes()
        if not legal:
            raise RuntimeError("expanding node with no legal actions")
        self.children = [
            UtttAiNode(self.state.execute(action), action=action) for action in legal
        ]


class UtttAiMCTS:
    """Port of uttt.ai's browser-side Neural MCTS loop.

    The JavaScript bundle uses PUCT-like selection:
      -child_value + c * max(0.01, prior) * sqrt(parent_visits) / (child_visits + 1)

    Values are scalar and relative to the player to move at the evaluated node.
    """

    def __init__(
        self,
        net: UtttAiPolicyValueNet,
        exploration_strength: float = 2.0,
        seed: int | None = None,
    ):
        self.net = net
        self.exploration_strength = exploration_strength
        self.rng = random.Random(seed)

    def search(self, state: UtttAiState, simulations: int) -> UtttAiNode:
        root = UtttAiNode(state)
        for _ in range(simulations):
            self.simulate(root)
        return root

    def simulate(self, root: UtttAiNode) -> None:
        path = self._select_leaf_path(root)
        leaf = path[-1]
        leaf.expand()
        self._evaluate(leaf)
        self._backprop(path, leaf.state_value if leaf.state_value is not None else 0.0)

    def _select_leaf_path(self, root: UtttAiNode) -> list[UtttAiNode]:
        path = []
        node = root
        while not node.is_leaf():
            path.append(node)
            scores = [
                -child.state_value_mean
                + self.exploration_strength
                * max(0.01, child.action_probability or 0.0)
                * np.sqrt(node.visit_count)
                / (child.visit_count + 1)
                for child in node.children
            ]
            best = max(scores)
            best_indexes = [i for i, score in enumerate(scores) if score >= best]
            node = node.children[self.rng.choice(best_indexes)]
        path.append(node)
        return path

    def _evaluate(self, node: UtttAiNode) -> None:
        if node.is_terminal():
            node.state_value = 0.0 if node.state.result == DRAW else -1.0
            return
        value, policy = self.net.evaluate(node.state)
        for child in node.children:
            child.action_probability = policy[child.action]
        node.state_value = value

    @staticmethod
    def _backprop(path: list[UtttAiNode], value: float) -> None:
        sign = 1.0
        for node in reversed(path):
            node.visit_count += 1
            node.state_value_sum += sign * value
            node.state_value_mean = node.state_value_sum / node.visit_count
            sign *= -1.0

    @staticmethod
    def display_policy(root: UtttAiNode) -> dict[int, float]:
        if not root.children or root.visit_count <= 0:
            return {}
        smoothing = max(0.0, -0.1 * root.visit_count + 100.0)
        denominator = root.visit_count + smoothing - 1.0
        if denominator <= 0:
            return {}
        return {
            child.action: (child.visit_count + (child.action_probability or 0.0) * smoothing)
            / denominator
            for child in root.children
            if child.action is not None
        }


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "uttt_ai" / MODEL_URL.rsplit("/", 1)[-1]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(default_model_path()))
    parser.add_argument("--state", help="93-digit uttt.ai /init?state= value")
    parser.add_argument("--moves", nargs="*", type=int, default=[])
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=0,
        help="Run local port of uttt.ai browser MCTS for this many simulations",
    )
    parser.add_argument("--exploration-strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    state = (
        UtttAiState.from_query_state(args.state)
        if args.state
        else UtttAiState.initial()
    )
    for move in args.moves:
        state = state.execute(move)
    net = UtttAiPolicyValueNet(args.model)
    if args.mcts_sims > 0:
        mcts = UtttAiMCTS(
            net,
            exploration_strength=args.exploration_strength,
            seed=args.seed,
        )
        root = mcts.search(state, args.mcts_sims)
        display_policy = mcts.display_policy(root)
        top = sorted(display_policy.items(), key=lambda item: item[1], reverse=True)[
            : args.top
        ]
        visits = {child.action: child.visit_count for child in root.children}
        priors = {
            child.action: child.action_probability
            for child in root.children
            if child.action_probability is not None
        }
        print(
            json.dumps(
                {
                    "simulations": root.visit_count,
                    "value": root.state_value_mean,
                    "top": [
                        {
                            "move": move,
                            "probability": prob,
                            "visits": visits.get(move, 0),
                            "prior": priors.get(move),
                        }
                        for move, prob in top
                    ],
                },
                indent=2,
            )
        )
    else:
        value, policy = net.evaluate(state)
        top = sorted(policy.items(), key=lambda item: item[1], reverse=True)[: args.top]
        print(json.dumps({"value": value, "top": top}, indent=2))
