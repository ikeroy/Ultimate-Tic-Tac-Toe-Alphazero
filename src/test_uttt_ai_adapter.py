import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alphazero
from game_runner import (
    _PROJECT_ROOT,
    _ensure_project_root_importable,
    _raw_policy_action,
    play_raw_policy_match,
)
from tools.uttt_ai_adapter import (
    O,
    UNCONSTRAINED,
    X,
    UtttAiMCTS,
    UtttAiState,
    UtttAiPolicyValueNet,
    ZN,
    _ort_providers,
    canonical_batch_to_model_input,
    default_model_path,
)


class _PolicyAgent:
    def __init__(self, policy):
        self.policy = np.asarray(policy, dtype=np.float32)

    def process(self, canonical_batch):
        import torch

        batch = canonical_batch.shape[0]
        value = torch.zeros((batch, 3), dtype=torch.float32)
        policy = torch.tensor(np.tile(self.policy, (batch, 1)), dtype=torch.float32)
        return value, policy


class _OneMoveGame:
    @staticmethod
    def NUM_PLAYERS():
        return 2

    def __init__(self):
        self.winner = None

    def current_player(self):
        return 0

    def valid_moves(self):
        return [1, 1]

    def canonicalized(self):
        return np.zeros((1, 1, 2), dtype=np.float32)

    def play_move(self, action):
        # Action 0 awards seat 0, action 1 awards seat 1.
        self.winner = int(action)

    def scores(self):
        if self.winner is None:
            return None
        out = [0.0, 0.0, 0.0]
        out[self.winner] = 1.0
        return out


class _OpeningGame:
    @staticmethod
    def NUM_PLAYERS():
        return 2

    def __init__(self):
        self.moves = []

    def current_player(self):
        return len(self.moves) % 2

    def valid_moves(self):
        return [1, 1, 1]

    def canonicalized(self):
        return np.zeros((1, 1, 3), dtype=np.float32)

    def play_move(self, action):
        self.moves.append(int(action))

    def scores(self):
        if len(self.moves) < 3:
            return None
        out = [0.0, 0.0, 0.0]
        out[0 if self.moves[-1] == 0 else 1] = 1.0
        return out


def test_initial_state_matches_uttt_ai_query_defaults():
    state = UtttAiState.initial()

    assert state.next_symbol == X
    assert state.constraint == UNCONSTRAINED
    assert state.result == 0
    assert state.legal_indexes() == list(range(81))
    assert state.query_state() == "0" * 90 + "190"


def test_move_constrains_next_player_to_cell_index_board():
    state = UtttAiState.initial().execute(40)

    assert state.state[40] == X
    assert state.next_symbol == O
    assert state.constraint == 4
    assert state.legal_indexes() == [36, 37, 38, 39, 41, 42, 43, 44]


def test_completed_target_board_makes_next_state_unconstrained():
    state = UtttAiState.initial()
    raw = state.state.copy()
    raw[81 + 4] = X
    state = UtttAiState(raw).execute(4)

    assert state.constraint == UNCONSTRAINED


def test_tensor_uses_uttt_ai_four_plane_encoding_and_layout_mapping():
    state = UtttAiState.initial().execute(40)
    tensor = state.encode_tensor().reshape(4, 81)

    assert tensor[1, ZN[40]] == 1.0
    assert np.all(tensor[2] == -1.0)
    assert tensor[3, ZN[36]] == 1.0
    assert tensor[3, ZN[40]] == 0.0
    assert int(tensor[3].sum()) == 8


def test_project_canonical_initial_state_matches_uttt_ai_input_encoding():
    game = alphazero.UltimateTicTacToeGS()
    canonical = np.asarray(game.canonicalized())

    converted = canonical_batch_to_model_input(canonical)

    assert converted.shape == (1, 4, 9, 9)
    assert np.array_equal(converted, UtttAiState.initial().encode_tensor())


def test_project_canonical_after_move_matches_uttt_ai_input_encoding():
    game = alphazero.UltimateTicTacToeGS()
    game.play_move(40)
    canonical = np.asarray(game.canonicalized())

    converted = canonical_batch_to_model_input(canonical)
    expected = UtttAiState.initial().execute(40).encode_tensor()

    assert np.array_equal(converted, expected)


def test_ort_provider_selection_can_require_cuda(monkeypatch):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    monkeypatch.delenv("UTTT_AI_ORT_PROVIDERS", raising=False)

    assert _ort_providers(require_cuda=True)[0] == "CUDAExecutionProvider"


def test_ort_provider_selection_fails_fast_when_cuda_required(monkeypatch):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    monkeypatch.delenv("UTTT_AI_ORT_PROVIDERS", raising=False)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        _ort_providers(require_cuda=True)


def test_training_hook_makes_tools_package_importable(monkeypatch):
    root = str(_PROJECT_ROOT)
    stripped = [entry for entry in sys.path if entry != root]
    monkeypatch.setattr(sys, "path", stripped)

    _ensure_project_root_importable()

    assert sys.path[0] == root
    __import__("tools.uttt_ai_adapter")


def test_cached_model_process_returns_nnwrapper_compatible_outputs():
    model_path = default_model_path()
    if not model_path.exists():
        pytest.skip(f"cached uttt.ai ONNX model not found: {model_path}")

    torch = pytest.importorskip("torch")
    game = alphazero.UltimateTicTacToeGS()
    canonical = torch.from_numpy(np.asarray(game.canonicalized()).copy()).unsqueeze(0)

    value, policy = UtttAiPolicyValueNet(model_path).process(canonical)

    assert tuple(value.shape) == (1, 3)
    assert tuple(policy.shape) == (1, 81)
    assert torch.allclose(value.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(policy.sum(dim=1), torch.ones(1), atol=1e-6)
    assert int(policy.argmax(dim=1).item()) == 40


def test_cached_model_mcts_one_simulation_matches_raw_opening_policy():
    model_path = default_model_path()
    if not model_path.exists():
        pytest.skip(f"cached uttt.ai ONNX model not found: {model_path}")

    net = UtttAiPolicyValueNet(model_path)
    root = UtttAiMCTS(net, seed=0).search(UtttAiState.initial(), simulations=1)
    policy = UtttAiMCTS.display_policy(root)

    assert root.visit_count == 1
    assert len(root.children) == 81
    assert max(policy, key=policy.get) == 40
    assert sum(policy.values()) == pytest.approx(1.0)


def test_raw_policy_action_masks_illegal_moves():
    rng = np.random.default_rng(0)

    action = _raw_policy_action([0.9, 0.1], [0, 1], temperature=0.0, rng=rng)

    assert action == 1


def test_raw_policy_match_alternates_seats_without_mcts():
    first_action_agent = _PolicyAgent([1.0, 0.0])
    second_action_agent = _PolicyAgent([1.0, 0.0])

    win_rate, draw_rate, avg_len = play_raw_policy_match(
        _OneMoveGame,
        [first_action_agent, second_action_agent],
        games=2,
        temperature=0.0,
        seed=0,
    )

    assert win_rate == pytest.approx(0.5)
    assert draw_rate == 0.0
    assert avg_len == 1.0


def test_raw_policy_match_applies_random_opening_before_policy_play():
    choose_zero = _PolicyAgent([1.0, 0.0, 0.0])
    other = _PolicyAgent([1.0, 0.0, 0.0])

    win_rate, draw_rate, avg_len = play_raw_policy_match(
        _OpeningGame,
        [choose_zero, other],
        games=2,
        temperature=0.0,
        seed=0,
        opening_plies=2,
    )

    assert win_rate == pytest.approx(0.5)
    assert draw_rate == 0.0
    assert avg_len == 3.0


def test_random_opening_seed_controls_eval_suite(monkeypatch):
    openings = []

    original_action = _OpeningGame.play_move

    def record_opening(self, action):
        if len(self.moves) < 2:
            openings.append(int(action))
        original_action(self, action)

    monkeypatch.setattr(_OpeningGame, "play_move", record_opening)
    agents = [_PolicyAgent([1.0, 0.0, 0.0]), _PolicyAgent([1.0, 0.0, 0.0])]

    play_raw_policy_match(
        _OpeningGame, agents, games=4, temperature=0.0, seed=7, opening_plies=2
    )
    first = list(openings)
    openings.clear()

    play_raw_policy_match(
        _OpeningGame, agents, games=4, temperature=0.0, seed=7, opening_plies=2
    )
    same = list(openings)
    openings.clear()

    play_raw_policy_match(
        _OpeningGame, agents, games=4, temperature=0.0, seed=8, opening_plies=2
    )
    different = list(openings)

    assert same == first
    assert different != first
