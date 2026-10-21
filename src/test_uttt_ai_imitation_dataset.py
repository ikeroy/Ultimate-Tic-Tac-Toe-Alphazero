import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_runner import glob_file_triples, load_compressed
from tools import generate_uttt_ai_imitation_dataset as gen


def test_policy_dict_to_vector_normalizes_soft_distribution():
    policy = gen.policy_dict_to_vector({40: 2.0, 41: 1.0, 999: 5.0, 7: -1.0})

    assert policy.shape == (81,)
    assert policy.sum() == pytest.approx(1.0)
    assert policy[40] == pytest.approx(2.0 / 3.0)
    assert policy[41] == pytest.approx(1.0 / 3.0)
    assert policy[7] == 0.0


def test_policy_dict_to_vector_can_fallback_to_one_hot():
    policy = gen.policy_dict_to_vector({}, fallback_action=12)

    assert policy.sum() == pytest.approx(1.0)
    assert int(policy.argmax()) == 12


def test_scalar_value_to_wld_is_current_player_distribution():
    assert np.allclose(gen.scalar_value_to_wld(1.0), [1.0, 0.0, 0.0])
    assert np.allclose(gen.scalar_value_to_wld(-1.0), [0.0, 1.0, 0.0])
    assert np.allclose(gen.scalar_value_to_wld(0.0), [0.5, 0.5, 0.0])


def test_choose_action_masks_illegal_moves_even_with_bad_teacher_policy():
    rng = np.random.default_rng(0)
    policy = np.zeros(81, dtype=np.float32)
    policy[0] = 1.0
    legal = np.zeros(81, dtype=bool)
    legal[5] = True

    assert gen.choose_action(policy, legal, rng, temperature=0.0) == 5


class _FakeNet:
    providers = ["FakeExecutionProvider"]

    def evaluate(self, state):
        legal = state.legal_indexes()
        policy = {action: float(i + 1) for i, action in enumerate(legal)}
        return 0.25, policy


def test_generate_dataset_writes_standard_history_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "UtttAiPolicyValueNet", lambda *args, **kwargs: _FakeNet())

    out = tmp_path / "dataset"
    metadata = gen.generate_dataset(
        output_dir=out,
        model_path=tmp_path / "fake.onnx",
        games=2,
        target="raw",
        mcts_sims=0,
        one_hot=False,
        move_temperature=0.0,
        move_policy="teacher",
        student_checkpoint=None,
        random_opening_plies=0,
        max_positions_per_game=3,
        shard_size=4,
        seed=1,
        require_cuda=False,
        overwrite=False,
        zstd_level=1,
    )

    triples = glob_file_triples(str(out / "history"), "*-canonical-*.ptz")
    assert metadata["positions"] == 6
    assert triples
    c_path, v_path, pi_path, size = triples[0]
    canonical = load_compressed(c_path).float()
    values = load_compressed(v_path).float()
    policy = load_compressed(pi_path).float()
    assert canonical.shape[1:] == (15, 9, 9)
    assert values.shape[1] == 3
    assert policy.shape[1] == 81
    assert size == canonical.shape[0]
    assert np.allclose(policy.sum(dim=1).numpy(), 1.0, atol=5e-4)


def test_generate_dataset_skips_when_teacher_state_has_no_legal_moves(tmp_path, monkeypatch):
    class TerminalTeacherState:
        @classmethod
        def initial(cls):
            return cls()

        def legal_indexes(self):
            return []

    monkeypatch.setattr(gen, "UtttAiPolicyValueNet", lambda *args, **kwargs: _FakeNet())
    monkeypatch.setattr(gen, "UtttAiState", TerminalTeacherState)

    out = tmp_path / "dataset"
    metadata = gen.generate_dataset(
        output_dir=out,
        model_path=tmp_path / "fake.onnx",
        games=1,
        target="raw",
        mcts_sims=0,
        one_hot=False,
        move_temperature=0.0,
        move_policy="teacher",
        student_checkpoint=None,
        random_opening_plies=0,
        max_positions_per_game=0,
        shard_size=4,
        seed=1,
        require_cuda=False,
        overwrite=False,
        zstd_level=1,
    )

    assert metadata["positions"] == 0
    assert glob_file_triples(str(out / "history"), "*-canonical-*.ptz") == []


def test_generate_dataset_can_roll_out_with_student_policy(tmp_path, monkeypatch):
    class FakeStudent:
        def enable_inference_optimizations(self):
            pass

        def process(self, canonical_batch):
            import torch

            policy = torch.zeros((1, 81), dtype=torch.float32)
            policy[0, 8] = 1.0
            return torch.zeros((1, 3), dtype=torch.float32), policy

    captured = {}

    def fake_choose_action(policy, legal, rng, temperature):
        captured["policy_argmax"] = int(np.argmax(policy))
        return 8

    monkeypatch.setattr(gen, "UtttAiPolicyValueNet", lambda *args, **kwargs: _FakeNet())
    monkeypatch.setattr(gen.neural_net.NNWrapper, "load_checkpoint", lambda *args, **kwargs: FakeStudent())
    monkeypatch.setattr(gen, "choose_action", fake_choose_action)

    out = tmp_path / "dataset"
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"fake")
    metadata = gen.generate_dataset(
        output_dir=out,
        model_path=tmp_path / "fake.onnx",
        games=1,
        target="raw",
        mcts_sims=0,
        one_hot=False,
        move_temperature=0.0,
        move_policy="student",
        student_checkpoint=checkpoint,
        random_opening_plies=0,
        max_positions_per_game=1,
        shard_size=4,
        seed=1,
        require_cuda=False,
        overwrite=False,
        zstd_level=1,
    )

    assert metadata["move_policy"] == "student"
    assert captured["policy_argmax"] == 8
