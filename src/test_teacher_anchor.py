"""Focused tests for fixed teacher-anchor training support."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_runner
import neural_net
from config import TrainConfig
from game_runner import (
    _seed_initial_checkpoint,
    save_compressed,
    resolve_history_dir,
    select_training_work_items,
)


def _triples():
    return [
        ("first-canonical-3.ptz", "first-v-3.ptz", "first-pi-3.ptz", 3),
        ("second-canonical-4.ptz", "second-v-4.ptz", "second-pi-4.ptz", 4),
    ]


def test_teacher_anchor_config_requires_directory_and_valid_fraction():
    config = TrainConfig()
    config.teacher_anchor_fraction = 0.2
    with pytest.raises(ValueError, match="teacher_anchor_dir"):
        config.validate()

    config.teacher_anchor_dir = "data/anchor"
    config.teacher_anchor_fraction = 1.0
    with pytest.raises(ValueError, match="teacher_anchor_fraction"):
        config.validate()

    config.teacher_anchor_fraction = 0.2
    config.validate()


def test_select_training_work_items_without_replacement_is_unique(monkeypatch):
    monkeypatch.setattr("game_runner.random.sample", lambda population, count: list(range(count)))

    work_items = select_training_work_items(_triples(), 5)

    selected = [(path, index) for path, _, _, indices in work_items for index in indices]
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert selected == [
        ("first-canonical-3.ptz", 0),
        ("first-canonical-3.ptz", 1),
        ("first-canonical-3.ptz", 2),
        ("second-canonical-4.ptz", 0),
        ("second-canonical-4.ptz", 1),
    ]


def test_select_training_work_items_can_preserve_mix_with_replacement(monkeypatch):
    monkeypatch.setattr("game_runner.random.choices", lambda population, k: [0] * k)

    work_items = select_training_work_items(_triples(), 12, allow_replacement=True)

    selected = [(path, index) for path, _, _, indices in work_items for index in indices]
    assert len(selected) == 12
    assert selected == [("first-canonical-3.ptz", 0)] * 12


def test_resolve_history_dir_accepts_dataset_or_history_directory(tmp_path):
    dataset = tmp_path / "dataset"
    history = dataset / "history"
    history.mkdir(parents=True)

    assert resolve_history_dir(str(dataset)) == str(history)
    assert resolve_history_dir(str(history)) == str(history)


def test_seed_initial_checkpoint_copies_exact_source_bytes(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint-bytes")
    checkpoint_dir = tmp_path / "trial" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)

    destination = _seed_initial_checkpoint(str(source), str(checkpoint_dir), "teacher-anchor")

    assert os.path.basename(destination) == "0000-teacher-anchor.pt"
    assert open(destination, "rb").read() == b"checkpoint-bytes"


def test_train_mixes_exact_teacher_anchor_fraction(tmp_path, monkeypatch):
    config = TrainConfig(
        game="uttt",
        hist_size=1,
        train_batch_size=4,
        train_sample_rate=2,
        loader_threads=0,
        teacher_anchor_fraction=0.25,
    )
    history = tmp_path / "history"
    anchor = tmp_path / "anchor" / "history"
    checkpoint = tmp_path / "checkpoint"
    history.mkdir(parents=True)
    anchor.mkdir(parents=True)
    checkpoint.mkdir()
    config.teacher_anchor_dir = str(anchor.parent)

    shape = tuple(config.Game.CANONICAL_SHAPE())
    moves = config.Game.NUM_MOVES()
    values = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    policy = torch.zeros((4, moves))
    policy[:, 0] = 1.0
    save_compressed(torch.zeros((4, *shape)), str(history / "0000-0000-canonical-4.ptz"))
    save_compressed(values, str(history / "0000-0000-v-4.ptz"))
    save_compressed(policy, str(history / "0000-0000-pi-4.ptz"))

    anchor_values = torch.tensor([[0.0, 1.0, 0.0]] * 8)
    save_compressed(torch.ones((8, *shape)), str(anchor / "0000-0000-canonical-8.ptz"))
    save_compressed(anchor_values, str(anchor / "0000-0000-v-8.ptz"))
    save_compressed(torch.cat((policy, policy)), str(anchor / "0000-0000-pi-8.ptz"))

    class FakeNN:
        def __init__(self):
            self.seen = []

        def set_lr(self, _lr):
            pass

        def train(self, batches, steps_to_train, *_args, **_kwargs):
            for step, batch in enumerate(batches):
                if step >= steps_to_train:
                    break
                self.seen.extend(batch[0][:, 0, 0, 0].tolist())
            return 1.0, 2.0

        def save_checkpoint(self, *_args, **_kwargs):
            pass

    fake_nn = FakeNN()
    monkeypatch.setattr(
        neural_net.NNWrapper,
        "load_checkpoint",
        staticmethod(lambda *_args, **_kwargs: fake_nn),
    )

    class FakeRun:
        def __init__(self):
            self.tracked = []

        def track(self, value, **kwargs):
            self.tracked.append((value, kwargs))

    run = FakeRun()
    v_loss, pi_loss, steps = game_runner.train(
        config,
        {"history": str(history), "checkpoint": str(checkpoint)},
        "anchor-trial",
        iteration=0,
        hist_size=1,
        run=run,
        total_train_steps=0,
    )

    assert (v_loss, pi_loss, steps) == (1.0, 2.0, 2)
    assert fake_nn.seen.count(1.0) == 2
    assert fake_nn.seen.count(0.0) == 6
    assert any(kwargs["name"] == "teacher_anchor_fraction" for _, kwargs in run.tracked)
