"""Tests for AlphaZero-style latest-network self-play selection."""

import pytest

from config import TrainConfig
from game_runner import selfplay_parent_iteration, update_uttt_ai_champion


def test_selfplay_parent_defaults_to_gated_best():
    config = TrainConfig()

    assert selfplay_parent_iteration(config, iteration=12, current_best=7) == 7


def test_selfplay_parent_uses_latest_when_enabled():
    config = TrainConfig(selfplay_use_latest=True)

    assert selfplay_parent_iteration(config, iteration=12, current_best=7) == 12


def test_uttt_ai_champion_uses_draws_as_half_a_win():
    state = {"iteration": 3, "score": 0.4}

    updated, improved = update_uttt_ai_champion(
        state, iteration=4, win_rate=0.25, draw_rate=0.5
    )

    assert improved is True
    assert updated == {
        "iteration": 4,
        "score": 0.5,
        "win_rate": 0.25,
        "draw_rate": 0.5,
    }


def test_uttt_ai_champion_does_not_replace_on_tie_or_regression():
    state = {
        "iteration": 3,
        "score": 0.5,
        "win_rate": 0.2,
        "draw_rate": 0.6,
    }

    tied, tie_improved = update_uttt_ai_champion(
        state, iteration=4, win_rate=0.3, draw_rate=0.4
    )
    weaker, weak_improved = update_uttt_ai_champion(
        state, iteration=5, win_rate=0.1, draw_rate=0.6
    )

    assert tie_improved is False
    assert weak_improved is False
    assert tied is state
    assert weaker is state


def test_external_selection_requires_uttt_and_enabled_eval():
    with pytest.raises(ValueError, match="Unknown game"):
        TrainConfig(game="unsupported", uttt_ai_select_best=True).validate()

    with pytest.raises(ValueError, match="must be positive"):
        TrainConfig(
            game="uttt",
            uttt_ai_select_best=True,
            uttt_ai_eval_interval=0,
        ).validate()
