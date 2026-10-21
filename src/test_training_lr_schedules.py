import math

import pytest

from config import TrainConfig
from game_runner import _default_lr_state, get_lr


def test_exponential_schedule_reaches_geometric_midpoint_and_final_factor():
    config = TrainConfig(
        lr=0.04,
        lr_schedule="exponential",
        lr_exponential_final_factor=0.25,
        lr_exponential_start_iter=60,
        lr_exponential_end_iter=100,
        lr_warmup_target=0,
    )
    state = _default_lr_state(config)

    assert get_lr(config, 60, state) == pytest.approx(0.04)
    assert get_lr(config, 80, state) == pytest.approx(0.04 * math.sqrt(0.25))
    assert get_lr(config, 100, state) == pytest.approx(0.04 * 0.25)
    assert get_lr(config, 120, state) == pytest.approx(0.04 * 0.25)
