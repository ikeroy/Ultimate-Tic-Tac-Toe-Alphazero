"""Tests for temp_decay_half_life plumbing.

Covers:
  - base_params() falls back to scalar when config holds an int
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainConfig
from game_runner import base_params


def test_base_params_scalar_when_int():
    cfg = TrainConfig(game="uttt", temp_decay_half_life=10)
    params = base_params(cfg, start_temp=1.0, bs=64, cb=2)
    assert params.temp_decay_half_life == 10.0
    assert list(params.temp_decay_half_life_by_variant) == []


def test_base_params_scalar_when_zero():
    """0 (decay disabled) still routes through the scalar path."""
    cfg = TrainConfig(game="uttt", temp_decay_half_life=0)
    params = base_params(cfg, start_temp=1.0, bs=64, cb=2)
    assert params.temp_decay_half_life == 0.0
    assert list(params.temp_decay_half_life_by_variant) == []
