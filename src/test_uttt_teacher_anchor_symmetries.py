"""Regression coverage for symmetry-expanded UTTT teacher anchors."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TrainConfig
from game_runner import glob_file_triples, load_compressed, save_compressed
from tools.expand_uttt_imitation_symmetries import expand_dataset_symmetries


def test_expand_dataset_symmetries_keeps_source_and_creates_all_uttt_orientations(tmp_path):
    source = tmp_path / "source"
    source_history = source / "history"
    source_history.mkdir(parents=True)
    canonical = torch.zeros((1, 15, 9, 9), dtype=torch.float32)
    canonical[0, 0, 0, 1] = 1.0
    values = torch.tensor([[0.6, 0.4, 0.0]], dtype=torch.float32)
    policy = torch.zeros((1, 81), dtype=torch.float32)
    policy[0, 1] = 1.0
    save_compressed(canonical, str(source_history / "0000-0000-canonical-1.ptz"))
    save_compressed(values, str(source_history / "0000-0000-v-1.ptz"))
    save_compressed(policy, str(source_history / "0000-0000-pi-1.ptz"))

    config = TrainConfig(game="uttt", hist_size=8, loader_threads=1, half_storage=False)
    output = tmp_path / "expanded"
    metadata = expand_dataset_symmetries(source, output, config)

    assert metadata["source_positions"] == 1
    assert metadata["positions"] == 8
    assert (source_history / "0000-0000-canonical-1.ptz").exists()

    triples = glob_file_triples(str(output / "history"), "*-canonical-*.ptz")
    assert sum(size for _, _, _, size in triples) == 8
    expanded_policies = torch.cat([load_compressed(p).float() for _, _, p, _ in triples])
    assert torch.allclose(expanded_policies.sum(dim=1), torch.ones(8))
    assert len(torch.unique(expanded_policies.argmax(dim=1))) == 8
