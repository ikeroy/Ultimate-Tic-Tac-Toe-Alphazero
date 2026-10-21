import numpy as np

from confidence_backfill import (
    action_coords,
    distribution_stats,
    masked_policy,
    spatial_legal_plane_to_action_mask,
)


def test_spatial_legal_plane_to_action_mask_uses_uttt_action_order():
    plane = np.zeros((9, 9), dtype=np.float32)
    actions = [0, 10, 44, 80]
    for action in actions:
        row, col = action_coords(action)
        plane[row, col] = 1

    mask = spatial_legal_plane_to_action_mask(plane)

    assert mask.shape == (81,)
    assert np.flatnonzero(mask).tolist() == actions


def test_spatial_legal_plane_to_action_mask_preserves_batch_shape():
    planes = np.zeros((2, 9, 9), dtype=np.float32)
    planes[0, action_coords(5)[0], action_coords(5)[1]] = 1
    planes[1, action_coords(76)[0], action_coords(76)[1]] = 1

    mask = spatial_legal_plane_to_action_mask(planes)

    assert mask.shape == (2, 81)
    assert np.flatnonzero(mask[0]).tolist() == [5]
    assert np.flatnonzero(mask[1]).tolist() == [76]


def test_distribution_stats_reports_entropy_and_top_mass():
    probs = np.array([
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0],
    ])

    stats = distribution_stats(probs, "p")

    assert np.isclose(stats["p_top1_mean"], 0.75)
    assert np.isclose(stats["p_entropy_mean"], np.log(2) / 2)
    assert np.isclose(stats["p_gt90_frac"], 0.5)


def test_masked_policy_renormalizes_to_legal_moves():
    raw = np.array([[0.1, 0.2, 0.7], [0.0, 0.0, 0.0]])
    legal = np.array([[True, True, False], [False, True, True]])

    masked = masked_policy(raw, legal)

    assert np.allclose(masked[0], [1 / 3, 2 / 3, 0])
    assert np.allclose(masked[1], [0, 0.5, 0.5])
