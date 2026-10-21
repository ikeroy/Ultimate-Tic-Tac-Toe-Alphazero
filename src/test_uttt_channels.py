import numpy as np
import pytest
import torch

import alphazero
from neural_net import NNArgs, NNArch


CURRENT_PLAYER_CELLS = 0
OPPONENT_CELLS = 1
LAST_MOVE = 2
LEGAL_MOVES = 3
FORCED_BOARD = 4
CURRENT_PLAYER_WON_BOARDS = 5
OPPONENT_WON_BOARDS = 6
CLOSED_BOARDS = 7
GIVES_OPPONENT_FREE_CHOICE = 8
CURRENT_LOCAL_WIN_THREATS = 9
OPPONENT_LOCAL_WIN_THREATS = 10
CURRENT_MACRO_WIN_THREATS = 11
OPPONENT_MACRO_WIN_THREATS = 12
SENDS_OPPONENT_TO_LOCAL_WIN = 13
OPPONENT_SENDS_CURRENT_TO_LOCAL_WIN = 14


def _action_coords(action):
    group, cell = divmod(action, 9)
    return (group // 3) * 3 + cell // 3, (group % 3) * 3 + cell % 3


def test_uttt_python_canonical_shape_and_empty_planes():
    game = alphazero.UltimateTicTacToeGS()
    canonical = np.asarray(game.canonicalized())

    assert alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE() == [15, 9, 9]
    assert canonical.shape == (15, 9, 9)
    assert canonical[LEGAL_MOVES].sum() == 81
    assert canonical[
        [
            CURRENT_PLAYER_CELLS,
            OPPONENT_CELLS,
            LAST_MOVE,
            FORCED_BOARD,
            CURRENT_PLAYER_WON_BOARDS,
            OPPONENT_WON_BOARDS,
            CLOSED_BOARDS,
            GIVES_OPPONENT_FREE_CHOICE,
            CURRENT_LOCAL_WIN_THREATS,
            OPPONENT_LOCAL_WIN_THREATS,
            CURRENT_MACRO_WIN_THREATS,
            OPPONENT_MACRO_WIN_THREATS,
            SENDS_OPPONENT_TO_LOCAL_WIN,
            OPPONENT_SENDS_CURRENT_TO_LOCAL_WIN,
        ]
    ].sum() == 0


def test_uttt_python_planes_after_first_move():
    game = alphazero.UltimateTicTacToeGS()
    game.play_move(40)  # board 4, cell 4: sends opponent to board 4
    canonical = np.asarray(game.canonicalized())

    assert canonical[OPPONENT_CELLS, 4, 4] == 1
    assert canonical[LAST_MOVE, 4, 4] == 1
    assert canonical[CURRENT_PLAYER_CELLS].sum() == 0
    assert canonical[LEGAL_MOVES].sum() == 8
    assert canonical[FORCED_BOARD, 3:6, 3:6].sum() == 9
    assert canonical[FORCED_BOARD].sum() == 9
    assert canonical[GIVES_OPPONENT_FREE_CHOICE].sum() == 0


def test_uttt_network_accepts_nine_channel_positions():
    game = alphazero.UltimateTicTacToeGS()
    canonical = torch.from_numpy(np.asarray(game.canonicalized()).copy())
    model = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(num_channels=4, depth=1, kernel_size=3, dense_net=True),
    )
    model.eval()

    with torch.no_grad():
        value, policy = model(canonical.unsqueeze(0))

    assert value.shape == (1, 3)
    assert policy.shape == (1, 81)
    assert torch.isfinite(value).all()
    assert torch.isfinite(policy).all()


def test_uttt_hierarchical_network_accepts_nine_channel_positions():
    game = alphazero.UltimateTicTacToeGS()
    canonical = torch.from_numpy(np.asarray(game.canonicalized()).copy())
    model = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(
            num_channels=6,
            depth=2,
            kernel_size=3,
            dense_net=True,
            uttt_hierarchical=True,
        ),
    )
    model.eval()

    with torch.no_grad():
        value, policy = model(canonical.unsqueeze(0))

    assert value.shape == (1, 3)
    assert policy.shape == (1, 81)
    assert torch.isfinite(value).all()
    assert torch.isfinite(policy).all()
    assert torch.allclose(value.exp().sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.allclose(policy.exp().sum(dim=1), torch.ones(1), atol=1e-5)


def test_uttt_hierarchical_macro_transformer_outputs_normalized_policy():
    canonical = torch.from_numpy(
        np.asarray(alphazero.UltimateTicTacToeGS().canonicalized()).copy()
    )
    model = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(
            num_channels=6,
            depth=2,
            kernel_size=3,
            dense_net=True,
            uttt_hierarchical=True,
            uttt_macro_transformer_layers=2,
            uttt_macro_transformer_heads=2,
            uttt_macro_transformer_ff_mult=2.0,
            uttt_macro_transformer_dropout=0.0,
        ),
    )
    model.eval()

    with torch.no_grad():
        value, policy = model(canonical.unsqueeze(0))

    assert value.shape == (1, 3)
    assert policy.shape == (1, 81)
    assert torch.isfinite(value).all()
    assert torch.isfinite(policy).all()
    assert torch.allclose(value.exp().sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.allclose(policy.exp().sum(dim=1), torch.ones(1), atol=1e-5)
    assert model.arch.macro_pos_embedding.shape == (1, 9, 6)


def test_uttt_hierarchical_split_macro_width_outputs_normalized_policy():
    canonical = torch.from_numpy(
        np.asarray(alphazero.UltimateTicTacToeGS().canonicalized()).copy()
    )
    model = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(
            num_channels=6,
            depth=4,
            kernel_size=3,
            dense_net=True,
            uttt_hierarchical=True,
            uttt_macro_channels=16,
            uttt_macro_transformer_layers=1,
            uttt_macro_transformer_heads=4,
            uttt_macro_transformer_ff_mult=2.0,
            uttt_macro_transformer_dropout=0.0,
            head_channels=10,
        ),
    )
    model.eval()

    with torch.no_grad():
        value, policy = model(canonical.unsqueeze(0))

    assert value.shape == (1, 3)
    assert policy.shape == (1, 81)
    assert torch.isfinite(value).all()
    assert torch.isfinite(policy).all()
    assert torch.allclose(value.exp().sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.allclose(policy.exp().sum(dim=1), torch.ones(1), atol=1e-5)
    assert model.arch.local_stem[0].weight.shape[0] == 6
    assert model.arch.macro_proj.weight.shape == (16, 6)
    assert model.arch.macro_pos_embedding.shape == (1, 9, 16)
    assert model.arch.macro_blocks[0].conv1.weight.shape[:2] == (16, 16)
    assert model.arch.fuse[0].weight.shape[1] == 6 + 16
    assert model.arch.v_fc1.weight.shape[1] == 16


def test_uttt_hierarchical_macro_transformer_requires_head_divisibility():
    with pytest.raises(ValueError, match="must divide uttt_macro_channels"):
        NNArch(
            alphazero.UltimateTicTacToeGS,
            NNArgs(
                num_channels=6,
                depth=2,
                kernel_size=3,
                dense_net=True,
                uttt_hierarchical=True,
                uttt_macro_channels=10,
                uttt_macro_transformer_layers=1,
                uttt_macro_transformer_heads=4,
            ),
        )


def test_uttt_hierarchical_macro_token_grid_round_trip():
    from neural_net import UTTTHierarchicalArch

    macro = torch.arange(2 * 3 * 3, dtype=torch.float32).view(1, 2, 3, 3)

    tokens = UTTTHierarchicalArch._macro_grid_to_tokens(macro)
    restored = UTTTHierarchicalArch._macro_tokens_to_grid(tokens)

    assert tokens.shape == (1, 9, 2)
    assert torch.equal(restored, macro)


def test_uttt_hierarchical_factorized_policy_heads_output_normalized_policy():
    game = alphazero.UltimateTicTacToeGS()
    canonical = torch.from_numpy(np.asarray(game.canonicalized()).copy())

    for policy_head in ("factorized_simple", "factorized_hybrid"):
        model = NNArch(
            alphazero.UltimateTicTacToeGS,
            NNArgs(
                num_channels=6,
                depth=2,
                kernel_size=3,
                dense_net=True,
                uttt_hierarchical=True,
                uttt_policy_head=policy_head,
            ),
        )
        model.eval()

        with torch.no_grad():
            value, policy = model(canonical.unsqueeze(0))

        assert value.shape == (1, 3)
        assert policy.shape == (1, 81)
        assert torch.isfinite(value).all()
        assert torch.isfinite(policy).all()
        assert torch.allclose(value.exp().sum(dim=1), torch.ones(1), atol=1e-5)
        assert torch.allclose(policy.exp().sum(dim=1), torch.ones(1), atol=1e-5)


def test_uttt_hierarchical_destination_context_outputs_normalized_policy():
    game = alphazero.UltimateTicTacToeGS()
    canonical = torch.from_numpy(np.asarray(game.canonicalized()).copy())
    model = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(
            num_channels=6,
            depth=2,
            kernel_size=3,
            dense_net=True,
            uttt_hierarchical=True,
            uttt_destination_context=True,
        ),
    )
    model.eval()

    with torch.no_grad():
        value, policy = model(canonical.unsqueeze(0))

    assert value.shape == (1, 3)
    assert policy.shape == (1, 81)
    assert torch.isfinite(value).all()
    assert torch.isfinite(policy).all()
    assert torch.allclose(value.exp().sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.allclose(policy.exp().sum(dim=1), torch.ones(1), atol=1e-5)


def test_uttt_hierarchical_policy_head_modes_create_expected_parameters():
    simple = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(
            num_channels=6,
            depth=2,
            kernel_size=3,
            dense_net=True,
            uttt_hierarchical=True,
            uttt_policy_head="factorized_simple",
        ),
    ).arch
    hybrid = NNArch(
        alphazero.UltimateTicTacToeGS,
        NNArgs(
            num_channels=6,
            depth=2,
            kernel_size=3,
            dense_net=True,
            uttt_hierarchical=True,
            uttt_policy_head="factorized_hybrid",
        ),
    ).arch

    assert hasattr(simple, "pi_board_conv")
    assert hasattr(simple, "pi_cell_conv")
    assert not hasattr(simple, "pi_fc1")
    assert hasattr(hybrid, "pi_board_conv")
    assert hasattr(hybrid, "pi_cell_conv")
    assert hasattr(hybrid, "pi_fc1")


def test_uttt_hierarchical_destination_context_maps_cell_to_destination_board():
    from neural_net import UTTTHierarchicalArch

    macro = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)

    dest_grid = UTTTHierarchicalArch._destination_context_grid(macro)

    assert dest_grid.shape == (1, 1, 9, 9)
    for action in range(81):
        group, cell = divmod(action, 9)
        row, col = _action_coords(action)
        assert dest_grid[0, 0, row, col] == cell


def test_uttt_hierarchical_board_split_merge_preserves_coordinates():
    from neural_net import UTTTHierarchicalArch

    x = torch.arange(9 * 9, dtype=torch.float32).view(1, 1, 9, 9)

    boards = UTTTHierarchicalArch._split_boards(x)

    assert boards.shape == (9, 1, 3, 3)
    for action in range(81):
        group, cell = divmod(action, 9)
        row, col = _action_coords(action)
        cell_row, cell_col = divmod(cell, 3)
        assert boards[group, 0, cell_row, cell_col] == x[0, 0, row, col]

    merged = UTTTHierarchicalArch._merge_boards(boards, 1)
    assert torch.equal(merged, x)


def test_uttt_planes_match_rules_through_reachable_games():
    for action_offset in range(5):
        game = alphazero.UltimateTicTacToeGS()
        while game.scores() is None:
            canonical = np.asarray(game.canonicalized())
            valid = np.asarray(game.valid_moves(), dtype=bool)

            assert canonical[LEGAL_MOVES].sum() == valid.sum()
            assert not np.any(canonical[CURRENT_PLAYER_WON_BOARDS]
                              * canonical[OPPONENT_WON_BOARDS])
            assert np.all(canonical[CURRENT_PLAYER_WON_BOARDS]
                          <= canonical[CLOSED_BOARDS])
            assert np.all(canonical[OPPONENT_WON_BOARDS]
                          <= canonical[CLOSED_BOARDS])

            for group in range(9):
                gr, gc = divmod(group, 3)
                for plane in (
                    FORCED_BOARD,
                    CURRENT_PLAYER_WON_BOARDS,
                    OPPONENT_WON_BOARDS,
                    CLOSED_BOARDS,
                ):
                    region = canonical[plane, gr * 3:(gr + 1) * 3,
                                       gc * 3:(gc + 1) * 3]
                    assert np.all(region == region[0, 0])

            for action in range(81):
                row, col = _action_coords(action)
                assert bool(canonical[LEGAL_MOVES, row, col]) == bool(valid[action])
                if not valid[action]:
                    assert canonical[GIVES_OPPONENT_FREE_CHOICE, row, col] == 0
                    assert canonical[SENDS_OPPONENT_TO_LOCAL_WIN, row, col] == 0
                    continue

                after = game.copy()
                after.play_move(action)
                after_canonical = np.asarray(after.canonicalized())
                grants_free_choice = after_canonical[FORCED_BOARD].sum() == 0
                assert bool(canonical[GIVES_OPPONENT_FREE_CHOICE, row, col]) == grants_free_choice

            legal_actions = np.flatnonzero(valid)
            game.play_move(int(legal_actions[action_offset % len(legal_actions)]))
