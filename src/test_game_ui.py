"""Tests for the generic and Ultimate Tic-Tac-Toe play interfaces."""

import io
import contextlib
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from game_ui import GameUI, UltimateTicTacToeUI, get_game_ui


# --- Generic GameUI ---


def test_generic_parse_move_valid():
    """Generic UI parses valid integer action IDs."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()
    # Action 0 is valid in the initial UTTT state.
    assert ui.parse_move(gs, "0", valids) == 0


def test_generic_parse_move_invalid_string():
    """Generic UI returns None for non-integer input."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()
    assert ui.parse_move(gs, "abc", valids) is None
    assert ui.parse_move(gs, "", valids) is None


def test_generic_parse_move_out_of_range():
    """Generic UI returns None for out-of-range action."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()
    assert ui.parse_move(gs, "9999", valids) is None
    assert ui.parse_move(gs, "-1", valids) is None


def test_generic_parse_move_invalid_action():
    """Generic UI returns None for in-range but not-valid action."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()
    # Find an invalid action if any
    invalid = [i for i in range(len(valids)) if not valids[i]]
    if invalid:
        assert ui.parse_move(gs, str(invalid[0]), valids) is None


def test_generic_format_move():
    """Generic UI formats moves as string of action ID."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    assert ui.format_move(gs, 0) == "0"
    assert ui.format_move(gs, 42) == "42"


def test_generic_display_board():
    """Generic UI returns non-empty string for board display."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    board = ui.display_board(gs)
    assert isinstance(board, str)
    assert len(board) > 0


def test_generic_valid_move_descriptions():
    """Generic UI lists all valid moves with descriptions."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()
    descs = ui.get_valid_move_descriptions(gs, valids)
    num_valid = sum(1 for v in valids if v)
    assert len(descs) == num_valid
    for action_id, desc in descs:
        assert valids[action_id]


def test_generic_display_actions_menu():
    """Generic UI display_actions_menu prints top-N summary."""
    ui = GameUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = np.array(gs.valid_moves(), dtype=np.float32)
    probs = np.zeros(len(valids))
    valid_idx = np.where(valids > 0)[0]
    probs[valid_idx] = 1.0 / len(valid_idx)
    # Should not raise
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ui.display_actions_menu(gs, probs, valids)
    output = f.getvalue()
    assert "Top moves:" in output


# --- GameUI Registry ---


def test_get_game_ui_uttt():
    """UTTT returns its base-9 coordinate UI."""
    ui = get_game_ui("uttt")
    assert isinstance(ui, UltimateTicTacToeUI)


def test_get_game_ui_unknown():
    """Unknown game falls back to generic GameUI."""
    ui = get_game_ui("nonexistent_game")
    assert isinstance(ui, GameUI)


# --- UltimateTicTacToeUI ---


def test_uttt_parse_move_uses_base9_group_cell():
    """UTTT text move 44 means group 4/cell 4, not decimal action 44."""
    ui = UltimateTicTacToeUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()

    assert ui.parse_move(gs, "44", valids) == 4 * 9 + 4
    assert ui.format_move(gs, 4 * 9 + 4) == "44"


def test_uttt_decimal_action_requires_prefix():
    """Raw decimal IDs are not accepted accidentally for UTTT."""
    ui = UltimateTicTacToeUI()
    gs = alphazero.UltimateTicTacToeGS()
    valids = gs.valid_moves()

    assert ui.parse_move(gs, "a44", valids) == 44
    assert ui.parse_move(gs, "99", valids) is None


def test_generic_select_variant_returns_none():
    """Generic GameUI has no variant selection."""
    ui = GameUI()
    assert ui.select_variant() is None
