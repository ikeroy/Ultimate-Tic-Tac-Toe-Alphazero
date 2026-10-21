"""Regression tests for play.py's terminal UI behavior."""

import io
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import action_selector
import play
from action_selector import ActionSelector


class RecordingUI:
    def __init__(self, board="board"):
        self.board = board
        self.calls = []

    def display_board(self, game):
        return self.board

    def build_action_menu(self, game, probs, valids, wld=None):
        self.calls.append((probs, valids, wld))
        entries = [("info", play.format_wld(wld))] if wld is not None else []
        entries.extend(
            ("action", i, f"move {i}") for i, valid in enumerate(valids) if valid
        )
        return entries


def make_ctx():
    return SimpleNamespace(
        game=object(),
        players=[play.PlayerConfig(), play.PlayerConfig()],
        cache=None,
        player_caches=[None, None],
        tree_reuse=True,
        auto_delay=0.0,
    )


class TurnState:
    def __init__(self, name, player):
        self.name = name
        self.player = player

    def current_player(self):
        return self.player


def test_player_defaults_show_wld_but_not_hints_or_autoselect():
    player = play.PlayerConfig()

    assert player.show_wld is True
    assert player.show_hints is False
    assert player.auto_select is False
    assert str(player) == "Human"


def test_autoselect_command_only_enables_ai_players(capsys):
    ctx = make_ctx()
    ctx.players[1].is_ai = True

    play.handle_config_command(["autoselect", "on"], ctx, "uttt", "data")

    assert ctx.players[0].auto_select is False
    assert ctx.players[1].auto_select is True
    capsys.readouterr()


def test_undo_rewinds_past_autoselect_ai_turn():
    ctx = make_ctx()
    ctx.players[1].is_ai = True
    ctx.players[1].auto_select = True
    ctx.players[0].mcts = object()
    ctx.players[1].mcts = object()
    human_turn = TurnState("human-before-move", 0)
    ai_turn = TurnState("ai-before-move", 1)
    ctx.game = TurnState("human-after-ai-move", 0)
    history = [human_turn, ai_turn]

    assert play.undo_move(ctx, history) is True

    assert ctx.game is human_turn
    assert history == []
    assert all(player.mcts is None for player in ctx.players)


def test_undo_stops_on_ai_turn_when_autoselect_is_off():
    ctx = make_ctx()
    ctx.players[1].is_ai = True
    human_turn = TurnState("human-before-move", 0)
    ai_turn = TurnState("ai-before-move", 1)
    ctx.game = TurnState("human-after-ai-move", 0)
    history = [human_turn, ai_turn]

    assert play.undo_move(ctx, history) is True

    assert ctx.game is ai_turn
    assert history == [human_turn]


def test_undo_without_history_leaves_game_unchanged():
    ctx = make_ctx()
    game = TurnState("initial", 0)
    ctx.game = game

    assert play.undo_move(ctx, []) is False
    assert ctx.game is game


def test_human_turn_shows_wld_without_hints(monkeypatch, capsys):
    ctx = make_ctx()
    ctx.players[0].eval_type = "playout"
    ui = RecordingUI()
    probs = np.array([0.8, 0.2])
    wld = np.array([0.6, 0.3, 0.1])
    monkeypatch.setattr(
        play, "get_ai_probs", lambda *_: (probs, "playout", 100, wld)
    )

    entries, highlighted, menu_probs, menu_wld = play.prepare_human_turn_menu(
        ctx, ui, 0, np.array([1, 1])
    )

    assert highlighted is None
    assert menu_probs is None
    assert np.array_equal(menu_wld, wld)
    assert ui.calls[-1][0] is None
    assert np.array_equal(ui.calls[-1][2], wld)
    assert entries[0][0] == "info"
    assert "Hints" not in capsys.readouterr().out


def test_human_hints_add_probs_and_highlight_while_wld_stays_visible(
    monkeypatch, capsys
):
    ctx = make_ctx()
    player = ctx.players[0]
    player.eval_type = "playout"
    player.show_hints = True
    player.greedy = True
    ui = RecordingUI()
    probs = np.array([0.2, 0.8])
    wld = np.array([0.6, 0.3, 0.1])
    monkeypatch.setattr(
        play, "get_ai_probs", lambda *_: (probs, "playout", 100, wld)
    )

    _, highlighted, menu_probs, menu_wld = play.prepare_human_turn_menu(
        ctx, ui, 0, np.array([1, 1])
    )

    assert highlighted == 1
    assert np.array_equal(menu_probs, probs)
    assert np.array_equal(menu_wld, wld)
    assert np.array_equal(ui.calls[-1][0], probs)
    assert np.array_equal(ui.calls[-1][2], wld)
    assert "Hints [playout]:" in capsys.readouterr().out


def test_wld_off_does_not_evaluate_when_hints_are_also_off(monkeypatch):
    ctx = make_ctx()
    player = ctx.players[0]
    player.eval_type = "playout"
    player.show_wld = False
    ui = RecordingUI()

    def unexpected_eval(*args):
        raise AssertionError("evaluator should not run")

    monkeypatch.setattr(play, "get_ai_probs", unexpected_eval)
    _, highlighted, probs, wld = play.prepare_human_turn_menu(
        ctx, ui, 0, np.array([1, 1])
    )

    assert highlighted is None
    assert probs is None
    assert wld is None
    assert ui.calls[-1][0] is None
    assert ui.calls[-1][2] is None


def test_selector_scrolling_keeps_rendered_line_count_constant():
    entries = [("header", "Moves")]
    entries.extend(("action", i, f"move {i}") for i in range(20))
    selector = ActionSelector(entries, max_visible_actions=7)
    line_counts = []

    for cursor in [0, 1, 5, 10, 18, 19]:
        selector.cursor = cursor
        lines = selector._render_lines()
        line_counts.append(len(lines))
        assert len(selector._visible_entry_indices()) == 8

    assert len(set(line_counts)) == 1


def test_selector_scroll_markers_replace_action_rows():
    entries = [("action", i, f"move {i}") for i in range(20)]
    selector = ActionSelector(entries, max_visible_actions=7)

    selector.cursor = 0
    top = selector._render_lines()
    selector.cursor = 10
    middle = selector._render_lines()
    selector.cursor = 19
    bottom = selector._render_lines()

    assert not any("moves above" in line for line in top)
    assert any("moves below" in line for line in top)
    assert any("moves above" in line for line in middle)
    assert any("moves below" in line for line in middle)
    assert any("moves above" in line for line in bottom)
    assert not any("moves below" in line for line in bottom)
    assert len(top) == len(middle) == len(bottom)


def test_selector_run_scrolls_and_returns_selected_action(monkeypatch):
    entries = [("action", i, f"move {i}") for i in range(12)]
    selector = ActionSelector(entries, max_visible_actions=5)
    keys = iter(["down"] * 8 + ["enter"])
    monkeypatch.setattr(action_selector, "read_key", lambda: next(keys))
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert selector.run() == 8


def test_make_action_selector_uses_terminal_height(monkeypatch):
    ctx = make_ctx()
    ui = RecordingUI(board="a\nb\nc\nd")
    entries = [("info", "W/L/D")]
    entries.extend(("action", i, f"move {i}") for i in range(20))
    monkeypatch.setattr(play.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        play.shutil, "get_terminal_size", lambda fallback: os.terminal_size((80, 20))
    )

    selector = play.make_action_selector(ctx, ui, entries)

    assert selector.max_visible_actions == 8
