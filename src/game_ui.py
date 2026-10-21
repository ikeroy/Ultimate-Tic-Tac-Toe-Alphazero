"""Game UI base class and registry for pluggable play interfaces."""

import numpy as np


class GameUI:
    """Base class for game-specific play UI."""

    def display_board(self, gs) -> str:
        """Return string representation of the board."""
        return str(gs)

    def parse_move(self, gs, input_str, valid_moves) -> int | None:
        """Parse human input into action ID. Return None if invalid."""
        try:
            action = int(input_str)
            if 0 <= action < len(valid_moves) and valid_moves[action]:
                return action
            return None
        except ValueError:
            return None

    def format_move(self, gs, action) -> str:
        """Return human-friendly name for an action."""
        return str(action)

    def format_move_short(self, gs, action) -> str:
        """Return compact (single-token, no description) name for an action.

        Used when many moves need to fit on one line, e.g. opening-tree
        sequence displays. Defaults to `format_move`; games with a verbose
        format_move should override to return only the short code.
        """
        return self.format_move(gs, action)

    def get_valid_move_descriptions(self, gs, valid_moves) -> list[tuple[int, str]]:
        """Return list of (action_id, description) for valid moves."""
        return [(i, str(i)) for i in range(len(valid_moves)) if valid_moves[i]]

    def show_help(self, gs):
        """Print game-specific help text."""
        print("Enter action ID (integer). Type 'help' for valid moves.")

    def display_actions_menu(self, gs, probs, valids, wld=None, top_n=5):
        """Display action menu with probabilities. Base: top-N summary."""
        valid_indices = np.where(np.asarray(valids) > 0)[0]
        if len(valid_indices) == 0:
            return
        if wld is not None:
            print(f"  WLD: {wld}")
        if probs is not None:
            sorted_idx = valid_indices[np.argsort(probs[valid_indices])[::-1]]
            print("  Top moves:")
            for idx in sorted_idx[:top_n]:
                desc = self.format_move(gs, idx)
                print(f"  {idx:4d}: {desc}  [{probs[idx]*100:5.1f}%]")
        else:
            print("  Valid moves:")
            for idx in valid_indices:
                desc = self.format_move(gs, idx)
                print(f"  {idx:4d}: {desc}")

    def build_action_menu(self, gs, probs, valids, wld=None) -> list[tuple]:
        """Build structured menu entries for interactive arrow-key selection.

        Returns list of entry tuples:
            ("header", "text")              - section header, not selectable
            ("action", action_id, "text")   - selectable action item
            ("info", "text")                - info line, not selectable
        """
        valid_indices = np.where(np.asarray(valids) > 0)[0]
        entries = []

        if wld is not None:
            entries.append(("info", f"Win: {wld[0]*100:.1f}%  Loss: {wld[1]*100:.1f}%  Draw: {wld[2]*100:.1f}%"))

        if probs is not None:
            sorted_idx = valid_indices[np.argsort(probs[valid_indices])[::-1]]
            entries.append(("header", "Actions:"))
            for idx in sorted_idx:
                desc = self.format_move(gs, idx)
                prob_str = f"  [{probs[idx]*100:5.1f}%]"
                entries.append(("action", int(idx), f"{desc}{prob_str}"))
        else:
            entries.append(("header", "Actions:"))
            for idx in valid_indices:
                desc = self.format_move(gs, idx)
                entries.append(("action", int(idx), desc))

        return entries

    def select_variant(self) -> str | None:
        """Optionally offer variant selection at startup. Returns game name or None."""
        return None


class UltimateTicTacToeUI(GameUI):
    """Human-facing UTTT moves use base-9 group/cell pairs.

    Internally the engine keeps action IDs as group * 9 + cell. In the play UI,
    entering "44" means group 4, cell 4, not decimal action 44.
    """

    def parse_move(self, gs, input_str, valid_moves) -> int | None:
        text = input_str.strip()
        if len(text) == 2 and text[0].isdigit() and text[1].isdigit():
            group = int(text[0])
            cell = int(text[1])
            if group < 9 and cell < 9:
                action = group * 9 + cell
                if valid_moves[action]:
                    return action
            return None
        if text.startswith("a"):
            try:
                action = int(text[1:])
                if 0 <= action < len(valid_moves) and valid_moves[action]:
                    return action
            except ValueError:
                pass
        return None

    def format_move(self, gs, action) -> str:
        group = action // 9
        cell = action % 9
        return f"{group}{cell}"

    def get_valid_move_descriptions(self, gs, valid_moves) -> list[tuple[int, str]]:
        return [
            (i, self.format_move(gs, i))
            for i in range(len(valid_moves))
            if valid_moves[i]
        ]

    def show_help(self, gs):
        print("Enter a two-digit base-9 move: <small-board><cell>, e.g. 44.")
        print("Use a decimal action ID only with an 'a' prefix, e.g. a40.")


# Registry mapping game names to UI factory functions.
GAME_UI_REGISTRY: dict[str, callable] = {
    "uttt": UltimateTicTacToeUI,
}


def get_game_ui(game_name: str) -> GameUI:
    """Get the appropriate GameUI for a game name."""
    if game_name in GAME_UI_REGISTRY:
        return GAME_UI_REGISTRY[game_name]()
    return GameUI()
