"""Interactive arrow-key action selector for terminal-based play.

Provides a menu UI where users navigate with arrow keys and select with Enter.
Uses only stdlib (tty, termios, sys) - no external dependencies.
"""

import sys
import tty
import termios
import shutil


def read_key():
    """Read a single keypress from stdin in raw mode.

    Returns:
        str: 'up', 'down', 'enter', 'tab', 'q', 'u', 'h', 'escape',
             or the character pressed.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        if ch == '\x1b':  # Escape sequence
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'A':
                    return 'up'
                if ch3 == 'B':
                    return 'down'
                if ch3 == 'C':
                    return 'right'
                if ch3 == 'D':
                    return 'left'
                return 'escape'
            return 'escape'
        if ch == '\r' or ch == '\n':
            return 'enter'
        if ch == '\t':
            return 'tab'
        if ch == '\x03':  # Ctrl-C
            return 'ctrl-c'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ANSI escape helpers
REVERSE = '\033[7m'
DIM = '\033[2m'
BOLD = '\033[1m'
RESET_ATTR = '\033[0m'
CLEAR_LINE = '\033[2K'


class ActionSelector:
    """Interactive menu selector for game actions.

    Entry types (tuples):
        ("header", "Moves:")              - section header, not selectable
        ("action", action_id, "text...")  - selectable action item
        ("info", "Win: 85.3% ...")        - info line, not selectable

    Usage:
        selector = ActionSelector(entries, preselect=best_action_id)
        result = selector.run()
        # result: int (action_id), str (meta-command), or None (Tab pressed)
    """

    def __init__(self, entries, preselect=None, max_visible_actions=None):
        self.entries = entries
        self.selectable = [
            i for i, e in enumerate(entries) if e[0] == 'action'
        ]
        if not self.selectable:
            raise ValueError("No selectable entries in menu")

        # Find preselected index
        self.cursor = 0
        if preselect is not None:
            for si, idx in enumerate(self.selectable):
                if entries[idx][1] == preselect:
                    self.cursor = si
                    break

        self.total_lines = 0  # tracks lines printed for cursor-up
        if max_visible_actions is None:
            max_visible_actions = shutil.get_terminal_size((80, 24)).lines - 6
        self.max_visible_actions = max(3, int(max_visible_actions))

    def _format_entry(self, entry_idx):
        """Format a single entry for display."""
        entry = self.entries[entry_idx]
        kind = entry[0]

        if kind == 'header':
            return f"{BOLD}{entry[1]}{RESET_ATTR}"
        if kind == 'info':
            return f"  {DIM}{entry[1]}{RESET_ATTR}"
        if kind == 'action':
            is_selected = (
                entry_idx == self.selectable[self.cursor]
            )
            text = entry[2]
            if is_selected:
                return f"  {REVERSE}> {text}{RESET_ATTR}"
            else:
                return f"    {text}"
        return str(entry)

    def _visible_entry_indices(self):
        """Return menu entry indices for a cursor-centered action viewport."""
        action_count = len(self.selectable)
        slots = min(self.max_visible_actions, action_count)
        above_marker = False
        below_marker = False
        start = 0
        end = slots
        for _ in range(4):
            visible_count = max(1, slots - int(above_marker) - int(below_marker))
            half = visible_count // 2
            start = self.cursor - half
            start = max(0, min(start, action_count - visible_count))
            end = start + visible_count
            next_above = start > 0
            next_below = end < action_count
            if next_above == above_marker and next_below == below_marker:
                break
            above_marker = next_above
            below_marker = next_below

        visible_actions = set(self.selectable[start:end])
        out = []
        inserted_above = False
        for i, entry in enumerate(self.entries):
            if entry[0] == 'action':
                if above_marker and not inserted_above:
                    out.append(("virtual", f"  {DIM}... {start} moves above ...{RESET_ATTR}"))
                    inserted_above = True
                if i in visible_actions:
                    out.append(i)
            else:
                out.append(i)
        if below_marker:
            hidden_below = action_count - end
            out.append(("virtual", f"  {DIM}... {hidden_below} moves below ...{RESET_ATTR}"))
        return out

    def _render_lines(self):
        """Render the current selector viewport."""
        lines = []
        for item in self._visible_entry_indices():
            if isinstance(item, tuple) and item[0] == "virtual":
                lines.append(item[1])
            else:
                lines.append(self._format_entry(item))

        lines.append('')
        position = f"{self.cursor + 1}/{len(self.selectable)}"
        lines.append(
            f"{DIM}[{RESET_ATTR}{position}{DIM}]  [{RESET_ATTR}"
            f"\u2191\u2193 Navigate{DIM}]  [{RESET_ATTR}"
            f"Enter Select{DIM}]  [{RESET_ATTR}"
            f"Tab Type{DIM}]  [{RESET_ATTR}"
            f"q Quit{DIM}]  [{RESET_ATTR}"
            f"u Undo{DIM}]{RESET_ATTR}"
        )
        return lines

    def draw(self):
        """Print the current menu viewport to stdout."""
        lines = self._render_lines()
        output = '\n'.join(lines)
        sys.stdout.write(output + '\n')
        sys.stdout.flush()
        self.total_lines = len(lines)

    def redraw(self):
        """Move cursor up and redraw the menu in place."""
        # Move cursor up to the start of our menu
        if self.total_lines > 0:
            sys.stdout.write(f'\033[{self.total_lines}A')

        old_total = self.total_lines
        lines = self._render_lines()
        for line in lines:
            sys.stdout.write(CLEAR_LINE + line + '\n')
        for _ in range(old_total - len(lines)):
            sys.stdout.write(CLEAR_LINE + '\n')
        sys.stdout.flush()
        self.total_lines = max(old_total, len(lines))

    def cleanup(self):
        """Ensure cursor is at the bottom of the menu."""
        # Nothing to do - cursor is already at the bottom after draw/redraw
        pass

    def run(self):
        """Run the interactive selector.

        Returns:
            int: action_id if user selected an action
            str: 'quit', 'undo', 'help', 'status' for meta-commands
            None: user pressed Tab (switch to text input mode)
        """
        self.draw()

        try:
            while True:
                key = read_key()

                if key == 'up':
                    if self.cursor > 0:
                        self.cursor -= 1
                        self.redraw()
                elif key == 'down':
                    if self.cursor < len(self.selectable) - 1:
                        self.cursor += 1
                        self.redraw()
                elif key == 'enter':
                    entry = self.entries[self.selectable[self.cursor]]
                    self.cleanup()
                    return entry[1]  # action_id
                elif key == 'tab':
                    self.cleanup()
                    return None
                elif key == 'q':
                    self.cleanup()
                    return 'quit'
                elif key == 'u':
                    self.cleanup()
                    return 'undo'
                elif key == 'h':
                    self.cleanup()
                    return 'help'
                elif key == 's':
                    self.cleanup()
                    return 'status'
                elif key == 'v':
                    self.cleanup()
                    return 'valid'
                elif key == 'ctrl-c':
                    self.cleanup()
                    return 'quit'
                elif key == 'escape':
                    self.cleanup()
                    return None
        except (KeyboardInterrupt, EOFError):
            self.cleanup()
            return 'quit'
