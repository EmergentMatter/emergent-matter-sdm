#!/usr/bin/env python3
"""Create a changelog.d/ note for this change.

Run it, pick a bump level, write a one-line user-facing summary. `q`,
`Esc`, or Ctrl-C backs out of the level picker without writing anything.
Stdlib only, no dependencies.

Keep this file at the repo root, as `scripts/changeset.py`, invoked by
path. Do not move it under `src/<pkg>/` and do not give it a
`[project.scripts]` entry: either one ships a contributor-only tool to
everyone who installs the package, and neither fails anything, so nothing
catches it but this note.
"""

from __future__ import annotations

import os
import secrets
import select
import sys
from pathlib import Path
from typing import Literal

LEVELS = [
    ("major", "Existing callers have to change something"),
    ("minor", "New capability, nothing breaks"),
    ("patch", "Fixes, documentation, anything else user-visible"),
]

NOTES_DIR = Path("changelog.d")

ESC = "\x1b"
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_INTERRUPT = "\x03"  # Ctrl-C
# Ctrl-D. Raw mode delivers it as a plain byte instead of closing stdin, so
# it has to be recognised explicitly here to mean what it already means at
# the numbered prompt: EOF.
KEY_EOT = "\x04"

# Bracketed paste markers. A terminal in this mode wraps pasted text between
# these; see _swallow_paste for why the payload has to be discarded.
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"

# A terminal only sends those markers to a program that asked for them, and
# the shell turns the mode off before running us, so the picker has to ask
# for itself. Terminals that don't support it ignore both of these.
PASTE_MODE_ON = "\x1b[?2004h"
PASTE_MODE_OFF = "\x1b[?2004l"

# How long to wait for the rest of an escape sequence before concluding the
# user pressed a bare Esc. Arrow keys arrive as one burst, so anything that
# hasn't landed within this is not part of the sequence.
ESC_SEQUENCE_TIMEOUT = 0.05

# What a keypress means to the picker.
Action = Literal["move", "select", "cancel", "interrupt", "ignore"]


def handle_key(ch: str, selected: int) -> tuple[int, Action]:
    """Pure state machine for the interactive picker.

    Takes the key just read (an arrow key arrives as its whole escape
    sequence; "" means EOF) and the current selection. Returns the new
    selection plus what to do about it. Kept free of terminal I/O so it can
    be tested without a pty.
    """
    if ch in (KEY_UP, "k"):  # k/j as well as the arrows, for vim hands
        return (selected - 1) % len(LEVELS), "move"
    if ch in (KEY_DOWN, "j"):
        return (selected + 1) % len(LEVELS), "move"
    if ch in ("\r", "\n"):
        return selected, "select"
    if ch.isdigit() and 1 <= int(ch) <= len(LEVELS):
        # The numbered fallback trains this muscle memory; honour it here too.
        return int(ch) - 1, "select"
    if ch in ("q", "Q", ESC, KEY_EOT, ""):
        # A bare Esc is a deliberate back-out; "" is EOF (stdin closing under
        # us) and must not be looped on.
        return selected, "cancel"
    if ch == KEY_INTERRUPT:
        return selected, "interrupt"
    return selected, "ignore"


def _read_byte(fd: int) -> str:
    """One byte from `fd`, decoded leniently. "" at EOF.

    `os.read` rather than `sys.stdin.read`, so the `_has_pending` peek below
    stays accurate. Stdin's text layer buffers a whole burst ("\x1b[A") on
    the first read; after that, `select` reports nothing pending and a real
    arrow key looks like a bare Esc.
    """
    return os.read(fd, 1).decode("utf-8", "replace")


def _has_pending(fd: int) -> bool:
    """Is there more input already on its way? Never blocks for long."""
    return bool(select.select([fd], [], [], ESC_SEQUENCE_TIMEOUT)[0])


def _swallow_paste(fd: int) -> None:
    """Discard a bracketed paste's payload, up to and including its end marker.

    Pasted text isn't keystrokes, and the picker has nowhere to paste into.
    Read as keys, the first "q" in the pasted text would cancel and the
    first newline would select. Bounded by the same pending-input timeout
    as everything else, so a paste that stalls mid-stream stops being
    swallowed instead of hanging the prompt.
    """
    tail = ""
    while _has_pending(fd):
        tail = (tail + _read_byte(fd))[-len(PASTE_END) :]
        if tail == PASTE_END:
            return


def read_key(fd: int) -> str:
    r"""Read one keypress from `fd`, resolving whole escape sequences.

    Returns "" at EOF, ESC for a lone Esc, KEY_UP/KEY_DOWN for the arrows,
    PASTE_START for a bracketed paste (whose payload is consumed and thrown
    away), and whatever was read for anything else.

    Sequences are consumed to their final byte rather than to a fixed length:
    Home/End/F-keys and paste markers are longer than an arrow key
    ("\x1b[1~", "\x1b[200~"), and stopping early would leave their tail bytes
    in the queue to be read as separate keypresses.
    """
    ch = _read_byte(fd)
    if ch != ESC or not _has_pending(fd):
        # Nothing followed the Esc, so it's a keypress on its own. Reading
        # further here would block waiting for bytes that never arrive.
        return ch

    ch2 = _read_byte(fd)
    if ch2 not in ("[", "O") or not _has_pending(fd):
        return ESC + ch2
    if ch2 == "O":  # SS3: exactly one byte follows (F1-F4 on many terminals)
        return ESC + ch2 + _read_byte(fd)

    seq = ESC + ch2  # CSI: parameter bytes, then a final byte in @-~
    while _has_pending(fd):
        byte = _read_byte(fd)
        seq += byte
        if "\x40" <= byte <= "\x7e":
            break
    if seq == PASTE_START:
        _swallow_paste(fd)
    return seq


def prompt_level_interactive() -> str | None:
    """Arrow-key selectable list, using raw terminal mode.

    Returns the chosen level, or None if the user cancelled (q, Esc,
    Ctrl-D, or EOF).
    """
    import termios
    import tty

    selected = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def render() -> None:
        # Raw mode turns off ONLCR, so "\n" is a bare line feed and leaves the
        # cursor in the current column. Every line must end "\r\n" or the list
        # walks diagonally across the screen on each redraw.
        lines = ["Bump level (↑↓ then Enter, or 1-3 to pick; q to cancel)"]
        for i, (level, desc) in enumerate(LEVELS):
            marker = "❯" if i == selected else " "
            lines.append(f"  {marker} {level:<8} {desc}")
        sys.stdout.write("".join(f"{line}\r\n" for line in lines))
        sys.stdout.flush()

    def clear(n: int) -> None:
        sys.stdout.write("\r")
        for _ in range(n):
            sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write(PASTE_MODE_ON)
        render()
        while True:
            selected, action = handle_key(read_key(fd), selected)
            if action == "ignore":
                continue
            clear(len(LEVELS) + 1)
            if action == "move":
                render()
                continue
            if action == "interrupt":
                raise KeyboardInterrupt  # the finally below restores the terminal
            if action == "cancel":
                return None  # menu already cleared: leave the screen as we found it
            render()  # a digit can pick a level the marker isn't on yet
            break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write(PASTE_MODE_OFF)
        sys.stdout.flush()
    print()
    return LEVELS[selected][0]


def ask(prompt: str) -> str | None:
    """`input()` with both ways out of a cooked-mode prompt handled.

    Returns None at EOF (Ctrl-D). On Ctrl-C it prints the newline the
    terminal doesn't, then re-raises for the top-level handler. Cooked mode
    echoes "^C" with no newline, so "Cancelled." would otherwise land glued
    to the prompt. The raw-mode picker needs none of this: it disables
    echo, and its last render already ends the line.
    """
    try:
        return input(prompt)
    except EOFError:
        print()
        return None
    except KeyboardInterrupt:
        print()
        raise


def prompt_level_numbered() -> str | None:
    """Fallback for non-TTY stdin/stdout: plain numbered prompt.

    Returns the chosen level, or None if the user cancelled with q/EOF.
    """
    print("Bump level:")
    for i, (level, desc) in enumerate(LEVELS, start=1):
        print(f"  {i}. {level} - {desc}")
    while True:
        choice = ask(f"Select 1-{len(LEVELS)} (or q to cancel): ")
        if choice is None:
            return None
        choice = choice.strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(LEVELS):
            return LEVELS[int(choice) - 1][0]
        print("Invalid choice, try again.")


def prompt_level() -> str | None:
    """Returns the chosen level, or None if the user cancelled.

    Cancelling is a return value rather than a `sys.exit`, so the broad
    `except` below can't swallow it, and it stays testable.
    """
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        try:
            return prompt_level_interactive()
        except Exception as exc:  # noqa: BLE001 - degrade to the numbered prompt on any
            # platform quirk (e.g. `import termios` failing on Windows) rather than
            # taking the script down. A cancel is None, not an exception, so it
            # returns above instead of falling through into a second prompt.
            print(f"interactive prompt unavailable ({exc}); using numbered prompt", file=sys.stderr)
    return prompt_level_numbered()


def prompt_summary() -> str | None:
    """Ask for the one-line summary. Returns None if the user cancelled.

    An empty summary re-prompts rather than cancelling: "q" is a perfectly
    good summary, so no printable string can mean "back out" here. Ctrl-C
    and Ctrl-D (EOFError) are the ways out, and the re-prompt says so.
    """
    prompt = "Summary (user-facing, one line): "
    while True:
        summary = ask(prompt)
        if summary is None:
            return None
        summary = summary.strip()
        if summary:
            return summary
        prompt = "Summary can't be empty (Ctrl-C to cancel): "


def main() -> int:
    level = prompt_level()
    if level is None:
        print("Cancelled.")
        return 0

    summary = prompt_summary()
    if summary is None:
        print("Cancelled.")
        return 0

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_path = NOTES_DIR / f"+{secrets.token_hex(4)}.{level}.md"
    note_path.write_text(summary + "\n", encoding="utf-8")

    print(f"Created {note_path}")
    print("Commit it with your change.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl-C at any prompt: no traceback, and 130 is the conventional
        # exit code for dying to SIGINT.
        print("Cancelled.")
        sys.exit(130)
