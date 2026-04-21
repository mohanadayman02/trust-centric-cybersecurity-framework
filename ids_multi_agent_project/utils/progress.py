"""Lightweight terminal progress logging helpers."""

from __future__ import annotations

import sys

_RESET = "\033[0m"
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"


def _use_color() -> bool:
    """Return True when ANSI colors are likely supported."""
    return sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    """Colorize text when terminal supports ANSI colors."""
    if not _use_color():
        return text
    return f"{color}{text}{_RESET}"


def log_section(title: str) -> None:
    """Print a large section header."""
    line = "=" * 57
    print(f"\n{line}")
    print(title)
    print(line)


def log_step(message: str) -> None:
    """Print a primary step line."""
    print(_colorize(f"[STEP] {message}", _BLUE))


def log_substep(message: str) -> None:
    """Print a nested sub-step line."""
    print(f"   ↳ {message}...")


def log_success(message: str) -> None:
    """Print a success line."""
    print(_colorize(f"   ✔ {message}", _GREEN))


def log_warning(message: str) -> None:
    """Print a warning line."""
    print(_colorize(f"   ⚠ {message}", _YELLOW))
