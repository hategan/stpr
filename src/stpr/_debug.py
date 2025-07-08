import time
from enum import Enum


DEBUG = False


_TS = None


def _ts():
    return time.time() - _TS


class Color(Enum):
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    GRAY = 8
    BRIGHT_BLACK = 60
    BRIGHT_RED = 61
    BRIGHT_GREEN = 62
    BRIGHT_YELLOW = 63
    BRIGHT_BLUE = 64
    BRIGHT_MAGENTA = 65
    BRIGHT_CYAN = 66
    BRIGHT_WHITE = 67


def debug_print(msg: str, color: Color = None, background: Color = None) -> None:
    if not DEBUG:
        return

    _print(msg, color=color, background=background)


def _print(msg: str, color: Color = None, background: Color = None) -> None:
    cstr1 = ''
    cstr2 = ''
    if color:
        cstr1 += f'\033[{30 + color.value}m'
    if background:
        cstr1 += f'\033[{40 + background.value}m'
    if color or background:
        cstr2 = '\033[0m'
    print(f'{cstr1}{msg}{cstr2}')
