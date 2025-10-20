# -*- coding: UTF-8 -*-
"""Tools for rendering textual progress bars in the terminal."""

import os
import shutil
import sys
from typing import Optional, Sequence, Tuple, Union


class ShowProcess(object):
    """Render a simple textual progress bar with optional metadata."""

    DEFAULT_BAR_WIDTH = 10
    MIN_BAR_WIDTH = 10
    MAX_BAR_WIDTH = 10
    FILL_CHAR = '#'
    EMPTY_CHAR = '-'
    POINTER_CHAR = '>'
    SPINNER_FRAMES: Sequence[str] = ('|', '/', '-', '\\')
    # ANSI styles
    _RESET = '\033[0m'
    _STYLE = {
        'info': '\033[1;35m',          # bold magenta
        'spinner': '\033[1;33m',       # bold yellow
        'bar_fill': '\033[1;34m',      # bold blue
        'bar_empty': '\033[90m',       # bright black (dim)
        'bar_pointer': '\033[1;36m',   # bold cyan
        'count': '\033[1;37m',         # bold white
        'percent': '\033[1;36m',       # bold cyan
        'sep': '\033[90m',             # dim separator
        'label': '\033[34m',           # blue labels (eta/step)
        'value': '\033[97m',           # bright white values
    }

    def __init__(self, max_steps: int, info: str = '', info_done: str = 'Done',
                 bar_width: Optional[int] = None) -> None:
        super().__init__()
        self.__info = info
        self.__max_steps = max(max_steps, 1)
        self.__target_steps = max_steps
        self.__counter = 0
        self.__info_done = info_done
        self.__spinner_index = 0
        self.__custom_bar_width = bar_width is not None
        self.__term_columns = None
        self.__bar_width = self.__resolve_bar_width(bar_width)
        self.__use_color = self.__detect_color_support()

    def __resolve_bar_width(self, requested_width: Optional[int]) -> int:
        columns = self.__detect_columns()
        self.__term_columns = columns
        if requested_width is not None:
            return max(self.MIN_BAR_WIDTH, min(self.MAX_BAR_WIDTH, requested_width))

        return self.DEFAULT_BAR_WIDTH

    def __refresh_layout(self) -> None:
        columns = self.__detect_columns()
        self.__term_columns = columns
        if not self.__custom_bar_width:
            self.__bar_width = self.DEFAULT_BAR_WIDTH

    def __detect_color_support(self) -> bool:
        # Allow forcing via env: JF_PROGRESS_COLOR=0/1; respect NO_COLOR
        env = os.environ
        if env.get('JF_PROGRESS_COLOR') == '0' or env.get('NO_COLOR'):
            return False
        if env.get('JF_PROGRESS_COLOR') == '1':
            return True
        try:
            if hasattr(sys, 'stdout') and sys.stdout and sys.stdout.isatty():
                return True
            # Fallback to original stdout, useful if fd 1 is temporarily piped
            if hasattr(sys, '__stdout__') and sys.__stdout__ and sys.__stdout__.isatty():
                return True
        except Exception:
            pass
        # If a pre-filter captured TTY state, trust it
        if os.environ.get('JF_STDOUT_WAS_TTY') == '1':
            return True
        return False

    def __fmt(self, text: str, style_key: Optional[str]) -> str:
        if not self.__use_color or not style_key:
            return text
        style = self._STYLE.get(style_key)
        if not style:
            return text
        return f"{style}{text}{self._RESET}"

    def __count(self, start_counter: Optional[int]) -> None:
        if start_counter is not None:
            self.__counter = max(0, start_counter)
        else:
            self.__counter += 1
        self.__counter = min(self.__counter, self.__max_steps)

    def __cal_bar(self) -> Tuple[int, int, float]:
        ratio = self.__counter / float(self.__max_steps)
        num_filled = int(ratio * self.__bar_width)
        percent = ratio * 100.0
        return num_filled, self.__bar_width - num_filled, percent

    def __generate_info_done(self) -> str:
        if self.__target_steps <= 0:
            return ''
        return f' | {self.__info_done}' if self.__counter >= self.__target_steps else ''

    def __build_bar(self, num_filled: int, num_empty: int) -> str:
        if num_filled == 0:
            pointer = ''
        elif num_filled == self.__bar_width:
            pointer = ''
        else:
            pointer = self.POINTER_CHAR
            num_filled -= 1
        return f"{self.FILL_CHAR * num_filled}{pointer}{self.EMPTY_CHAR * num_empty}"

    def __next_spinner(self) -> str:
        frame = self.SPINNER_FRAMES[self.__spinner_index % len(self.SPINNER_FRAMES)]
        self.__spinner_index += 1
        return frame

    def show_process(self, start_counter: Optional[int] = None, show_info: str = '',
                     rest_time: Union[str, float] = '', duration: Union[str, float] = '',
                     queue_size: Union[str, int] = '') -> None:
        self.__refresh_layout()
        self.__count(start_counter)
        num_filled, num_empty, percent = self.__cal_bar()
        info_done = self.__generate_info_done()
        spinner = self.__next_spinner()

        # Build uncoloured components for correct width calculation
        uncolored_bar = self.__build_bar(num_filled, num_empty)
        uncolored_prefix = (f"[{self.__info}] {spinner} |{uncolored_bar}| "
                            f"{self.__counter}/{self.__max_steps} {percent:.1f}%")
        detail_segments = self.__compose_detail_segments(show_info, queue_size, rest_time, duration)
        detail_uncolored = self.__format_detail(detail_segments)
        detail_uncolored = self.__clip_detail(detail_uncolored, uncolored_prefix, info_done)

        # Apply colour to each segment when enabled
        colored_spinner = self.__fmt(spinner, 'spinner')
        pointer_present = (num_filled != 0 and num_filled != self.__bar_width)
        fill_count = max(num_filled - (1 if pointer_present else 0), 0)
        fill_str = self.FILL_CHAR * fill_count
        empty_str = self.EMPTY_CHAR * num_empty
        colored_bar = (
            self.__fmt(fill_str, 'bar_fill') +
            (self.__fmt(self.POINTER_CHAR, 'bar_pointer') if pointer_present else '') +
            self.__fmt(empty_str, 'bar_empty')
        )

        # Colour counts and percent
        colored_count = self.__fmt(f"{self.__counter}", 'count')
        colored_max = self.__fmt(f"{self.__max_steps}", 'count')
        colored_percent = self.__fmt(f"{percent:.1f}%", 'percent')
        colored_info = self.__fmt(f"[{self.__info}]", 'info')
        sep_bar = self.__fmt('|', 'sep')

        # Colour detail: colon labels (eta/step/queue) and separators
        detail_colored = detail_uncolored
        if self.__use_color and detail_uncolored:
            # Replace separators
            detail_colored = detail_colored.replace(' :: ', f" {self.__fmt('::', 'sep')} ")
            detail_colored = detail_colored.replace(' | ', f" {self.__fmt('|', 'sep')} ")
            # Highlight common labels
            for lbl in ('eta:', 'step:', 'queue:'):
                detail_colored = detail_colored.replace(lbl, self.__fmt(lbl, 'label'))

        base_prefix_colored = (f"{colored_info} {colored_spinner} {sep_bar}{colored_bar}{sep_bar} "
                               f"{colored_count}/{colored_max} {colored_percent}")
        process_str = f"{base_prefix_colored}{detail_colored}{info_done}    \r"
        self.__print(process_str)

    def close(self) -> None:
        self.__print('\n')
        self.__counter = 0

    def check_finish(self) -> None:
        if self.__counter >= self.__target_steps:
            self.close()

    @staticmethod
    def __format_optional_seconds(value: Union[str, float]) -> Optional[str]:
        if value in ('', None):
            return None
        if isinstance(value, str):
            return value

        seconds = max(float(value), 0.0)
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if seconds >= 60:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02d}m {secs:04.1f}s"
        if seconds >= 1:
            return f"{seconds:0.2f}s"
        return f"{seconds * 1000:0.0f}ms"

    @staticmethod
    def __compose_detail_segments(show_info: str, queue_size: Union[str, int],
                                  rest_time: Union[str, float],
                                  duration: Union[str, float]) -> Tuple[str, ...]:
        details = []
        if show_info:
            details.append(show_info.strip())

        if queue_size not in ('', None):
            details.append(f"queue: {queue_size}")

        rest_time_str = ShowProcess.__format_optional_seconds(rest_time)
        if rest_time_str:
            details.append(f"eta: {rest_time_str}")

        duration_str = ShowProcess.__format_optional_seconds(duration)
        if duration_str:
            details.append(f"step: {duration_str}")

        return tuple(details)

    @staticmethod
    def __format_detail(segments: Tuple[str, ...]) -> str:
        if not segments:
            return ''
        normalized = []
        for segment in segments:
            stripped = segment.strip()
            if not stripped:
                continue
            normalized.append(' '.join(stripped.split()))
        if not normalized:
            return ''
        return ' :: ' + ' | '.join(normalized)

    def __clip_detail(self, detail: str, base_prefix: str, info_done: str) -> str:
        if not detail or self.__term_columns is None:
            return detail

        available = self.__term_columns - len(base_prefix) - len(info_done) - 4
        if available <= len(' :: '):
            return ''

        if len(detail) <= available:
            return detail

        trimmed = detail[:max(available - 3, len(' :: '))]
        if len(trimmed) <= len(' :: '):
            return ''
        trimmed = trimmed.rstrip()
        if len(trimmed) + len('...') <= available:
            return trimmed + '...'
        final = trimmed[:max(available - len('...'), len(' :: '))].rstrip()
        if len(final) <= len(' :: '):
            return ''
        return final + '...'

    @staticmethod
    def __detect_columns() -> int:
        fallback = shutil.get_terminal_size((80, 20)).columns
        for stream in (sys.__stdout__, sys.__stderr__, sys.__stdin__):
            if stream is None:
                continue
            try:
                size = os.get_terminal_size(stream.fileno())
            except (OSError, ValueError):
                continue
            fallback = max(fallback, size.columns)
        env_override = (os.environ.get('JF_PROGRESS_COLUMNS') or
                        os.environ.get('COLUMNS'))
        if env_override:
            try:
                override_value = int(env_override)
                fallback = max(override_value, fallback)
            except ValueError:
                pass
        return fallback

    @staticmethod
    def __print(process_str: str) -> None:
        sys.stdout.write(process_str)
        sys.stdout.flush()
