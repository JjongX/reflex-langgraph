"""
The env submodule reads and creates ``.env`` files: one place for an app's API keys and
settings.

A ``.env`` file looks like this::

    # .env  (git-ignored; commit a .env.example without real values)
    GOOGLE_API_KEY=...          # the app's own settings: name them after the app
    MYAPP_DATA_DIR=data
    RLG_UI_TITLE=My agents      # read by reflex-langgraph ui

``reflex-langgraph ui`` reads ``./.env`` by default -- only ``RLG_API_URL`` and ``RLG_UI_*``,
so an app's API keys in the same file never reach the UI process. A backend calls
:func:`load_env_file` before reading its settings. Variables already set in the environment
win, so a one-off ``FOO=bar uvicorn ...`` still overrides the file. ``reflex-langgraph init-env`` creates the
file from the example, readable by its owner only.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env(text: str) -> Dict[str, str]:
    """
    Parse the text of a .env file.

    Lines are ``KEY=value``, optionally after ``export``. Blank lines and ``#`` comments
    are skipped. An unquoted value ends at `` #``; a quoted value is taken as written,
    without its quotes.

    Parameters
    ----------
    text : str
        Text of the file.

    Returns
    -------
    dict
        Variable name to value.

    Raises
    ------
    ValueError
        If a line is not a ``KEY=value`` line.

    Examples
    --------

    >>> from reflex_langgraph.env import parse_env
    >>> parse_env('''
    ... # Gemini
    ... export GOOGLE_API_KEY=abc
    ... RLG_UI_TITLE="My agents"
    ... RLG_UI_PORT=3000   # the default
    ... ''')
    {'GOOGLE_API_KEY': 'abc', 'RLG_UI_TITLE': 'My agents', 'RLG_UI_PORT': '3000'}
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            raise ValueError(f"not a KEY=value line: {raw.strip()!r}")
        key, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        out[key] = value
    return out


def load_env_file(path: Union[str, Path], *, override: bool = False,
                  prefix: Union[str, Tuple[str, ...], None] = None) -> Dict[str, str]:
    """
    Load a .env file into the environment.

    Variables already set in the environment win over the file, so a one-off
    ``FOO=bar uvicorn ...`` still overrides it.

    Parameters
    ----------
    path : str or pathlib.Path
        The .env file. A missing file loads nothing.
    override : bool, default: False
        Replace variables that are already set.
    prefix : str or tuple of str, optional
        Load only the variables whose names start with it (or with one of them).

    Returns
    -------
    dict
        The variables taken from the file, including those not set because they were
        already set.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    values = parse_env(path.read_text())
    if prefix is not None:
        values = {k: v for k, v in values.items() if k.startswith(prefix)}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def init_env_file(example: Union[str, Path], out: Union[str, Path]) -> Path:
    """
    Create a .env file as a copy of its example.

    The new file is readable by its owner only (mode 600), since it is where API keys go.

    Parameters
    ----------
    example : str or pathlib.Path
        The template, e.g. ``.env.example``.
    out : str or pathlib.Path
        The file to create, e.g. ``.env``.

    Returns
    -------
    pathlib.Path
        The new file.

    Raises
    ------
    FileExistsError
        If ``out`` already exists: it is never overwritten.
    FileNotFoundError
        If ``example`` does not exist.
    ValueError
        If ``example`` is not a valid .env file (nothing is created then).
    """
    example, out = Path(example), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; edit it, or delete it first")
    text = example.read_text()
    parse_env(text)  # refuse a malformed example before creating anything
    # Create with 0600 from the start, so keys pasted in later are never readable by others.
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return out


def env_path(name: str, default: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Resolve a path setting.

    A relative path is taken from ``base`` (usually the folder of the .env file), not
    from wherever the server was started.

    Parameters
    ----------
    name : str
        Environment variable holding the path.
    default : str or pathlib.Path
        Used when the variable is unset or empty.
    base : str or pathlib.Path
        Folder that relative paths are relative to.

    Returns
    -------
    pathlib.Path
        The absolute path.

    Examples
    --------

    >>> from reflex_langgraph import env_path
    >>> str(env_path('MYAPP_UNSET_DATA_DIR', 'data', '/srv/myapp'))
    '/srv/myapp/data'
    """
    value: Optional[str] = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default)
    return path if path.is_absolute() else (Path(base) / path).resolve()
