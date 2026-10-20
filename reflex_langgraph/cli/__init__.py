"""
The cli submodule holds the ``reflex-langgraph`` commands, one module per command.

Each command module defines ``description``, ``epilog`` (examples),
``create_parser(subparsers)`` and ``main(args)``. :data:`commands` maps each command's
name (its module name, with dashes) to its module; a new module is picked up
automatically.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Optional


def add_parser(subparsers, module: str, *, description: str, help: str,
               epilog: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Add a command's parser, formatted like every other command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Where the commands are registered.
    module : str
        The command module's ``__name__``; the command is its last part, with dashes.
    description : str
        Shown by ``reflex-langgraph <command> -h``.
    help : str
        One line, shown in the list of commands.
    epilog : str, optional
        Examples, shown after the options.

    Returns
    -------
    argparse.ArgumentParser
        The command's parser.
    """
    parser = subparsers.add_parser(
        module.rsplit(".", 1)[-1].replace("_", "-"),
        description=description.strip(),
        epilog=epilog.strip() if epilog else None,
        help=help,
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser._positionals.title = "Positional arguments"
    parser._optionals.title = "Optional arguments"
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    return parser


#: Command name -> command module, e.g. ``{"init-env": ..., "ui": ...}``.
commands = {}

for _file in sorted(Path(__file__).parent.glob("*.py")):
    if not _file.stem.startswith("_"):
        commands[_file.stem.replace("_", "-")] = import_module(f".{_file.stem}", __package__)
