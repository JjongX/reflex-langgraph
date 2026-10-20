"""
The ``reflex-langgraph`` command line (also ``python -m reflex_langgraph``).

Each command lives in its own module under :mod:`reflex_langgraph.cli`.
"""

import argparse
import sys

from .cli import commands
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    """
    Build the ``reflex-langgraph`` parser with every command.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="reflex-langgraph",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
    )
    for command in commands.values():
        command.create_parser(subparsers)
    return parser


def main(argv=None) -> int:
    """
    Run a ``reflex-langgraph`` command.

    Parameters
    ----------
    argv : list of str, optional
        The arguments, without the program name. By default, ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status.
    """
    args = build_parser().parse_args(argv)
    return commands[args.command].main(args)


if __name__ == "__main__":
    sys.exit(main())
