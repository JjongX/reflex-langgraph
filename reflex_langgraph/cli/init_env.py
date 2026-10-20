import sys

from ..env import init_env_file
from . import add_parser

description = """
Create a .env file from a .env.example file.

The new file is readable by you only (mode 600), since it is where API keys go,
and an existing file is never overwritten. Fill in the values left empty.
"""

epilog = """
[Example] To create ./.env from ./.env.example:
  $ reflex-langgraph init-env

[Example] To create the file somewhere else:
  $ reflex-langgraph init-env \\
  --example backend/.env.example \\
  --out backend/.env
"""


def create_parser(subparsers):
    parser = add_parser(
        subparsers,
        __name__,
        description=description,
        epilog=epilog,
        help=
"""Create a .env file from a .env.example file."""
    )
    parser.add_argument(
        "--example",
        metavar="PATH",
        default=".env.example",
        help=
"""Template to copy (default: .env.example)."""
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=".env",
        help=
"""File to create (default: .env)."""
    )


def main(args):
    try:
        out = init_env_file(args.example, args.out)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {out} (readable by you only). Fill in the values left empty; never commit it.")
    return 0
