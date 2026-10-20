import os
import sys

import httpx

from .. import launcher
from . import add_parser

description = """
Run the chat UI for a running backend.

The UI needs no frontend code: the page is built from the backend's /health
(its title, and whether it has jobs) and runs as a small generated Reflex
project. It has no login, so anyone who can open it can use it: keep it
private (an SSH tunnel, or a firewall open to known addresses only).

Each setting comes from its flag, else from its environment variable
(RLG_API_URL, RLG_UI_*), else from its default. The variables are also read
from ./.env, but only those names: the backend's API keys in the same file
never reach the UI process.
"""

epilog = """
[Example] To run the UI for a backend on this machine, as the production
build (to try it or show it; the page loads in a fraction of a second):
  $ reflex-langgraph ui --prod --port 3000 --backend-port 3000

[Example] To run it on Reflex's development server, while changing UI code
(the first page load takes seconds):
  $ reflex-langgraph ui

[Example] To run it for another backend, with its own title:
  $ reflex-langgraph ui \\
  --api http://10.0.0.5:8100 \\
  --title "Lab assistant"
"""


def _setting(value, name: str, default=None):
    """The flag's value, else the environment variable ``name``, else ``default``."""
    if value is not None:
        return value
    return os.environ.get(name) or default


def create_parser(subparsers):
    parser = add_parser(
        subparsers,
        __name__,
        description=description,
        epilog=epilog,
        help=
"""Run the chat UI for a running backend."""
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help=
"""File to read RLG_API_URL and RLG_UI_* from
(default: ./.env, if it exists)."""
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help=
"""Do not read a .env file."""
    )
    parser.add_argument(
        "--api",
        metavar="URL",
        help=
"""Backend URL, as seen from this machine
(default: $RLG_API_URL, or http://localhost:8100)."""
    )
    parser.add_argument(
        "--title",
        metavar="TEXT",
        help=
"""Page title (default: $RLG_UI_TITLE, or the
backend's title)."""
    )
    parser.add_argument(
        "--subtitle",
        metavar="TEXT",
        help=
"""Text under the title (default: $RLG_UI_SUBTITLE)."""
    )
    parser.add_argument(
        "--placeholder",
        metavar="TEXT",
        help=
"""Hint text in the message box
(default: $RLG_UI_PLACEHOLDER)."""
    )
    parser.add_argument(
        "--greeting",
        metavar="TEXT",
        help=
"""Shown above the message box in an empty chat
(default: $RLG_UI_GREETING, or "What can I help
with?")."""
    )
    parser.add_argument(
        "--jobs",
        choices=["auto", "on", "off"],
        help=
"""Show the jobs panel (default: auto, which shows it
when the backend has jobs)."""
    )
    parser.add_argument(
        "--accent",
        metavar="COLOR",
        help=
"""Radix accent colour (default: $RLG_UI_ACCENT, or
indigo)."""
    )
    parser.add_argument(
        "--port",
        type=int,
        metavar="INT",
        help=
"""Frontend port (default: $RLG_UI_PORT, or 3000)."""
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        metavar="INT",
        help=
"""Port of Reflex's own backend
(default: $RLG_UI_BACKEND_PORT, or 8000)."""
    )
    parser.add_argument(
        "--backend-host",
        metavar="HOST",
        help=
"""Interface Reflex's backend binds to."""
    )
    parser.add_argument(
        "--public-url",
        metavar="URL",
        help=
"""URL the browser uses to reach Reflex's backend,
when it is not localhost, e.g.
http://<public-ip>:8000 (default: $RLG_UI_PUBLIC_URL)."""
    )
    parser.add_argument(
        "--single-port",
        action="store_true",
        help=
"""Only check that --port and --backend-port match:
a production build always serves the page and
Reflex's backend on one port."""
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help=
"""Run a production build instead of the dev server:
use it to try or show the UI. Give --port and
--backend-port the same number."""
    )
    parser.add_argument(
        "--workdir",
        metavar="PATH",
        help=
"""Directory of the generated Reflex project
(default: ~/.cache/reflex-langgraph/ui)."""
    )


def options(args) -> dict:
    """Every setting, resolved: the flag, else the environment (after reading the UI's
    lines of the .env file), else the default. Raises FileNotFoundError for a missing
    ``--env-file``."""
    if not args.no_env_file:
        path = args.env_file or ".env"
        taken = launcher.load_ui_env(path, required=args.env_file is not None)
        if taken:
            print(f"loaded {len(taken)} UI setting(s) (RLG_API_URL, RLG_UI_*) from {path}",
                  file=sys.stderr)
    return {
        "api": _setting(args.api, "RLG_API_URL", launcher.DEFAULT_API),
        "title": _setting(args.title, "RLG_UI_TITLE"),
        "subtitle": _setting(args.subtitle, "RLG_UI_SUBTITLE"),
        "placeholder": _setting(args.placeholder, "RLG_UI_PLACEHOLDER"),
        "greeting": _setting(args.greeting, "RLG_UI_GREETING"),
        "jobs": {"auto": None, "on": True, "off": False}[args.jobs or "auto"],
        "accent": _setting(args.accent, "RLG_UI_ACCENT", "indigo"),
        "port": int(_setting(args.port, "RLG_UI_PORT", 3000)),
        "backend_port": int(_setting(args.backend_port, "RLG_UI_BACKEND_PORT", 8000)),
        "backend_host": args.backend_host,
        "public_url": _setting(args.public_url, "RLG_UI_PUBLIC_URL"),
        "single_port": args.single_port,
        "prod": args.prod,
        "workdir": args.workdir,
    }


def main(args):
    try:
        opts = options(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        import reflex  # noqa: F401
    except ImportError:
        print("reflex is not installed: pip install reflex-langgraph", file=sys.stderr)
        return 2
    api = opts["api"]
    try:
        health = launcher.fetch_health(api)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"cannot reach the backend at {api}/health ({exc}).\n"
              f"Start it first, or point --api at it.", file=sys.stderr)
        return 2
    try:
        settings = launcher.page_settings(
            health, title=opts["title"], subtitle=opts["subtitle"],
            placeholder=opts["placeholder"], greeting=opts["greeting"], jobs=opts["jobs"],
            accent=opts["accent"])
    except launcher.SettingsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return launcher.run_ui(
        api, settings, workdir=opts["workdir"], port=opts["port"],
        backend_port=opts["backend_port"], backend_host=opts["backend_host"],
        public_url=opts["public_url"], single_port=opts["single_port"], prod=opts["prod"])
