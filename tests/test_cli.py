import py_compile

import httpx
import pytest

from reflex_langgraph import JobRunner, __version__, jobs_router
from reflex_langgraph.agents import ToolRegistry, build_graph
from reflex_langgraph.__main__ import build_parser, main
from reflex_langgraph.cli import commands
from reflex_langgraph.launcher import SettingsError, page_settings, write_project
from reflex_langgraph.config import load_config
from reflex_langgraph.server import create_server
from reflex_langgraph.server.schemas import API_VERSION

pytestmark = pytest.mark.anyio


def _graph():
    return build_graph(load_config({"model": "x:y", "agents": [{"name": "a"}]}), ToolRegistry(),
                       models={"a": object()})


async def _health(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return (await c.get("/health")).json()


async def test_health_declares_auth_mode_and_jobs(tmp_path):
    plain = await _health(create_server(_graph(), title="Plain"))
    assert plain == {"ok": True, "api": API_VERSION, "version": __version__, "title": "Plain",
                     "auth": "none", "jobs": False, "approvals": True, "files": False,
                     "max_file_mb": 0, "max_files": 0, "accept": {}}

    full = await _health(create_server(_graph(), title="Full",
                                       routers=[jobs_router(JobRunner(tmp_path, []))]))
    assert (full["auth"], full["jobs"]) == ("none", True)

    # A hand-made /jobs router is not mistaken for the jobs capability.
    from fastapi import APIRouter
    other = APIRouter(prefix="/jobs")
    other.add_api_route("", lambda: [], methods=["GET"])
    custom = await _health(create_server(_graph(), auth=lambda r: {"id": "x"}, routers=[other]))
    assert (custom["auth"], custom["jobs"]) == ("external", False)


def _settings(argv, health):
    opts = commands["ui"].options(build_parser().parse_args(["ui", "--no-env-file", *argv]))
    return page_settings({"api": API_VERSION, **health}, title=opts["title"], jobs=opts["jobs"])


def test_flags_override_backend_settings(monkeypatch):
    monkeypatch.delenv("RLG_UI_TITLE", raising=False)
    health = {"title": "From backend", "auth": "none", "jobs": True}
    auto = _settings([], health)
    assert (auto["title"], auto["jobs"]) == ("From backend", True)
    forced = _settings(["--title", "Mine", "--jobs", "off"], health)
    assert (forced["title"], forced["jobs"]) == ("Mine", False)
    assert "login" not in auto
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ui", "--jobs", "maybe"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ui", "--login", "off"])   # there is no login


def test_ui_refuses_backends_with_auth_and_api_mismatch():
    with pytest.raises(SettingsError, match="no login"):
        _settings([], {"auth": "external"})
    with pytest.raises(SettingsError, match="API version"):
        page_settings({"api": 99, "auth": "none"})


def test_every_command_is_listed_with_its_help(capsys):
    with pytest.raises(SystemExit):
        main(["-h"])
    out = capsys.readouterr().out
    assert "init-env     Create a .env file from a .env.example file." in out
    assert "ui           Run the chat UI for a running backend." in out
    with pytest.raises(SystemExit):
        main(["ui", "-h"])
    out = capsys.readouterr().out
    assert "Optional arguments:" in out and "[Example] To run the UI" in out


def test_generated_project_compiles(tmp_path):
    write_project(tmp_path)
    write_project(tmp_path)  # rewriting is safe
    for f in (tmp_path / "rxconfig.py", tmp_path / "rlg_ui" / "rlg_ui.py"):
        py_compile.compile(str(f), doraise=True)


def test_ui_refuses_when_backend_is_down(tmp_path, capsys):
    assert main(["ui", "--no-env-file", "--api", "http://127.0.0.1:9",
                 "--workdir", str(tmp_path)]) == 2
    assert "cannot reach the backend" in capsys.readouterr().err
    assert not (tmp_path / "rxconfig.py").exists()
