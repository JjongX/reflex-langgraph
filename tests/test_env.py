import os
import stat

import pytest

from reflex_langgraph.__main__ import build_parser, main
from reflex_langgraph.cli import commands
from reflex_langgraph.env import env_path, init_env_file, load_env_file, parse_env
from reflex_langgraph.launcher import load_ui_env


def ui_options(argv):
    return commands["ui"].options(build_parser().parse_args(["ui", *argv]))


def test_parse_env_forms():
    text = """
# comment
export A=1
B = two words   # trailing comment
C="quoted # not a comment"
D='single'
E=
"""
    assert parse_env(text) == {"A": "1", "B": "two words", "C": "quoted # not a comment",
                               "D": "single", "E": ""}
    with pytest.raises(ValueError):
        parse_env("not a pair")


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("RLG_T_X=from_file\nRLG_T_Y=from_file\n")
    monkeypatch.setenv("RLG_T_X", "from_shell")
    monkeypatch.delenv("RLG_T_Y", raising=False)
    assert load_env_file(f) == {"RLG_T_X": "from_file", "RLG_T_Y": "from_file"}
    assert os.environ["RLG_T_X"] == "from_shell" and os.environ["RLG_T_Y"] == "from_file"
    assert load_env_file(tmp_path / "missing") == {}


def test_init_env_copies_the_example_private_and_never_overwrites(tmp_path):
    example = tmp_path / ".env.example"
    example.write_text("# keys\nGOOGLE_API_KEY=\nKEEP=x")
    out = init_env_file(example, tmp_path / ".env")
    assert out.read_text() == "# keys\nGOOGLE_API_KEY=\nKEEP=x\n"
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        init_env_file(example, out)
    bad = tmp_path / "bad.example"
    bad.write_text("not a pair\n")
    with pytest.raises(ValueError):
        init_env_file(bad, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_env_path_is_relative_to_the_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RLG_T_DIR", "data")
    assert env_path("RLG_T_DIR", "x", tmp_path) == (tmp_path / "data").resolve()
    monkeypatch.setenv("RLG_T_DIR", "/abs/dir")
    assert env_path("RLG_T_DIR", "x", tmp_path) == env_path("RLG_T_DIR", "y", "/elsewhere")
    monkeypatch.delenv("RLG_T_DIR")
    assert env_path("RLG_T_DIR", "runs", tmp_path) == (tmp_path / "runs").resolve()


def test_ui_options_come_from_env_file_and_flags_win(tmp_path, monkeypatch):
    for k in ("RLG_UI_TITLE", "RLG_UI_ACCENT", "RLG_UI_PORT", "RLG_API_URL"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "ui.env").write_text("RLG_UI_TITLE=From file\nRLG_UI_ACCENT=teal\n"
                                     "RLG_UI_PORT=3100\nRLG_API_URL=http://api:9\n")
    opts = ui_options(["--env-file", str(tmp_path / "ui.env"), "--title", "Flag"])
    assert (opts["title"], opts["accent"], opts["port"], opts["api"]) == (
        "Flag", "teal", 3100, "http://api:9")


def test_ui_default_env_file_is_optional_but_a_named_one_must_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # no ./.env here: fine
    ui_options([])
    with pytest.raises(FileNotFoundError):
        ui_options(["--env-file", "nope.env"])
    assert main(["ui", "--env-file", "nope.env"]) == 2


def test_init_env_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("GOOGLE_API_KEY=\n")
    assert main(["init-env"]) == 0
    assert main(["init-env"]) == 1 and "already exists" in capsys.readouterr().err


def test_ui_loads_only_its_own_settings_from_a_shared_env_file(tmp_path, monkeypatch):
    names = ("GOOGLE_API_KEY", "APP_SECRET", "RLG_SECRET", "RLG_UI_TITLE", "RLG_API_URL")
    for k in names:
        monkeypatch.delenv(k, raising=False)
    shared = tmp_path / ".env"
    shared.write_text("GOOGLE_API_KEY=sk-real\nAPP_SECRET=s1\nRLG_SECRET=s2\n"
                      "RLG_UI_TITLE=Mine\nRLG_API_URL=http://api:1\n")
    load_ui_env(shared)
    # The UI hands this environment to the Reflex process: only the UI's own settings
    assert {k: os.environ.get(k) for k in names} == {
        "GOOGLE_API_KEY": None, "APP_SECRET": None, "RLG_SECRET": None,
        "RLG_UI_TITLE": "Mine", "RLG_API_URL": "http://api:1"}
