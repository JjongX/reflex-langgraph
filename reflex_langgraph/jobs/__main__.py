"""
The detached job process: ``python -m reflex_langgraph.jobs <job_dir>``.

Started by :meth:`JobRunner.submit`. Everything it runs was frozen into ``state.json`` at
submit time, so it imports no user code.

Cancel sends SIGTERM to this worker, which owns the command: before the command starts it
stops right there; once it has started, the worker terminates the command's process group
(the command runs in its own session) and records the job as cancelled. There is no window
in which a cancel can miss a command that is just being launched.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from ._files import fingerprint, read_json, write_json
from .runner import MANIFEST, STATE, _now, _record_error, _update


def _versions(commands: dict, env: dict) -> dict:
    found = {}
    for name, cmd in commands.items():
        try:
            out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
            lines = (out.stdout or out.stderr).strip().splitlines()
            found[name] = lines[0] if lines else f"exit {out.returncode}, no output"
        except (OSError, subprocess.TimeoutExpired) as exc:
            found[name] = f"unavailable ({type(exc).__name__})"
    return found


class _Cancelled(Exception):
    pass


def main(job_dir: str) -> int:
    """
    Run a submitted job's command until it ends, recording its state.

    Parameters
    ----------
    job_dir : str
        The job's directory, holding the plan frozen at submission.

    Returns
    -------
    int
        The command's exit code; 143 when cancelled before it started.
    """
    job_dir = Path(job_dir)
    state = read_json(job_dir / STATE)
    plan = state["plan"]
    env = {**os.environ, **plan.get("env", {})}
    manifest_path = job_dir / MANIFEST
    launch = {"started": False, "proc": None, "cancel": False}

    def on_term(signum, frame):
        launch["cancel"] = True
        if not launch["started"]:
            raise _Cancelled()  # still fingerprinting: nothing to stop but ourselves
        if launch["proc"] is not None:
            _stop(launch["proc"])
        # else: Popen is returning right now; checked again just after it

    signal.signal(signal.SIGTERM, on_term)
    try:
        _update(job_dir, status="fingerprinting", worker_pid=os.getpid())
        manifest = read_json(manifest_path)
        manifest["fingerprints"] = {label: fingerprint(Path(p))
                                    for label, p in plan.get("fingerprint", {}).items()}
        manifest["versions"] = _versions(plan.get("version_commands", {}), env)
        write_json(manifest_path, manifest)

        if read_json(job_dir / STATE).get("cancel_requested"):
            raise _Cancelled()

        with open(job_dir / plan["log_name"], "ab") as log:
            launch["started"] = True
            proc = subprocess.Popen(plan["command"], stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, cwd=str(job_dir), env=env,
                                    start_new_session=True)
            launch["proc"] = proc
            if launch["cancel"] or read_json(job_dir / STATE).get("cancel_requested"):
                _stop(proc)  # cancelled while launching
            _update(job_dir, status="running", command_pid=proc.pid, started_at=_now())
            rc = proc.wait()

        cancelled = launch["cancel"] or read_json(job_dir / STATE).get("cancel_requested")
        _update(job_dir, status="cancelled" if cancelled else ("succeeded" if rc == 0 else "failed"),
                returncode=rc, finished_at=_now())
        return rc
    except _Cancelled:
        _update(job_dir, status="cancelled", finished_at=_now())
        return 143
    except Exception as exc:
        ref = _record_error(job_dir, exc)  # the detail stays on the server
        _update(job_dir, status="error", error=f"the job runner failed (ref {ref})",
                finished_at=_now())
        raise


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
