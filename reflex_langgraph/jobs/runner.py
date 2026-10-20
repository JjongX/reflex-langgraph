"""
The runner submodule runs jobs as detached processes, with a concurrency limit, cancel and
provenance.

Layout under ``root``::

    root/
      .jobs.guard                 flock target serialising submissions
      <job_id>/
        state.json                status, pids, timestamps, the frozen command
        state.json.lock           flock target serialising updates of state.json
        manifest.json             params, requester, approver, input/tool fingerprints
        error.txt                 only if preparing the job failed: the detail, for staff
        <spec.log_name>           the command's stdout/stderr

The command runs in its own session in a separate Python process (``python -m
reflex_langgraph.jobs``), so restarting the web server never stops a job, and cancel
signals the whole process group. A job directory is never reused.

The ``*_for(user, ...)`` methods apply the runner's
:class:`~reflex_langgraph.policy.AccessPolicy`; the HTTP routes and the agent tools use
only those, so both paths enforce the same rules.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, ValidationError

from ..policy import AccessPolicy, Principal
from ._files import read_json, update_json, write_json
from .spec import JobSpec, Preflight

log = logging.getLogger("reflex_langgraph.jobs")

#: Name of each job's state file (status, process ids, timestamps).
STATE = "state.json"
#: Name of each job's manifest (parameters, requester, approver, fingerprints).
MANIFEST = "manifest.json"
#: Statuses of a job that has not ended.
RUNNING = ("starting", "fingerprinting", "running")
#: How long a job may stay ``starting`` before its worker's pid is recorded (normally a few
#: milliseconds); after that, it died before it could start.
START_GRACE = timedelta(seconds=30)
#: A valid job id: letters, digits, ``.``, ``_`` and ``-``; at most 64 characters.
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class JobError(ValueError):
    """
    Raised when a job request is refused.

    The message is safe to show any user: it never names another user's jobs and never
    contains a server path.
    """


class JobForbidden(JobError):
    """Raised when a user may see a job but not do this to it."""


class JobNotRunning(JobError):
    """Raised when cancelling a job that is not running."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _since(stamp: Optional[str]) -> timedelta:
    """Time since an ISO 8601 stamp written by :func:`_now`; very long when unreadable."""
    try:
        return datetime.now(timezone.utc) - datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return timedelta.max


def pid_alive(pid: Optional[int]) -> bool:
    """
    Check whether a process is alive.

    Parameters
    ----------
    pid : int, optional
        Process id.

    Returns
    -------
    bool
        Whether the process exists (also when it belongs to another user).
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobRunner:
    """
    Run jobs as detached processes, each in a fresh directory under ``root``.

    The ``*_for(user, ...)`` methods apply the access ``policy``; the HTTP routes and the
    agent tools use only those.

    Parameters
    ----------
    root : str or pathlib.Path
        Folder that holds one directory per job.
    specs : iterable of JobSpec
        The kinds of job, with unique names.
    max_concurrent : int, default: 1
        How many jobs may run at once; more are refused (not queued).
    min_free_bytes : int, default: 0
        Disk to keep free beyond a job's own estimate; less is a preflight blocker.
    include_unmanaged : bool, default: False
        Also list directories under ``root`` that this runner did not create (read-only).
    hidden : iterable of str, default: ('archive',)
        Directory names under ``root`` that are never listed nor used as job ids.
    policy : AccessPolicy, optional
        Who may see and cancel which job; pass the server's. By default, everyone
        everything.

    Examples
    --------

    >>> import tempfile, time
    >>> from reflex_langgraph.jobs import JobRunner, JobSpec
    >>> class Hello(JobSpec):
    ...     name = 'hello'
    ...     def command(self, params, job_dir):
    ...         return ['echo', 'hello']
    >>> root = tempfile.TemporaryDirectory()
    >>> runner = JobRunner(root.name, [Hello()])
    >>> runner.submit('hello', 'job1', {}, approved_by='me')['job_id']
    'job1'
    >>> for _ in range(100):
    ...     if runner.summary('job1')['status'] not in ('starting', 'fingerprinting', 'running'):
    ...         break
    ...     time.sleep(0.1)
    >>> runner.summary('job1')['status'], runner.log_tail('job1')
    ('succeeded', 'hello')
    >>> root.cleanup()
    """

    def __init__(self, root: Path, specs: Iterable[JobSpec], *, max_concurrent: int = 1,
                 min_free_bytes: int = 0, include_unmanaged: bool = False,
                 hidden: Iterable[str] = ("archive",), policy: Optional[AccessPolicy] = None):
        self.root = Path(root).resolve()
        self.specs: Dict[str, JobSpec] = {}
        for s in specs:
            if not s.name or s.name in self.specs:
                raise ValueError(f"job specs need unique, non-empty names: {s.name!r}")
            self.specs[s.name] = s
        self.max_concurrent = max_concurrent
        self.min_free_bytes = min_free_bytes
        #: List directories under root that this runner did not create (read-only).
        self.include_unmanaged = include_unmanaged
        self.hidden = set(hidden)
        #: Who may see and cancel which job; shared with the server (see AccessPolicy).
        self.policy = policy or AccessPolicy()

    # --- lookup -------------------------------------------------------------------

    def spec(self, name: str) -> JobSpec:
        """
        Return the spec of a kind of job.

        Parameters
        ----------
        name : str
            The spec's name.

        Returns
        -------
        JobSpec
            The spec.

        Raises
        ------
        JobError
            If there is no such spec.
        """
        if name not in self.specs:
            raise JobError(f"unknown job type {name!r}; known: {sorted(self.specs)}")
        return self.specs[name]

    def job_dir(self, job_id: str, spec: Optional[JobSpec] = None) -> Path:
        """
        Return a job's directory, after checking the id.

        Parameters
        ----------
        job_id : str
            The job id.
        spec : JobSpec, optional
            Also refuse a directory in or under the spec's protected paths.

        Returns
        -------
        pathlib.Path
            ``root/<job_id>`` (it may not exist yet).

        Raises
        ------
        JobError
            If the id is invalid or reserved, or the directory would be protected.
        """
        if not JOB_ID_RE.match(job_id or ""):
            raise JobError(f"invalid job id {job_id!r}: use letters, digits, '.', '_', '-'")
        if job_id in self.hidden:
            raise JobError(f"job id {job_id!r} is reserved")
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise JobError(f"job id escapes the jobs root: {job_id!r}")
        for protected in (spec.protected_paths if spec else ()):
            protected = Path(protected).resolve()
            if path == protected or protected in path.parents:
                raise JobError("refusing to create a job under a protected directory")
        return path

    def _state(self, job_dir: Path) -> dict:
        return read_json(job_dir / STATE)

    def _alive(self, state: dict) -> bool:
        if state.get("status") not in RUNNING:
            return False
        worker, command = state.get("worker_pid"), state.get("command_pid")
        if not (worker or command):
            # submit() has written the plan and is starting the worker right now.
            return _since(state.get("updated_at")) < START_GRACE
        # The worker records the outcome only after the command exits, and a command can
        # outlive a killed worker: the job is alive while either process is.
        return pid_alive(worker) or pid_alive(command)

    def active(self) -> List[str]:
        """
        Return the ids of the jobs running now.

        Returns
        -------
        list of str
            The job ids.
        """
        if not self.root.is_dir():
            return []
        return [p.name for p in self.root.iterdir()
                if p.is_dir() and self._alive(self._state(p))]

    # --- preflight / submit ---------------------------------------------------------

    def parse_params(self, spec: JobSpec, params) -> BaseModel:
        """
        Validate a job's parameters.

        Parameters
        ----------
        spec : JobSpec
            The kind of job.
        params : dict or pydantic.BaseModel
            The parameters.

        Returns
        -------
        pydantic.BaseModel
            The parameters, as the spec's ``params_model``.

        Raises
        ------
        JobError
            If they are invalid.
        """
        if isinstance(params, spec.params_model):
            return params
        try:
            return spec.params_model.model_validate(params or {})
        except ValidationError as exc:
            raise JobError(f"invalid parameters: {exc.errors(include_url=False)}") from None

    def preflight(self, spec_name: str, params) -> Preflight:
        """
        Check whether a job could start now, without starting it.

        Adds the runner's own checks to the spec's: the concurrency limit and free disk.

        Parameters
        ----------
        spec_name : str
            The kind of job.
        params : dict or pydantic.BaseModel
            The job's parameters.

        Returns
        -------
        Preflight
            Blockers, warnings and facts.

        Raises
        ------
        JobError
            If the kind of job is unknown or the parameters are invalid.
        """
        spec = self.spec(spec_name)
        result = spec.preflight(self.parse_params(spec, params))
        busy = self.active()
        if len(busy) >= self.max_concurrent:
            # Counts only: the running jobs may belong to other users.
            result.blockers.append(
                f"the server is busy: {len(busy)} job(s) running, the limit is "
                f"{self.max_concurrent}; try again when one finishes")
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        need = result.required_bytes + self.min_free_bytes
        result.info.setdefault("free disk (GB)", round(free / 1e9, 1))
        if result.required_bytes:
            result.info.setdefault("estimated disk (GB)", round(result.required_bytes / 1e9, 1))
        if need and free < need:
            result.blockers.append(f"not enough disk: need ~{need / 1e9:.1f} GB, "
                                   f"{free / 1e9:.1f} GB free")
        return result

    def submit(self, spec_name: str, job_id: str, params, *, approved_by: str,
               requested_by: str = "") -> dict:
        """
        Start a job, detached.

        The preflight runs again first; any blocker refuses the job. The job's directory
        is created, the spec prepares it, and a separate process runs the command, so
        nothing here waits for the job.

        Parameters
        ----------
        spec_name : str
            The kind of job.
        job_id : str
            A new, unique id: letters, digits, ``.``, ``_`` and ``-``.
        params : dict or pydantic.BaseModel
            The job's parameters.
        approved_by : str
            Who approved it; recorded in the manifest. Required.
        requested_by : str, optional
            Who asked for it; recorded in the manifest.

        Returns
        -------
        dict
            ``job_id``, ``spec``, ``job_dir``, and the preflight's ``warnings`` and
            ``info``.

        Raises
        ------
        JobError
            If the job is refused: no approver, invalid parameters, an id that is taken,
            a preflight blocker, or a failure while preparing it.
        """
        spec = self.spec(spec_name)
        if not (approved_by or "").strip():
            raise JobError("a named approver is required")
        parsed = self.parse_params(spec, params)
        job_dir = self.job_dir(job_id, spec)
        self.root.mkdir(parents=True, exist_ok=True)

        with open(self.root / ".jobs.guard", "w") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if job_dir.exists():
                # Job ids are one namespace for all users: do not confirm that someone else's
                # job exists. (Directories are never reused either way.)
                raise JobError(f"job id {job_id!r} is not available; choose another")
            check = self.preflight(spec_name, parsed)
            if not check.ok:
                raise JobError("preflight failed: " + "; ".join(check.blockers))

            job_dir.mkdir()
            # Ownership first: whatever happens next, the requester can see this job.
            write_json(job_dir / MANIFEST, {
                "job_id": job_id, "spec": spec.name,
                "params": parsed.model_dump(mode="json"),
                "approved_by": approved_by.strip(), "requested_by": (requested_by or "").strip(),
                "submitted_at": _now(), "preflight": {"warnings": check.warnings,
                                                       "info": check.info},
            })
            try:
                spec.prepare(parsed, job_dir)
                command = spec.command(parsed, job_dir)
                plan = {
                    "command": command,
                    "env": spec.env(parsed, job_dir),
                    "log_name": spec.log_name,
                    "fingerprint": {k: str(v) for k, v in spec.fingerprint(parsed, job_dir).items()},
                    "version_commands": spec.version_commands(parsed, job_dir),
                }
            except Exception as exc:
                # The detail (often a server path) stays on the server: in error.txt and the
                # log. The state, which users see, gets a reference only.
                ref = _record_error(job_dir, exc)
                write_json(job_dir / STATE, {"status": "failed_to_start", "spec": spec.name,
                                             "error": f"could not be prepared (ref {ref})",
                                             "finished_at": _now(), "updated_at": _now()})
                log.exception("job %s could not be prepared [ref %s]", job_id, ref)
                raise JobError(f"job {job_id!r} could not be prepared (ref {ref}); the error "
                               "is recorded on the server") from exc

            write_json(job_dir / STATE, {"status": "starting", "spec": spec.name,
                                         "plan": plan, "updated_at": _now()})
            with open(job_dir / "worker.log", "ab") as wlog:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "reflex_langgraph.jobs", str(job_dir)],
                    stdin=subprocess.DEVNULL, stdout=wlog, stderr=subprocess.STDOUT,
                    start_new_session=True, cwd=str(job_dir), env=_child_env(),
                )
            _update(job_dir, worker_pid=worker.pid)
            # Reap the worker when it exits, or it lingers as a zombie of this process and
            # still looks alive to pid_alive(). (After a restart of this process, init reaps.)
            threading.Thread(target=worker.wait, name=f"reap-{job_id}", daemon=True).start()

        return {"job_id": job_id, "spec": spec.name, "job_dir": str(job_dir),
                "warnings": check.warnings, "info": check.info}

    # --- cancel ---------------------------------------------------------------------

    def cancel(self, job_id: str, *, requested_by: str) -> dict:
        """
        Stop a running job. Its directory is kept.

        Parameters
        ----------
        job_id : str
            The job id.
        requested_by : str
            Who cancelled it; recorded in its state.

        Returns
        -------
        dict
            ``job_id`` and ``signalled``.

        Raises
        ------
        JobNotRunning
            If the job is not running.
        """
        job_dir = self.job_dir(job_id)
        state = self._state(job_dir)
        if not self._alive(state):
            raise JobNotRunning(f"job {job_id!r} is not running")
        _update(job_dir, cancel_requested=True, cancelled_by=requested_by, cancelled_at=_now())
        # The worker owns the command and stops it at whatever stage it is in (see
        # jobs/__main__.py). Only if the worker is gone do we signal the command directly.
        worker, command = state.get("worker_pid"), state.get("command_pid")
        try:
            if pid_alive(worker):
                os.kill(worker, signal.SIGTERM)
            elif pid_alive(command):
                os.killpg(command, signal.SIGTERM)  # the command runs in its own session
        except ProcessLookupError:
            pass  # it ended in between
        return {"job_id": job_id, "signalled": True}

    # --- views ----------------------------------------------------------------------

    def list(self) -> List[dict]:
        """
        List every job, with no access check (see :meth:`list_for`).

        Returns
        -------
        list of dict
            Per job: ``job_id``, ``spec``, ``managed``, ``requested_by``, ``approved_by``,
            ``status``, ``submitted_at`` and ``finished_at``.
        """
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name in self.hidden or path.name.startswith("."):
                continue
            state = self._state(path)
            if not state and not self.include_unmanaged:
                continue
            manifest = read_json(path / MANIFEST)
            out.append({**self._ref(path.name, state, manifest),
                        "status": self._status(state), "submitted_at": manifest.get("submitted_at"),
                        "finished_at": state.get("finished_at")})
        return out

    @staticmethod
    def _ref(job_id: str, state: dict, manifest: dict) -> dict:
        """What an access policy gets to decide on."""
        return {"job_id": job_id, "spec": state.get("spec"), "managed": bool(state),
                "requested_by": manifest.get("requested_by") or None,
                "approved_by": manifest.get("approved_by") or None}

    def _status(self, state: dict) -> str:
        if not state:
            return "unmanaged"
        status = state.get("status", "unknown")
        if status in RUNNING and not self._alive(state):
            # the process is gone without recording an outcome
            return "cancelled" if state.get("cancel_requested") else "died"
        return status

    def summary(self, job_id: str, *, spec_name: Optional[str] = None,
                public: bool = True) -> dict:
        """
        Return a job's state, manifest and progress.

        There is no access check here (see :meth:`summary_for`). A job that exited 0 while its log reports a failed step is
        ``completed_with_failures``.

        Parameters
        ----------
        job_id : str
            The job id.
        spec_name : str, optional
            Spec to parse an unmanaged directory's log with.
        public : bool, default: True
            Filter the summary through the spec's public view.

        Returns
        -------
        dict
            ``job_id``, ``spec``, ``managed``, ``status``, ``state``, ``manifest`` and
            ``progress``.

        Raises
        ------
        JobError
            If there is no such job.
        """
        job_dir = self.job_dir(job_id)
        if not job_dir.is_dir() or job_id in self.hidden:
            raise JobError(f"no such job: {job_id!r}")
        state = self._state(job_dir)
        if not state and not self.include_unmanaged:
            raise JobError(f"no such job: {job_id!r}")
        spec = self.specs.get(state.get("spec") or spec_name or "")
        manifest = read_json(job_dir / MANIFEST)
        summary = {
            "job_id": job_id, "spec": spec.name if spec else None, "managed": bool(state),
            "status": self._status(state),
            "state": {k: v for k, v in state.items() if k != "plan"},
            "manifest": manifest, "progress": None,
        }
        if spec and spec.progress is not None:
            log = job_dir / (state.get("plan", {}).get("log_name") or spec.log_name)
            text = log.read_text(errors="replace") if log.is_file() else ""
            summary["progress"] = spec.progress.parse(text, self._alive(state)).to_dict()
            # Many drivers exit 0 even when a step failed; the log is the better witness.
            if summary["status"] == "succeeded" and summary["progress"]["counts"]["fail"]:
                summary["status"] = "completed_with_failures"
        return spec.public_view(summary) if (spec and public) else summary

    # --- as a user: the policy applies -------------------------------------------------

    def ref(self, job_id: str) -> dict:
        """
        Return what an access policy decides on for a job.

        Parameters
        ----------
        job_id : str
            The job id.

        Returns
        -------
        dict
            ``job_id``, ``spec``, ``managed``, ``requested_by`` and ``approved_by``.
        """
        job_dir = self.job_dir(job_id)
        return self._ref(job_id, self._state(job_dir), read_json(job_dir / MANIFEST))

    def _visible(self, user: Optional[Principal], job_id: str) -> dict:
        ref = self.ref(job_id)
        if job_id in self.hidden or not self.policy.can_view_job(user, ref):
            raise JobError(f"no such job: {job_id!r}")  # do not reveal that it exists
        return ref

    def list_for(self, user: Optional[Principal]) -> List[dict]:
        """
        List the jobs a user may see.

        Parameters
        ----------
        user : dict, optional
            The caller (``None`` without auth).

        Returns
        -------
        list of dict
            As in :meth:`list`.
        """
        return [row for row in self.list() if self.policy.can_view_job(user, row)]

    def summary_for(self, user: Optional[Principal], job_id: str) -> dict:
        """
        Return a job's summary as a user may see it.

        Parameters
        ----------
        user : dict, optional
            The caller.
        job_id : str
            The job id.

        Returns
        -------
        dict
            The summary, through the policy's
            :meth:`~reflex_langgraph.policy.AccessPolicy.job_view`.

        Raises
        ------
        JobError
            If there is no such job, or the user may not see it (indistinguishably).
        """
        self._visible(user, job_id)
        summary = self.summary(job_id, public=False)
        return self.policy.job_view(user, summary, self.specs.get(summary.get("spec") or ""))

    def cancel_for(self, user: Optional[Principal], job_id: str, *, requested_by: str) -> dict:
        """
        Stop a running job, if the user may.

        Parameters
        ----------
        user : dict, optional
            The caller.
        job_id : str
            The job id.
        requested_by : str
            Who cancelled it; recorded in its state.

        Returns
        -------
        dict
            As in :meth:`cancel`.

        Raises
        ------
        JobError
            If there is no such job, or the user may not see it.
        JobForbidden
            If the user may see it but not cancel it.
        JobNotRunning
            If it is not running.
        """
        ref = self._visible(user, job_id)
        if not self.policy.can_cancel_job(user, ref):
            raise JobForbidden(f"you are not allowed to cancel job {job_id!r}")
        return self.cancel(job_id, requested_by=requested_by)

    def log_tail(self, job_id: str, lines: int = 200) -> str:
        """
        Return the end of a job's log.

        Parameters
        ----------
        job_id : str
            The job id.
        lines : int, default: 200
            How many lines, at most (from the last 256 KiB of the log).

        Returns
        -------
        str
            The lines; empty when there is no log yet.
        """
        job_dir = self.job_dir(job_id)
        state = self._state(job_dir)
        spec = self.specs.get(state.get("spec", ""))
        name = state.get("plan", {}).get("log_name") or (spec.log_name if spec else "job.log")
        path = job_dir / name
        if not path.is_file():
            return ""
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256 * 1024))
            data = fh.read().decode(errors="replace")
        return "\n".join(data.splitlines()[-lines:])


def _update(job_dir: Path, **fields) -> None:
    """Merge ``fields`` into the job's state; safe against concurrent updaters."""
    update_json(job_dir / STATE, {**fields, "updated_at": _now()})


def _record_error(job_dir: Path, exc: BaseException) -> str:
    """Keep an error's detail on the server (``error.txt``); return a reference for users."""
    ref = uuid.uuid4().hex[:8]
    with open(job_dir / "error.txt", "a") as fh:
        fh.write(f"[{_now()}] ref {ref}\n"
                 + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return ref


def _child_env() -> dict:
    env = dict(os.environ)
    pkg_parent = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{pkg_parent}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else pkg_parent
    return env
