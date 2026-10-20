"""
The tools submodule gives agents tools over a :class:`JobRunner`.

``job_tools(runner)`` returns, for every spec ``<name>``:

* ``<name>_preflight(params)`` -- read-only readiness check
* ``<name>_submit(job_id, params)`` -- starts a job; put it under ``approve:`` in the config
* plus shared ``list_jobs()``, ``job_status(job_id)`` and ``cancel_job(job_id)``

Identities come from the server, never from arguments the model writes:

* ``approved_by`` is the user who triggered the run that executes the submit -- with the
  submit tool under ``approve:``, the person who answered the approval. Without an auth
  hook nobody is identified, and ``anonymous_approver`` (default ``"anonymous"``) is
  recorded;
* ``requested_by`` is the conversation's owner (``thread_owner``);
* reads (``list_jobs``, ``job_status``) are checked against the conversation's owner,
  because their output lands in the owner's chat even when a reviewer resumed the run;
* ``cancel_job`` is authorised as the user who triggered the run.

All checks go through ``runner.policy``, as the HTTP routes do. Outputs land in a user's
chat, so they never carry server paths, and unexpected errors are reported by reference
only (the detail goes to the server log).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from ..policy import principal_id
from .runner import JobError, JobRunner
from .spec import JobSpec


def _configurable(config: Optional[RunnableConfig]) -> dict:
    return (config or {}).get("configurable") or {}


def _user(config: Optional[RunnableConfig]):
    """Who triggered this run (sent the message, or answered the approval)."""
    return _configurable(config).get("user")


def _audience(config: Optional[RunnableConfig]):
    """Whose conversation this is: what a tool returns ends up in their chat."""
    c = _configurable(config)
    return c.get("thread_owner") if c.get("thread_owner") is not None else c.get("user")


log = logging.getLogger("reflex_langgraph.jobs")


def _dump(data) -> str:
    return json.dumps(data, indent=2, default=str)


def _failed() -> str:
    """Call inside ``except``: log the exception, give the user a reference only."""
    ref = uuid.uuid4().hex[:8]
    log.exception("job tool failed [ref %s]", ref)
    return f"FAILED: an internal error occurred (ref {ref}); the details are in the server log."


def _spec_tools(runner: JobRunner, spec: JobSpec, anonymous_approver: Optional[str]) -> List[BaseTool]:
    Params = spec.params_model

    async def preflight(config: RunnableConfig, **params) -> str:
        try:
            result = runner.preflight(spec.name, params)
        except JobError as exc:
            return f"REFUSED: {exc}"
        except Exception:
            return _failed()
        return _dump({"ok": result.ok, "blockers": result.blockers,
                      "warnings": result.warnings, "info": result.info})

    SubmitArgs = create_model(
        f"{spec.name.title().replace('_', '')}SubmitArgs",
        job_id=(str, Field(description="New, unique job id: letters, digits, '.', '_', '-'.")),
        __base__=Params,
    )

    async def submit(job_id: str, config: RunnableConfig, **params) -> str:
        approver = principal_id(_user(config)) or anonymous_approver
        if not approver:
            return "REFUSED: no approver could be identified, so jobs cannot be started here."
        requester = principal_id(_configurable(config).get("thread_owner")) or approver
        try:
            out = runner.submit(spec.name, job_id, params, approved_by=approver,
                                requested_by=requester)
        except JobError as exc:
            return f"REFUSED: {exc}"
        except Exception:
            return _failed()
        # Not the job directory: a server path is no business of the chat.
        return _dump({"started": True, "job_id": out["job_id"], "spec": out["spec"],
                      "warnings": out["warnings"], "info": out["info"]})

    return [
        StructuredTool.from_function(
            coroutine=preflight, name=f"{spec.name}_preflight", args_schema=Params,
            description=f"Check whether a '{spec.name}' job could start now, without starting it. "
                        f"{spec.description}"),
        StructuredTool.from_function(
            coroutine=submit, name=f"{spec.name}_submit", args_schema=SubmitArgs,
            description=f"Start a '{spec.name}' job. Runs detached; check it with job_status. "
                        f"{spec.description}"),
    ]


class _JobId(BaseModel):
    job_id: str


def job_tools(runner: JobRunner, *,
              anonymous_approver: Optional[str] = "anonymous") -> List[BaseTool]:
    """
    Create agent tools for every kind of job in a runner.

    For each spec ``<name>``: ``<name>_preflight`` and ``<name>_submit`` (put it under
    ``approve:``); plus ``list_jobs``, ``job_status`` and ``cancel_job``. Who approved and
    who asked come from the server, never from the model's arguments.

    Parameters
    ----------
    runner : JobRunner
        The runner; its policy decides what each caller may see and do.
    anonymous_approver : str, optional, default: 'anonymous'
        Recorded as the approver when the server identifies nobody (no auth hook).
        ``None`` refuses submissions then.

    Returns
    -------
    list of BaseTool
        The tools, for a :class:`~reflex_langgraph.agents.ToolRegistry`.

    Examples
    --------

    >>> import tempfile
    >>> from reflex_langgraph.jobs import JobRunner, JobSpec, job_tools
    >>> class Train(JobSpec):
    ...     name = 'train'
    ...     def command(self, params, job_dir):
    ...         return ['true']
    >>> tools = job_tools(JobRunner(tempfile.gettempdir(), [Train()]))
    >>> [t.name for t in tools]
    ['train_preflight', 'train_submit', 'list_jobs', 'job_status', 'cancel_job']
    """
    tools: List[BaseTool] = []
    for spec in runner.specs.values():
        tools += _spec_tools(runner, spec, anonymous_approver)

    async def list_jobs(config: RunnableConfig) -> str:
        try:
            return _dump(runner.list_for(_audience(config)))
        except Exception:
            return _failed()

    async def job_status(job_id: str, config: RunnableConfig) -> str:
        try:
            return _dump(runner.summary_for(_audience(config), job_id))
        except JobError as exc:
            return f"REFUSED: {exc}"
        except Exception:
            return _failed()

    async def cancel_job(job_id: str, config: RunnableConfig) -> str:
        user = _user(config)
        try:
            return _dump(runner.cancel_for(user, job_id,
                                           requested_by=principal_id(user) or "anonymous"))
        except JobError as exc:
            return f"REFUSED: {exc}"
        except Exception:
            return _failed()

    tools += [
        StructuredTool.from_function(coroutine=list_jobs, name="list_jobs",
                                     description="List jobs and their status."),
        StructuredTool.from_function(coroutine=job_status, name="job_status", args_schema=_JobId,
                                     description="Status and step-by-step progress of one job."),
        StructuredTool.from_function(coroutine=cancel_job, name="cancel_job", args_schema=_JobId,
                                     description="Stop a running job. Its directory is kept."),
    ]
    return tools
