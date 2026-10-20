"""
The router submodule serves a :class:`JobRunner` over HTTP, for the UI's jobs panel.

    GET  /jobs                 jobs the caller may see
    GET  /jobs/{id}            summary, as the policy lets the caller see it
    GET  /jobs/{id}/log        tail of the log -- only when expose_logs=True
    POST /jobs/{id}/cancel

Access follows ``runner.policy``, the same object the agent tools use. Submission is
deliberately not an HTTP route: jobs start through the agent's submit tool, behind the
approval step.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..policy import principal_id
from ..server.schemas import JobCancelled, JobRow, JobSummary
from .runner import JobError, JobForbidden, JobNotRunning, JobRunner


def jobs_router(runner: JobRunner, *, expose_logs: bool = False,
                can_read_logs: Optional[Callable[[Optional[dict]], bool]] = None) -> APIRouter:
    """
    Create the ``/jobs`` routes for a runner.

    Parameters
    ----------
    runner : JobRunner
        The runner; its policy decides what each caller may see and do.
    expose_logs : bool, default: False
        Add ``GET /jobs/{id}/log``. Raw logs often contain server paths that the spec's
        public view hides, so it is off by default.
    can_read_logs : callable, optional
        ``can_read_logs(user) -> bool``, limiting the log route further.

    Returns
    -------
    fastapi.APIRouter
        The routes, for ``create_server(routers=[...])``.
    """
    router = APIRouter(prefix="/jobs", tags=["jobs"])
    router.rlg_capability = "jobs"  # declared in /health

    async def user(request: Request) -> Optional[dict]:
        hook = getattr(request.app.state, "current_user", None)
        return await hook(request) if hook else None

    @router.get("", response_model=List[JobRow])
    async def list_jobs(u=Depends(user)):
        return runner.list_for(u)

    @router.get("/{job_id}", response_model=JobSummary)
    async def get_job(job_id: str, u=Depends(user)):
        try:
            return runner.summary_for(u, job_id)
        except JobError as exc:
            raise HTTPException(404, str(exc))

    if expose_logs:
        @router.get("/{job_id}/log")
        async def get_log(job_id: str, lines: int = 200, u=Depends(user)):
            if can_read_logs is not None and not can_read_logs(u):
                raise HTTPException(403, "not allowed to read job logs")
            try:
                runner.summary_for(u, job_id)  # same visibility as the summary
                return {"job_id": job_id, "log": runner.log_tail(job_id, min(lines, 2000))}
            except JobError as exc:
                raise HTTPException(404, str(exc))

    @router.post("/{job_id}/cancel", response_model=JobCancelled)
    async def cancel(job_id: str, u=Depends(user)):
        try:
            return runner.cancel_for(u, job_id, requested_by=principal_id(u) or "anonymous")
        except JobForbidden as exc:
            raise HTTPException(403, str(exc))
        except JobNotRunning as exc:
            raise HTTPException(409, str(exc))
        except JobError as exc:
            raise HTTPException(404, str(exc))

    return router
