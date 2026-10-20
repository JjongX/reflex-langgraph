"""
The spec submodule describes a kind of job: subclass :class:`JobSpec` once per kind of
long-running work.

.. code-block:: python

    class TrainParams(BaseModel):
        dataset: str

    class Train(JobSpec):
        name = "train"
        description = "Train a model on a dataset directory."
        params_model = TrainParams

        def command(self, params, job_dir):
            # The job runs inside job_dir, so name the script by its full path.
            script = Path(__file__).parent / "train.py"
            return ["python", str(script), "--data", params.dataset, "--out", str(job_dir)]

The runner takes care of everything else: a fresh directory per job, detaching, the
concurrency limit, cancel, provenance hashing and progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from .progress import ProgressParser


class NoParams(BaseModel):
    """The parameters of a job that takes none (the default ``params_model``)."""


@dataclass
class Preflight:
    """
    The result of checking whether a job could start.

    Attributes
    ----------
    blockers : list of str
        Reasons the job must not start. Any blocker refuses the submission.
    warnings : list of str
        Concerns worth showing, which do not refuse it.
    info : dict
        Facts worth showing the reviewer, e.g. ``{"input files": 384}``.
    required_bytes : int
        Disk the job will need under the runner's root; the runner checks it.

    Examples
    --------

    >>> from reflex_langgraph.jobs import Preflight
    >>> check = Preflight(info={'input files': 30})
    >>> check.ok
    True
    >>> check.blockers.append('input folder is empty')
    >>> check.ok
    False
    """

    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, object] = field(default_factory=dict)
    required_bytes: int = 0

    @property
    def ok(self) -> bool:
        """bool : True when there are no blockers."""
        return not self.blockers


class JobSpec:
    """
    A kind of job. Subclass it, set the attributes, and implement :meth:`command`.

    Agents get tools named after :attr:`name` (see
    :func:`~reflex_langgraph.jobs.job_tools`); everything else (a fresh directory per job,
    detaching, the concurrency limit, cancel, provenance and progress) is the runner's.

    Attributes
    ----------
    name : str
        Unique name; agents get ``<name>_preflight`` and ``<name>_submit`` tools.
    description : str
        What the job does, for the agents' tool descriptions.
    params_model : type of pydantic.BaseModel
        The job's parameters; their schema becomes the tools' arguments. By default,
        :class:`NoParams`.
    log_name : str
        File the command's output goes to, inside the job directory.
    progress : ProgressParser, optional
        Turns the log into step-by-step progress (e.g. a :class:`StepMarkerParser`).
    protected_paths : sequence of pathlib.Path
        Directories a job directory may never be created in or under (inputs, the
        pipeline itself).
    """

    name: str = ""
    description: str = ""
    params_model: Type[BaseModel] = NoParams
    log_name: str = "job.log"
    progress: Optional[ProgressParser] = None
    protected_paths: Sequence[Path] = ()

    def preflight(self, params: BaseModel) -> Preflight:
        """
        Check whether a job could start, without creating anything.

        Parameters
        ----------
        params : pydantic.BaseModel
            The job's parameters.

        Returns
        -------
        Preflight
            Blockers refuse the submission. By default, none.
        """
        return Preflight()

    def prepare(self, params: BaseModel, job_dir: Path) -> None:
        """
        Lay out the fresh job directory before the command starts (e.g. link inputs).

        Parameters
        ----------
        params : pydantic.BaseModel
            The job's parameters.
        job_dir : pathlib.Path
            The job's new, empty directory.
        """

    def command(self, params: BaseModel, job_dir: Path) -> List[str]:
        """
        Return the command that does the work. Subclasses must implement it.

        Parameters
        ----------
        params : pydantic.BaseModel
            The job's parameters.
        job_dir : pathlib.Path
            The job's directory, also the command's working directory.

        Returns
        -------
        list of str
            The command and its arguments.
        """
        raise NotImplementedError

    def env(self, params: BaseModel, job_dir: Path) -> Dict[str, str]:
        """
        Return extra environment variables for the command.

        Parameters
        ----------
        params : pydantic.BaseModel
            The job's parameters.
        job_dir : pathlib.Path
            The job's directory.

        Returns
        -------
        dict
            Variable name to value. By default, none.
        """
        return {}

    def fingerprint(self, params: BaseModel, job_dir: Path) -> Dict[str, Path]:
        """
        Return the files or folders whose content is hashed into the job's manifest.

        Hashing runs in the detached job process, so large inputs do not slow submission.

        Parameters
        ----------
        params : pydantic.BaseModel
            The job's parameters.
        job_dir : pathlib.Path
            The job's directory.

        Returns
        -------
        dict
            Label to file or folder. By default, none.
        """
        return {}

    def version_commands(self, params: BaseModel, job_dir: Path) -> Dict[str, List[str]]:
        """
        Return commands whose first output line records a tool's version in the manifest.

        Parameters
        ----------
        params : pydantic.BaseModel
            The job's parameters.
        job_dir : pathlib.Path
            The job's directory.

        Returns
        -------
        dict
            Tool name to command, e.g. ``{"bash": ["bash", "--version"]}``. By default,
            none.
        """
        return {}

    def public_view(self, summary: dict) -> dict:
        """
        Filter a job's summary before it leaves the server.

        Parameters
        ----------
        summary : dict
            The full summary (see :meth:`JobRunner.summary
            <reflex_langgraph.jobs.JobRunner.summary>`).

        Returns
        -------
        dict
            What people see. By default, the summary unchanged; override it to leave out
            server paths and other internals.
        """
        return summary
