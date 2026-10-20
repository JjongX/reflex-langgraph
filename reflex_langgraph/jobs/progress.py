"""
The progress submodule turns a job's log into step-by-step progress.

:class:`StepMarkerParser` covers the common shape: a known list of steps, a line printed
when each starts, and optionally a closing summary with each step's outcome.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Pattern, Protocol, Sequence, Tuple, Union


@dataclass
class Step:
    """
    One step of a job.

    Attributes
    ----------
    key : str
        Matched against the log's start lines, e.g. ``"02"``.
    label : str
        Shown to people, e.g. ``"02 align reads"``.
    status : str
        ``pending``, ``running``, ``done`` (finished, outcome not reported yet),
        ``pass``, ``fail``, ``interrupted`` (running when the process died) or
        ``not_reached``.
    started : str, optional
        When it started, as the log printed it.
    returncode : int, optional
        Its exit code, when the log reports one.
    """

    key: str
    label: str
    status: str = "pending"
    started: Optional[str] = None
    returncode: Optional[int] = None


@dataclass
class Progress:
    """
    A job's progress.

    Attributes
    ----------
    steps : list of Step
        The steps, in execution order.
    finished : bool
        Whether the log has reported how every step ended.
    """

    steps: List[Step] = field(default_factory=list)
    finished: bool = False

    @property
    def current(self) -> Optional[Step]:
        """Step or None : The step running now."""
        return next((s for s in self.steps if s.status == "running"), None)

    def counts(self) -> dict:
        """
        Count the steps by status.

        Returns
        -------
        dict
            Status to number of steps, for every status.
        """
        out = {k: 0 for k in ("pass", "fail", "done", "running", "interrupted", "pending",
                                "not_reached")}
        for s in self.steps:
            out[s.status] += 1
        return out

    def to_dict(self) -> dict:
        """
        Return the progress as plain data, as the API sends it.

        Returns
        -------
        dict
            ``finished``, ``counts``, ``current`` (the running step's label) and
            ``steps``.
        """
        return {"finished": self.finished, "counts": self.counts(),
                "current": self.current.label if self.current else None,
                "steps": [asdict(s) for s in self.steps]}


class ProgressParser(Protocol):
    """Anything with a ``parse(log_text, alive)`` method returning :class:`Progress`."""

    def parse(self, log_text: str, alive: bool) -> Progress:
        """
        Read a job's progress from its log.

        Parameters
        ----------
        log_text : str
            The log so far.
        alive : bool
            Whether the job's process is still running.

        Returns
        -------
        Progress
            The progress.
        """


def _compile(p: Union[str, Pattern]) -> Pattern:
    return re.compile(p) if isinstance(p, str) else p


class StepMarkerParser:
    r"""
    Progress from the lines a log prints as steps start, and from its closing summary.

    Each step prints a line when it starts; an optional summary at the end gives each
    step's outcome. Before the summary appears, finished steps are reported as ``done``, never ``pass``:
    the log has not said how they ended.

    Parameters
    ----------
    plan : sequence of (str, str)
        ``(key, label)`` for each step, in execution order; keys may repeat.
    start : str or re.Pattern
        Matches a step's start line, with a ``key`` group (and optionally ``time``).
    summary_start : str or re.Pattern, optional
        Matches the line that opens the result summary.
    result : str or re.Pattern, optional
        Matches a result line, with ``label`` and ``status`` groups (and optionally
        ``rc``).
    pass_value : str, default: 'PASS'
        The ``status`` that means success.

    Examples
    --------

    >>> from reflex_langgraph.jobs import StepMarkerParser
    >>> parser = StepMarkerParser(
    ...     plan=[('01', 'download'), ('02', 'align')],
    ...     start=r'^=== step (?P<key>\d+)',
    ...     summary_start=r'^RESULT',
    ...     result=r'^(?P<status>PASS|FAIL)\s+(?P<label>\w+)')
    >>> running = parser.parse('=== step 01\n=== step 02\n', alive=True)
    >>> [(s.label, s.status) for s in running.steps]
    [('download', 'done'), ('align', 'running')]
    >>> ended = parser.parse('=== step 01\n=== step 02\nRESULT\nPASS download\nFAIL align\n',
    ...                      alive=False)
    >>> [(s.label, s.status) for s in ended.steps], ended.finished
    ([('download', 'pass'), ('align', 'fail')], True)
    """

    def __init__(self, plan: Sequence[Tuple[str, str]], start: Union[str, Pattern],
                 summary_start: Optional[Union[str, Pattern]] = None,
                 result: Optional[Union[str, Pattern]] = None, pass_value: str = "PASS"):
        self.plan = list(plan)
        self.start = _compile(start)
        self.summary_start = _compile(summary_start) if summary_start else None
        self.result = _compile(result) if result else None
        self.pass_value = pass_value

    def parse(self, log_text: str, alive: bool) -> Progress:
        """
        Read a job's progress from its log.

        Parameters
        ----------
        log_text : str
            The log so far.
        alive : bool
            Whether the job's process is still running; the last started step is
            ``running`` if so, else ``interrupted``.

        Returns
        -------
        Progress
            The progress.
        """
        steps = [Step(key=k, label=l) for k, l in self.plan]
        progress = Progress(steps=steps)
        lines = log_text.splitlines()

        summary_at = None
        if self.summary_start is not None:
            summary_at = next((i for i, l in enumerate(lines) if self.summary_start.search(l)), None)

        # Markers map onto plan entries in order, matched by key. A step that dies before
        # printing its marker leaves its slot unmatched instead of shifting later steps.
        cursor = 0
        for line in lines[:summary_at]:
            m = self.start.search(line)
            if not m:
                continue
            key = m.group("key")
            for j in range(cursor, len(steps)):
                if steps[j].key == key and steps[j].started is None:
                    steps[j].started = m.groupdict().get("time") or ""
                    cursor = j + 1
                    break

        if summary_at is not None and self.result is not None:
            progress.finished = True
            results = {}
            for line in lines[summary_at + 1:]:
                m = self.result.search(line)
                if m:
                    results[m.group("label").strip()] = m
            for s in steps:
                m = results.get(s.label)
                if m is None:
                    s.status = "not_reached"
                elif m.group("status") == self.pass_value:
                    s.status, s.returncode = "pass", 0
                else:
                    s.status = "fail"
                    rc = m.groupdict().get("rc")
                    s.returncode = int(rc) if rc and rc.lstrip("-").isdigit() else None
            return progress

        started = [i for i, s in enumerate(steps) if s.started is not None]
        if started:
            for i in started[:-1]:
                steps[i].status = "done"
            steps[started[-1]].status = "running" if alive else "interrupted"
        return progress
