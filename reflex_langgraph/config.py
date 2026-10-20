"""
The config submodule describes every agent of an app in one YAML file (or dict).

The topology comes from the ``workers`` field: an agent with workers is a supervisor that
delegates to them, and a worker may have workers of its own, which gives hierarchies of
any depth. The ``entry`` agent is the one users talk to.

.. code-block:: yaml

    model: google_genai:gemini-2.5-flash
    entry: supervisor
    agents:
      - name: supervisor
        prompt: You coordinate the team.
        workers: [researcher, operator]
      - name: researcher
        description: Looks things up.
        tools: [search]
      - name: operator
        description: Starts jobs.
        tools: [submit_job]
        approve:
          submit_job: [approve, reject, edit]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: A decision a person can take on a tool call that needs approval.
Decision = Literal["approve", "edit", "reject", "respond"]
#: Every decision, in the order the UI offers them.
ALL_DECISIONS: Tuple[Decision, ...] = ("approve", "edit", "reject", "respond")


class ModelSpec(BaseModel):
    """
    A chat model, created with ``langchain.chat_models.init_chat_model``.

    In YAML, a plain string is short for ``{id: <string>}``.

    Attributes
    ----------
    id : str
        ``"<provider>:<model>"``, e.g. ``google_genai:gemini-2.5-flash``,
        ``openai:gpt-4.1-mini`` or ``anthropic:claude-sonnet-5-5``.
    temperature : float, optional
        Sampling temperature. By default, the provider's.
    kwargs : dict
        Other arguments for the model. To stream the model's thinking (shown as "Thinking"
        in the UI): ``{include_thoughts: true}`` for Gemini, ``{thinking: {type: adaptive}}``
        for Claude, ``{reasoning: {effort: medium, summary: auto}}`` for OpenAI.
    reads : list of str, optional
        Kinds of attached file the model reads as content: ``image``, ``pdf``. By default,
        both for Gemini, Claude and OpenAI, none for other providers (text files always go
        in as text; files a model cannot read are described to it instead).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    temperature: Optional[float] = None
    kwargs: Dict[str, object] = Field(default_factory=dict)
    reads: Optional[List[Literal["image", "pdf"]]] = None


class AgentSpec(BaseModel):
    """
    One agent, with its instructions, model, tools, workers and approvals.

    Attributes
    ----------
    name : str
        Unique name: letters, digits, ``_`` and ``-``. A worker named ``x`` becomes the
        ``ask_x`` tool of its supervisors.
    description : str
        What the agent does. Supervisors read it to decide what to delegate.
    prompt : str
        System prompt.
    model : ModelSpec, optional
        The agent's model. By default, the app's ``model``.
    tools : list of str
        Names of tools in the :class:`~reflex_langgraph.agents.ToolRegistry`.
    workers : list of str
        Agents this one may delegate to.
    approve : dict
        Tools that pause for a human decision before running, mapped to the decisions
        offered. In YAML, a list of tool names (or ``tool: true``) offers every decision.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
    description: str = ""
    prompt: str = ""
    model: Optional[ModelSpec] = None
    tools: List[str] = Field(default_factory=list)
    workers: List[str] = Field(default_factory=list)
    approve: Dict[str, List[Decision]] = Field(default_factory=dict)

    @field_validator("model", mode="before")
    @classmethod
    def _model_shorthand(cls, v):
        return {"id": v} if isinstance(v, str) else v

    @field_validator("approve", mode="before")
    @classmethod
    def _approve_shorthand(cls, v):
        # `approve: [tool_a, tool_b]` or `approve: {tool_a: true}` offer every decision.
        if isinstance(v, list):
            return {name: list(ALL_DECISIONS) for name in v}
        if isinstance(v, dict):
            return {k: (list(ALL_DECISIONS) if d is True else d) for k, d in v.items()}
        return v

    @model_validator(mode="after")
    def _approve_only_own_tools(self):
        unknown = set(self.approve) - set(self.tools)
        if unknown:
            raise ValueError(f"agent {self.name!r}: approve names tools it does not have: "
                             f"{sorted(unknown)}")
        return self


class AppConfig(BaseModel):
    """
    Every agent of an app, validated as a whole.

    Validation fails for: no agents, duplicate names, an unknown ``entry`` or worker, an
    agent without a model when there is no default, approvals for tools an agent does not
    have, and worker cycles.

    Attributes
    ----------
    model : ModelSpec, optional
        Default model for agents that do not set one.
    entry : str, optional
        The agent users talk to. By default, the first agent.
    agents : list of AgentSpec
        The agents.
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[ModelSpec] = None
    entry: Optional[str] = None
    agents: List[AgentSpec]

    @field_validator("model", mode="before")
    @classmethod
    def _model_shorthand(cls, v):
        return {"id": v} if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_graph(self):
        if not self.agents:
            raise ValueError("at least one agent is required")
        by_name = {a.name: a for a in self.agents}
        if len(by_name) != len(self.agents):
            raise ValueError("agent names must be unique")
        if self.entry is None:
            self.entry = self.agents[0].name
        if self.entry not in by_name:
            raise ValueError(f"entry agent {self.entry!r} is not defined")
        for a in self.agents:
            missing = [w for w in a.workers if w not in by_name]
            if missing:
                raise ValueError(f"agent {a.name!r} has undefined workers: {missing}")
            if a.model is None and self.model is None:
                raise ValueError(f"agent {a.name!r} has no model and no default model is set")
        self._check_acyclic(by_name)
        return self

    def _check_acyclic(self, by_name: Dict[str, AgentSpec]) -> None:
        state: Dict[str, int] = {}

        def visit(name: str, path: List[str]) -> None:
            if state.get(name) == 1:
                raise ValueError("worker cycle: " + " -> ".join(path + [name]))
            if state.get(name) == 2:
                return
            state[name] = 1
            for w in by_name[name].workers:
                visit(w, path + [name])
            state[name] = 2

        for a in self.agents:
            visit(a.name, [])

    def agent(self, name: str) -> AgentSpec:
        """
        Return the agent with this name.

        Parameters
        ----------
        name : str
            Agent name.

        Returns
        -------
        AgentSpec
            The agent.
        """
        return next(a for a in self.agents if a.name == name)


def load_config(source: Union[str, Path, dict]) -> AppConfig:
    """
    Load and validate an app's configuration.

    Parameters
    ----------
    source : str, pathlib.Path or dict
        A ``.yaml``/``.yml`` file, YAML text, or the configuration as a dict.

    Returns
    -------
    AppConfig
        The validated configuration.

    Raises
    ------
    pydantic.ValidationError
        If the configuration is invalid (see :class:`AppConfig`).

    Examples
    --------

    >>> from reflex_langgraph import load_config
    >>> config = load_config('''
    ... model: google_genai:gemini-2.5-flash
    ... agents:
    ...   - name: supervisor
    ...     workers: [operator]
    ...   - name: operator
    ...     tools: [submit]
    ...     approve: [submit]
    ... ''')
    >>> config.entry
    'supervisor'
    >>> config.agent('operator').approve
    {'submit': ['approve', 'edit', 'reject', 'respond']}
    >>> load_config({'model': 'x:y', 'agents': [{'name': 'a', 'workers': ['b']}]})
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for AppConfig
    ...
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        text = path.read_text() if path.suffix in (".yaml", ".yml") and path.exists() else str(source)
        data = yaml.safe_load(text)
    return AppConfig.model_validate(data)
