"""
The builder submodule compiles an :class:`~reflex_langgraph.config.AppConfig` into one
LangGraph graph.

Each agent is a ``langchain.agents.create_agent`` graph. A worker is exposed to its
supervisor as a delegation tool, ``ask_<worker>``, which runs the worker on the task and
returns its final answer. Workers run inside the supervisor's tool node and inherit its
checkpointer, so an approval pause anywhere in the tree surfaces on the entry graph and a
resume continues the paused worker where it stopped.

Every agent gets :class:`~.blocked.BlockedResponseMiddleware`: a reply the model provider
blocked is marked (and streamed as a ``blocked`` event), and blocked turns are left out of
later model calls. Every agent also gets :class:`~.attachments.AttachmentMiddleware`, which
shows its model the files attached to messages; a supervisor hands files on to a worker by
their ids.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from ..config import AgentSpec, AppConfig, ModelSpec
from ..files import ATTACHMENTS_KEY, FILES_KEY
from .attachments import AttachmentMiddleware, default_reads
from .blocked import BlockedResponseMiddleware, block_info
from .tools import ToolRegistry

#: Makes an agent's chat model from its model spec: ``factory(spec, agent)``.
ModelFactory = Callable[[ModelSpec, AgentSpec], BaseChatModel]

#: Extra agent middleware: one list for every agent, ``{agent: [...]}``, or
#: ``factory(agent_spec) -> [...]`` for a fresh list per agent.
MiddlewareOption = Union[Sequence[object], Mapping[str, Sequence[object]],
                         Callable[[AgentSpec], Sequence[object]]]

#: Delegation tools are named ``ask_<worker>``.
DELEGATE_PREFIX = "ask_"


def default_model_factory(spec: ModelSpec, agent: AgentSpec) -> BaseChatModel:
    """
    Create a chat model with ``langchain.chat_models.init_chat_model``.

    Parameters
    ----------
    spec : ModelSpec
        The model: ``id``, ``temperature`` and other ``kwargs``.
    agent : AgentSpec
        The agent it is for (unused here; a custom factory may use it).

    Returns
    -------
    BaseChatModel
        The model.
    """
    kwargs = dict(spec.kwargs)
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    return init_chat_model(spec.id, **kwargs)


class _Task(BaseModel):
    task: str = Field(description="What the worker should do, with every detail it needs.")
    files: List[str] = Field(default_factory=list, description=(
        "Ids of attached files the worker needs (from the [Attached file: ..., file id f_...] "
        "labels); it receives them with the task."))


def _delegation_tool(worker_name: str, worker_spec: AgentSpec, worker_graph) -> BaseTool:
    async def _run(task: str, config: RunnableConfig, files: Optional[List[str]] = None) -> str:
        # The supervisor's config carries its own agent name, which would label everything
        # the worker streams as the supervisor's.
        config = {**config, "metadata": {**(config.get("metadata") or {}),
                                         "lc_agent_name": worker_name}}
        configurable = config.get("configurable") or {}
        store, thread_id = configurable.get(FILES_KEY), configurable.get("thread_id")
        records = [store.get(thread_id, f) for f in files or []] if store else []
        refs = [r.ref() for r in records if r is not None]
        message = HumanMessage(task, additional_kwargs={ATTACHMENTS_KEY: refs} if refs else {})
        result = await worker_graph.ainvoke({"messages": [message]}, config)
        messages = result.get("messages", [])
        if not messages:
            return ""
        blocked = block_info(messages[-1])
        if blocked:
            what = "request" if blocked["scope"] == "prompt" else "response"
            return (f"BLOCKED: the model provider blocked the {worker_name} agent's {what} "
                    f"({blocked['reason']}). Do not send the same task again; rephrase it or "
                    "tell the user.")
        # The answer text only: with reasoning on, content also holds thought blocks.
        return messages[-1].text

    return StructuredTool.from_function(
        coroutine=_run,
        name=f"{DELEGATE_PREFIX}{worker_name}",
        description=(f"Delegate a task to the '{worker_name}' agent and get its answer. "
                     f"{worker_spec.description}").strip(),
        args_schema=_Task,
    )


def _system_prompt(agent: AgentSpec, config: AppConfig) -> str:
    parts = [agent.prompt.strip()] if agent.prompt.strip() else []
    if agent.workers:
        team = "\n".join(f"- {w}: {config.agent(w).description or '(no description)'}"
                         for w in agent.workers)
        parts.append(f"You can delegate to these agents with the ask_<name> tools:\n{team}")
    return "\n\n".join(parts)


def build_graph(config: AppConfig, registry: Optional[ToolRegistry] = None, *,
                checkpointer: Optional[BaseCheckpointSaver] = None,
                model_factory: ModelFactory = default_model_factory,
                models: Optional[Dict[str, BaseChatModel]] = None,
                middleware: Optional[MiddlewareOption] = None):
    """
    Compile the configuration into the entry agent's graph.

    Parameters
    ----------
    config : AppConfig
        The agents (see :func:`~reflex_langgraph.config.load_config`).
    registry : ToolRegistry, optional
        The tools agents name in ``tools``.
    checkpointer : BaseCheckpointSaver, optional
        Where conversations' graph state is kept. By default, in memory (see
        :func:`~reflex_langgraph.agents.sqlite_checkpointer` for a durable one).
    model_factory : callable, default: default_model_factory
        Makes each agent's model from its model spec.
    models : dict, optional
        Agent name to a ready model, used instead of the factory (e.g. a fake model in
        tests).
    middleware : list, dict or callable, optional
        Your own LangChain agent middleware (logging, guardrails, prompt changes...): a
        list for every agent, ``{agent_name: [...]}`` for some, or ``factory(agent_spec)``
        returning a fresh list per agent. Each agent's middleware is, in order:
        :class:`~.blocked.BlockedResponseMiddleware`,
        :class:`~.attachments.AttachmentMiddleware`, yours, then LangChain's
        ``HumanInTheLoopMiddleware`` when the agent has ``approve:``. Model-call wrappers
        nest in that order (the first is outermost), so yours see the request with the
        attached files already added, and a reply yours return is still checked for
        provider blocks; ``after_model`` hooks run in reverse order, so approval pauses
        are decided before yours run.

    Returns
    -------
    CompiledStateGraph
        The entry agent's graph; workers run inside it.

    Examples
    --------

    >>> from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    >>> from langchain_core.messages import AIMessage
    >>> from reflex_langgraph import build_graph, load_config
    >>> config = load_config({'model': 'x:y', 'agents': [{'name': 'assistant'}]})
    >>> model = GenericFakeChatModel(messages=iter([AIMessage('Hello!')]))  # no API key
    >>> graph = build_graph(config, models={'assistant': model})
    >>> state = graph.invoke({'messages': [('user', 'hi')]}, {'configurable': {'thread_id': '1'}})
    >>> state['messages'][-1].text
    'Hello!'
    """
    registry = registry or ToolRegistry()
    models = models or {}
    if isinstance(middleware, Mapping):
        unknown = set(middleware) - {a.name for a in config.agents}
        if unknown:
            raise ValueError(f"middleware for unknown agents: {sorted(unknown)}")

    def extra_middleware(spec: AgentSpec) -> list:
        if middleware is None:
            return []
        if isinstance(middleware, Mapping):
            return list(middleware.get(spec.name, []))
        if callable(middleware):
            return list(middleware(spec))
        return list(middleware)
    checkpointer = checkpointer or InMemorySaver()
    built: Dict[str, object] = {}

    def build(name: str):
        if name in built:
            return built[name]
        spec = config.agent(name)
        tools = registry.get(spec.tools)
        tools += [_delegation_tool(w, config.agent(w), build(w)) for w in spec.workers]
        model_spec = spec.model or config.model
        model = models.get(name) or model_factory(model_spec, spec)
        reads = model_spec.reads if model_spec.reads is not None else default_reads(model_spec.id)
        stack = [BlockedResponseMiddleware(), AttachmentMiddleware(reads), *extra_middleware(spec)]
        if spec.approve:
            stack.append(HumanInTheLoopMiddleware(interrupt_on={
                tool: {"allowed_decisions": list(decisions),
                       "description": f"The {name} agent wants to run {tool}."}
                for tool, decisions in spec.approve.items()
            }))
        graph = create_agent(
            model, tools,
            system_prompt=_system_prompt(spec, config) or None,
            middleware=stack,
            name=name,
            # Only the entry graph owns the checkpointer; workers inherit it at run time.
            checkpointer=checkpointer if name == config.entry else None,
        )
        built[name] = graph
        return graph

    return build(config.entry)
