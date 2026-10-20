"""
The tools submodule holds the registry of tools that agent configurations name.

A registry is an ordinary object, not global state, so one process can serve several apps.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Iterable, List, Optional, Union

from langchain_core.tools import BaseTool, StructuredTool

#: A LangChain tool, or a plain (sync or async) function to make one from.
ToolLike = Union[BaseTool, Callable]


class ToolRegistry:
    """
    Tools by name, for agent configurations to refer to.

    A plain function becomes a tool named after it, described by its docstring, with
    arguments taken from its signature; an ``async def`` function runs as a coroutine.

    Parameters
    ----------
    tools : iterable of BaseTool or callable, optional
        Tools to register at once.

    Examples
    --------

    >>> from reflex_langgraph import ToolRegistry
    >>> registry = ToolRegistry()
    >>> @registry.tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> registry.names()
    ['add']
    >>> registry.get('add')[0].invoke({'a': 2, 'b': 3})
    5
    """

    def __init__(self, tools: Iterable[ToolLike] = ()):
        self._tools: Dict[str, BaseTool] = {}
        for t in tools:
            self.add(t)

    def add(self, tool: ToolLike, *, name: Optional[str] = None,
            description: Optional[str] = None) -> BaseTool:
        """
        Register a tool.

        Parameters
        ----------
        tool : BaseTool or callable
            A LangChain tool, or a function to make one from.
        name : str, optional
            Name to register it under. By default, the tool's (or function's) name.
        description : str, optional
            What the model reads about the tool. By default, the docstring.

        Returns
        -------
        BaseTool
            The registered tool.

        Raises
        ------
        ValueError
            If a tool with this name is already registered.
        """
        if not isinstance(tool, BaseTool):
            # An async function must be passed as the coroutine: given as func, calling it
            # would return an un-awaited coroutine instead of running it.
            kind = "coroutine" if inspect.iscoroutinefunction(tool) else "func"
            tool = StructuredTool.from_function(**{kind: tool}, name=name, description=description)
        elif name or description:
            tool = tool.model_copy(update={k: v for k, v in
                                           (("name", name), ("description", description)) if v})
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, func: Optional[Callable] = None, *, name: Optional[str] = None,
             description: Optional[str] = None):
        """
        Register a function, as a decorator.

        Use it as ``@registry.tool``, or as ``@registry.tool(name=...)``.

        Parameters
        ----------
        func : callable, optional
            The function (given when used without parentheses).
        name : str, optional
            As in :meth:`add`.
        description : str, optional
            As in :meth:`add`.

        Returns
        -------
        callable
            The function itself, unchanged, or the decorator when called with options.
        """
        def wrap(f: Callable) -> Callable:
            self.add(f, name=name, description=description)
            return f
        return wrap(func) if func is not None else wrap

    def extend(self, tools: Iterable[ToolLike]) -> None:
        """
        Register several tools.

        Parameters
        ----------
        tools : iterable of BaseTool or callable
            The tools, as in :meth:`add`.
        """
        for t in tools:
            self.add(t)

    def get(self, names: Union[str, Iterable[str]]) -> List[BaseTool]:
        """
        Return tools by name.

        Parameters
        ----------
        names : str or iterable of str
            One name, or several.

        Returns
        -------
        list of BaseTool
            The tools, in the order of ``names``.

        Raises
        ------
        KeyError
            If a name is not registered.
        """
        names = [names] if isinstance(names, str) else list(names)
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"unregistered tools: {missing}; registered: {sorted(self._tools)}")
        return [self._tools[n] for n in names]

    def __contains__(self, name: str) -> bool:
        """bool : Whether a tool with this name is registered."""
        return name in self._tools

    def names(self) -> List[str]:
        """
        Return the registered names.

        Returns
        -------
        list of str
            The names, sorted.
        """
        return sorted(self._tools)
