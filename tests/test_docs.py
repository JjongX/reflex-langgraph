"""The docstrings are part of the package: every example in them runs, and every public
module, function, class and method is documented in NumPy style:

- the opening quotes stand alone; then a one-line summary, a full sentence, with no ``x: y``
  (Sphinx's summary tables cut there);
- sections in order: Parameters, Returns (or Yields), Raises, Examples; every parameter is
  described, as ``name : type, default: value`` (or ``, optional``); classes describe their
  constructor's parameters, or their fields under Attributes;
- examples are doctests that run; one that needs a backslash goes in a raw docstring;
- properties get a one-line docstring, e.g. ``str : Its title.``;
- a module's docstring starts with "The <name> submodule ..." (the API page lists it).
"""

import doctest
import importlib
import inspect
import pkgutil
import re
import warnings
from pathlib import Path

import pytest

import reflex_langgraph

MODULES = sorted(m.name for m in pkgutil.walk_packages(reflex_langgraph.__path__,
                                                       "reflex_langgraph."))
#: Command modules document themselves through argparse (docs/cli.rst is made from it).
COMMANDS = {"reflex_langgraph.cli.init_env", "reflex_langgraph.cli.ui"}
#: Framework hooks: their signature and meaning are LangChain's.
HOOKS = {"wrap_model_call", "awrap_model_call"}
SECTION = re.compile(r"^(\w[\w ]*)\n-{3,}$", re.MULTILINE)


def _documented_modules():
    for name in MODULES:
        last = name.rsplit(".", 1)[-1]
        if name not in COMMANDS and (not last.startswith("_") or last == "__main__"):
            yield name


def _function(member):
    """The plain function behind a class member, if it is one we document."""
    fn = getattr(member, "fn", None) or getattr(member, "_fget", None) or member  # Reflex
    if isinstance(fn, property):
        fn = fn.fget
    if isinstance(fn, (staticmethod, classmethod)):
        fn = fn.__func__
    return fn if inspect.isfunction(fn) else None


def _public_objects():
    for name in _documented_modules():
        module = importlib.import_module(name)
        for attr, obj in vars(module).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != name:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                yield f"{name}.{attr}", obj
            if inspect.isclass(obj):
                for member_name, member in vars(obj).items():
                    fn = _function(member)
                    # Only what the package defines (Reflex adds a ``setvar`` to each state).
                    if fn is not None and fn.__module__ == name \
                            and not member_name.startswith("_") and member_name not in HOOKS:
                        yield f"{name}.{attr}.{member_name}", fn


def _sections(doc):
    """Section name -> its text, for a NumPy docstring."""
    found = list(SECTION.finditer(doc))
    return {m.group(1): doc[m.end():found[i + 1].start() if i + 1 < len(found) else None]
            for i, m in enumerate(found)}


def _parameters(obj):
    """Names a docstring must describe: a function's parameters, or a class's
    constructor arguments when the class defines its constructor."""
    if inspect.isclass(obj):
        if "__init__" not in vars(obj) or getattr(obj, "__pydantic_fields__", None) is not None \
                or getattr(obj, "_is_protocol", False):
            return []
        obj = obj.__init__
    params = inspect.signature(obj).parameters
    return [p for p in params if p not in ("self", "cls")]


@pytest.mark.parametrize("name", MODULES)
def test_examples_in_docstrings_run(name):
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0, f"{result.failed} example(s) failed in {name}"


@pytest.mark.parametrize("name", list(_documented_modules()))
def test_modules_say_what_they_are(name):
    doc = importlib.import_module(name).__doc__
    assert doc and doc.startswith("\n"), f"{name}: no docstring, or it starts on its first line"
    assert doc.strip().startswith("The "), f"{name}: should start with 'The ...'"


def test_public_api_is_documented_in_numpy_style():
    problems = []
    for qualname, obj in _public_objects():
        own = obj.__doc__
        doc = inspect.getdoc(obj)
        if not doc:
            problems.append(f"{qualname}: no docstring")
            continue
        summary = doc.strip().splitlines()[0]
        if not summary.endswith("."):
            problems.append(f"{qualname}: summary line should be a sentence: {summary!r}")
        if re.search(r"(?<! ): ", summary):  # "type : text" (properties) is fine
            problems.append(f"{qualname}: autosummary cuts a summary at 'x: y': {summary!r}")
        if not own:
            continue  # inherited from the method it overrides
        if "\n" in own.strip() and not own.startswith("\n"):
            problems.append(f"{qualname}: a long docstring starts on the line after the quotes")
        sections = _sections(inspect.cleandoc(own))
        expected = _parameters(obj)
        if expected:
            text = sections.get("Parameters", "") + sections.get("Attributes", "")
            missing = [p for p in expected if not re.search(rf"(^|[\s,*]){p}\b", text)]
            if missing:
                problems.append(f"{qualname}: parameters not described: {missing}")
        if inspect.isfunction(obj) and not isinstance(inspect.getattr_static(
                obj, "__wrapped__", None), property):
            returns = inspect.signature(obj).return_annotation
            one_liner = "\n" not in own.strip()
            if returns not in (inspect.Signature.empty, None, "None") and not one_liner \
                    and not {"Returns", "Yields"} & set(sections):
                problems.append(f"{qualname}: returns {returns} but has no Returns section")
    assert not problems, "\n".join(problems)


def test_no_invalid_escapes_in_the_source():
    # A docstring example with "\d" or "\n" belongs in a raw docstring.
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for path in Path(reflex_langgraph.__file__).parent.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")
