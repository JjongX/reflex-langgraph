"""Sphinx configuration for the reflex-langgraph documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from reflex_langgraph import __version__  # noqa: E402

project = "reflex-langgraph"
author = "Jong Ha Shin"
copyright = "2026, Jong Ha Shin"
version = release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "autodocsumm",
    "myst_parser",
]

# Docstrings are NumPy style; types live in them, so signatures stay plain.
autodoc_default_options = {"autosummary": True}
autodoc_typehints = "none"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

autosectionlabel_prefix_document = True
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
# The README and changelog link to files in the repository, which are not doc pages.
suppress_warnings = ["myst.xref_missing"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"


def setup(app):
    """Teach autodoc about Reflex states, whose event handlers are wrapped functions."""
    import reflex as rx
    from reflex.event import EventHandler
    from sphinx.ext.autodoc import MethodDocumenter

    class EventHandlerDocumenter(MethodDocumenter):
        """Documents an event handler as the method it wraps."""

        objtype = "eventhandler"
        directivetype = "method"
        # Above the attribute documenters, which would claim the (callable) wrapper.
        priority = 100

        @classmethod
        def can_document_member(cls, member, membername, isattr, parent):
            return isinstance(member, EventHandler)

        def import_object(self, raiseerror=False):
            ok = super().import_object(raiseerror)
            if ok and isinstance(self.object, EventHandler):
                self.object = self.object.fn
            return ok

    def keep_documented_handlers(app, what, name, obj, skip, options):
        # Autodoc sees the wrapper, which has no docstring; keep the package's documented
        # handlers, and none of the ones Reflex adds to every state.
        if isinstance(obj, EventHandler):
            fn = obj.fn
            return not (fn.__doc__ and fn.__module__.startswith("reflex_langgraph"))
        return None

    def plain_state_signature(app, what, name, obj, options, signature, return_annotation):
        # States are never constructed by hand; Reflex's constructor arguments are noise.
        if what == "class" and isinstance(obj, type) and issubclass(obj, rx.State):
            return "", return_annotation
        return None

    # Summary tables read the wrapper's own docstring: give it the function's.
    from reflex_langgraph.ui import state

    for cls in vars(state).values():
        if isinstance(cls, type) and issubclass(cls, rx.State):
            for handler in vars(cls).values():
                if isinstance(handler, EventHandler) and handler.fn.__doc__:
                    object.__setattr__(handler, "__doc__", handler.fn.__doc__)

    app.add_autodocumenter(EventHandlerDocumenter)
    app.connect("autodoc-skip-member", keep_documented_handlers)
    app.connect("autodoc-process-signature", plain_state_signature)
