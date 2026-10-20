"""
The policy submodule decides who may see and do what.

One :class:`AccessPolicy` object is shared by the HTTP routes (:func:`create_server`,
:func:`jobs_router`) and the agent tools (:func:`job_tools`), so a rule cannot be enforced
on one path and forgotten on the other.

Two identities matter in a run:

* the **user** who triggered it -- the person who sent the message, or who answered the
  approval that resumed it. Actions (submit, cancel) are authorised as this user.
* the **thread owner** -- whose conversation it is. Whatever the agent reads lands in the
  owner's chat, so read access (list jobs, job status) is checked against the owner.

Both reach tools through ``config["configurable"]`` as ``user`` and ``thread_owner``. Both
carry current permissions only: ``thread_owner`` is the signed-in principal when the owner
is the one acting, and otherwise the owner's id and name without roles (a conversation never
stores roles). A reviewer acting in a client's conversation therefore gets the client's view
of jobs in that chat.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Whatever an ``auth`` hook returns for a caller, usually ``{"id", "name", "roles"}``.
Principal = Dict[str, Any]


def principal_id(user: Optional[Principal]) -> Optional[str]:
    """
    Return the stable id of a principal.

    Parameters
    ----------
    user : dict or str, optional
        The principal, as an ``auth`` hook returned it; ``None`` when anonymous.

    Returns
    -------
    str or None
        Its ``id``, else its ``name``; ``None`` when anonymous.

    Examples
    --------

    >>> from reflex_langgraph import principal_id
    >>> principal_id({'id': 'carol', 'name': 'Carol'})
    'carol'
    >>> principal_id(None) is None
    True
    """
    if user is None:
        return None
    if isinstance(user, dict):
        return str(user.get("id") or user.get("name") or "") or None
    return str(user) or None


def has_role(user: Optional[Principal], role: str) -> bool:
    """
    Check whether a principal has a role.

    Parameters
    ----------
    user : dict, optional
        The principal, with its roles under ``roles``.
    role : str
        Role name.

    Returns
    -------
    bool
        Whether ``role`` is among the principal's roles.

    Examples
    --------

    >>> from reflex_langgraph import has_role
    >>> has_role({'id': 'rita', 'roles': ['approver']}, 'approver')
    True
    >>> has_role(None, 'approver')
    False
    """
    return isinstance(user, dict) and role in (user.get("roles") or [])


class AccessPolicy:
    """
    Default rules for one team, with no reviewers.

    Conversations are private to their owner and the owner answers their own approvals.
    Everyone sees and may cancel every job. Without an ``auth`` hook (no identities) this
    means: everyone who reaches the server sees and does everything. Subclass and override
    what your app needs; every method receives the principal your ``auth`` hook returned
    (``None`` without auth).

    ``job`` arguments are dicts with ``job_id``, ``spec``, ``managed``, ``requested_by`` and
    ``approved_by`` (the ids recorded at submission; ``None`` for directories the runner did
    not create).
    """

    # --- conversations ---------------------------------------------------------------

    def can_view_thread(self, user: Optional[Principal], owner: Optional[str]) -> bool:
        """
        Decide whether a user may read and write a conversation.

        Parameters
        ----------
        user : dict, optional
            The caller.
        owner : str, optional
            Id of the conversation's owner; ``None`` for an anonymous conversation.

        Returns
        -------
        bool
            Whether the user may.
        """
        return owner is None or owner == principal_id(user)

    def can_decide(self, user: Optional[Principal], owner: Optional[str]) -> bool:
        """
        Decide whether a user may answer the approvals of a conversation.

        Parameters
        ----------
        user : dict, optional
            The caller.
        owner : str, optional
            Id of the conversation's owner.

        Returns
        -------
        bool
            Whether the user may.
        """
        return self.can_view_thread(user, owner)

    def can_review(self, user: Optional[Principal]) -> bool:
        """
        Decide whether a user works from the approvals queue across conversations.

        Parameters
        ----------
        user : dict, optional
            The caller.

        Returns
        -------
        bool
            Whether the user is a reviewer.
        """
        return False

    # --- jobs ------------------------------------------------------------------------

    def can_view_job(self, user: Optional[Principal], job: dict) -> bool:
        """
        Decide whether a user may see a job.

        Parameters
        ----------
        user : dict, optional
            The caller.
        job : dict
            The job's ``job_id``, ``spec``, ``managed``, ``requested_by`` and
            ``approved_by``.

        Returns
        -------
        bool
            Whether the user may.
        """
        return True

    def can_cancel_job(self, user: Optional[Principal], job: dict) -> bool:
        """
        Decide whether a user may cancel a job.

        Parameters
        ----------
        user : dict, optional
            The caller.
        job : dict
            As in :meth:`can_view_job`.

        Returns
        -------
        bool
            Whether the user may.
        """
        return self.can_view_job(user, job)

    def job_view(self, user: Optional[Principal], summary: dict, spec) -> dict:
        """
        Decide how much of a job's summary a user sees.

        Parameters
        ----------
        user : dict, optional
            The caller.
        summary : dict
            The job's full summary (see :meth:`JobRunner.summary
            <reflex_langgraph.jobs.JobRunner.summary>`).
        spec : JobSpec, optional
            The job's spec, whose :meth:`~reflex_langgraph.jobs.JobSpec.public_view`
            filters the summary.

        Returns
        -------
        dict
            What the user sees. By default, the spec's public view.
        """
        return spec.public_view(summary) if spec is not None else summary


class RolePolicy(AccessPolicy):
    """
    Users isolated from each other, plus an optional reviewer role.

    Everyone sees only their own conversations and the jobs they requested (through the
    spec's public view). By default only reviewers answer approvals:

    * ``owners_can_decide=True`` -- each user approves the requests in their own
      conversations (self-service); nobody else's.
    * ``owners_can_cancel=True`` -- each user may cancel the jobs they requested.

    Users with ``reviewer_role`` (if anyone has it) see every job in full, may cancel any
    job, and answer any approval from the approvals queue, without reading the chat.

    Parameters
    ----------
    reviewer_role : str, default: 'approver'
        Role that makes a user a reviewer.
    owners_can_decide : bool, default: False
        Let each user answer the approvals in their own conversations.
    owners_can_cancel : bool, default: False
        Let each user cancel the jobs they requested.

    Examples
    --------

    >>> from reflex_langgraph import RolePolicy
    >>> policy = RolePolicy('approver', owners_can_decide=True)
    >>> carol = {'id': 'carol', 'roles': []}
    >>> policy.can_decide(carol, owner='carol'), policy.can_decide(carol, owner='dave')
    (True, False)
    >>> policy.can_view_job(carol, {'requested_by': 'dave'})
    False
    >>> policy.can_view_job({'id': 'rita', 'roles': ['approver']}, {'requested_by': 'dave'})
    True
    """

    def __init__(self, reviewer_role: str = "approver", *, owners_can_decide: bool = False,
                 owners_can_cancel: bool = False):
        self.reviewer_role = reviewer_role
        self.owners_can_decide = owners_can_decide
        self.owners_can_cancel = owners_can_cancel

    def is_reviewer(self, user: Optional[Principal]) -> bool:
        """
        Check whether a user has the reviewer role.

        Parameters
        ----------
        user : dict, optional
            The caller.

        Returns
        -------
        bool
            Whether the user is a reviewer.
        """
        return has_role(user, self.reviewer_role)

    def can_decide(self, user, owner) -> bool:
        if self.is_reviewer(user):
            return True
        return self.owners_can_decide and owner is not None and owner == principal_id(user)

    def can_review(self, user) -> bool:
        return self.is_reviewer(user)

    def _owns(self, user, job: dict) -> bool:
        uid = principal_id(user)
        return uid is not None and job.get("requested_by") == uid

    def can_view_job(self, user, job: dict) -> bool:
        return self.is_reviewer(user) or self._owns(user, job)

    def can_cancel_job(self, user, job: dict) -> bool:
        return self.is_reviewer(user) or (self.owners_can_cancel and self._owns(user, job))

    def job_view(self, user, summary: dict, spec) -> dict:
        if self.is_reviewer(user):
            return summary
        return super().job_view(user, summary, spec)
