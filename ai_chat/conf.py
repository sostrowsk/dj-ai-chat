"""Host-configurable indirection for ai_chat's project/data_room coupling.

ai_chat chats over a host "project" model and a host "document" model and
filters them for chat-eligibility. Hosts repoint these via settings; the
defaults match the leasing monorepo:

- ``AI_CHAT_PROJECT_MODEL`` (default ``project.Project``)
- ``AI_CHAT_DOCUMENT_MODEL`` (default ``data_room.ProtectedProjectDocument``)
- ``AI_CHAT_ACCESSIBLE_PROJECTS_FUNC`` (optional dotted path ``func(user)``;
  default ``None`` -> package default implementation in ``views/chat.py``)
- ``AI_CHAT_INDEXED_DOCUMENT_FILTERS`` (queryset filters marking a document
  as chat-eligible; default: indexed, reviewed, not disabled)

Both models must expose ``check_permissions(user)`` (duck-typed host
contract). Note: the model settings are also read at model-definition time
for the ``ChatSession`` FKs and at import time for the module-level aliases
in consumers/views/services — hosts that override them need their own
migrations (see README of dj-ai-chat).
"""

from django.apps import apps
from django.conf import settings
from django.utils.module_loading import import_string

DEFAULT_PROJECT_MODEL = "project.Project"
DEFAULT_DOCUMENT_MODEL = "data_room.ProtectedProjectDocument"
DEFAULT_INDEXED_DOCUMENT_FILTERS = {
    "indexing_status": "indexed",
    "reviewed": True,
    "disabled": False,
}


def get_project_model():
    """Return the host model chat sessions can be scoped to."""
    return apps.get_model(getattr(settings, "AI_CHAT_PROJECT_MODEL", DEFAULT_PROJECT_MODEL))


def get_document_model():
    """Return the host model document-scoped chats talk about."""
    return apps.get_model(getattr(settings, "AI_CHAT_DOCUMENT_MODEL", DEFAULT_DOCUMENT_MODEL))


def get_accessible_projects_func():
    """Return the host's accessible-projects hook, or None for the default."""
    dotted_path = getattr(settings, "AI_CHAT_ACCESSIBLE_PROJECTS_FUNC", None)
    if not dotted_path:
        return None
    return import_string(dotted_path)


def get_indexed_document_filters():
    """Return a copy of the chat-eligibility queryset filters."""
    return dict(getattr(settings, "AI_CHAT_INDEXED_DOCUMENT_FILTERS", DEFAULT_INDEXED_DOCUMENT_FILTERS))
