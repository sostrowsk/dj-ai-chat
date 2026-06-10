"""Decoupling tests for ai_chat.conf — host-configurable project/data_room indirection.

ai_chat must not hard-import project/data_room at module level; the host
models, the accessible-projects hook and the indexed-document filters are
resolved via settings with leasing-defaults (AI_CHAT_PROJECT_MODEL,
AI_CHAT_DOCUMENT_MODEL, AI_CHAT_ACCESSIBLE_PROJECTS_FUNC,
AI_CHAT_INDEXED_DOCUMENT_FILTERS).
"""

import ast
from pathlib import Path

from django.test import RequestFactory, SimpleTestCase, override_settings

import ai_chat
from ai_chat import conf
from ai_chat.models import ChatMessage, ChatSession

_HOOK_SENTINEL = ["hook-result"]


def _fake_accessible_projects(user):
    """Hook target for the dotted-path tests — never touches the DB."""
    return _HOOK_SENTINEL


class TestConfDefaults(SimpleTestCase):
    """conf resolves to the leasing defaults when no setting is present."""

    def test_get_project_model_defaults_to_project_app(self):
        from project.models import Project

        self.assertIs(conf.get_project_model(), Project)

    def test_get_document_model_defaults_to_data_room(self):
        from data_room.models import ProtectedProjectDocument

        self.assertIs(conf.get_document_model(), ProtectedProjectDocument)

    def test_get_accessible_projects_func_defaults_to_none(self):
        self.assertIsNone(conf.get_accessible_projects_func())

    def test_get_indexed_document_filters_default_matches_legacy_dict(self):
        self.assertEqual(
            conf.get_indexed_document_filters(),
            {"indexing_status": "indexed", "reviewed": True, "disabled": False},
        )

    def test_get_indexed_document_filters_returns_copy(self):
        filters = conf.get_indexed_document_filters()
        filters["disabled"] = True
        self.assertEqual(conf.get_indexed_document_filters()["disabled"], False)


class TestConfOverrides(SimpleTestCase):
    """Hosts can repoint models/hook/filters via settings."""

    @override_settings(AI_CHAT_PROJECT_MODEL="ai_chat.ChatSession")
    def test_project_model_setting_overrides_default(self):
        self.assertIs(conf.get_project_model(), ChatSession)

    @override_settings(AI_CHAT_DOCUMENT_MODEL="ai_chat.ChatMessage")
    def test_document_model_setting_overrides_default(self):
        self.assertIs(conf.get_document_model(), ChatMessage)

    @override_settings(AI_CHAT_ACCESSIBLE_PROJECTS_FUNC="ai_chat.tests.unit.test_conf._fake_accessible_projects")
    def test_accessible_projects_func_setting_resolves_dotted_path(self):
        self.assertIs(conf.get_accessible_projects_func(), _fake_accessible_projects)

    @override_settings(AI_CHAT_INDEXED_DOCUMENT_FILTERS={"indexing_status": "indexed"})
    def test_indexed_document_filters_setting_overrides_default(self):
        self.assertEqual(conf.get_indexed_document_filters(), {"indexing_status": "indexed"})


class TestModelFKTargets(SimpleTestCase):
    """ChatSession FK targets stay byte-stable on the leasing defaults."""

    def test_session_project_fk_points_to_configured_model(self):
        field = ChatSession._meta.get_field("project")
        self.assertIs(field.remote_field.model, conf.get_project_model())
        self.assertEqual(field.remote_field.model._meta.label, "project.Project")

    def test_session_document_fk_points_to_configured_model(self):
        field = ChatSession._meta.get_field("document")
        self.assertIs(field.remote_field.model, conf.get_document_model())
        self.assertEqual(field.remote_field.model._meta.label, "data_room.ProtectedProjectDocument")

    def test_message_llm_log_fk_stays_on_peer_package(self):
        field = ChatMessage._meta.get_field("llm_log")
        self.assertEqual(field.remote_field.model._meta.label, "ai_router.LLMLog")


class TestAccessibleProjectsHook(SimpleTestCase):
    """ChatView delegates to the configured hook instead of the default query."""

    @override_settings(AI_CHAT_ACCESSIBLE_PROJECTS_FUNC="ai_chat.tests.unit.test_conf._fake_accessible_projects")
    def test_chat_view_uses_accessible_projects_hook(self):
        from ai_chat.views.chat import ChatView

        view = ChatView()
        view.request = RequestFactory().get("/")
        view.request.user = object()  # hook never touches the DB
        self.assertIs(view._get_accessible_projects(), _HOOK_SENTINEL)


class TestNoModuleLevelHostImports(SimpleTestCase):
    """No ai_chat production module may import project/data_room directly."""

    def test_ai_chat_sources_have_no_host_app_imports(self):
        package_dir = Path(ai_chat.__file__).resolve().parent
        forbidden = ("project", "data_room")
        offenders = []
        for path in sorted(package_dir.rglob("*.py")):
            relative = path.relative_to(package_dir)
            if "tests" in relative.parts or "__pycache__" in relative.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if any(name == app or name.startswith(f"{app}.") for app in forbidden):
                        offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], f"host app imports found in ai_chat: {offenders}")
