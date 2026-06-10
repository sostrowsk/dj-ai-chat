"""Tests fuer die persistenten ai_chat-Models (Phase B1).

ChatSession: eine aktive Session pro (user, scope, project, document) —
inkl. NULL-Semantik (zwei aktive General-Sessions desselben Users sind
verboten, obwohl project/document NULL sind).
ChatMessage: Ordering nach timestamp, unabhaengige JSON-Defaults.
RetrievalLog: Roundtrip aller Diagnose-Felder.
"""

import datetime

import pytest
from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone

from ai_chat.models import ChatMessage, ChatSession
from ai_chat.tests.factories import (
    ChatMessageFactory,
    ChatSessionFactory,
    RetrievalLogFactory,
)
from data_room.tests.factories import ProtectedDocumentFactory
from project.tests.factories import ProjectFactory
from users.tests.factories import ClientFactory


@pytest.mark.django_db
class TestChatSessionUniqueConstraint:
    def test_second_active_general_session_for_same_user_raises_integrity_error(self):
        """NULL-Semantik: project/document sind NULL — Constraint muss trotzdem greifen."""
        user = ClientFactory()
        ChatSessionFactory(user=user, scope=ChatSession.Scope.GENERAL)
        with pytest.raises(IntegrityError):
            with transaction.atomic():
                ChatSessionFactory(user=user, scope=ChatSession.Scope.GENERAL)

    def test_second_active_session_for_same_project_raises_integrity_error(self):
        user = ClientFactory()
        project = ProjectFactory()
        ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=project)
        with pytest.raises(IntegrityError):
            with transaction.atomic():
                ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=project)

    def test_second_active_session_for_same_document_raises_integrity_error(self):
        user = ClientFactory()
        document = ProtectedDocumentFactory()
        ChatSessionFactory(user=user, scope=ChatSession.Scope.DOCUMENT, document=document)
        with pytest.raises(IntegrityError):
            with transaction.atomic():
                ChatSessionFactory(user=user, scope=ChatSession.Scope.DOCUMENT, document=document)

    def test_new_active_session_allowed_after_previous_is_deactivated(self):
        user = ClientFactory()
        old = ChatSessionFactory(user=user, scope=ChatSession.Scope.GENERAL)
        old.is_active = False
        old.save()
        new = ChatSessionFactory(user=user, scope=ChatSession.Scope.GENERAL)
        assert new.pk != old.pk
        assert ChatSession.objects.filter(user=user).count() == 2

    def test_active_sessions_for_different_projects_are_allowed(self):
        user = ClientFactory()
        ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=ProjectFactory())
        ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=ProjectFactory())
        assert ChatSession.objects.filter(user=user, is_active=True).count() == 2

    def test_active_sessions_for_different_users_are_allowed(self):
        ChatSessionFactory(scope=ChatSession.Scope.GENERAL)
        ChatSessionFactory(scope=ChatSession.Scope.GENERAL)
        assert ChatSession.objects.filter(is_active=True).count() == 2

    def test_general_and_project_scope_can_be_active_simultaneously(self):
        user = ClientFactory()
        ChatSessionFactory(user=user, scope=ChatSession.Scope.GENERAL)
        ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=ProjectFactory())
        assert ChatSession.objects.filter(user=user, is_active=True).count() == 2

    def test_deleting_two_documents_with_active_sessions_does_not_collide(self):
        """Regression (Codex P2): SET_NULL beim Loeschen zweier Dokumente setzte
        beide aktiven Document-Sessions auf document=NULL — unter
        nulls_distinct=False kollidierte das zweite Update mit IntegrityError
        und blockierte die Dokument-Loeschung."""
        user = ClientFactory()
        doc1 = ProtectedDocumentFactory()
        doc2 = ProtectedDocumentFactory()
        s1 = ChatSessionFactory(user=user, scope=ChatSession.Scope.DOCUMENT, document=doc1)
        s2 = ChatSessionFactory(user=user, scope=ChatSession.Scope.DOCUMENT, document=doc2)

        doc1.delete()
        doc2.delete()  # raised IntegrityError before the fix

        s1.refresh_from_db()
        s2.refresh_from_db()
        assert s1.document is None and s2.document is None

    def test_deleting_two_projects_with_active_sessions_does_not_collide(self):
        """Gleiche NULL-Kollision fuer projekt-scoped Sessions."""
        user = ClientFactory()
        p1, p2 = ProjectFactory(), ProjectFactory()
        ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=p1)
        ChatSessionFactory(user=user, scope=ChatSession.Scope.PROJECT, project=p2)

        p1.delete()
        p2.delete()  # raised IntegrityError before the fix

        assert ChatSession.objects.filter(user=user, project__isnull=True).count() == 2


@pytest.mark.django_db
class TestChatSessionRelations:
    def test_deleting_user_cascades_to_sessions(self):
        session = ChatSessionFactory()
        session.user.delete()
        assert not ChatSession.objects.filter(pk=session.pk).exists()

    def test_deleting_project_sets_session_project_null(self):
        project = ProjectFactory()
        session = ChatSessionFactory(scope=ChatSession.Scope.PROJECT, project=project)
        project.delete()
        session.refresh_from_db()
        assert session.project is None

    def test_user_reverse_accessor_is_ai_chat_sessions(self):
        session = ChatSessionFactory()
        assert list(session.user.ai_chat_sessions.all()) == [session]


@pytest.mark.django_db
class TestChatMessage:
    def test_messages_are_ordered_by_timestamp(self):
        session = ChatSessionFactory()
        first = ChatMessageFactory(session=session, content="erste")
        second = ChatMessageFactory(session=session, content="zweite", role=ChatMessage.Role.ASSISTANT)
        # Timestamps explizit gegen die Insert-Reihenfolge setzen
        now = timezone.now()
        ChatMessage.objects.filter(pk=first.pk).update(timestamp=now)
        ChatMessage.objects.filter(pk=second.pk).update(timestamp=now - datetime.timedelta(minutes=1))
        assert [m.pk for m in session.messages.all()] == [second.pk, first.pk]

    def test_json_defaults_are_independent_lists(self):
        msg_a = ChatMessageFactory()
        msg_b = ChatMessageFactory()
        msg_a.retrieved_chunks.append({"content": "x"})
        assert msg_b.retrieved_chunks == []
        assert msg_a.used_documents == []

    def test_optional_llm_fields_default_to_empty_or_null(self):
        msg = ChatMessageFactory()
        msg.refresh_from_db()
        assert msg.prompt_tokens is None
        assert msg.completion_tokens is None
        assert msg.duration_ms is None
        assert msg.provider == ""
        assert msg.model == ""
        assert msg.llm_log is None

    def test_deleting_session_cascades_to_messages(self):
        msg = ChatMessageFactory()
        msg.session.delete()
        assert not ChatMessage.objects.filter(pk=msg.pk).exists()


@pytest.mark.django_db
class TestRetrievalLog:
    def test_roundtrip_persists_all_diagnostics_fields(self):
        session = ChatSessionFactory()
        log = RetrievalLogFactory(
            session=session,
            user=session.user,
            query="Wie hoch ist die Leasingrate?",
            scope=ChatSession.Scope.PROJECT,
            project_id=42,
            document_id=7,
            collection="project_42",
            candidate_scores=[0.032, 0.030, 0.011],
            final_k=2,
            cutoff_config={"rel_floor": 0.35, "elbow_drop": 0.45, "min_k": 3, "max_k": 50, "backend": "pgvector"},
            response_time_ms=123.4,
        )
        log.refresh_from_db()
        assert log.query == "Wie hoch ist die Leasingrate?"
        assert log.scope == "project"
        assert log.project_id == 42
        assert log.document_id == 7
        assert log.collection == "project_42"
        assert log.candidate_scores == [0.032, 0.030, 0.011]
        assert log.final_k == 2
        assert log.cutoff_config["backend"] == "pgvector"
        assert log.response_time_ms == 123.4
        assert log.created_at is not None

    def test_deleting_session_keeps_log_with_null_session(self):
        log = RetrievalLogFactory()
        log.session.delete()
        log.refresh_from_db()
        assert log.session is None

    def test_deleting_user_keeps_log_with_null_user(self):
        log = RetrievalLogFactory()
        log.user.delete()
        log.refresh_from_db()
        assert log.user is None

    def test_json_defaults_are_independent(self):
        log_a = RetrievalLogFactory()
        log_b = RetrievalLogFactory()
        log_a.candidate_scores.append(0.5)
        log_a.cutoff_config["min_k"] = 1
        assert log_b.candidate_scores == []
        assert log_b.cutoff_config == {}


class TestSettings:
    def test_ai_chat_max_history_setting_defaults_to_20(self):
        assert settings.AI_CHAT_MAX_HISTORY == 20
