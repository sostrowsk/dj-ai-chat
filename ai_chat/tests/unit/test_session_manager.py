"""Tests fuer den SessionManager-Service (Phase B2).

Sync-Hilfsmethoden (_validate_*, _generate_session_title) werden direkt
getestet; die async API via pytest.mark.asyncio + django_db(transaction=True).
"""

from unittest import mock

import pytest
from asgiref.sync import sync_to_async
from django.db import IntegrityError

from ai_chat.models import ChatSession
from ai_chat.services.session_manager import SessionManager
from ai_chat.tests.factories import ChatMessageFactory, ChatSessionFactory
from project.tests.factories import ProjectFactory
from users.tests.factories import ClientFactory

acreate_user = sync_to_async(ClientFactory)
acreate_project = sync_to_async(ProjectFactory)
acreate_session = sync_to_async(ChatSessionFactory)
acreate_message = sync_to_async(ChatMessageFactory)


# =========================================================================
# Sync-Hilfsmethoden
# =========================================================================


class TestValidateRetrievedChunks:
    def test_keeps_valid_entry_with_all_known_keys(self):
        chunks = [
            {
                "content": "Maschinenleasing Vertragstext",
                "document_id": "42",
                "page_number": "7",
                "score": "0.91",
                "document_path": "vertrag.pdf",
            }
        ]
        result = SessionManager._validate_retrieved_chunks(chunks)

        assert result == [
            {
                "content": "Maschinenleasing Vertragstext",
                "document_id": 42,
                "page_number": 7,
                "score": 0.91,
                "document_path": "vertrag.pdf",
            }
        ]

    def test_truncates_chunk_content_to_1000_chars(self):
        chunks = [{"content": "A" * 1500}]
        result = SessionManager._validate_retrieved_chunks(chunks)

        assert len(result[0]["content"]) == 1000

    def test_discards_entries_without_content_or_non_dict(self):
        chunks = ["kein dict", 5, None, {"score": 0.5}, {"content": ""}]
        assert SessionManager._validate_retrieved_chunks(chunks) == []

    def test_invalid_optional_values_are_dropped_not_fatal(self):
        chunks = [{"content": "ok", "document_id": "abc", "score": "nan?"}]
        result = SessionManager._validate_retrieved_chunks(chunks)

        assert result == [{"content": "ok"}]

    def test_none_and_empty_input_yield_empty_list(self):
        assert SessionManager._validate_retrieved_chunks(None) == []
        assert SessionManager._validate_retrieved_chunks([]) == []


class TestValidateUsedDocuments:
    def test_casts_id_score_and_page_number(self):
        docs = [{"id": "12", "name": "Bilanz 2024", "score": "0.8", "page_number": "3"}]
        result = SessionManager._validate_used_documents(docs)

        assert result == [{"id": 12, "name": "Bilanz 2024", "score": 0.8, "page_number": 3}]

    def test_discards_entries_without_valid_id(self):
        docs = [{"name": "ohne id"}, {"id": "abc"}, "kein dict", None]
        assert SessionManager._validate_used_documents(docs) == []

    def test_none_and_empty_input_yield_empty_list(self):
        assert SessionManager._validate_used_documents(None) == []
        assert SessionManager._validate_used_documents([]) == []


@pytest.mark.django_db
class TestGenerateSessionTitle:
    def test_uses_first_message_content(self):
        session = ChatSessionFactory(title="")
        assert SessionManager._generate_session_title(session, "Was steht im Vertrag?") == "Was steht im Vertrag?"

    def test_truncates_to_255_chars(self):
        session = ChatSessionFactory(title="")
        assert len(SessionManager._generate_session_title(session, "B" * 400)) == 255

    def test_falls_back_to_date_for_empty_or_whitespace_content(self):
        session = ChatSessionFactory(title="")
        assert "Chat vom" in SessionManager._generate_session_title(session, "")
        assert "Chat vom" in SessionManager._generate_session_title(session, "   ")


# =========================================================================
# Async API
# =========================================================================


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestGetOrCreateSession:
    async def test_creates_new_session_with_scope_fields(self):
        user = await acreate_user()
        project = await acreate_project()

        session = await SessionManager.get_or_create_session(user, ChatSession.Scope.PROJECT, project=project)

        assert session.pk is not None
        assert session.user_id == user.id
        assert session.scope == ChatSession.Scope.PROJECT
        assert session.project_id == project.id
        assert session.document_id is None
        assert session.is_active is True

    async def test_returns_existing_active_session(self):
        user = await acreate_user()
        existing = await SessionManager.get_or_create_session(user, ChatSession.Scope.GENERAL)

        again = await SessionManager.get_or_create_session(user, ChatSession.Scope.GENERAL)

        assert again.pk == existing.pk

    async def test_creates_fresh_session_after_close(self):
        user = await acreate_user()
        first = await SessionManager.get_or_create_session(user, ChatSession.Scope.GENERAL)

        await SessionManager.close_session(first)
        second = await SessionManager.get_or_create_session(user, ChatSession.Scope.GENERAL)

        assert second.pk != first.pk

    async def test_integrity_error_race_falls_back_to_concurrent_session(self):
        """Race: zweiter Request hat die Session bereits angelegt -> Fallback-Lookup."""
        user = await acreate_user()
        real_create = ChatSession.objects.create

        def racing_create(**kwargs):
            real_create(**kwargs)  # der "andere" Request gewinnt
            raise IntegrityError("duplicate key value violates unique constraint")

        with mock.patch.object(ChatSession.objects, "create", side_effect=racing_create):
            session = await SessionManager.get_or_create_session(user, ChatSession.Scope.GENERAL)

        assert session is not None
        assert session.user_id == user.id
        assert await ChatSession.objects.filter(user=user, is_active=True).acount() == 1


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestSaveMessage:
    async def test_persists_user_message_and_sets_title_from_first_message(self):
        session = await acreate_session(title="")

        message = await SessionManager.save_message(session, "user", "Wie hoch ist die Leasingrate?")

        assert message.pk is not None
        assert message.role == "user"
        assert message.content == "Wie hoch ist die Leasingrate?"
        await session.arefresh_from_db()
        assert session.title == "Wie hoch ist die Leasingrate?"

    async def test_assistant_message_does_not_set_title(self):
        session = await acreate_session(title="")

        await SessionManager.save_message(session, "assistant", "Die Rate betraegt 500 EUR.")

        await session.arefresh_from_db()
        assert session.title == ""

    async def test_existing_title_is_not_overwritten(self):
        session = await acreate_session(title="Bestehender Titel")

        await SessionManager.save_message(session, "user", "Neue Frage")

        await session.arefresh_from_db()
        assert session.title == "Bestehender Titel"

    async def test_empty_user_content_gets_date_fallback_title(self):
        session = await acreate_session(title="")

        await SessionManager.save_message(session, "user", "   ")

        await session.arefresh_from_db()
        assert "Chat vom" in session.title

    async def test_sanitizes_chunks_and_documents(self):
        session = await acreate_session()

        message = await SessionManager.save_message(
            session,
            "assistant",
            "Antwort",
            retrieved_chunks=[{"content": "X" * 2000, "score": 0.7}, "invalid"],
            used_documents=[{"id": "9", "name": "Vertrag"}, {"name": "ohne id"}],
        )

        assert len(message.retrieved_chunks) == 1
        assert len(message.retrieved_chunks[0]["content"]) == 1000
        assert message.used_documents == [{"id": 9, "name": "Vertrag"}]

    async def test_extra_llm_metadata_is_persisted(self):
        session = await acreate_session()

        message = await SessionManager.save_message(
            session,
            "assistant",
            "Antwort",
            prompt_tokens=120,
            completion_tokens=45,
            provider="bedrock",
            model="claude-sonnet",
            duration_ms=850,
        )

        await message.arefresh_from_db()
        assert message.prompt_tokens == 120
        assert message.completion_tokens == 45
        assert message.provider == "bedrock"
        assert message.model == "claude-sonnet"
        assert message.duration_ms == 850

    async def test_unknown_extra_keys_are_ignored_not_fatal(self):
        session = await acreate_session()

        message = await SessionManager.save_message(session, "assistant", "Antwort", totally_unknown_field="x")

        assert message.pk is not None


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestLoadHistory:
    async def test_returns_role_content_dicts_in_chronological_order(self):
        session = await acreate_session()
        await acreate_message(session=session, role="user", content="Frage 1")
        await acreate_message(session=session, role="assistant", content="Antwort 1")

        history = await SessionManager.load_history(session)

        assert history == [
            {"role": "user", "content": "Frage 1"},
            {"role": "assistant", "content": "Antwort 1"},
        ]

    async def test_caps_to_latest_limit_messages(self):
        session = await acreate_session()
        for i in range(6):
            await acreate_message(session=session, role="user", content=f"Nachricht {i}")

        history = await SessionManager.load_history(session, limit=2)

        assert [msg["content"] for msg in history] == ["Nachricht 4", "Nachricht 5"]

    async def test_default_limit_comes_from_settings(self, settings):
        settings.AI_CHAT_MAX_HISTORY = 3
        session = await acreate_session()
        for i in range(5):
            await acreate_message(session=session, role="user", content=f"Nachricht {i}")

        history = await SessionManager.load_history(session)

        assert len(history) == 3
        assert history[0]["content"] == "Nachricht 2"


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestLoadHistoryEntries:
    async def test_assistant_entry_includes_sources_from_used_documents(self):
        session = await acreate_session()
        await acreate_message(session=session, role="user", content="Frage")
        await acreate_message(
            session=session,
            role="assistant",
            content="Antwort",
            used_documents=[{"id": 7, "name": "Bilanz.pdf", "score": 0.42, "page_number": 3}],
        )

        entries = await SessionManager.load_history_entries(session)

        assert entries == [
            {"role": "user", "content": "Frage"},
            {
                "role": "assistant",
                "content": "Antwort",
                "sources": [{"id": 7, "name": "Bilanz.pdf", "score": 0.42, "page_number": 3}],
            },
        ]

    async def test_assistant_without_used_documents_has_no_sources_key(self):
        session = await acreate_session()
        await acreate_message(session=session, role="assistant", content="Antwort")

        entries = await SessionManager.load_history_entries(session)

        assert entries == [{"role": "assistant", "content": "Antwort"}]

    async def test_caps_to_latest_limit_messages(self):
        session = await acreate_session()
        for i in range(4):
            await acreate_message(session=session, role="user", content=f"Nachricht {i}")

        entries = await SessionManager.load_history_entries(session, limit=2)

        assert [entry["content"] for entry in entries] == ["Nachricht 2", "Nachricht 3"]


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestCloseSession:
    async def test_marks_session_inactive(self):
        session = await acreate_session(is_active=True)

        await SessionManager.close_session(session)

        await session.arefresh_from_db()
        assert session.is_active is False
