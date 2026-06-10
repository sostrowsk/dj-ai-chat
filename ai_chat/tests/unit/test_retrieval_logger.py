"""Tests fuer den fail-safe RetrievalLogger-Service (Phase B2).

log_retrieval persistiert die Diagnostik aus
``SCRIBE.search_similar_chunks(return_diagnostics=True)`` und darf
NIEMALS raisen — Logging-Fehler duerfen den Chat-Stream nicht brechen.
"""

from unittest import mock

import pytest
from asgiref.sync import sync_to_async

from ai_chat.models import RetrievalLog
from ai_chat.services.retrieval_logger import log_retrieval
from ai_chat.tests.factories import ChatSessionFactory
from users.tests.factories import ClientFactory

acreate_user = sync_to_async(ClientFactory)
acreate_session = sync_to_async(ChatSessionFactory)

DIAGNOSTICS = {
    "candidate_scores": [0.032, 0.03, 0.012],
    "cutoff_config": {
        "rel_floor": 0.35,
        "elbow_drop": 0.45,
        "min_k": 3,
        "max_k": 50,
        "backend": "pgvector",
    },
    "final_k": 2,
}


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestLogRetrieval:
    async def test_persists_diagnostics_fields(self):
        session = await acreate_session()

        log = await log_retrieval(
            session=session,
            user=session.user,
            query="Wie hoch ist die Leasingrate?",
            scope="project",
            collection="project_7",
            diagnostics=DIAGNOSTICS,
            project_id=7,
            document_id=None,
            response_time_ms=123.4,
        )

        assert log is not None
        await log.arefresh_from_db()
        assert log.session_id == session.id
        assert log.user_id == session.user_id
        assert log.query == "Wie hoch ist die Leasingrate?"
        assert log.scope == "project"
        assert log.collection == "project_7"
        assert log.candidate_scores == [0.032, 0.03, 0.012]
        assert log.cutoff_config["backend"] == "pgvector"
        assert log.final_k == 2
        assert log.project_id == 7
        assert log.document_id is None
        assert log.response_time_ms == 123.4

    async def test_works_without_session_and_user(self):
        log = await log_retrieval(
            session=None,
            user=None,
            query="Frage",
            scope="general",
            collection="general_chat",
            diagnostics=DIAGNOSTICS,
        )

        assert log is not None
        assert log.session_id is None
        assert log.user_id is None

    async def test_truncates_query_to_5000_chars(self):
        log = await log_retrieval(
            session=None,
            user=None,
            query="Q" * 6000,
            scope="general",
            collection="general_chat",
            diagnostics=DIAGNOSTICS,
        )

        assert len(log.query) == 5000

    async def test_malformed_diagnostics_does_not_raise(self):
        log = await log_retrieval(
            session=None,
            user=None,
            query="Frage",
            scope="general",
            collection="general_chat",
            diagnostics=None,
        )

        assert log is not None
        assert log.candidate_scores == []
        assert log.cutoff_config == {}
        assert log.final_k == 0

    async def test_database_error_is_swallowed_and_returns_none(self, caplog):
        with mock.patch.object(RetrievalLog.objects, "create", side_effect=RuntimeError("db down")):
            log = await log_retrieval(
                session=None,
                user=None,
                query="Frage",
                scope="general",
                collection="general_chat",
                diagnostics=DIAGNOSTICS,
            )

        assert log is None
        assert any("Retrieval-Logging fehlgeschlagen" in rec.message for rec in caplog.records)
