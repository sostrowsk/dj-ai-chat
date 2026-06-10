from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ai_router.types import Document
from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.conf import settings as django_settings
from django_otp.plugins.otp_static.models import StaticDevice

from ai_chat.consumers.chat import ChatConsumer
from ai_chat.models import ChatMessage, ChatSession, RetrievalLog
from ai_chat.services import SessionManager
from ai_chat.tests.factories import ChatSessionFactory
from data_room.tests.factories import ProtectedDocumentFactory
from project.tests.factories import ProjectFactory
from project.tests.project_utils import create_project
from users.factories import create_client


@pytest.fixture(autouse=True)
def use_in_memory_channel_layer(settings):
    """Use in-memory channel layer for all tests in this module."""
    settings.CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        },
    }


@pytest.fixture
def verified_client_user():
    user = create_client()
    device = StaticDevice.objects.create(user=user, confirmed=True)
    # Expose the device's persistent id so tests can mark the WS scope as
    # OTP-verified (the consumer enforces 2FA on connect()).
    user._ws_otp_device_id = device.persistent_id
    return user


@pytest.fixture
def project(verified_client_user):
    proj = create_project(user=verified_client_user)
    verified_client_user.client_company = proj.client_company
    verified_client_user.save()
    return proj


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestChatConsumer:

    async def test_chat_consumer_connect_authenticated(self, verified_client_user):
        communicator = WebsocketCommunicator(
            ChatConsumer.as_asgi(),
            "/ws/chat/",
        )
        communicator.scope["user"] = verified_client_user
        communicator.scope["url_route"] = {"kwargs": {}}
        communicator.scope["otp_device_id"] = verified_client_user._ws_otp_device_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager:
            mock_instance = AsyncMock()
            mock_room_manager.return_value = mock_instance
            mock_instance.chat_room = MagicMock(project_id=None)

            connected, _ = await communicator.connect()
            assert connected
            await communicator.disconnect()

    async def test_chat_consumer_connect_unauthenticated(self):
        from django.contrib.auth.models import AnonymousUser

        communicator = WebsocketCommunicator(
            ChatConsumer.as_asgi(),
            "/ws/chat/",
        )
        communicator.scope["user"] = AnonymousUser()
        communicator.scope["url_route"] = {"kwargs": {}}

        try:
            connected, _ = await communicator.connect()
            if connected:
                await communicator.disconnect()
                assert False, "Anonymous user should not be able to connect"
        except Exception:
            pass

    async def test_chat_consumer_general_chat_setup(self, verified_client_user):
        communicator = WebsocketCommunicator(
            ChatConsumer.as_asgi(),
            "/ws/chat/",
        )
        communicator.scope["user"] = verified_client_user
        communicator.scope["url_route"] = {"kwargs": {}}
        communicator.scope["otp_device_id"] = verified_client_user._ws_otp_device_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager, patch(
            "ai_chat.consumers.chat.VectorStoreManager"
        ) as mock_vector_store:

            mock_room_instance = AsyncMock()
            mock_room_manager.return_value = mock_room_instance
            mock_room_instance.chat_room = MagicMock(project_id=None)

            mock_vector_instance = AsyncMock()
            mock_vector_store.return_value = mock_vector_instance
            mock_vector_instance.initialize = AsyncMock(return_value=True)

            await communicator.connect()
            # Connect richtet die persistente Session ein und sendet History.
            history = await communicator.receive_json_from()
            assert history["type"] == "history"

            await communicator.send_json_to({"type": "general"})

            response = await communicator.receive_json_from()
            assert response["type"] == "general_connected"
            assert response["status"] == "success"

            await communicator.disconnect()

    @pytest.mark.skip(reason="Fixture transaction issue - to be fixed separately")
    async def test_chat_consumer_project_chat_setup(self, verified_client_user, project):
        communicator = WebsocketCommunicator(
            ChatConsumer.as_asgi(),
            "/ws/chat/",
        )
        communicator.scope["user"] = verified_client_user
        communicator.scope["url_route"] = {"kwargs": {}}
        communicator.scope["otp_device_id"] = verified_client_user._ws_otp_device_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager, patch(
            "ai_chat.consumers.chat.VectorStoreManager"
        ) as mock_vector_store:

            mock_room_instance = AsyncMock()
            mock_room_manager.return_value = mock_room_instance
            mock_room_instance.chat_room = MagicMock(project_id=None)

            mock_vector_instance = AsyncMock()
            mock_vector_store.return_value = mock_vector_instance
            mock_vector_instance.initialize = AsyncMock(return_value=True)

            await communicator.connect()
            await communicator.send_json_to({"type": "project", "project_id": project.id})

            response = await communicator.receive_json_from()
            assert response["type"] == "project_connected"
            assert response["project_id"] == project.id
            assert response["status"] == "success"

            await communicator.disconnect()

    async def test_chat_consumer_message_handling(self, verified_client_user):
        communicator = WebsocketCommunicator(
            ChatConsumer.as_asgi(),
            "/ws/chat/",
        )
        communicator.scope["user"] = verified_client_user
        communicator.scope["url_route"] = {"kwargs": {}}
        communicator.scope["otp_device_id"] = verified_client_user._ws_otp_device_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager, patch(
            "ai_chat.consumers.chat.get_llm_client"
        ) as mock_openai:

            mock_room_instance = AsyncMock()
            mock_room_manager.return_value = mock_room_instance
            mock_room_instance.chat_room = MagicMock(project_id=None, document_id=None)

            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            async def mock_stream(*args, **kwargs):
                chunk = MagicMock()
                chunk.content = "Test response"
                yield chunk

            mock_client.astream = mock_stream

            await communicator.connect()
            # Connect richtet die persistente Session ein und sendet History.
            history = await communicator.receive_json_from()
            assert history["type"] == "history"

            await communicator.send_json_to({"type": "message", "message": "Hello"})

            response = await communicator.receive_json_from()
            assert response["type"] == "message"
            assert "message" in response

            await communicator.disconnect()

    async def test_chat_consumer_disconnect(self, verified_client_user):
        communicator = WebsocketCommunicator(
            ChatConsumer.as_asgi(),
            "/ws/chat/",
        )
        communicator.scope["user"] = verified_client_user
        communicator.scope["url_route"] = {"kwargs": {}}
        communicator.scope["otp_device_id"] = verified_client_user._ws_otp_device_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager:
            mock_instance = AsyncMock()
            mock_room_manager.return_value = mock_instance
            mock_instance.chat_room = MagicMock(project_id=None)
            mock_instance.leave_chat_group = AsyncMock()

            await communicator.connect()
            await communicator.disconnect()

            mock_instance.leave_chat_group.assert_called_once()


def _build_communicator(user):
    communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
    communicator.scope["user"] = user
    communicator.scope["url_route"] = {"kwargs": {}}
    communicator.scope["otp_device_id"] = user._ws_otp_device_id
    return communicator


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestChatConsumerSessionLifecycle:
    """Phase B3: persistente ChatSession + History-Replay ueber WebSocket."""

    async def test_connect_general_sends_empty_history_and_creates_session(self, verified_client_user):
        communicator = _build_communicator(verified_client_user)
        connected, _ = await communicator.connect()
        assert connected

        response = await communicator.receive_json_from()
        assert response == {"type": "history", "messages": []}

        session = await ChatSession.objects.aget(user=verified_client_user, is_active=True)
        assert session.scope == ChatSession.Scope.GENERAL
        await communicator.disconnect()

    async def test_reconnect_replays_persisted_turns_with_sources(self, verified_client_user):
        session = await sync_to_async(ChatSessionFactory)(user=verified_client_user)
        await SessionManager.save_message(session, "user", "Wie hoch ist der Umsatz?")
        await SessionManager.save_message(
            session,
            "assistant",
            "Der Umsatz betraegt 5 Mio. EUR.",
            used_documents=[{"id": 7, "name": "Bilanz.pdf", "score": 0.42, "page_number": 3}],
        )

        communicator = _build_communicator(verified_client_user)
        connected, _ = await communicator.connect()
        assert connected

        response = await communicator.receive_json_from()
        assert response["type"] == "history"
        messages = response["messages"]
        assert messages[0] == {"role": "user", "content": "Wie hoch ist der Umsatz?"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Der Umsatz betraegt 5 Mio. EUR."
        assert messages[1]["sources"] == [{"id": 7, "name": "Bilanz.pdf", "score": 0.42, "page_number": 3}]

        # Reconnect nutzt die bestehende aktive Session, legt keine neue an.
        assert await ChatSession.objects.filter(user=verified_client_user).acount() == 1
        await communicator.disconnect()

    async def test_clear_history_closes_session_and_starts_new_one(self, verified_client_user):
        communicator = _build_communicator(verified_client_user)
        await communicator.connect()
        await communicator.receive_json_from()  # history-Frame vom Connect

        old_session = await ChatSession.objects.aget(user=verified_client_user, is_active=True)

        await communicator.send_json_to({"type": "clear"})
        response = await communicator.receive_json_from()
        assert response == {"type": "history_cleared", "status": "success"}

        await old_session.arefresh_from_db()
        assert old_session.is_active is False
        new_session = await ChatSession.objects.aget(user=verified_client_user, is_active=True)
        assert new_session.pk != old_session.pk
        await communicator.disconnect()

    async def test_project_scope_switch_loads_project_scoped_session(self, verified_client_user):
        project = await database_sync_to_async(self._create_project_for)(verified_client_user)
        project_session = await sync_to_async(ChatSessionFactory)(
            user=verified_client_user,
            scope=ChatSession.Scope.PROJECT,
            project=project,
        )
        await SessionManager.save_message(project_session, "user", "Projektfrage")
        await SessionManager.save_message(project_session, "assistant", "Projektantwort")

        communicator = _build_communicator(verified_client_user)
        with patch("ai_chat.consumers.chat.VectorStoreManager") as mock_vector_store:
            mock_vector_instance = AsyncMock()
            mock_vector_store.return_value = mock_vector_instance
            mock_vector_instance.initialize = AsyncMock(return_value=True)

            await communicator.connect()
            general_history = await communicator.receive_json_from()
            assert general_history == {"type": "history", "messages": []}

            await communicator.send_json_to({"type": "project", "project_id": project.id})
            connected_frame = await communicator.receive_json_from()
            assert connected_frame["type"] == "project_connected"

            history_frame = await communicator.receive_json_from()
            assert history_frame["type"] == "history"
            assert [m["content"] for m in history_frame["messages"]] == ["Projektfrage", "Projektantwort"]

            await communicator.disconnect()

    @staticmethod
    def _create_project_for(user):
        # ProjectFactory statt create_project: letzteres triggert den
        # broker_company-Default (id=1), der in transaction=True-Laeufen
        # mit verbrauchten Sequenzen nicht existiert (FK-Violation).
        return ProjectFactory(client_company=user.client_company)


def _make_streaming_client(mock_get_client, chunks, fail_after_chunks=False, usage=None):
    """get_llm_client-Mock: ``client.stream`` liefert einen sync Generator.

    ``usage`` simuliert ``CachedAnthropicClient.last_stream_usage`` — wird
    wie im echten Client erst nach Generator-Ende gesetzt (Phase B5).
    """
    client = MagicMock()
    client.last_stream_usage = None

    def make_stream(*args, **kwargs):
        def gen():
            for chunk in chunks:
                yield chunk
            if fail_after_chunks:
                raise RuntimeError("LLM mid-stream failure")
            client.last_stream_usage = usage

        return gen()

    client.stream = MagicMock(side_effect=make_stream)
    mock_get_client.return_value = client
    return client


def _make_vector_store_mock(mock_vs_cls, results, diagnostics, collection_name="general_chat"):
    instance = AsyncMock()
    instance.initialize = AsyncMock(return_value=True)
    instance.search_similar_chunks = AsyncMock(return_value=(results, diagnostics))
    instance.collection_name = collection_name
    instance.close = MagicMock()
    mock_vs_cls.return_value = instance
    return instance


async def _receive_until_eos(communicator):
    frames = []
    while True:
        frame = await communicator.receive_json_from()
        frames.append(frame)
        if frame.get("message") == "[EOS]":
            return frames


async def _wait_for_handler_completion(communicator):
    """Barrier: Consumer verarbeitet Frames sequenziell — sobald die Antwort
    auf das noop-Frame da ist, ist der vorherige Handler (inkl. Persistenz
    nach EOS) garantiert fertig."""
    await communicator.send_json_to({"type": "__noop__"})
    frame = await communicator.receive_json_from()
    assert frame["type"] == "error"


@database_sync_to_async
def _create_documents(count):
    project = ProjectFactory()
    return [ProtectedDocumentFactory(project=project) for _ in range(count)]


def _chunk(document_id, score, page_number, content="Maschinen-Leasing Vertragsdetails"):
    metadata = {"document_id": document_id, "page_number": page_number}
    return (Document(page_content=content, metadata=metadata), score)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestChatConsumerPersistence:
    """Phase B4: Persistenz vor/nach dem Stream + adaptive Sources im EOS-Frame."""

    async def _connect_general_with_vector_store(self, user):
        communicator = _build_communicator(user)
        connected, _ = await communicator.connect()
        assert connected
        await communicator.receive_json_from()  # history-Frame vom Connect
        await communicator.send_json_to({"type": "general"})
        connected_frame = await communicator.receive_json_from()
        assert connected_frame["type"] == "general_connected"
        await communicator.receive_json_from()  # history-Frame vom Scope-Setup
        return communicator

    async def test_eos_frame_lists_sources_for_all_adaptive_chunks(self, verified_client_user):
        docs = await _create_documents(3)
        chunks = [_chunk(docs[i % 3].pk, 0.032 - i * 0.001, page_number=i + 1) for i in range(12)]
        diagnostics = {
            "candidate_scores": [score for _, score in chunks] + [0.001],
            "cutoff_config": {"rel_floor": 0.35, "elbow_drop": 0.45, "min_k": 3, "max_k": 50, "backend": "pgvector"},
            "final_k": 12,
        }

        with patch("ai_chat.consumers.chat.VectorStoreManager") as mock_vs_cls, patch(
            "ai_chat.consumers.chat.get_llm_client"
        ) as mock_get_client:
            _make_vector_store_mock(mock_vs_cls, chunks, diagnostics)
            _make_streaming_client(mock_get_client, ["Hallo ", "Welt"])

            communicator = await self._connect_general_with_vector_store(verified_client_user)
            await communicator.send_json_to({"type": "message", "message": "Wie hoch ist der Umsatz?"})

            frames = await _receive_until_eos(communicator)
            eos = frames[-1]
            sources = eos["sources"]
            assert len(sources) == 12
            for i, source in enumerate(sources):
                doc = docs[i % 3]
                # Finales Wire-Format (Phase B7): exakt diese Keys,
                # der Legacy-Key "distance" ist entfernt.
                assert set(source) == {"name", "id", "score", "page_number", "content"}
                assert source["name"] == doc.name
                assert source["id"] == doc.pk
                assert source["score"] == round(0.032 - i * 0.001, 4)
                assert source["page_number"] == i + 1
                assert "Maschinen-Leasing" in source["content"]

            await communicator.disconnect()

    async def test_midstream_llm_error_persists_user_message_but_no_assistant(self, verified_client_user):
        with patch("ai_chat.consumers.chat.get_llm_client") as mock_get_client:
            _make_streaming_client(mock_get_client, ["Teil"], fail_after_chunks=True)

            communicator = _build_communicator(verified_client_user)
            await communicator.connect()
            await communicator.receive_json_from()  # history-Frame

            await communicator.send_json_to({"type": "message", "message": "Frage vor dem Crash"})

            partial = await communicator.receive_json_from()
            assert partial == {"type": "message", "message": "Teil", "status": "success"}
            error_frame = await communicator.receive_json_from()
            assert error_frame["type"] == "error"

            await _wait_for_handler_completion(communicator)
            assert await ChatMessage.objects.filter(role="user", content="Frage vor dem Crash").aexists()
            assert await ChatMessage.objects.filter(role="assistant").acount() == 0

            await communicator.disconnect()

    async def test_eos_persists_assistant_message_and_retrieval_log(self, verified_client_user):
        docs = await _create_documents(1)
        chunks = [
            _chunk(docs[0].pk, 0.03, page_number=2, content="X" * 1500),
            _chunk(docs[0].pk, 0.02, page_number=5),
        ]
        diagnostics = {
            "candidate_scores": [0.03, 0.02, 0.001],
            "cutoff_config": {"rel_floor": 0.35, "elbow_drop": 0.45, "min_k": 3, "max_k": 50, "backend": "pgvector"},
            "final_k": 2,
        }

        with patch("ai_chat.consumers.chat.VectorStoreManager") as mock_vs_cls, patch(
            "ai_chat.consumers.chat.get_llm_client"
        ) as mock_get_client:
            _make_vector_store_mock(mock_vs_cls, chunks, diagnostics)
            _make_streaming_client(mock_get_client, ["Hallo ", "Welt"])

            communicator = await self._connect_general_with_vector_store(verified_client_user)
            await communicator.send_json_to({"type": "message", "message": "Wie hoch ist der Umsatz?"})
            await _receive_until_eos(communicator)
            await _wait_for_handler_completion(communicator)

            assistant = await ChatMessage.objects.select_related("llm_log").aget(role="assistant")
            assert assistant.content == "Hallo Welt"
            assert assistant.retrieved_chunks[0]["content"] == "X" * 1000  # truncated
            assert assistant.retrieved_chunks[1]["page_number"] == 5
            assert assistant.used_documents[0]["id"] == docs[0].pk
            assert assistant.used_documents[0]["name"] == docs[0].name
            assert assistant.used_documents[0]["score"] == 0.03
            assert assistant.used_documents[0]["page_number"] == 2
            assert assistant.model == django_settings.DEFAULT_MODEL_AI_CHAT
            assert assistant.provider == "bedrock"
            assert assistant.duration_ms is not None
            assert assistant.llm_log is not None

            log = await RetrievalLog.objects.aget()
            assert log.query == "Wie hoch ist der Umsatz?"
            assert log.scope == ChatSession.Scope.GENERAL
            assert log.collection == "general_chat"
            assert log.candidate_scores == [0.03, 0.02, 0.001]
            assert log.final_k == 2
            assert log.cutoff_config == diagnostics["cutoff_config"]
            assert log.response_time_ms is not None
            assert log.user_id == verified_client_user.id
            assert log.session_id is not None

            await communicator.disconnect()

    async def test_eos_persists_streaming_token_usage_in_llm_log_and_chat_message(self, verified_client_user):
        """Phase B5: last_stream_usage des Clients landet nach dem EOS-Frame
        in LLMLog.input_tokens/output_tokens und
        ChatMessage.prompt_tokens/completion_tokens."""
        usage = SimpleNamespace(
            input_tokens=17,
            output_tokens=42,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        )
        with patch("ai_chat.consumers.chat.get_llm_client") as mock_get_client:
            _make_streaming_client(mock_get_client, ["Hallo ", "Welt"], usage=usage)

            communicator = _build_communicator(verified_client_user)
            await communicator.connect()
            await communicator.receive_json_from()  # history-Frame

            await communicator.send_json_to({"type": "message", "message": "Wie hoch ist der Umsatz?"})
            await _receive_until_eos(communicator)
            await _wait_for_handler_completion(communicator)

            assistant = await ChatMessage.objects.select_related("llm_log").aget(role="assistant")
            assert assistant.prompt_tokens == 17
            assert assistant.completion_tokens == 42
            assert assistant.llm_log is not None
            assert assistant.llm_log.input_tokens == 17
            assert assistant.llm_log.output_tokens == 42
            assert assistant.llm_log.cache_creation_input_tokens == 3
            assert assistant.llm_log.cache_read_input_tokens == 5

            await communicator.disconnect()

    async def test_stream_without_usage_leaves_token_fields_none(self, verified_client_user):
        """Ohne usage-Events (last_stream_usage=None) bleiben die
        Token-Felder None — kein Crash, keine geratenen Werte."""
        with patch("ai_chat.consumers.chat.get_llm_client") as mock_get_client:
            _make_streaming_client(mock_get_client, ["Antwort"], usage=None)

            communicator = _build_communicator(verified_client_user)
            await communicator.connect()
            await communicator.receive_json_from()  # history-Frame

            await communicator.send_json_to({"type": "message", "message": "Frage ohne usage"})
            await _receive_until_eos(communicator)
            await _wait_for_handler_completion(communicator)

            assistant = await ChatMessage.objects.select_related("llm_log").aget(role="assistant")
            assert assistant.prompt_tokens is None
            assert assistant.completion_tokens is None
            assert assistant.llm_log is not None
            assert assistant.llm_log.input_tokens is None
            assert assistant.llm_log.output_tokens is None

            await communicator.disconnect()

    async def test_no_results_message_is_german(self, verified_client_user):
        diagnostics = {"candidate_scores": [], "cutoff_config": {}, "final_k": 0}

        with patch("ai_chat.consumers.chat.VectorStoreManager") as mock_vs_cls:
            _make_vector_store_mock(mock_vs_cls, [], diagnostics)

            communicator = await self._connect_general_with_vector_store(verified_client_user)
            await communicator.send_json_to({"type": "message", "message": "Frage ohne Treffer"})

            response = await communicator.receive_json_from()
            assert response["type"] == "message"
            assert "keine relevanten Informationen" in response["message"]

            await communicator.disconnect()
