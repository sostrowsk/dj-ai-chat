"""Regression tests for WebSocket chat consumer authorization (P1 security).

Covers:
- [P1] connect() had ZERO authorization: any anonymous/other-tenant user could
  connect to ws/chat/ and stream RAG answers for arbitrary project/document IDs.
- [P1] _handle_initial trusted a client-supplied ``user_id`` (IDOR): a client
  could enumerate/act as any user's projects.

Strategy:
- Test A uses Channels' WebsocketCommunicator (full async path) to prove the
  anonymous handshake is rejected.
- Tests B and C unit-test the consumer's authorization logic synchronously by
  driving the handlers directly with real DB objects. This avoids the
  transaction=True / cross-connection FK flakiness that already forced the
  existing project-chat integration test to be skipped, while remaining a real
  RED -> GREEN regression for the per-resource permission check and the IDOR.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import AnonymousUser
from django_otp.plugins.otp_static.models import StaticDevice

from ai_chat.consumers.chat import ChatConsumer
from data_room.tests.factories import ProtectedDocumentFactory
from project.tests.project_utils import create_project
from users.factories import create_client


@pytest.fixture(autouse=True)
def use_in_memory_channel_layer(settings):
    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }


def _verified_client():
    user = create_client()
    StaticDevice.objects.create(user=user, confirmed=True)
    return user


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestAnonymousConnectionRejected:

    async def test_anonymous_connection_is_rejected(self):
        """Test A: anonymous scope -> connection rejected, not accepted."""
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = AnonymousUser()
        communicator.scope["url_route"] = {"kwargs": {}}

        connected, _ = await communicator.connect()
        await communicator.disconnect()

        assert connected is False, "Anonymous user must NOT be able to connect"


@pytest.mark.django_db
class TestProjectPermissionLogic:
    """The consumer reuses the SAME Project.check_permissions(user) the HTTP
    view uses. These call the sync permission function directly (no
    database_sync_to_async thread boundary) so they see test-transaction data.
    """

    def test_owning_client_is_granted_project_access(self):
        owner = _verified_client()
        project = create_project(user=owner)
        owner.client_company = project.client_company
        owner.save()

        consumer = ChatConsumer()
        # Bypass the database_sync_to_async wrapper to run the sync body inline.
        granted = ChatConsumer._can_access_project.__wrapped__(consumer, owner, project.id)
        assert granted is True

    def test_foreign_client_is_denied_project_access(self):
        """Test B (logic): a verified client of a DIFFERENT company is denied
        access to project X — even though the project exists."""
        owner = _verified_client()
        project = create_project(user=owner)
        owner.client_company = project.client_company
        owner.save()

        foreign = _verified_client()  # different client_company
        consumer = ChatConsumer()
        granted = ChatConsumer._can_access_project.__wrapped__(consumer, foreign, project.id)
        assert granted is False


class TestProjectChatHandlerWiring:
    """The handler must consult _can_access_project and refuse to set up the
    room / vector store when access is denied. No DB needed — the permission
    check is stubbed to isolate the authorization wiring.
    """

    def _make_consumer(self, user):
        consumer = ChatConsumer()
        consumer.scope = {"user": user, "url_route": {"kwargs": {}}}
        consumer.send_json = AsyncMock()
        consumer._handle_error = AsyncMock()
        consumer.room_manager = MagicMock()
        consumer.room_manager.setup_chat_room = AsyncMock()
        return consumer

    def test_denied_access_blocks_room_and_vector_store(self):
        consumer = self._make_consumer(MagicMock(is_authenticated=True))
        consumer._can_access_project = AsyncMock(return_value=False)

        with patch("ai_chat.consumers.chat.VectorStoreManager") as mock_vs:
            asyncio.run(consumer._handle_project_chat({"project_id": 999}))

            mock_vs.assert_not_called()
            consumer.room_manager.setup_chat_room.assert_not_called()
            consumer._handle_error.assert_awaited()
            sent_types = [c.args[0].get("type") for c in consumer.send_json.call_args_list if c.args]
            assert "project_connected" not in sent_types

    def test_granted_access_sets_up_project_chat(self):
        consumer = self._make_consumer(MagicMock(is_authenticated=True))
        consumer._can_access_project = AsyncMock(return_value=True)

        with patch("ai_chat.consumers.chat.VectorStoreManager") as mock_vs:
            mock_vs_instance = MagicMock()
            mock_vs.return_value = mock_vs_instance
            mock_vs_instance.initialize = AsyncMock(return_value=True)

            asyncio.run(consumer._handle_project_chat({"project_id": 42}))

            consumer._handle_error.assert_not_awaited()
            sent_types = [c.args[0].get("type") for c in consumer.send_json.call_args_list if c.args]
            assert "project_connected" in sent_types


@pytest.mark.django_db
class TestHandleInitialIgnoresClientUserId:

    def test_handle_initial_uses_scope_user_not_client_user_id(self):
        """Test C: client-supplied user_id is ignored; consumer uses scope['user'].

        A client sends a foreign victim's user_id. The consumer must derive the
        identity from scope['user'] and never query the victim's projects.
        """
        actor = create_client()
        victim = create_client()

        consumer = ChatConsumer()
        consumer.scope = {"user": actor}
        consumer.send_json = AsyncMock()
        consumer._handle_error = AsyncMock()

        captured = {}

        class FakeQS:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def values(self, *args, **kwargs):
                return self

            def __await__(self):
                async def _coro():
                    return []

                return _coro().__await__()

        with patch("ai_chat.consumers.chat.Project") as mock_project:
            mock_project.objects.filter.side_effect = lambda **kwargs: FakeQS(**kwargs)

            asyncio.run(consumer._handle_initial({"user_id": victim.id}))

            assert captured.get("user") == actor, (
                f"_handle_initial must use scope['user'] ({actor.id}), "
                f"not client-supplied user_id ({victim.id}). Got: {captured.get('user')}"
            )
            assert captured.get("user") != victim


@pytest.mark.django_db
class TestDocumentPermissionLogic:
    """[P1] Document chat must enforce the DOCUMENT-level permission check, not
    just the parent project's. ``ProtectedProjectDocument.check_permissions``
    additionally denies disabled/unreviewed uploads and legacy rows without an
    owner — a user with project access can otherwise read a hidden document's
    indexed chunks.

    These call the sync permission body inline (bypassing the
    database_sync_to_async wrapper) so they see test-transaction data.
    """

    def _owning_client_for_project(self, project):
        owner = _verified_client()
        owner.client_company = project.client_company
        owner.save()
        return owner

    def test_project_member_denied_disabled_document(self):
        """A verified client WITH project access is DENIED a disabled,
        non-client-owned document — even though project.check_permissions()
        would grant access to the project itself."""
        document = ProtectedDocumentFactory(
            user_type="broker",  # not the requesting client -> review/disabled gate applies
            reviewed=True,
            disabled=True,
        )
        owner = self._owning_client_for_project(document.project)

        # Sanity: the PROJECT check (the old, insufficient gate) would grant.
        assert document.project.check_permissions(owner) is True
        # The DOCUMENT check (the correct gate) must deny.
        assert document.check_permissions(owner) is False

        consumer = ChatConsumer()
        granted = ChatConsumer._can_access_document.__wrapped__(consumer, owner, document.id)
        assert granted is False, "Disabled document must be denied at WS document-chat gate"

    def test_project_member_denied_unreviewed_document(self):
        """A verified client WITH project access is DENIED an unreviewed,
        non-client-owned document."""
        document = ProtectedDocumentFactory(
            user_type="broker",
            reviewed=False,
            disabled=False,
        )
        owner = self._owning_client_for_project(document.project)

        assert document.project.check_permissions(owner) is True
        assert document.check_permissions(owner) is False

        consumer = ChatConsumer()
        granted = ChatConsumer._can_access_document.__wrapped__(consumer, owner, document.id)
        assert granted is False, "Unreviewed document must be denied at WS document-chat gate"

    def test_project_member_granted_visible_document(self):
        """A verified client WITH project access IS granted a reviewed, enabled
        document of their own company — the gate must not over-block."""
        document = ProtectedDocumentFactory(
            user_type="broker",
            reviewed=True,
            disabled=False,
        )
        owner = self._owning_client_for_project(document.project)

        assert document.check_permissions(owner) is True

        consumer = ChatConsumer()
        granted = ChatConsumer._can_access_document.__wrapped__(consumer, owner, document.id)
        assert granted is True


def _confirmed_device(user):
    """A confirmed OTP device the WS connect path can verify against."""
    return StaticDevice.objects.create(user=user, name="default", confirmed=True)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestOtpEnforcedOnConnect:
    """[P2] The HTTP chat view is behind OTPRequiredMixin, but the WS connect()
    accepts any authenticated session because django_otp's middleware does not
    run in the WS stack. connect() must additionally require that 2FA was
    completed in this session (a valid otp_device_id in the scope/session),
    mirroring OTPRequiredMixin.
    """

    async def test_authenticated_without_otp_is_rejected(self):
        user = await database_sync_to_async(create_client)()
        await database_sync_to_async(_confirmed_device)(user)  # device exists, but 2FA NOT completed this session

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = user
        communicator.scope["url_route"] = {"kwargs": {}}
        # No otp_device_id in scope -> session did not pass 2FA.

        connected, _ = await communicator.connect()
        await communicator.disconnect()

        assert connected is False, "Authenticated-but-not-OTP-verified session must be rejected"

    async def test_otp_verified_session_is_accepted(self):
        user = await database_sync_to_async(create_client)()
        device = await database_sync_to_async(_confirmed_device)(user)

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = user
        communicator.scope["url_route"] = {"kwargs": {}}
        # Session passed 2FA: the verified device's persistent id is in scope.
        communicator.scope["otp_device_id"] = device.persistent_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager:
            mock_instance = AsyncMock()
            mock_room_manager.return_value = mock_instance
            mock_instance.chat_room = MagicMock(project_id=None)

            connected, _ = await communicator.connect()
            await communicator.disconnect()

        assert connected is True, "OTP-verified session must be accepted"

    async def test_legacy_persistent_id_session_is_accepted(self):
        """[P2] django_otp's OTPMiddleware NORMALIZES the session device id
        before lookup: a session whose otp_device_id is the LEGACY full
        import-path persistent id (``<full.import.path>.Model/<pk>``) still
        resolves to the device on the HTTP path. The WS gate must mirror that
        normalization, otherwise an already-verified user whose session carries
        the legacy id gets the handshake rejected as OTP_REQUIRED while the HTTP
        chat page loads fine.
        """
        user = await database_sync_to_async(create_client)()
        device = await database_sync_to_async(_confirmed_device)(user)
        # Legacy form: full import path of the model class + "/<pk>".
        legacy_id = f"django_otp.plugins.otp_static.models.StaticDevice/{device.id}"

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = user
        communicator.scope["url_route"] = {"kwargs": {}}
        communicator.scope["otp_device_id"] = legacy_id

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager:
            mock_instance = AsyncMock()
            mock_room_manager.return_value = mock_instance
            mock_instance.chat_room = MagicMock(project_id=None)

            connected, _ = await communicator.connect()
            await communicator.disconnect()

        assert connected is True, (
            "Session carrying a legacy full-import-path otp_device_id must be "
            "accepted (HTTP OTPMiddleware normalizes & resolves it)"
        )


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestDebugOtpBypassOnConnect:
    """[P3] In DEBUG, this repo's ``DebugOTPMiddleware`` auto-verifies any
    authenticated request WITHOUT writing an otp_device_id into the session, so
    the HTTP chat page is served. The WS connect path must mirror that exact
    DEBUG-only bypass so local/default dev chat connects, while production
    (DEBUG=False) still enforces OTP.
    """

    async def test_debug_bypass_accepts_authenticated_without_device(self, settings):
        settings.DEBUG = True
        user = await database_sync_to_async(create_client)()  # no confirmed device, no otp_device_id in scope

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = user
        communicator.scope["url_route"] = {"kwargs": {}}

        with patch("ai_chat.consumers.chat.ChatRoomManager") as mock_room_manager:
            mock_instance = AsyncMock()
            mock_room_manager.return_value = mock_instance
            mock_instance.chat_room = MagicMock(project_id=None)

            connected, _ = await communicator.connect()
            await communicator.disconnect()

        assert connected is True, (
            "With DEBUG=True an authenticated session without an otp_device_id "
            "must connect (mirrors DebugOTPMiddleware)"
        )

    async def test_no_debug_bypass_when_debug_false(self, settings):
        """Regression guard: the SAME session (authenticated, no device) is
        still rejected when DEBUG=False — production keeps enforcing OTP."""
        settings.DEBUG = False
        user = await database_sync_to_async(create_client)()  # no confirmed device, no otp_device_id in scope

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = user
        communicator.scope["url_route"] = {"kwargs": {}}

        connected, _ = await communicator.connect()
        await communicator.disconnect()

        assert connected is False, (
            "With DEBUG=False an authenticated-but-not-OTP-verified session " "must still be rejected"
        )


@pytest.mark.django_db
class TestDebugOtpBypassMirrorsMiddlewareForPermissionChecks:
    """[P3] ``DebugOTPMiddleware`` patches ``request.user.is_verified`` to
    ``True`` for DEBUG. Downstream ``check_permissions`` (Project /
    ProtectedProjectDocument) consult ``user.is_verified()``, so the WS DEBUG
    bypass must patch the SCOPE user the SAME way — otherwise the chat page
    loads but project/document chat setup fails with "No access" for a local-dev
    user that has no OTP device.

    These drive connect() inline (with super().connect() and ChatRoomManager
    mocked) and exercise the permission-check bodies directly via __wrapped__,
    so they see test-transaction data without the transaction=True /
    cross-connection FK flakiness that the WebsocketCommunicator-based tests
    above must use.
    """

    def test_debug_connect_patches_scope_user_for_downstream_permission_checks(self, settings):
        """In DEBUG, a no-OTP user with access to a project can set up PROJECT
        and DOCUMENT chat: connect() must patch ``scope['user'].is_verified``
        exactly like DebugOTPMiddleware, so check_permissions passes.

        RED before the fix: ``is_verified()`` (== user_has_device) stays False
        for a user with no confirmed device, so ``_can_access_project`` /
        ``_can_access_document`` deny with "No access".
        """
        settings.DEBUG = True
        user = create_client()  # no confirmed OTP device
        project = create_project(user=user)
        user.client_company = project.client_company
        user.save()
        document = ProtectedDocumentFactory(
            project=project,
            user=user,
            user_type="client",
            reviewed=True,
            disabled=False,
        )

        # Sanity: a no-OTP user is NOT verified out of the box.
        assert user.is_verified() is False

        consumer = ChatConsumer()
        consumer.scope = {"user": user, "url_route": {"kwargs": {}}}
        # Stub the ASGI channel machinery a bare consumer lacks.
        consumer.channel_layer = None
        consumer.channel_name = "test-channel"

        # connect() must mirror DebugOTPMiddleware: patch is_verified on the
        # scope user so ALL downstream check_permissions see a verified user.
        # super().connect() + ChatRoomManager are mocked to isolate the gate.
        with patch("ai_chat.consumers.base.AsyncWebsocketConsumer.connect", new=AsyncMock()), patch(
            "ai_chat.consumers.chat.ChatRoomManager"
        ) as mock_room_manager:
            mock_instance = MagicMock()
            mock_instance.setup_chat_room = AsyncMock()
            mock_instance.chat_room = MagicMock(project_id=None)
            mock_room_manager.return_value = mock_instance
            asyncio.run(consumer.connect())

        scope_user = consumer.scope["user"]
        assert scope_user.is_verified() is True, (
            "DEBUG connect() must patch scope['user'].is_verified to True, " "mirroring DebugOTPMiddleware"
        )

        # Downstream permission checks (the SAME bodies the queued project /
        # document handlers call) now grant access for the no-OTP user.
        assert ChatConsumer._can_access_project.__wrapped__(consumer, scope_user, project.id) is True, (
            "PROJECT chat setup must succeed in DEBUG for a no-OTP owner " "(is_verified patched)"
        )
        assert ChatConsumer._can_access_document.__wrapped__(consumer, scope_user, document.id) is True, (
            "DOCUMENT chat setup must succeed in DEBUG for a no-OTP owner " "(is_verified patched)"
        )

    def test_no_is_verified_patch_when_debug_false(self, settings):
        """Regression guard: with DEBUG=False the scope user's is_verified is
        NOT patched — real OTP enforcement (user_has_device) is preserved."""
        settings.DEBUG = False
        user = create_client()  # no confirmed OTP device -> is_verified() False

        consumer = ChatConsumer()
        consumer.scope = {"user": user, "url_route": {"kwargs": {}}}
        # connect() rejects (no otp_device_id); self.close is mocked because the
        # ASGI send machinery isn't wired up for a bare consumer.
        consumer.close = AsyncMock()

        asyncio.run(consumer.connect())

        assert (
            consumer.scope["user"].is_verified() is False
        ), "With DEBUG=False connect() must NOT patch is_verified — OTP stays enforced"
