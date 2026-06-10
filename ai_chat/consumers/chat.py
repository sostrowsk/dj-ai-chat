# ai_chat/consumers/chat.py
import asyncio
import json
import logging
import time
from typing import Any, Optional

from ai_router.client import get_llm_client
from ai_router.logging import allm_log
from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from django.conf import settings
from django.db import DatabaseError
from django.utils.translation import gettext
from django_otp import DEVICE_ID_SESSION_KEY

from ai_chat import conf
from ai_chat.models import ChatMessage, ChatSession
from ai_chat.services import SessionManager, build_retrieved_chunks, build_sources, log_retrieval, resolve_provider

from .base import BaseConsumer
from .chat_room import ChatRoomManager
from .config import ErrorCodes
from .message_handler import MessageHandler
from .vector_store import VectorStoreManager

Project = conf.get_project_model()
ProtectedProjectDocument = conf.get_document_model()

logger = logging.getLogger(__name__)


class ChatConsumer(BaseConsumer):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.room_manager: Optional[ChatRoomManager] = None
        self.vector_store: Optional[VectorStoreManager] = None
        self.message_handler: Optional[MessageHandler] = None
        self.session: Optional[ChatSession] = None
        self._session_scope: str = ChatSession.Scope.GENERAL
        self._session_project: Optional[Project] = None
        self._session_document: Optional[ProtectedProjectDocument] = None

    async def connect(self) -> None:
        user = self.scope.get("user")
        if user is None or not getattr(user, "is_authenticated", False):
            # Reject unauthenticated/anonymous handshakes before accept().
            await self.close(code=ErrorCodes.SETUP_FAILED)
            return
        # Mirror leasing.middleware_otp.DebugOTPMiddleware EXACTLY: in DEBUG,
        # patch the (authenticated) scope user so is_verified() returns True.
        # The middleware only patches the HTTP request.user; the WS stack does
        # not run it, so without this patch the queued project/document chat
        # handlers' check_permissions() -> user.is_verified() would deny a
        # local-dev user that has no OTP device ("No access"). Production
        # (DEBUG=False) is untouched: real OTP stays enforced below.
        if settings.DEBUG:
            user.is_verified = lambda: True
        # Enforce 2FA on the WS handshake. The HTTP chat view sits behind
        # OTPRequiredMixin, but django_otp's middleware does NOT run in the WS
        # stack, so an authenticated-but-not-OTP-verified session would
        # otherwise bypass 2FA here. Mirror the HTTP gate by requiring a valid
        # otp_device_id (DEVICE_ID_SESSION_KEY) carried in the WS scope.
        if not await self._is_otp_verified(user):
            await self.close(code=ErrorCodes.OTP_REQUIRED)
            return
        try:
            url_kwargs = self.scope["url_route"]["kwargs"]
            if not await self._user_can_access_url_route(user, url_kwargs):
                await self.close(code=ErrorCodes.SETUP_FAILED)
                return
            await super().connect()
            self.room_manager = ChatRoomManager(self.channel_layer, self.channel_name)
            await self.room_manager.setup_chat_room(url_kwargs)
            self.message_handler = MessageHandler(system_prompt=settings.AI_CHAT_SYSTEM_PROMPT)
            if self.room_manager.chat_room.project_id:
                collection_name = f"project_{self.room_manager.chat_room.project_id}"
                self.vector_store = VectorStoreManager(collection_name)
                if not await self.vector_store.initialize():
                    await self._handle_error(
                        "Failed to initialize vector store",
                        close_connection=True,
                        error_code=ErrorCodes.VECTOR_STORE_FAILED,
                    )
                    return
            await self._setup_session_from_kwargs(url_kwargs)
        except (AttributeError, KeyError, RuntimeError, ConnectionError) as e:
            await self._handle_error(
                f"Connection setup failed: {str(e)}",
                close_connection=True,
                error_code=ErrorCodes.CONNECT_FAILED,
            )

    @database_sync_to_async
    def _is_otp_verified(self, user: Any) -> bool:
        """Confirm 2FA was completed in THIS session, mirroring django_otp's
        OTPMiddleware: the session-stored ``otp_device_id`` must resolve to a
        confirmed device owned by ``user``. Mere existence of a device
        (``user.is_verified()`` is overridden to ``user_has_device``) is NOT
        sufficient — that would let an un-stepped-up session in.

        Two HTTP behaviours are mirrored so the WS gate accepts every session
        the HTTP chat view accepts:

        - DEBUG bypass (``leasing.middleware_otp.DebugOTPMiddleware``): in DEBUG
          any authenticated user is auto-verified WITHOUT an otp_device_id in
          the session. Production (DEBUG=False) keeps enforcing OTP.
        - Legacy persistent ids (``OTPMiddleware._normalize_persistent_id``):
          sessions storing django_otp's old full-import-path id are normalized
          before lookup, otherwise ``Device.from_persistent_id`` returns None.
        """
        from django_otp.middleware import OTPMiddleware
        from django_otp.models import Device

        # Mirror DebugOTPMiddleware: DEBUG auto-verifies authenticated users.
        if settings.DEBUG:
            return True

        device_id = self.scope.get("otp_device_id") or self.scope.get("session", {}).get(DEVICE_ID_SESSION_KEY)
        if not device_id:
            return False
        # Mirror OTPMiddleware: normalize legacy full-import-path ids first.
        device_id = OTPMiddleware._normalize_persistent_id(device_id)
        try:
            device = Device.from_persistent_id(device_id)
        except (ValueError, LookupError):
            return False
        return bool(device and device.confirmed and device.user_id == user.id)

    @database_sync_to_async
    def _can_access_project(self, user: Any, project_id: int) -> bool:
        """Reuse the SAME permission check the HTTP view/Project model uses."""
        try:
            project = Project.objects.get(id=int(project_id))
        except (Project.DoesNotExist, ValueError, TypeError):
            return False
        return bool(project.check_permissions(user))

    @database_sync_to_async
    def _can_access_document(self, user: Any, document_id: int) -> bool:
        """Authorize via the DOCUMENT's own permission check, not just the
        parent project's. ``ProtectedProjectDocument.check_permissions`` also
        denies disabled/unreviewed uploads and legacy rows without an owner —
        the project check alone would expose those indexed chunks.
        """
        try:
            document = ProtectedProjectDocument.objects.select_related("project").get(id=int(document_id))
        except (ProtectedProjectDocument.DoesNotExist, ValueError, TypeError):
            return False
        return bool(document.check_permissions(user))

    async def _user_can_access_url_route(self, user: Any, url_kwargs: dict) -> bool:
        """Authorize the project/document referenced in the connect URL (if any)."""
        if project_id := url_kwargs.get("project_id"):
            return await self._can_access_project(user, project_id)
        if document_id := url_kwargs.get("document_id"):
            return await self._can_access_document(user, document_id)
        return True

    @database_sync_to_async
    def _resolve_session_context(self, url_kwargs: dict) -> tuple:
        """Mappt project_id/document_id-Kwargs auf (scope, project, document)."""
        if document_id := url_kwargs.get("document_id"):
            document = ProtectedProjectDocument.objects.select_related("project").get(id=int(document_id))
            return ChatSession.Scope.DOCUMENT, document.project, document
        if project_id := url_kwargs.get("project_id"):
            return ChatSession.Scope.PROJECT, Project.objects.get(id=int(project_id)), None
        return ChatSession.Scope.GENERAL, None, None

    async def _setup_session_from_kwargs(self, kwargs: dict) -> None:
        """Session-Setup nach Scope-Wechsel — fail-soft.

        Persistenz/History-Replay sind Enhancements: schlaegt das Setup fehl,
        laeuft der Chat ohne Persistenz weiter (geloggt, kein Error-Frame).
        """
        try:
            scope_value, project, document = await self._resolve_session_context(kwargs)
            await self._setup_session(scope_value, project=project, document=document)
        except Exception:
            logger.exception(f"Session-Setup fehlgeschlagen ({kwargs}) — Chat laeuft ohne Persistenz weiter")
            if self.message_handler is None:
                self.message_handler = MessageHandler(system_prompt=settings.AI_CHAT_SYSTEM_PROMPT)

    async def _setup_session(self, scope: str, project=None, document=None) -> None:
        """Persistente Session holen/anlegen, MessageHandler seeden, History senden.

        Wird nach jedem Scope-Setup aufgerufen (connect/general/project/
        document). Assistant-Messages enthalten ``sources`` (rekonstruiert
        aus ``used_documents``).
        """
        self._session_scope = scope
        self._session_project = project
        self._session_document = document
        self.session = await SessionManager.get_or_create_session(
            self.scope.get("user"), scope, project=project, document=document
        )
        if self.session is None:
            logger.warning(f"Keine Chat-Session verfuegbar (scope={scope}) — History-Replay uebersprungen")
            self._seed_message_handler([])
            return
        entries = await SessionManager.load_history_entries(self.session)
        self._seed_message_handler(entries)
        await self.send_json({"type": "history", "messages": entries})

    def _seed_message_handler(self, entries: list) -> None:
        """Frischen MessageHandler mit der persistierten History befuellen."""
        self.message_handler = MessageHandler(system_prompt=settings.AI_CHAT_SYSTEM_PROMPT)
        for entry in entries:
            if entry["role"] == "user":
                self.message_handler.chat_history.add_user_message(entry["content"])
            elif entry["role"] == "assistant":
                self.message_handler.chat_history.add_ai_message(entry["content"])

    async def receive(self, text_data: str) -> None:
        try:
            data = json.loads(text_data)
            message_type = data.get("type", "message")
            handlers = {
                "initial": self._handle_initial,
                "message": self._handle_chat_message,
                "clear": self._handle_clear_history,
                "general": self._handle_general_chat,
                "project": self._handle_project_chat,
                "document": self._handle_document_chat,
            }
            handler = handlers.get(message_type)
            if handler:
                await handler(data)
            else:
                await self._handle_error("Unknown message type", error_code=ErrorCodes.INVALID_MESSAGE)
        except json.JSONDecodeError:
            await self._handle_error("Invalid JSON data", error_code=ErrorCodes.INVALID_JSON)
        except (KeyError, TypeError, RuntimeError) as e:
            await self._handle_error(
                f"Error processing message: {str(e)}",
                error_code=ErrorCodes.PROCESSING_FAILED,
            )

    async def _handle_initial(self, data: dict) -> None:
        # SECURITY: never trust a client-supplied user_id (IDOR). Always derive
        # the identity from the authenticated WebSocket scope.
        try:
            user = self.scope.get("user")
            if user is None or not getattr(user, "is_authenticated", False):
                await self._handle_error("User not found", error_code=ErrorCodes.USER_NOT_FOUND)
                return
            projects = await Project.objects.filter(user=user, active=True).values("id", "name")
            await self.send_json({"type": "initial_projects", "projects": list(projects)})
        except (ValueError, TypeError, DatabaseError) as e:
            await self._handle_error(f"Initial setup failed: {str(e)}", error_code=ErrorCodes.SETUP_FAILED)

    async def _handle_chat_message(self, data: dict) -> None:
        query = data.get("message")
        if not query or not query.strip():
            await self._handle_error("Empty message", error_code=ErrorCodes.INVALID_MESSAGE)
            return
        if len(query) > 4000:
            await self._handle_error("Message too long (max 4000 chars)", error_code=ErrorCodes.INVALID_MESSAGE)
            return
        try:
            relevant_context = []
            diagnostics = None
            search_ms = None
            if self.vector_store:
                search_start = time.monotonic()
                relevant_context, diagnostics = await self.vector_store.search_similar_chunks(
                    query,
                    self.room_manager.chat_room.project_id,
                    self.room_manager.chat_room.document_id,
                    return_diagnostics=True,
                )
                search_ms = (time.monotonic() - search_start) * 1000
                if not relevant_context and not self.message_handler.chat_history.messages:
                    await self._handle_no_results()
                    return
            messages = self.message_handler._prepare_messages(query, relevant_context)
            # Sources fuer ALLE adaptiv zurueckgegebenen Chunks — eine
            # gebatchte Doc-Query statt N+1 aget-Loop.
            sources = await sync_to_async(build_sources)(relevant_context)
            await self._save_user_message(query)
            stream_info = await self._stream_openai_response(messages, query, sources)
            if stream_info is not None:
                await self._persist_assistant_turn(
                    query, relevant_context, sources, diagnostics, search_ms, stream_info
                )
        except (ConnectionError, ValueError, DatabaseError) as e:
            await self._handle_error(
                f"Chat message processing failed: {str(e)}",
                error_code=ErrorCodes.PROCESSING_FAILED,
            )

    async def _save_user_message(self, query: str) -> None:
        """User-Message VOR dem Stream persistieren — fail-soft."""
        if not self.session:
            return
        try:
            await SessionManager.save_message(self.session, ChatMessage.Role.USER, query)
        except Exception:
            logger.exception("User-Message konnte nicht persistiert werden")

    async def _persist_assistant_turn(
        self,
        query: str,
        relevant_context: list,
        sources: list,
        diagnostics: Optional[dict],
        search_ms: Optional[float],
        stream_info: dict,
    ) -> None:
        """Persistenz NACH dem EOS-Frame — blockiert das Streaming nie."""
        if self.session:
            try:
                extra = {
                    "model": settings.DEFAULT_MODEL_AI_CHAT,
                    "provider": resolve_provider(settings.DEFAULT_MODEL_AI_CHAT),
                    "duration_ms": stream_info["duration_ms"],
                }
                if stream_info.get("llm_log") is not None:
                    extra["llm_log"] = stream_info["llm_log"]
                # Phase B5: Streaming-Token-Usage in die ChatMessage.
                for token_key in ("prompt_tokens", "completion_tokens"):
                    if stream_info.get(token_key) is not None:
                        extra[token_key] = stream_info[token_key]
                await SessionManager.save_message(
                    self.session,
                    ChatMessage.Role.ASSISTANT,
                    stream_info["response"],
                    retrieved_chunks=build_retrieved_chunks(relevant_context),
                    used_documents=sources,
                    **extra,
                )
            except Exception:
                logger.exception("Assistant-Message konnte nicht persistiert werden")
        if diagnostics is not None:
            # log_retrieval ist fail-safe (raised nie).
            await log_retrieval(
                session=self.session,
                user=self.scope.get("user"),
                query=query,
                scope=self._session_scope,
                collection=getattr(self.vector_store, "collection_name", "") or "",
                diagnostics=diagnostics,
                project_id=self.room_manager.chat_room.project_id,
                document_id=self.room_manager.chat_room.document_id,
                response_time_ms=search_ms,
            )

    async def _handle_clear_history(self, data: dict) -> None:
        if not self.message_handler:
            await self._handle_error("Message handler not initialized", error_code=ErrorCodes.NOT_INITIALIZED)
            return
        self.message_handler.clear_history()
        if self.session:
            await SessionManager.close_session(self.session)
            self.session = await SessionManager.get_or_create_session(
                self.scope.get("user"),
                self._session_scope,
                project=self._session_project,
                document=self._session_document,
            )
        await self.send_json({"type": "history_cleared", "status": "success"})

    async def _initialize_vector_store(self, collection_name: str) -> bool:
        self.vector_store = VectorStoreManager(collection_name)
        if not await self.vector_store.initialize():
            await self._handle_error(
                "Failed to initialize vector store",
                close_connection=True,
                error_code=ErrorCodes.VECTOR_STORE_FAILED,
            )
            return False
        return True

    async def _handle_general_chat(self, data: dict) -> None:
        try:
            await self.room_manager.setup_chat_room({})
            if not await self._initialize_vector_store("general_chat"):
                return
            await self.send_json({"type": "general_connected", "status": "success"})
            await self._setup_session_from_kwargs({})
        except (ConnectionError, RuntimeError) as e:
            await self._handle_error(
                f"General chat setup failed: {str(e)}",
                error_code=ErrorCodes.SETUP_FAILED,
            )

    async def _handle_project_chat(self, data: dict) -> None:
        try:
            project_id = data.get("project_id")
            if not project_id:
                await self._handle_error("Missing project ID", error_code=ErrorCodes.INVALID_MESSAGE)
                return

            if not await self._can_access_project(self.scope.get("user"), project_id):
                await self._handle_error(
                    "No access to this project",
                    close_connection=True,
                    error_code=ErrorCodes.SETUP_FAILED,
                )
                return

            await self.room_manager.setup_chat_room({"project_id": project_id})
            if not await self._initialize_vector_store(f"project_{project_id}"):
                return
            await self.send_json(
                {
                    "type": "project_connected",
                    "project_id": project_id,
                    "status": "success",
                }
            )
            await self._setup_session_from_kwargs({"project_id": project_id})
        except Project.DoesNotExist:
            await self._handle_error("Project not found", error_code=ErrorCodes.PROJECT_NOT_FOUND)
        except (ConnectionError, RuntimeError) as e:
            await self._handle_error(
                f"Project chat setup failed: {str(e)}",
                error_code=ErrorCodes.SETUP_FAILED,
            )

    async def _handle_document_chat(self, data: dict) -> None:
        try:
            document_id = data.get("document_id")
            if not document_id:
                await self._handle_error("Missing document ID", error_code=ErrorCodes.INVALID_MESSAGE)
                return

            if not await self._can_access_document(self.scope.get("user"), document_id):
                await self._handle_error(
                    "No access to this document",
                    close_connection=True,
                    error_code=ErrorCodes.SETUP_FAILED,
                )
                return

            await self.room_manager.setup_chat_room({"document_id": document_id})
            document = await self.room_manager._get_document(document_id)
            if not await self._initialize_vector_store(f"project_{document.project.id}"):
                return
            await self.send_json(
                {
                    "type": "document_connected",
                    "document_id": document_id,
                    "status": "success",
                }
            )
            await self._setup_session_from_kwargs({"document_id": document_id})
        except ProtectedProjectDocument.DoesNotExist:
            await self._handle_error("Document not found", error_code=ErrorCodes.DOCUMENT_NOT_FOUND)
        except (ConnectionError, RuntimeError) as e:
            await self._handle_error(
                f"Document chat setup failed: {str(e)}",
                error_code=ErrorCodes.SETUP_FAILED,
            )

    async def _handle_no_results(self) -> None:
        try:
            response = gettext(
                "Zu Ihrer Anfrage konnte ich in den verfügbaren Dokumenten "
                "keine relevanten Informationen finden. Bitte formulieren Sie "
                "Ihre Frage um oder stellen Sie eine Frage zu einem anderen Thema."
            )
            await self.send_json({"type": "message", "message": response, "status": "success"})
        except (TypeError, json.JSONDecodeError, ConnectionError) as e:
            await self._handle_error(f"Error sending no results message: {str(e)}", close_connection=False)
        except Exception as e:
            logger.exception(f"Unexpected error handling no results: {str(e)}")
            await self._handle_error(f"Error handling no results: {str(e)}", close_connection=False)

    async def _stream_openai_response(self, messages: list, query: str, sources: list) -> Optional[dict]:
        """Streamt die LLM-Antwort; sendet das EOS-Frame mit den Sources.

        Rueckgabe bei Erfolg: ``{"response", "duration_ms", "llm_log"}``
        (Input fuer die Persistenz NACH dem EOS-Frame). Bei Fehlern wird
        ein Error-Frame gesendet und ``None`` zurueckgegeben — der Caller
        persistiert dann keine Assistant-Message.
        """
        client = get_llm_client(model=settings.DEFAULT_MODEL_AI_CHAT)

        system_prompt = ""
        chat_messages = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_prompt = msg.get("content", "")
            else:
                chat_messages.append(msg)

        try:
            full_response = ""
            llm_log_obj = None
            stream_started = time.monotonic()
            async with allm_log("ai_chat", settings.DEFAULT_MODEL_AI_CHAT, user_prompt=query) as log:
                # Run sync stream in thread, push chunks back via queue
                chunk_queue = asyncio.Queue()
                sentinel = object()

                async def _run_stream():
                    try:
                        stream_gen = await asyncio.to_thread(client.stream, system_prompt, messages=chat_messages)
                        for text_chunk in stream_gen:
                            await chunk_queue.put(text_chunk)
                    finally:
                        await chunk_queue.put(sentinel)

                stream_task = asyncio.create_task(_run_stream())

                while True:
                    item = await chunk_queue.get()
                    if item is sentinel:
                        break
                    full_response += item
                    await self.send_json(
                        {
                            "type": "message",
                            "message": item,
                            "status": "success",
                        }
                    )

                await stream_task
                log.output = full_response
                # Phase B5: Streaming-Token-Usage (cached_llm setzt
                # last_stream_usage nach Generator-Ende) in den LLMLog
                # kopieren — allm_log speichert die Felder beim Exit.
                usage_tokens = self._extract_stream_usage(client)
                for field, value in usage_tokens.items():
                    setattr(log, field, value)
                llm_log_obj = log if getattr(log, "pk", None) else None
            duration_ms = int((time.monotonic() - stream_started) * 1000)
            await self.send_json(
                {
                    "type": "message",
                    "message": "[EOS]",
                    "sources": sources,
                    "status": "success",
                }
            )
            self.message_handler.update_history(query, full_response)
            return {
                "response": full_response,
                "duration_ms": duration_ms,
                "llm_log": llm_log_obj,
                "prompt_tokens": usage_tokens.get("input_tokens"),
                "completion_tokens": usage_tokens.get("output_tokens"),
            }
        except (ConnectionError, TimeoutError, OSError) as e:
            await self._handle_error(f"API connection error: {str(e)}")
        except (ValueError, TypeError) as e:
            await self._handle_error(f"Invalid API parameters: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected streaming error: {str(e)}")
            await self._handle_error(f"Streaming failed: {str(e)}")
        return None

    @staticmethod
    def _extract_stream_usage(client: Any) -> dict:
        """Phase B5: Token-Usage des letzten stream()-Calls auslesen.

        ``CachedAnthropicClient`` setzt ``last_stream_usage`` nach
        Generator-Ende; Clients ohne das Attribut (z.B. Gemini) liefern
        ein leeres Dict. Nur int-Werte werden uebernommen — keine
        geratenen Werte ("a wrong answer is 3x worse than an empty field").
        """
        usage = getattr(client, "last_stream_usage", None)
        tokens = {}
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = getattr(usage, field, None)
            if isinstance(value, int):
                tokens[field] = value
        return tokens

    async def disconnect(self, close_code: int) -> None:
        try:
            if self.room_manager:
                await self.room_manager.leave_chat_group()
            if self.vector_store:
                self.vector_store.close()
            await super().disconnect(close_code)
        except Exception as e:
            logger.exception(f"Error during disconnect: {str(e)}")
