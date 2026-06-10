"""Session- und Message-Persistenz fuer den AI-Chat.

Port von arznei-muster-mello ``ai_chat/services/session_manager.py``,
ohne LangChain-Message-Klassen: ``load_history`` liefert plain
``{"role", "content"}``-Dicts, kompatibel zu
``ai_chat/consumers/message_handler.py``.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from asgiref.sync import sync_to_async
from django.conf import settings
from django.db import IntegrityError

from ai_chat.models import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

#: Chunk-Text wird truncated gespeichert — der Volltext bleibt im
#: Vector-Store (DocumentChunk) reproduzierbar.
CHUNK_TEXT_MAX_LENGTH = 1000

#: Erlaubte Passthrough-Felder fuer ``save_message(**extra)``.
ALLOWED_MESSAGE_EXTRA_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "provider",
    "model",
    "duration_ms",
    "llm_log",
}


class SessionManager:
    """Verwaltet Chat-Sessions und Message-Persistenz."""

    MAX_TITLE_LENGTH = 255

    @classmethod
    async def get_or_create_session(
        cls,
        user,
        scope: str,
        project=None,
        document=None,
    ) -> ChatSession:
        """Holt die aktive Session fuer (user, scope, project, document) oder erstellt sie.

        Race-Conditions auf den partiellen UniqueConstraint werden via
        IntegrityError-Fallback aufgeloest.
        """

        @sync_to_async
        def _get_or_create():
            lookup = {
                "user": user,
                "scope": scope,
                "project": project,
                "document": document,
                "is_active": True,
            }
            session = ChatSession.objects.filter(**lookup).first()
            if session:
                return session

            try:
                return ChatSession.objects.create(
                    user=user,
                    scope=scope,
                    project=project,
                    document=document,
                )
            except IntegrityError:
                # Race: ein paralleler Request hat die Session bereits angelegt.
                logger.debug(f"Race bei Session-Erstellung: user={user.pk}, scope={scope}")
                return ChatSession.objects.filter(**lookup).first()

        return await _get_or_create()

    @classmethod
    def _validate_retrieved_chunks(cls, chunks: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """Sanitized retrieved_chunks-Eintraege.

        Pflicht: ``content`` (nicht-leerer str, truncated auf
        ``CHUNK_TEXT_MAX_LENGTH``). Optional mit Type-Cast: ``document_id``
        (int), ``page_number`` (int), ``score`` (float), ``document_path``
        (str). Ungueltige Eintraege werden verworfen, ungueltige optionale
        Werte still weggelassen.
        """
        if not chunks:
            return []

        validated = []
        discarded = 0
        for chunk in chunks:
            if not isinstance(chunk, dict):
                discarded += 1
                continue
            content = chunk.get("content")
            if not content or not isinstance(content, str):
                discarded += 1
                continue

            entry: Dict[str, Any] = {"content": content[:CHUNK_TEXT_MAX_LENGTH]}
            for key, caster in (("document_id", int), ("page_number", int), ("score", float)):
                if chunk.get(key) is not None:
                    try:
                        entry[key] = caster(chunk[key])
                    except (ValueError, TypeError):
                        pass
            if chunk.get("document_path"):
                entry["document_path"] = str(chunk["document_path"])

            validated.append(entry)

        if discarded:
            logger.debug(f"{discarded}/{len(chunks)} ungueltige retrieved_chunks verworfen")
        return validated

    @classmethod
    def _validate_used_documents(cls, documents: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """Sanitized used_documents-Eintraege (Sources-Format des Consumers).

        Pflicht: ``id`` (int-castbar). Optional: ``name`` (str), ``score``
        (float), ``page_number`` (int). Eintraege ohne valide id werden
        verworfen.
        """
        if not documents:
            return []

        validated = []
        discarded = 0
        for doc in documents:
            if not isinstance(doc, dict):
                discarded += 1
                continue
            try:
                entry: Dict[str, Any] = {"id": int(doc["id"])}
            except (KeyError, ValueError, TypeError):
                discarded += 1
                continue

            if doc.get("name") is not None:
                entry["name"] = str(doc["name"])
            for key, caster in (("score", float), ("page_number", int)):
                if doc.get(key) is not None:
                    try:
                        entry[key] = caster(doc[key])
                    except (ValueError, TypeError):
                        pass

            validated.append(entry)

        if discarded:
            logger.debug(f"{discarded}/{len(documents)} ungueltige used_documents verworfen")
        return validated

    @classmethod
    def _generate_session_title(cls, session: ChatSession, content: str) -> str:
        """Titel aus erster User-Message, Fallback auf Datum."""
        if content and content.strip():
            return content[: cls.MAX_TITLE_LENGTH]
        return f"Chat vom {datetime.now().strftime('%d.%m.%Y')}"

    @classmethod
    async def save_message(
        cls,
        session: ChatSession,
        role: str,
        content: str,
        retrieved_chunks: Optional[List[dict]] = None,
        used_documents: Optional[List[dict]] = None,
        **extra,
    ) -> ChatMessage:
        """Speichert eine Nachricht; setzt den Session-Titel bei der ersten User-Message.

        ``extra`` erlaubt LLM-Metadaten (``ALLOWED_MESSAGE_EXTRA_FIELDS``);
        unbekannte Keys werden mit Warnung ignoriert statt zu raisen.
        """
        validated_chunks = cls._validate_retrieved_chunks(retrieved_chunks)
        validated_docs = cls._validate_used_documents(used_documents)

        unknown_keys = set(extra) - ALLOWED_MESSAGE_EXTRA_FIELDS
        if unknown_keys:
            logger.warning(f"save_message: unbekannte extra-Felder ignoriert: {sorted(unknown_keys)}")
        extra_fields = {key: value for key, value in extra.items() if key in ALLOWED_MESSAGE_EXTRA_FIELDS}

        @sync_to_async
        def _save():
            message = ChatMessage.objects.create(
                session=session,
                role=role,
                content=content,
                retrieved_chunks=validated_chunks,
                used_documents=validated_docs,
                **extra_fields,
            )

            if role == ChatMessage.Role.USER and not session.title:
                session.title = cls._generate_session_title(session, content)
                session.save(update_fields=["title", "updated_at"])
            else:
                session.save(update_fields=["updated_at"])

            return message

        return await _save()

    @classmethod
    async def load_history(
        cls,
        session: ChatSession,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Laedt die letzten ``limit`` Nachrichten chronologisch.

        Rueckgabe: ``[{"role": ..., "content": ...}]`` — kompatibel zum
        MessageHandler-Wire-Format.
        """
        if limit is None:
            limit = settings.AI_CHAT_MAX_HISTORY

        @sync_to_async
        def _load():
            messages = list(session.messages.order_by("-timestamp", "-pk")[:limit])
            messages.reverse()
            return [{"role": msg.role, "content": msg.content} for msg in messages]

        return await _load()

    @classmethod
    async def load_history_entries(
        cls,
        session: ChatSession,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Wie ``load_history``, aber inkl. ``sources`` fuer Assistant-Messages.

        ``sources`` wird aus ``used_documents`` rekonstruiert (Wire-Format des
        Consumers: ``id``, ``name``, ``score``, ``page_number``) — fuer das
        History-Replay-Frame ``{"type": "history", "messages": [...]}``.
        """
        if limit is None:
            limit = settings.AI_CHAT_MAX_HISTORY

        @sync_to_async
        def _load():
            messages = list(session.messages.order_by("-timestamp", "-pk")[:limit])
            messages.reverse()
            entries: List[Dict[str, Any]] = []
            for msg in messages:
                entry: Dict[str, Any] = {"role": msg.role, "content": msg.content}
                if msg.role == ChatMessage.Role.ASSISTANT and msg.used_documents:
                    entry["sources"] = msg.used_documents
                entries.append(entry)
            return entries

        return await _load()

    @classmethod
    async def close_session(cls, session: ChatSession) -> None:
        """Markiert eine Session als inaktiv."""

        @sync_to_async
        def _close():
            session.is_active = False
            session.save(update_fields=["is_active", "updated_at"])

        await _close()
