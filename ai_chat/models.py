"""Persistente Chat-Models: Session, Message und Retrieval-Diagnostik.

ChatSession/ChatMessage persistieren den Chat-Verlauf ueber
WebSocket-Disconnects hinweg. RetrievalLog speichert die
Pre-Cutoff-Diagnostik jeder Hybrid-Suche (Vertrag aus
``SCRIBE.search_similar_chunks(return_diagnostics=True)``) als
Datengrundlage fuer das ``tune_retrieval``-Command.
"""

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _


class ChatSession(models.Model):
    """Chat-Session mit optionalem Projekt-/Dokument-Kontext."""

    class Scope(models.TextChoices):
        GENERAL = "general", _("Allgemein")
        PROJECT = "project", _("Projekt")
        DOCUMENT = "document", _("Dokument")

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ai_chat_sessions",
    )
    scope = models.CharField(
        max_length=10,
        choices=Scope.choices,
        default=Scope.GENERAL,
    )
    project = models.ForeignKey(
        getattr(settings, "AI_CHAT_PROJECT_MODEL", "project.Project"),
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ai_chat_sessions",
    )
    document = models.ForeignKey(
        getattr(settings, "AI_CHAT_DOCUMENT_MODEL", "data_room.ProtectedProjectDocument"),
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ai_chat_sessions",
    )
    title = models.CharField(
        _("Titel"),
        max_length=255,
        blank=True,
        help_text=_("Automatisch aus erster Nachricht generiert"),
    )
    system_prompt = models.TextField(
        _("System-Prompt"),
        blank=True,
        help_text=_("Gerenderter System-Prompt-Snapshot fuer diese Session"),
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _("Chat-Session")
        verbose_name_plural = _("Chat-Sessions")
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "is_active"]),
            models.Index(fields=["user", "-updated_at"]),
        ]
        constraints = [
            # Ein Constraint pro Scope statt nulls_distinct=False ueber alles:
            # SET_NULL beim Loeschen von Projekt/Dokument darf scoped Sessions
            # nicht in eine NULL-Kollision laufen lassen (Codex P2) — genullte
            # FKs fallen via isnull-Bedingung aus dem partiellen Index.
            models.UniqueConstraint(
                fields=["user"],
                condition=models.Q(is_active=True, scope="general"),
                name="unique_active_general_chat_session",
            ),
            models.UniqueConstraint(
                fields=["user", "project"],
                condition=models.Q(is_active=True, scope="project", project__isnull=False),
                name="unique_active_project_chat_session",
            ),
            models.UniqueConstraint(
                fields=["user", "document"],
                condition=models.Q(is_active=True, scope="document", document__isnull=False),
                name="unique_active_document_chat_session",
            ),
        ]

    def __str__(self):
        return f"Chat {self.pk}: {self.title or 'Neue Unterhaltung'}"


class ChatMessage(models.Model):
    """Einzelne Chat-Nachricht mit RAG- und LLM-Metadaten."""

    class Role(models.TextChoices):
        USER = "user", _("Benutzer")
        ASSISTANT = "assistant", _("AI-Assistent")
        SYSTEM = "system", _("System")

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(
        max_length=10,
        choices=Role.choices,
    )
    content = models.TextField()
    retrieved_chunks = models.JSONField(
        default=list,
        blank=True,
        help_text=_(
            "RAG-Chunks mit Score/Seite; Chunk-Text auf 1000 Zeichen "
            "truncated — Volltext bleibt im Vector-Store reproduzierbar."
        ),
    )
    used_documents = models.JSONField(
        default=list,
        blank=True,
        help_text=_("Liste der verwendeten Dokument-IDs und Scores"),
    )
    prompt_tokens = models.PositiveIntegerField(null=True, blank=True)
    completion_tokens = models.PositiveIntegerField(null=True, blank=True)
    provider = models.CharField(
        max_length=50,
        blank=True,
        help_text=_("LLM-Provider (z.B. bedrock, azure, vertex)"),
    )
    model = models.CharField(max_length=100, blank=True)
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    llm_log = models.ForeignKey(
        "ai_router.LLMLog",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chat_messages",
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = _("Chat-Nachricht")
        verbose_name_plural = _("Chat-Nachrichten")
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["session", "timestamp"]),
        ]

    def __str__(self):
        preview = self.content[:50] if self.content else ""
        return f"[{self.role}] {preview}..."


class RetrievalLog(models.Model):
    """Diagnostik einer Hybrid-Suche (alle Pre-Cutoff-Scores, absteigend)."""

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="retrieval_logs",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="retrieval_logs",
    )
    query = models.TextField()
    scope = models.CharField(max_length=10, blank=True)
    project_id = models.IntegerField(null=True, blank=True)
    document_id = models.IntegerField(null=True, blank=True)
    collection = models.CharField(max_length=128)
    candidate_scores = models.JSONField(
        default=list,
        blank=True,
        help_text=_("Alle fused-RRF-Scores vor dem Cutoff, absteigend"),
    )
    final_k = models.PositiveIntegerField()
    cutoff_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("rel_floor, elbow_drop, min_k, max_k, backend"),
    )
    response_time_ms = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = _("Retrieval-Log")
        verbose_name_plural = _("Retrieval-Logs")
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["collection", "created_at"]),
        ]

    def __str__(self):
        return f"RetrievalLog {self.pk}: {self.collection} k={self.final_k}"
