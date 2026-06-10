"""Read-only Admin fuer Chat-Sessions, -Messages und Retrieval-Logs.

Pattern wie ai_router/admin.py: kein Add/Change, Delete erlaubt
(Aufraeumen alter Logs).
"""

from django.contrib import admin

from ai_chat.models import ChatMessage, ChatSession, RetrievalLog


class ReadOnlyAdmin(admin.ModelAdmin):
    """Disable add/change; allow delete for cleanup."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return True


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    can_delete = False
    fields = ["timestamp", "role", "content", "provider", "model", "prompt_tokens", "completion_tokens"]
    readonly_fields = fields

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ChatSession)
class ChatSessionAdmin(ReadOnlyAdmin):
    list_display = ["id", "user", "scope", "project", "document", "title", "is_active", "updated_at"]
    list_filter = ["scope", "is_active", "updated_at"]
    search_fields = ["title", "user__email", "project__name", "document__name"]
    ordering = ["-updated_at"]
    date_hierarchy = "updated_at"
    inlines = [ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(ReadOnlyAdmin):
    list_display = ["id", "session", "role", "content_preview", "provider", "model", "duration_ms", "timestamp"]
    list_filter = ["role", "provider", "timestamp"]
    search_fields = ["content", "session__title", "session__user__email"]
    ordering = ["-timestamp"]
    date_hierarchy = "timestamp"

    @admin.display(description="Inhalt")
    def content_preview(self, obj):
        return obj.content[:80]


@admin.register(RetrievalLog)
class RetrievalLogAdmin(ReadOnlyAdmin):
    list_display = ["id", "collection", "scope", "query_preview", "final_k", "response_time_ms", "created_at"]
    list_filter = ["scope", "collection", "created_at"]
    search_fields = ["query", "collection", "user__email"]
    ordering = ["-created_at"]
    date_hierarchy = "created_at"

    @admin.display(description="Query")
    def query_preview(self, obj):
        return obj.query[:80]
