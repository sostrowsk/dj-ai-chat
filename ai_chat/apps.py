from django.apps import AppConfig
from django.apps import apps as django_apps
from django.core import checks

#: Peer Django apps ai_chat imports at runtime (scribe.scribe_milvus /
#: scribe.retrieval, ai_router.client / ai_router.logging / ai_router.types).
#: Per architecture rule the package does NOT declare them in pyproject —
#: the host pins all dj-* packages and this system check fails fast when a
#: peer is missing.
PEER_APPS = ("scribe", "ai_router")


def check_peer_apps(app_configs, **kwargs):
    errors = []
    for index, peer in enumerate(PEER_APPS, start=1):
        if not django_apps.is_installed(peer):
            errors.append(
                checks.Error(
                    f"ai_chat requires the '{peer}' Django app to be installed.",
                    hint=f"Add '{peer}' to INSTALLED_APPS (host pins the dj-* package).",
                    id=f"ai_chat.E{index:03d}",
                )
            )
    return errors


class AiChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_chat"

    def ready(self):
        checks.register(check_peer_apps)
