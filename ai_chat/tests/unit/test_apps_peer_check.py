"""Peer-requirement system check: ai_chat requires scribe + ai_router in the host.

ai_chat imports ``scribe.scribe_milvus`` / ``scribe.retrieval`` and
``ai_router.client`` / ``ai_router.logging`` / ``ai_router.types`` at runtime
but (per architecture rule) does NOT declare them as package dependencies —
the host pins all dj-* packages. The system check makes a missing peer fail
fast at ``manage.py check`` time.
"""

from unittest import mock

from django.test import SimpleTestCase


class TestAiChatPeerCheck(SimpleTestCase):
    def test_check_passes_when_peers_installed(self):
        from ai_chat.apps import check_peer_apps

        self.assertEqual(check_peer_apps(app_configs=None), [])

    def test_check_reports_one_error_per_missing_peer(self):
        from ai_chat.apps import check_peer_apps

        with mock.patch("ai_chat.apps.django_apps.is_installed", return_value=False):
            errors = check_peer_apps(app_configs=None)

        self.assertEqual({e.id for e in errors}, {"ai_chat.E001", "ai_chat.E002"})
        joined = " ".join(e.msg for e in errors)
        self.assertIn("scribe", joined)
        self.assertIn("ai_router", joined)

    def test_check_is_registered_with_django(self):
        from django.core.checks.registry import registry

        from ai_chat.apps import check_peer_apps

        self.assertIn(check_peer_apps, registry.registered_checks)
