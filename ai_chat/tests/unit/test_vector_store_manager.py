"""Phase A6: VectorStoreManager is backend-agnostic — readiness comes from the
configured search backend and the hard max_k=20 cap is gone (the SCRIBE facade
applies the adaptive cutoff)."""

import asyncio
from unittest.mock import AsyncMock, patch

from django.test import TestCase

from ai_chat.consumers.vector_store import VectorStoreManager


class VectorStoreManagerTests(TestCase):
    def _manager_with_mock_scribe(self, mock_scribe_cls):
        manager = VectorStoreManager("project_1")
        return manager, mock_scribe_cls.return_value

    def test_initialize_returns_backend_readiness(self):
        with patch("ai_chat.consumers.vector_store.SCRIBE") as mock_scribe_cls:
            manager, scribe = self._manager_with_mock_scribe(mock_scribe_cls)
            scribe.search_backend.is_ready.return_value = True

            self.assertTrue(asyncio.run(manager.initialize()))
            scribe.search_backend.is_ready.assert_called_once()

    def test_initialize_false_when_backend_not_ready(self):
        with patch("ai_chat.consumers.vector_store.SCRIBE") as mock_scribe_cls:
            manager, scribe = self._manager_with_mock_scribe(mock_scribe_cls)
            scribe.search_backend.is_ready.return_value = False

            self.assertFalse(asyncio.run(manager.initialize()))

    def test_initialize_false_without_collection(self):
        manager = VectorStoreManager()
        self.assertFalse(asyncio.run(manager.initialize()))

    def test_search_does_not_force_hard_max_k(self):
        with patch("ai_chat.consumers.vector_store.SCRIBE") as mock_scribe_cls:
            manager, scribe = self._manager_with_mock_scribe(mock_scribe_cls)
            scribe.search_similar_chunks = AsyncMock(return_value=[])

            asyncio.run(manager.search_similar_chunks("Frage", project_id=3, document_id=None))

        kwargs = scribe.search_similar_chunks.await_args.kwargs
        self.assertNotIn("max_k", kwargs)
        self.assertEqual(kwargs["project_id"], 3)
        self.assertIsNone(kwargs["document_id"])

    def test_manager_exposes_collection_name(self):
        with patch("ai_chat.consumers.vector_store.SCRIBE"):
            manager = VectorStoreManager("general_chat")
        self.assertEqual(manager.collection_name, "general_chat")

    def test_search_with_diagnostics_passes_flag_and_returns_tuple(self):
        diagnostics = {"candidate_scores": [0.03], "cutoff_config": {}, "final_k": 1}
        with patch("ai_chat.consumers.vector_store.SCRIBE") as mock_scribe_cls:
            manager, scribe = self._manager_with_mock_scribe(mock_scribe_cls)
            scribe.search_similar_chunks = AsyncMock(return_value=([("doc", 0.03)], diagnostics))

            results, diag = asyncio.run(manager.search_similar_chunks("Frage", return_diagnostics=True))

        self.assertTrue(scribe.search_similar_chunks.await_args.kwargs["return_diagnostics"])
        self.assertEqual(results, [("doc", 0.03)])
        self.assertEqual(diag, diagnostics)

    def test_search_with_diagnostics_degrades_to_empty_tuple_on_error(self):
        with patch("ai_chat.consumers.vector_store.SCRIBE") as mock_scribe_cls:
            manager, scribe = self._manager_with_mock_scribe(mock_scribe_cls)
            scribe.search_similar_chunks = AsyncMock(side_effect=ConnectionError("down"))

            results, diag = asyncio.run(manager.search_similar_chunks("Frage", return_diagnostics=True))

        self.assertEqual(results, [])
        self.assertIsNone(diag)
