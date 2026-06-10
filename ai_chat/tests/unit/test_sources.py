"""Phase B4/B7: adaptive Sources + Persistenz-Payloads fuer den Chat-Consumer.

``build_sources`` ersetzt den alten ``relevant_context[:5]``-Loop mit
N+1-``aget``-Queries: EINE gebatchte Doc-Query, ein Source-Eintrag pro
adaptiv zurueckgegebenem Chunk (``score`` statt ``distance``; der
uebergangsweise behaltene ``distance``-Key wurde in Phase B7 entfernt).
"""

import pytest
from ai_router.types import Document

from ai_chat.services.sources import build_retrieved_chunks, build_sources, resolve_provider
from data_room.tests.factories import ProtectedDocumentFactory
from project.tests.factories import ProjectFactory


def _chunk(document_id, score, page_number=None, content="Maschinen-Leasing Vertragsdetails", **extra_meta):
    metadata = {"document_id": document_id, **extra_meta}
    if page_number is not None:
        metadata["page_number"] = page_number
    return (Document(page_content=content, metadata=metadata), score)


@pytest.mark.django_db
class TestBuildSources:
    def test_returns_source_per_chunk_with_score_and_page(self):
        project = ProjectFactory()
        docs = [ProtectedDocumentFactory(project=project) for _ in range(3)]
        chunks = [_chunk(docs[i % 3].pk, 0.032 - i * 0.001, page_number=i + 1) for i in range(12)]

        sources = build_sources(chunks)

        assert len(sources) == 12
        for i, source in enumerate(sources):
            doc = docs[i % 3]
            assert source["name"] == doc.name
            assert source["id"] == doc.pk
            assert source["score"] == round(0.032 - i * 0.001, 4)
            assert source["page_number"] == i + 1
            assert "Maschinen-Leasing" in source["content"]

    def test_source_keys_are_final_wire_format_without_distance(self):
        """Phase B7: der Legacy-Key ``distance`` ist entfernt — das
        Wire-Format ist exakt {name, id, score, page_number, content}."""
        doc = ProtectedDocumentFactory()
        sources = build_sources([_chunk(doc.pk, 0.03, page_number=2)])

        assert set(sources[0]) == {"name", "id", "score", "page_number", "content"}

    def test_uses_single_batched_document_query(self, django_assert_num_queries):
        project = ProjectFactory()
        docs = [ProtectedDocumentFactory(project=project) for _ in range(3)]
        chunks = [_chunk(docs[i % 3].pk, 0.03 - i * 0.001) for i in range(12)]

        with django_assert_num_queries(1):
            build_sources(chunks)

    def test_skips_chunks_with_unknown_or_missing_document_id(self):
        doc = ProtectedDocumentFactory()
        chunks = [
            _chunk(doc.pk, 0.03),
            _chunk(doc.pk + 99999, 0.02),  # unbekanntes Dokument
            (Document(page_content="ohne id", metadata={}), 0.01),  # keine document_id
        ]

        sources = build_sources(chunks)

        assert [s["id"] for s in sources] == [doc.pk]

    def test_content_preview_truncated_to_150_chars(self):
        doc = ProtectedDocumentFactory()
        chunks = [_chunk(doc.pk, 0.03, content="A" * 300)]

        sources = build_sources(chunks)

        assert "A" * 150 in sources[0]["content"]
        assert "A" * 151 not in sources[0]["content"]

    def test_empty_context_returns_empty_list_without_query(self, django_assert_num_queries):
        with django_assert_num_queries(0):
            assert build_sources([]) == []


class TestBuildRetrievedChunks:
    def test_maps_content_metadata_and_score(self):
        chunks = [
            _chunk(7, 0.025, page_number=3, content="Volltext des Chunks", document_path="docs/bilanz.pdf"),
        ]

        payload = build_retrieved_chunks(chunks)

        assert payload == [
            {
                "content": "Volltext des Chunks",
                "document_id": 7,
                "page_number": 3,
                "score": 0.025,
                "document_path": "docs/bilanz.pdf",
            }
        ]

    def test_missing_optional_metadata_yields_none_values(self):
        payload = build_retrieved_chunks([(Document(page_content="nur Text", metadata={}), 0.01)])

        assert payload[0]["content"] == "nur Text"
        assert payload[0]["document_id"] is None
        assert payload[0]["page_number"] is None
        assert payload[0]["document_path"] is None


class TestResolveProvider:
    def test_bedrock_model(self):
        assert resolve_provider("claude-sonnet-4-6") == "bedrock"

    def test_vertex_model(self):
        assert resolve_provider("gemini-3.1-pro-preview") == "vertex"

    def test_azure_model(self):
        assert resolve_provider("gpt-5.4") == "azure"

    def test_unknown_model_returns_empty_string(self):
        assert resolve_provider("unbekanntes-modell") == ""
