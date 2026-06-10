"""Source- und Persistenz-Payloads fuer den Chat-Consumer (Phase B4).

``build_sources`` ersetzt den alten ``relevant_context[:5]``-Loop mit
N+1-``aget``-Queries: EINE gebatchte Doc-Query (``in_bulk``), ein
Source-Eintrag pro adaptiv zurueckgegebenem Chunk. ``score`` ersetzt
``distance`` (Legacy-Key in Phase B7 entfernt).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ai_router.types import Document
from ai_router.utils.safe_markdown import safe_markdown_to_html

from ai_chat import conf

ProtectedProjectDocument = conf.get_document_model()

logger = logging.getLogger(__name__)

#: Laenge der Content-Preview im Source-Eintrag (vor Markdown-Rendering).
SOURCE_PREVIEW_LENGTH = 150

RelevantContext = List[Tuple[Document, float]]


def _document_id(chunk: Document) -> Optional[int]:
    try:
        return int(chunk.metadata.get("document_id"))
    except (TypeError, ValueError):
        return None


def build_sources(relevant_context: RelevantContext) -> List[Dict[str, Any]]:
    """Baut die Source-Liste fuer das EOS-Frame — eine gebatchte Doc-Query.

    Chunks ohne aufloesbares ``document_id`` (z.B. geloeschte Dokumente)
    werden uebersprungen statt den Chat-Turn zu brechen.
    """
    doc_ids = {doc_id for chunk, _ in relevant_context if (doc_id := _document_id(chunk)) is not None}
    if not doc_ids:
        return []

    documents = ProtectedProjectDocument.objects.in_bulk(doc_ids)
    sources = []
    skipped = 0
    for chunk, score in relevant_context:
        document = documents.get(_document_id(chunk))
        if document is None:
            skipped += 1
            continue
        sources.append(
            {
                "name": document.name,
                "id": document.pk,
                "score": round(float(score), 4),
                "page_number": chunk.metadata.get("page_number"),
                "content": safe_markdown_to_html(chunk.page_content[:SOURCE_PREVIEW_LENGTH]),
            }
        )
    if skipped:
        logger.debug(f"{skipped}/{len(relevant_context)} Chunks ohne aufloesbares Dokument uebersprungen")
    return sources


def build_retrieved_chunks(relevant_context: RelevantContext) -> List[Dict[str, Any]]:
    """Persistenz-Payload fuer ``ChatMessage.retrieved_chunks``.

    Volltext bleibt erhalten — die 1000-Zeichen-Truncation macht
    ``SessionManager._validate_retrieved_chunks`` beim Speichern.
    """
    return [
        {
            "content": chunk.page_content,
            "document_id": chunk.metadata.get("document_id"),
            "page_number": chunk.metadata.get("page_number"),
            "score": float(score),
            "document_path": chunk.metadata.get("document_path"),
        }
        for chunk, score in relevant_context
    ]


def resolve_provider(model: str) -> str:
    """Mappt einen Modellnamen auf den Provider (bedrock/vertex/azure)."""
    from ai_router.azure_client import AZURE_MODEL_CONFIG
    from ai_router.bedrock_client import BEDROCK_MODEL_CONFIG
    from ai_router.vertex_client import VERTEX_MODEL_CONFIG

    if model in BEDROCK_MODEL_CONFIG:
        return "bedrock"
    if model in VERTEX_MODEL_CONFIG:
        return "vertex"
    if model in AZURE_MODEL_CONFIG:
        return AZURE_MODEL_CONFIG[model].get("provider", "azure")
    return ""
