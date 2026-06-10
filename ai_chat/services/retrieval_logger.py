"""Fail-safe Persistenz der Retrieval-Diagnostik.

Schreibt pro Hybrid-Suche einen ``RetrievalLog`` aus dem diagnostics-Dict
von ``SCRIBE.search_similar_chunks(return_diagnostics=True)``. Logging darf
den Chat-Stream NIEMALS brechen — jede Exception wird geschluckt und nur
als Warning geloggt.
"""

import logging
from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async

from ai_chat.models import RetrievalLog

logger = logging.getLogger(__name__)

QUERY_MAX_LENGTH = 5000


async def log_retrieval(
    session,
    user,
    query: str,
    scope: str,
    collection: str,
    diagnostics: Optional[Dict[str, Any]],
    project_id: Optional[int] = None,
    document_id: Optional[int] = None,
    response_time_ms: Optional[float] = None,
) -> Optional[RetrievalLog]:
    """Persistiert die Suche-Diagnostik; gibt bei Fehlern None zurueck statt zu raisen."""
    try:
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        candidate_scores = diagnostics.get("candidate_scores") or []
        cutoff_config = diagnostics.get("cutoff_config") or {}
        final_k = diagnostics.get("final_k") or 0

        @sync_to_async
        def _create():
            return RetrievalLog.objects.create(
                session=session,
                user=user,
                query=(query or "")[:QUERY_MAX_LENGTH],
                scope=scope or "",
                collection=collection or "",
                candidate_scores=candidate_scores,
                final_k=final_k,
                cutoff_config=cutoff_config,
                project_id=project_id,
                document_id=document_id,
                response_time_ms=response_time_ms,
            )

        return await _create()
    except Exception:
        logger.warning("Retrieval-Logging fehlgeschlagen", exc_info=True)
        return None
