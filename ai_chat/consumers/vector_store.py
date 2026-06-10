# ai_chat/consumers/vector_store.py
import logging
from typing import List, Optional, Tuple, Union

from asgiref.sync import sync_to_async
from django.conf import settings
from scribe.scribe_milvus import SCRIBE

from ai_chat import conf

Document = conf.get_document_model()

logger = logging.getLogger(__name__)

SearchResults = List[Tuple[Document, float]]


class VectorStoreManager:
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name: Optional[str] = collection_name
        self.scribe: Optional[SCRIBE] = None
        if collection_name:
            self.scribe = SCRIBE(collection_name)

    async def initialize(self) -> bool:
        """Backend-agnostic readiness check (pgvector or Milvus)."""
        try:
            if not self.scribe:
                return False
            ready = await sync_to_async(self.scribe.search_backend.is_ready)()
            if not ready:
                logger.error(f"Search backend '{settings.VECTORSTORE_BACKEND}' is not ready")
                return False
            logger.info("Vector store initialized successfully")
            return True
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Connection error initializing vector store: {str(e)}")
            return False
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Configuration error initializing vector store: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error initializing vector store: {str(e)}")
            return False

    async def search_similar_chunks(
        self,
        query: str,
        project_id: Optional[int] = None,
        document_id: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[SearchResults, Tuple[SearchResults, Optional[dict]]]:
        """Hybrid search via the SCRIBE facade.

        With ``return_diagnostics=True`` returns ``(results, diagnostics)``
        — the diagnostics dict feeds ai_chat's RetrievalLog. Errors degrade
        to ``([], None)`` so the chat stream never breaks.
        """
        empty = ([], None) if return_diagnostics else []
        if not self.scribe:
            return empty
        try:
            # No hard max_k: the SCRIBE facade applies the adaptive cutoff
            # (settings VECTORSTORE_MIN_K/MAX_K/RELATIVE_CUTOFF/ELBOW_DROP).
            result = await self.scribe.search_similar_chunks(
                query=query,
                project_id=project_id,
                document_id=document_id,
                return_diagnostics=return_diagnostics,
            )
            results = result[0] if return_diagnostics else result
            logger.info(f"Found {len(results)} similar chunks")
            return result
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Connection error during search: {str(e)}")
            return empty
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid search parameters: {str(e)}")
            return empty
        except Exception as e:
            logger.exception(f"Unexpected error during search: {str(e)}")
            return empty

    def close(self) -> None:
        if self.scribe:
            self.scribe.close()
