# dj-ai-chat

RAG-backed AI chat for Django: WebSocket (Channels) chat consumer with
OTP/2FA gate, chat session + history management, retrieval source rendering
with scores/pages, retrieval logging and a `tune_retrieval` management
command.

Python package name: **`ai_chat`** (the repo name `dj-ai-chat` is only the
distribution name — app label, import path, DB tables and migrations stay
`ai_chat`).

## Installation

Installed by the host project as a Poetry git dependency (single lock
authority lives in the host):

```toml
[tool.poetry.dependencies]
dj-ai-chat = { git = "ssh://git@github.com/sostrowsk/dj-ai-chat.git", branch = "main" }
```

```python
INSTALLED_APPS = [
    ...
    "ai_chat.apps.AiChatConfig",
]
```

```python
# urls.py
path("ai-chat/", include("ai_chat.urls", namespace="ai_chat")),
```

```python
# asgi.py
from ai_chat.routing import websocket_urlpatterns as ai_chat_urlpatterns
# add ai_chat_urlpatterns to your ProtocolTypeRouter websocket URLRouter
# WebSocket consumer path: ws/chat/
```

## Peer requirements

These Django apps must be installed in the host (enforced fail-fast via
system checks `ai_chat.E001` / `ai_chat.E002` — see `ai_chat/apps.py`).
Per architecture rule they are NOT declared in `pyproject.toml`; only the
host pins dj-* packages:

| Peer app    | Package      | Used for                                              |
| ----------- | ------------ | ----------------------------------------------------- |
| `scribe`    | dj-rag-db    | `scribe.scribe_milvus.SCRIBE`, `scribe.retrieval`     |
| `ai_router` | dj-ai-router | `get_llm_client`, `allm_log`, `Document`, model config |

`ChatMessage.llm_log` is a literal FK to `ai_router.LLMLog`.

## Host contract

The host project MUST provide:

- **`AUTH_USER_MODEL`** — `ChatSession.user` is a FK to the swappable user
  model.
- **A "project" model** (`AI_CHAT_PROJECT_MODEL`) and a **"document" model**
  (`AI_CHAT_DOCUMENT_MODEL`) — both must expose a duck-typed
  `check_permissions(user)` method (raises/denies for unauthorized users).
- **URL namespace `project:detail`** — the chat template links back to the
  project detail page via `{% url "project:detail" project.id %}`.
- **Template `base_container_fluid.html`** — `ai_chat/chat.html` extends it
  and expects Bootstrap 5 styling plus `{% block scripts %}`.
- **`CHANNEL_LAYERS`** — a working Channels channel layer (e.g.
  `channels_redis`).
- **OTP/2FA** — `django-otp` middleware + `two_factor` (the chat view uses
  `OTPRequiredMixin`; the WebSocket consumer rejects non-OTP-verified
  sessions).
- **Vector store** — configured `scribe` backend (`VECTORSTORE_BACKEND`
  etc., see dj-rag-db README).

## Settings

Required (no package defaults):

| Setting                 | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `AI_CHAT_MAX_HISTORY`   | number of messages replayed as LLM context           |
| `AI_CHAT_SYSTEM_PROMPT` | system prompt template for the chat LLM              |
| `DEFAULT_MODEL_AI_CHAT` | model key (see dj-ai-router `ALL_MODEL_CONFIG`)      |
| `VECTORSTORE_BACKEND`   | scribe vector store backend (read by vector_store)   |

Optional (defaults via `getattr`, see `ai_chat/conf.py`):

| Setting                            | Default                                          |
| ---------------------------------- | ------------------------------------------------ |
| `AI_CHAT_PROJECT_MODEL`            | `"project.Project"`                              |
| `AI_CHAT_DOCUMENT_MODEL`           | `"data_room.ProtectedProjectDocument"`           |
| `AI_CHAT_ACCESSIBLE_PROJECTS_FUNC` | `None` (package default impl in `views/chat.py`) |
| `AI_CHAT_INDEXED_DOCUMENT_FILTERS` | `{"indexing_status": "indexed", "reviewed": True, "disabled": False}` |

**Caveat:** the model settings are read at model-definition time (FK
strings) and at import time (module-level aliases in consumers/views/
services). `override_settings` has no effect on them, and foreign hosts
that override them need their own migrations.

### Migrations in foreign hosts

`ai_chat` migrations pin host app labels
(`data_room.0028_...`, `project.0069_...`). For the original host these are
byte-identical no-ops. Foreign hosts without those apps must supply their
own migrations via `MIGRATION_MODULES = {"ai_chat": "<your_pkg>.migrations_ai_chat"}`.

## Frontend bundle

`ai_chat/static/ai_chat/chat.bundle.js` is a **committed, prebuilt** esbuild
bundle of `frontend/index.js` (bundles `marked` for Markdown rendering; the
template loads it via `{% static "ai_chat/chat.bundle.js" %}`).

After ANY change to `frontend/index.js`:

```bash
npm install        # once; devDeps: esbuild, marked
npm run build      # writes ai_chat/static/ai_chat/chat.bundle.js
git add ai_chat/static/ai_chat/chat.bundle.js && git commit
```

Do not gitignore the bundle — Poetry excludes gitignored files from the
wheel/sdist.

## Tests

Tests live in `ai_chat/tests/` and run from the host via:

```bash
pytest --pyargs ai_chat.tests
```

Note: several tests use host factories (`project.tests.factories`,
`users.factories`, `data_room.tests.factories`) — they run against the
original host (contract: tests run via the leasing host), not standalone.

## Development workflow

- Local override in the host:
  `poetry run pip install -e ../dj-ai-chat`
  (note: a later `poetry install` in the host resets to the locked git ref).
- Release: commit + push to `main`, then in the host
  `poetry update dj-ai-chat`.
