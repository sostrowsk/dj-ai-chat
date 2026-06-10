# dj-ai-chat

Django app package `ai_chat` (app label, import path, DB-Tabellen bleiben
`ai_chat`). Host-Projekte pinnen dieses Repo als Poetry-git-Dependency auf
`main` — jeder Push auf main ist sofort releasebar.

## TDD-Regeln (Pflicht)

- **Test zuerst, RED bestaetigen, dann implementieren, GREEN bestaetigen.**
- Bugfix = Regressionstest, der den Bug reproduziert und VOR dem Fix failt.
- Reine Moves: Import-Smoke-Tests.
- Tests laufen aus dem Host-Projekt: `pytest --pyargs ai_chat.tests`
  (das Package hat keine eigene Settings-/pytest-Infrastruktur; einige
  Tests nutzen Host-Factories aus project/users/data_room).
- LLM-/Netzwerk-/Milvus-Calls IMMER mocken — kein Test darf echte
  Provider-APIs oder einen echten Vector-Store treffen.

## Architektur-Regeln

- Keine Imports aus Host-Apps (users, project, leasing, ai_agents,
  data_room, ...). Host-Models NUR ueber `ai_chat/conf.py`
  (`AI_CHAT_PROJECT_MODEL`, `AI_CHAT_DOCUMENT_MODEL`,
  `AI_CHAT_ACCESSIBLE_PROJECTS_FUNC`, `AI_CHAT_INDEXED_DOCUMENT_FILTERS`).
- Peer-Apps `scribe` und `ai_router` duerfen direkt importiert werden —
  System-Check-gesichert (`ai_chat.E001`/`E002` in `ai_chat/apps.py`),
  aber NICHT in pyproject deklariert (nur der Host pinnt dj-* Packages).
- **Migrations-Byte-Stabilitaet:** Aenderungen duerfen keine neuen
  Migrationen im Host erzeugen (`makemigrations --check --dry-run` muss im
  Host clean bleiben). Modul-Level-Settings-FKs nicht "dynamisieren".
- **Frontend:** `frontend/index.js` wird per esbuild nach
  `ai_chat/static/ai_chat/chat.bundle.js` gebaut. Bundle ist COMMITTED —
  nach jeder JS-Aenderung `npm run build` + Bundle mit-committen
  (niemals gitignoren, Poetry exkludiert gitignorte Dateien).
- Settings-Katalog im README aktuell halten, wenn neue Settings dazukommen.
