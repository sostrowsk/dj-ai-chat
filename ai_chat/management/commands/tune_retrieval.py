"""Tuning der adaptiven Cutoff-Thresholds aus geloggten Retrievals (Phase B6).

Drei Modi (das Command schreibt NIE Settings, es druckt nur Empfehlungen):

1. Offline-Replay (Default, kostenlos): replayt alle ``RetrievalLog``-Rows
   mit gespeicherten ``candidate_scores`` gegen das Config-Grid aus
   :mod:`ai_chat.services.threshold_tuning` und empfiehlt pro Collection +
   global die Config mit minimalem mean-k, die das Coverage-Ziel haelt.
2. ``--eval-set path.json`` (supervised): ``[{query,
   relevant_document_ids, collection?}]`` -> Live-Suche via SCRIBE,
   Precision/Recall am adaptiven Cutoff (aktuelle Settings).
3. ``--judge --sample N`` (optional, kostet LLM-Calls): re-runnt die
   juengsten N geloggten Queries live und laesst ein LLM via
   ``get_llm_client`` die Relevanz der behaltenen Chunks labeln.

Usage:
    python manage.py tune_retrieval [--coverage 0.9] [--json]
    python manage.py tune_retrieval --eval-set eval.json [--json]
    python manage.py tune_retrieval --judge --sample 20 [--json]
"""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List, Optional

from ai_router import get_llm_client
from django.core.management.base import BaseCommand, CommandError
from pydantic import BaseModel, Field
from scribe.scribe_milvus import SCRIBE

from ai_chat.models import RetrievalLog
from ai_chat.services.threshold_tuning import precision_recall, tune

logger = logging.getLogger(__name__)

DEFAULT_EVAL_COLLECTION = "general_chat"
JUDGE_CHUNK_PREVIEW_CHARS = 500

JUDGE_SYSTEM_PROMPT = (
    "Du bewertest die Relevanz von Suchergebnissen fuer eine Nutzer-Anfrage. "
    "Ein Chunk ist relevant, wenn er zur Beantwortung der Anfrage inhaltlich "
    "beitraegt. Antworte ausschliesslich im geforderten JSON-Format."
)


class RelevanceJudgement(BaseModel):
    """LLM-Label: welche der nummerierten Chunks sind relevant."""

    relevant_indices: List[int] = Field(
        default_factory=list,
        description="0-basierte Indizes der relevanten Chunks",
    )


class Command(BaseCommand):
    help = "Empfiehlt VECTORSTORE_*-Cutoff-Werte aus geloggten Retrievals (schreibt nie Settings)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--coverage",
            type=float,
            default=0.9,
            help="Score-Mass-Coverage-Ziel fuer den Offline-Replay (Default 0.9).",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="as_json",
            help="Report als JSON statt Tabelle ausgeben.",
        )
        parser.add_argument(
            "--eval-set",
            type=str,
            default=None,
            help="Pfad zu JSON [{query, relevant_document_ids, collection?}] fuer supervised Live-Eval.",
        )
        parser.add_argument(
            "--judge",
            action="store_true",
            help="Relevanz-Labels via LLM fuer eine Stichprobe geloggter Queries (kostet LLM-Calls).",
        )
        parser.add_argument(
            "--sample",
            type=int,
            default=20,
            help="Stichprobengroesse fuer --judge (Default 20).",
        )

    def handle(self, *args, **options):
        if options["eval_set"]:
            payload = self._run_eval_set(options["eval_set"])
        elif options["judge"]:
            payload = self._run_judge(options["sample"])
        else:
            payload = self._run_replay(options["coverage"])

        if options["as_json"]:
            self.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            self._render_table(payload)

    # -- Modus 1: Offline-Replay ------------------------------------------------

    def _run_replay(self, coverage_target: float) -> dict:
        score_lists_by_collection = defaultdict(list)
        rows = RetrievalLog.objects.exclude(candidate_scores=[]).values_list("collection", "candidate_scores")
        for collection, raw_scores in rows.iterator():
            scores = self._sanitize_scores(raw_scores)
            if scores:
                score_lists_by_collection[collection].append(scores)

        report = tune(dict(score_lists_by_collection), coverage_target=coverage_target)
        report["mode"] = "replay"
        report["env"] = self._env_values(report["global"])
        return report

    @staticmethod
    def _sanitize_scores(raw_scores) -> List[float]:
        """Geloggte JSON-Scores defensiv saeubern: nur Zahlen, absteigend."""
        if not isinstance(raw_scores, list):
            return []
        numeric = [float(s) for s in raw_scores if isinstance(s, (int, float)) and not isinstance(s, bool)]
        return sorted(numeric, reverse=True)

    @staticmethod
    def _env_values(recommendation: Optional[dict]) -> Optional[dict]:
        if recommendation is None:
            return None
        config = recommendation["config"]
        return {
            "VECTORSTORE_RELATIVE_CUTOFF": config["rel_floor"],
            "VECTORSTORE_ELBOW_DROP": config["elbow_drop"],
            "VECTORSTORE_MIN_K": config["min_k"],
            "VECTORSTORE_MAX_K": config["max_k"],
        }

    # -- Modus 2: --eval-set ------------------------------------------------------

    def _run_eval_set(self, eval_set_path: str) -> dict:
        path = Path(eval_set_path)
        if not path.exists():
            raise CommandError(f"Eval-Set nicht gefunden: {eval_set_path}")
        try:
            entries = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise CommandError(f"Eval-Set ist kein valides JSON: {exc}")
        if not isinstance(entries, list) or not all(
            isinstance(e, dict) and "query" in e and "relevant_document_ids" in e for e in entries
        ):
            raise CommandError("Eval-Set muss eine Liste aus {query, relevant_document_ids[, collection]} sein.")

        results = asyncio.run(self._evaluate_entries(entries))
        return {
            "mode": "eval",
            "n_queries": len(results),
            "mean_precision": mean(r["precision"] for r in results) if results else 0.0,
            "mean_recall": mean(r["recall"] for r in results) if results else 0.0,
            "results": results,
        }

    async def _evaluate_entries(self, entries: List[dict]) -> List[dict]:
        scribes = {}
        results = []
        for entry in entries:
            collection = entry.get("collection") or DEFAULT_EVAL_COLLECTION
            if collection not in scribes:
                scribes[collection] = SCRIBE(collection)
            hits = await scribes[collection].search_similar_chunks(entry["query"])
            retrieved_ids = [doc.metadata.get("document_id") for doc, _score in hits]
            retrieved_ids = [doc_id for doc_id in retrieved_ids if doc_id is not None]
            precision, recall = precision_recall(retrieved_ids, entry["relevant_document_ids"])
            results.append(
                {
                    "query": entry["query"],
                    "collection": collection,
                    "retrieved_k": len(hits),
                    "precision": precision,
                    "recall": recall,
                }
            )
        return results

    # -- Modus 3: --judge ---------------------------------------------------------

    def _run_judge(self, sample: int) -> dict:
        logs = list(RetrievalLog.objects.exclude(query="").order_by("-created_at")[:sample])
        results = asyncio.run(self._judge_logs(logs)) if logs else []
        return {
            "mode": "judge",
            "n_queries": len(results),
            "mean_precision": mean(r["precision"] for r in results) if results else 0.0,
            "results": results,
        }

    async def _judge_logs(self, logs: List[RetrievalLog]) -> List[dict]:
        client = get_llm_client()
        scribes = {}
        results = []
        for log in logs:
            if log.collection not in scribes:
                scribes[log.collection] = SCRIBE(log.collection)
            hits = await scribes[log.collection].search_similar_chunks(log.query)
            if not hits:
                results.append({"query": log.query, "collection": log.collection, "retrieved_k": 0, "precision": 0.0})
                continue

            chunk_list = "\n\n".join(
                f"[{i}] {doc.page_content[:JUDGE_CHUNK_PREVIEW_CHARS]}" for i, (doc, _score) in enumerate(hits)
            )
            user_prompt = (
                f"Anfrage: {log.query}\n\n"
                f"Suchergebnisse (nummeriert):\n{chunk_list}\n\n"
                "Welche Chunks sind fuer die Anfrage relevant?"
            )
            _result, judgement = await asyncio.to_thread(
                client.invoke, JUDGE_SYSTEM_PROMPT, user_prompt, output_schema=RelevanceJudgement
            )
            if judgement is None:
                logger.warning("LLM-Judge lieferte kein parsebares Label fuer Query %r", log.query)
                relevant = set()
            else:
                relevant = {i for i in judgement.relevant_indices if 0 <= i < len(hits)}
            results.append(
                {
                    "query": log.query,
                    "collection": log.collection,
                    "retrieved_k": len(hits),
                    "precision": len(relevant) / len(hits),
                }
            )
        return results

    # -- Output -------------------------------------------------------------------

    def _render_table(self, payload: dict) -> None:
        mode = payload["mode"]
        if mode == "replay":
            self._render_replay_table(payload)
        else:
            self._render_eval_table(payload)

    def _render_replay_table(self, payload: dict) -> None:
        self.stdout.write(
            f"=== Offline-Replay: {payload['n_queries']} geloggte Queries, "
            f"Coverage-Ziel {payload['coverage_target']:.2f} ===\n"
        )
        if payload["global"] is None:
            self.stdout.write("Keine RetrievalLog-Eintraege mit candidate_scores gefunden — nichts zu tunen.")
            return

        header = (
            f"{'Collection':<24} {'n':>5} {'rel_floor':>9} {'elbow':>6} {'min_k':>5} "
            f"{'max_k':>5} {'coverage':>9} {'noise':>7} {'mean_k':>7} {'median_k':>8}"
        )
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        rows = sorted(payload["collections"].items()) + [("GLOBAL", payload["global"])]
        for name, rec in rows:
            if rec is None:
                self.stdout.write(f"{name:<24} {'—':>5}")
                continue
            cfg = rec["config"]
            self.stdout.write(
                f"{name:<24} {rec['n_queries']:>5} {cfg['rel_floor']:>9.2f} {cfg['elbow_drop']:>6.1f} "
                f"{cfg['min_k']:>5} {cfg['max_k']:>5} {rec['mean_coverage']:>9.3f} "
                f"{rec['mean_tail_noise']:>7.3f} {rec['mean_k']:>7.2f} {rec['median_k']:>8.1f}"
            )

        self.stdout.write("\nEmpfohlene Env-Werte (global; werden NICHT automatisch gesetzt):")
        for key, value in payload["env"].items():
            self.stdout.write(f"  {key}={value}")

    def _render_eval_table(self, payload: dict) -> None:
        label = "Eval-Set" if payload["mode"] == "eval" else "LLM-Judge"
        self.stdout.write(f"=== {label}: {payload['n_queries']} Queries ===\n")
        if not payload["results"]:
            self.stdout.write("Keine Queries ausgewertet.")
            return
        for row in payload["results"]:
            recall_part = f"  recall={row['recall']:.3f}" if "recall" in row else ""
            self.stdout.write(
                f"{row['collection']:<24} k={row['retrieved_k']:<3} "
                f"precision={row['precision']:.3f}{recall_part}  {row['query'][:60]}"
            )
        self.stdout.write(f"\nMean Precision @cutoff: {payload['mean_precision']:.3f}")
        if "mean_recall" in payload:
            self.stdout.write(f"Mean Recall   @cutoff: {payload['mean_recall']:.3f}")
