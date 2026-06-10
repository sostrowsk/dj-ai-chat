"""Tests fuer Phase B6: tune_retrieval Command + threshold_tuning Service.

Pure-function-Tests (Grid, Replay-Metriken, Empfehlung, Precision/Recall)
laufen ohne DB. Die call_command-Tests nutzen RetrievalLogFactory-Daten;
eval-set-/judge-Pfade mocken SCRIBE bzw. get_llm_client im Command-Modul.
"""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from ai_router.types import Document
from django.core.management import call_command

from ai_chat.services.threshold_tuning import (
    evaluate_config,
    generate_grid,
    precision_recall,
    recommend,
    replay_grid,
    tune,
)
from ai_chat.tests.factories import RetrievalLogFactory

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class TestGenerateGrid:
    def test_grid_covers_documented_ranges(self):
        grid = generate_grid()

        # rel_floor 0.3..0.8 step 0.05 (11) x elbow_drop 0.1..0.5 step 0.1 (5)
        # x min_k {1,3,5} x max_k {10,20,30,50}
        assert len(grid) == 11 * 5 * 3 * 4

        rel_floors = sorted({c["rel_floor"] for c in grid})
        elbow_drops = sorted({c["elbow_drop"] for c in grid})
        assert rel_floors[0] == 0.3
        assert rel_floors[-1] == 0.8
        assert len(rel_floors) == 11
        assert elbow_drops == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert sorted({c["min_k"] for c in grid}) == [1, 3, 5]
        assert sorted({c["max_k"] for c in grid}) == [10, 20, 30, 50]


# ---------------------------------------------------------------------------
# evaluate_config (Replay einer Config ueber geloggte Score-Listen)
# ---------------------------------------------------------------------------


class TestEvaluateConfig:
    def test_plateau_keeps_everything_with_full_coverage(self):
        score_lists = [[1.0] * 8]

        metrics = evaluate_config(score_lists, rel_floor=0.3, elbow_drop=0.5, min_k=3, max_k=10)

        assert metrics["mean_k"] == 8.0
        assert metrics["median_k"] == 8.0
        assert metrics["mean_coverage"] == pytest.approx(1.0)
        assert metrics["mean_tail_noise"] == pytest.approx(0.0)
        assert metrics["n_queries"] == 1

    def test_sharp_elbow_cuts_after_drop(self):
        scores = [1.0, 0.95, 0.9, 0.88, 0.2, 0.18, 0.15]

        metrics = evaluate_config([scores], rel_floor=0.1, elbow_drop=0.45, min_k=1, max_k=10)

        assert metrics["mean_k"] == 4.0
        assert metrics["mean_coverage"] == pytest.approx(sum(scores[:4]) / sum(scores))
        assert metrics["mean_tail_noise"] == pytest.approx(0.0)

    def test_flat_tail_counts_min_k_filler_as_noise(self):
        # rel_floor schneidet nach dem Top-Hit, min_k=3 erzwingt zwei
        # Filler unterhalb des Floors — die zaehlen als Tail-Noise.
        scores = [1.0, 0.1, 0.09, 0.08]

        metrics = evaluate_config(
            [scores],
            rel_floor=0.35,
            elbow_drop=0.95,
            min_k=3,
            max_k=10,
        )

        assert metrics["mean_k"] == 3.0
        kept_mass = 1.0 + 0.1 + 0.09
        assert metrics["mean_coverage"] == pytest.approx(kept_mass / sum(scores))
        assert metrics["mean_tail_noise"] == pytest.approx((0.1 + 0.09) / kept_mass)

    def test_single_result_yields_full_coverage(self):
        metrics = evaluate_config([[0.5]], rel_floor=0.35, elbow_drop=0.45, min_k=3, max_k=10)

        assert metrics["mean_k"] == 1.0
        assert metrics["median_k"] == 1.0
        assert metrics["mean_coverage"] == pytest.approx(1.0)
        assert metrics["mean_tail_noise"] == pytest.approx(0.0)

    def test_returns_none_without_usable_score_lists(self):
        assert evaluate_config([], rel_floor=0.35, elbow_drop=0.45, min_k=3, max_k=10) is None
        assert evaluate_config([[]], rel_floor=0.35, elbow_drop=0.45, min_k=3, max_k=10) is None


# ---------------------------------------------------------------------------
# replay_grid + recommend
# ---------------------------------------------------------------------------


def _result(mean_k, coverage, tail_noise=0.0, **config):
    cfg = {"rel_floor": 0.35, "elbow_drop": 0.45, "min_k": 3, "max_k": 50}
    cfg.update(config)
    return {
        "config": cfg,
        "mean_coverage": coverage,
        "mean_tail_noise": tail_noise,
        "mean_k": mean_k,
        "median_k": mean_k,
        "n_queries": 10,
    }


class TestRecommend:
    def test_picks_minimal_mean_k_that_holds_coverage(self):
        results = [
            _result(12.0, 0.97, rel_floor=0.3),
            _result(5.0, 0.92, rel_floor=0.5),
            _result(3.0, 0.7, rel_floor=0.8),  # verfehlt Coverage-Ziel
        ]

        best = recommend(results, coverage_target=0.9)

        assert best["config"]["rel_floor"] == 0.5
        assert best["mean_k"] == 5.0

    def test_tie_breaks_on_lower_tail_noise(self):
        results = [
            _result(5.0, 0.92, tail_noise=0.2, rel_floor=0.3),
            _result(5.0, 0.92, tail_noise=0.05, rel_floor=0.4),
        ]

        best = recommend(results, coverage_target=0.9)

        assert best["config"]["rel_floor"] == 0.4

    def test_falls_back_to_best_coverage_when_target_unreachable(self):
        results = [
            _result(3.0, 0.6, rel_floor=0.8),
            _result(4.0, 0.85, rel_floor=0.5),
        ]

        best = recommend(results, coverage_target=0.9)

        assert best["mean_coverage"] == 0.85

    def test_returns_none_for_empty_results(self):
        assert recommend([], coverage_target=0.9) is None

    def test_replay_grid_returns_one_result_per_config(self):
        grid = generate_grid()[:7]

        results = replay_grid([[1.0, 0.9, 0.05]], grid=grid)

        assert len(results) == 7
        assert all("config" in r and "mean_k" in r for r in results)


# ---------------------------------------------------------------------------
# tune (pro Collection + global)
# ---------------------------------------------------------------------------


class TestTune:
    def test_recommends_per_collection_and_global(self):
        sharp = [1.0, 0.9, 0.85, 0.8, 0.05, 0.04, 0.03]
        report = tune(
            {"project_1": [sharp, sharp], "general_chat": [[0.5, 0.5, 0.5]]},
            coverage_target=0.9,
            grid=generate_grid()[:30],
        )

        assert set(report["collections"]) == {"project_1", "general_chat"}
        assert report["collections"]["project_1"]["n_queries"] == 2
        assert report["global"]["n_queries"] == 3
        assert report["coverage_target"] == 0.9

    def test_empty_input_yields_no_recommendation(self):
        report = tune({}, coverage_target=0.9)

        assert report["collections"] == {}
        assert report["global"] is None


# ---------------------------------------------------------------------------
# precision / recall
# ---------------------------------------------------------------------------


class TestPrecisionRecall:
    def test_partial_overlap(self):
        precision, recall = precision_recall([1, 2, 3], [2, 3, 4])

        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)

    def test_empty_retrieved_scores_zero(self):
        assert precision_recall([], [1, 2]) == (0.0, 0.0)

    def test_no_relevant_ids_means_perfect_recall(self):
        precision, recall = precision_recall([1], [])

        assert precision == 0.0
        assert recall == 1.0

    def test_duplicate_retrieved_ids_counted_once(self):
        precision, recall = precision_recall([1, 1, 2], [1])

        assert precision == pytest.approx(1 / 2)
        assert recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Command: Offline-Replay (Default)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestTuneRetrievalCommandReplay:
    SHARP = [1.0, 0.9, 0.85, 0.8, 0.05, 0.04, 0.03]

    def test_json_output_contains_recommendations_and_env(self):
        for _ in range(3):
            RetrievalLogFactory(collection="project_1", candidate_scores=self.SHARP, final_k=4)
        RetrievalLogFactory(collection="general_chat", candidate_scores=[0.5, 0.5, 0.5], final_k=3)

        out = StringIO()
        call_command("tune_retrieval", "--json", stdout=out)
        data = json.loads(out.getvalue())

        assert data["mode"] == "replay"
        assert set(data["collections"]) == {"project_1", "general_chat"}
        # 3x sharp elbow (k=4) + 1x Plateau mit 3 Kandidaten (k=3)
        assert data["global"]["mean_k"] == pytest.approx(3.75)
        for key in ("rel_floor", "elbow_drop", "min_k", "max_k"):
            assert key in data["global"]["config"]
        assert set(data["env"]) == {
            "VECTORSTORE_RELATIVE_CUTOFF",
            "VECTORSTORE_ELBOW_DROP",
            "VECTORSTORE_MIN_K",
            "VECTORSTORE_MAX_K",
        }

    def test_table_output_prints_collections_and_env_values(self):
        RetrievalLogFactory(collection="project_7", candidate_scores=self.SHARP, final_k=4)

        out = StringIO()
        call_command("tune_retrieval", stdout=out)
        output = out.getvalue()

        assert "project_7" in output
        assert "GLOBAL" in output
        assert "VECTORSTORE_RELATIVE_CUTOFF=" in output
        assert "VECTORSTORE_MAX_K=" in output

    def test_without_logs_reports_missing_data(self):
        out = StringIO()
        call_command("tune_retrieval", "--json", stdout=out)
        data = json.loads(out.getvalue())

        assert data["global"] is None
        assert data["collections"] == {}

    def test_ignores_logs_with_empty_candidate_scores(self):
        RetrievalLogFactory(collection="project_1", candidate_scores=[], final_k=0)

        out = StringIO()
        call_command("tune_retrieval", "--json", stdout=out)
        data = json.loads(out.getvalue())

        assert data["global"] is None

    def test_coverage_option_is_passed_through(self):
        RetrievalLogFactory(collection="project_1", candidate_scores=self.SHARP, final_k=4)

        out = StringIO()
        call_command("tune_retrieval", "--json", "--coverage", "0.5", stdout=out)
        data = json.loads(out.getvalue())

        assert data["coverage_target"] == 0.5


# ---------------------------------------------------------------------------
# Command: --eval-set (supervised, Live-Suche gemockt)
# ---------------------------------------------------------------------------


def _hit(document_id, score):
    return (Document(page_content=f"Chunk {document_id}", metadata={"document_id": document_id}), score)


def _mock_scribe(hits):
    instance = Mock()
    instance.search_similar_chunks = AsyncMock(return_value=hits)
    return Mock(return_value=instance), instance


@pytest.mark.django_db
class TestTuneRetrievalCommandEvalSet:
    def test_eval_set_reports_precision_and_recall(self, tmp_path):
        eval_file = tmp_path / "eval.json"
        eval_file.write_text(
            json.dumps(
                [
                    {"query": "Maschinen", "relevant_document_ids": [1, 2], "collection": "project_1"},
                ]
            )
        )
        scribe_cls, instance = _mock_scribe([_hit(1, 0.9), _hit(99, 0.5)])

        out = StringIO()
        with patch("ai_chat.management.commands.tune_retrieval.SCRIBE", scribe_cls):
            call_command("tune_retrieval", "--json", "--eval-set", str(eval_file), stdout=out)
        data = json.loads(out.getvalue())

        scribe_cls.assert_called_once_with("project_1")
        instance.search_similar_chunks.assert_awaited_once_with("Maschinen")
        assert data["mode"] == "eval"
        assert data["mean_precision"] == pytest.approx(0.5)
        assert data["mean_recall"] == pytest.approx(0.5)
        assert data["results"][0]["retrieved_k"] == 2


# ---------------------------------------------------------------------------
# Command: --judge --sample N (LLM-Labels gemockt)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestTuneRetrievalCommandJudge:
    def test_judge_samples_logs_and_reports_llm_precision(self):
        RetrievalLogFactory(collection="project_1", query="Maschinenpark", candidate_scores=[0.9], final_k=1)
        RetrievalLogFactory(collection="project_1", query="Leasingrate", candidate_scores=[0.8], final_k=1)

        scribe_cls, _ = _mock_scribe([_hit(1, 0.9), _hit(2, 0.5)])
        llm_client = Mock()
        llm_client.invoke.return_value = (Mock(), SimpleNamespace(relevant_indices=[0]))

        out = StringIO()
        with (
            patch("ai_chat.management.commands.tune_retrieval.SCRIBE", scribe_cls),
            patch("ai_chat.management.commands.tune_retrieval.get_llm_client", return_value=llm_client),
        ):
            call_command("tune_retrieval", "--json", "--judge", "--sample", "2", stdout=out)
        data = json.loads(out.getvalue())

        assert data["mode"] == "judge"
        assert data["n_queries"] == 2
        # 1 von 2 Chunks pro Query als relevant gelabelt
        assert data["mean_precision"] == pytest.approx(0.5)
        assert llm_client.invoke.call_count == 2

    def test_judge_without_logs_reports_empty(self):
        out = StringIO()
        with patch("ai_chat.management.commands.tune_retrieval.get_llm_client"):
            call_command("tune_retrieval", "--json", "--judge", stdout=out)
        data = json.loads(out.getvalue())

        assert data["mode"] == "judge"
        assert data["n_queries"] == 0
