"""Pure Tuning-Logik fuer den adaptiven Retrieval-Cutoff (Phase B6).

Replayt geloggte ``RetrievalLog.candidate_scores`` (fused-RRF, absteigend)
gegen ein Grid von Cutoff-Configs via :func:`scribe.retrieval.adaptive_cutoff`
und empfiehlt pro Collection + global die Config mit dem kleinsten mean-k,
die das Score-Mass-Coverage-Ziel haelt. Keine DB-/Settings-Zugriffe — alles
hier ist ohne Django testbar.

Metrik-Definitionen:
- coverage: behaltene Score-Masse / gesamte Score-Masse einer Query.
- tail_noise: Anteil der behaltenen Score-Masse aus Chunks unterhalb des
  relativen Floors (``score/top < rel_floor``) — das sind min_k-Filler,
  die nur wegen des Clampings behalten wurden.
"""

from statistics import mean, median
from typing import Dict, List, Optional, Sequence

from scribe.retrieval import adaptive_cutoff

GRID_REL_FLOORS = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30 .. 0.80
GRID_ELBOW_DROPS = [round(0.1 * i, 1) for i in range(1, 6)]  # 0.1 .. 0.5
GRID_MIN_KS = [1, 3, 5]
GRID_MAX_KS = [10, 20, 30, 50]


def generate_grid() -> List[dict]:
    """Alle Cutoff-Config-Kandidaten fuer den Offline-Replay."""
    return [
        {"rel_floor": rel_floor, "elbow_drop": elbow_drop, "min_k": min_k, "max_k": max_k}
        for rel_floor in GRID_REL_FLOORS
        for elbow_drop in GRID_ELBOW_DROPS
        for min_k in GRID_MIN_KS
        for max_k in GRID_MAX_KS
    ]


def evaluate_config(
    score_lists: Sequence[Sequence[float]],
    *,
    rel_floor: float,
    elbow_drop: float,
    min_k: int,
    max_k: int,
) -> Optional[dict]:
    """Replay einer Cutoff-Config ueber viele geloggte Score-Listen.

    Args:
        score_lists: Pro Query die Pre-Cutoff-Scores, absteigend sortiert.

    Returns:
        ``{"mean_coverage", "mean_tail_noise", "mean_k", "median_k",
        "n_queries"}`` oder ``None``, wenn keine nicht-leere Liste vorliegt.
    """
    coverages: List[float] = []
    tail_noises: List[float] = []
    ks: List[int] = []

    for scores in score_lists:
        if not scores:
            continue
        k = adaptive_cutoff(scores, max_k=max_k, min_k=min_k, rel_floor=rel_floor, elbow_drop=elbow_drop)
        kept = scores[:k]
        total_mass = sum(scores)
        kept_mass = sum(kept)
        top = scores[0]

        ks.append(k)
        coverages.append(kept_mass / total_mass if total_mass > 0 else 1.0)
        noise_mass = sum(s for s in kept if top > 0 and s / top < rel_floor)
        tail_noises.append(noise_mass / kept_mass if kept_mass > 0 else 0.0)

    if not ks:
        return None

    return {
        "mean_coverage": mean(coverages),
        "mean_tail_noise": mean(tail_noises),
        "mean_k": mean(ks),
        "median_k": float(median(ks)),
        "n_queries": len(ks),
    }


def replay_grid(
    score_lists: Sequence[Sequence[float]],
    grid: Optional[List[dict]] = None,
) -> List[dict]:
    """Alle Grid-Configs ueber dieselben Score-Listen replayen."""
    results = []
    for config in grid if grid is not None else generate_grid():
        metrics = evaluate_config(score_lists, **config)
        if metrics is not None:
            results.append({"config": config, **metrics})
    return results


def recommend(results: List[dict], coverage_target: float = 0.9) -> Optional[dict]:
    """Beste Config: minimales mean-k, das das Coverage-Ziel haelt.

    Tie-Break: geringere Tail-Noise, dann hoehere Coverage. Haelt keine
    Config das Ziel, gewinnt die mit der hoechsten Coverage (Fallback).
    """
    if not results:
        return None
    eligible = [r for r in results if r["mean_coverage"] >= coverage_target]
    if eligible:
        return min(eligible, key=lambda r: (r["mean_k"], r["mean_tail_noise"], -r["mean_coverage"]))
    return max(results, key=lambda r: (r["mean_coverage"], -r["mean_k"]))


def tune(
    score_lists_by_collection: Dict[str, List[List[float]]],
    coverage_target: float = 0.9,
    grid: Optional[List[dict]] = None,
) -> dict:
    """Empfehlung pro Collection + global ueber alle geloggten Queries."""
    grid = grid if grid is not None else generate_grid()

    collections = {
        name: recommend(replay_grid(score_lists, grid=grid), coverage_target=coverage_target)
        for name, score_lists in score_lists_by_collection.items()
    }
    all_lists = [scores for score_lists in score_lists_by_collection.values() for scores in score_lists]
    return {
        "coverage_target": coverage_target,
        "n_queries": len(all_lists),
        "collections": collections,
        "global": recommend(replay_grid(all_lists, grid=grid), coverage_target=coverage_target),
    }


def precision_recall(retrieved_ids: Sequence, relevant_ids: Sequence) -> tuple:
    """Precision/Recall ueber Dokument-IDs (Duplikate zaehlen einmal).

    Ohne retrieved IDs ist die Precision 0; ohne relevante IDs ist der
    Recall 1 (es gab nichts zu finden).
    """
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    hits = len(retrieved & relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(relevant) if relevant else 1.0
    return precision, recall
