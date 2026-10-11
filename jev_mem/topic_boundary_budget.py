"""Local bounded candidate proposal for a later sparse-boundary experiment.

Consumes precomputed raw-turn embeddings; makes no API call, chooses no semantic
cut, and invokes no writer. Unselected gaps remain unassessed. The running QA and
all-gap boundary implementation are unchanged.
"""
import math

from jev_mem.topic_boundaries import _session_ranges, boundary_requests
from jev_mem.topic_grouping import _validate_units


def _normalized(vector):
    if not isinstance(vector, list) or len(vector) != 768:
        raise ValueError("One 768-dimensional vector per raw turn is required")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise ValueError("Embedding values must be finite numbers")
    norm = math.sqrt(sum(x * x for x in vector))
    if not norm:
        raise ValueError("Embedding norm must be positive")
    return [x / norm for x in vector]


def _mean_direction(vectors):
    values = [sum(column) / len(vectors) for column in zip(*vectors)]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else None


def propose_sparse_boundaries(raw_turns, embeddings, *, initial_budget=4,
                              refinement_budget=2, window_turns=2):
    """Reserve two fractional coverage gaps, then fill by embedding change.

    Initial and refinement budgets count logical JEV requests, not paid attempts.
    Refinement is only RESERVED here, not automatically spent or executed. The
    downstream scorer must share that budget across expanded rejudgments and new
    nearby gaps. No hard peak spacing is imposed: adjacent entry/exit boundaries
    of a short inserted topic must remain eligible.
    """
    for name, value in (("initial_budget", initial_budget), ("refinement_budget", refinement_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    units = _validate_units(raw_turns)
    if not isinstance(embeddings, list) or len(embeddings) != len(units):
        raise ValueError("Raw-turn and embedding counts differ")
    vectors = [_normalized(vector) for vector in embeddings]
    requests = {r["right_source_index"]: r for r in boundary_requests(units, window_turns=window_turns)}
    sessions = []
    for start, end in _session_ranges(units):
        gaps = list(range(start + 1, end))
        scores = {}
        for gap in gaps:
            left = _mean_direction(vectors[max(start, gap - window_turns):gap])
            right = _mean_direction(vectors[gap:min(end, gap + window_turns)])
            if left is None or right is None:
                raise ValueError("Zero window-mean embedding; do not silently assign a gap score")
            cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))
            scores[gap] = 1.0 - cosine
        chosen, reasons = [], {}
        if gaps:
            for numerator in (1, 2):
                gap = start + min(end - start - 1, max(1, round((end - start) * numerator / 3)))
                if gap not in chosen and len(chosen) < initial_budget:
                    chosen.append(gap)
                    reasons[gap] = "fractional_coverage_anchor"
            for gap in sorted(gaps, key=lambda i: (-scores[i], i)):
                if len(chosen) >= initial_budget:
                    break
                if gap not in chosen:
                    chosen.append(gap)
                    reasons[gap] = "largest_remaining_embedding_window_change"
        sessions.append({
            "session_id": units[start]["session_id"], "turn_count": end - start,
            "all_gap_count": len(gaps), "initial_requests": [requests[i] for i in sorted(chosen)],
            "selection_reasons": {str(i): reasons[i] for i in chosen},
            "embedding_change_scores": {str(i): scores[i] for i in gaps},
            "unassessed_gap_indices": [i for i in gaps if i not in chosen],
            "refinement_requests_reserved": refinement_budget if chosen else 0,
            "initial_logical_requests": len(chosen),
            "max_logical_requests_including_refinement": len(chosen) + (refinement_budget if chosen else 0),
        })
    return {"version": "sparse-boundary-proposal-v1", "sessions": sessions,
            "initial_logical_requests": sum(s["initial_logical_requests"] for s in sessions),
            "max_logical_requests_including_refinement": sum(s["max_logical_requests_including_refinement"] for s in sessions),
            "model_calls_made": 0, "embedding_cost_excluded": True,
            "scores_are_embedding_change_not_JEV": True, "qa_visible": False,
            "cuts_or_groups_emitted": False, "sparse_scorer_integration_pending": True}
