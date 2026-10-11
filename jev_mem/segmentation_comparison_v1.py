"""Offline partition agreement, not accuracy against a human gold standard.

No API clients, credentials, model calls or QA answers are used. Segmentation and
event grouping are evaluated separately: grouping cannot erase a cut retrospectively.
All metrics require a complete, unchanged source partition for one conversation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

from jev_mem.topic_grouping import _validate_units, render_group

VERSION = "segmentation-comparison-v1"
INTERPRETATION = "agreement_to_llm_not_accuracy"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")


def validate_partition(raw_turns, groups, *, conversation_id, contiguous=False):
    """Return source-ordered labels after strict source and partition validation.

    Source IDs are scoped to the explicit conversation_id. Optional conversation
    fields must agree. _group_record's redundant date/order/render fields, when
    present, are checked too. Contiguous segments cannot cross original sessions;
    event groups may contain nonadjacent turns and span dates in this conversation.
    """
    _nonnegative_int(conversation_id, "conversation_id")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError("A nonempty ordered raw-turn list is required")
    stripped = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            raise ValueError("Raw turns must be mappings")
        for field in ("conversation", "conversation_id"):
            if field in turn and (type(turn[field]) is not int or turn[field] != conversation_id):
                raise ValueError("Cross-conversation raw source")
        stripped.append({k: v for k, v in turn.items() if k not in {"conversation", "conversation_id"}})
    _validate_units(stripped)
    # A session may not reappear after another session in ordered raw input.
    sessions_seen, previous = set(), None
    for turn in raw_turns:
        session = turn["session_id"]
        if session != previous:
            if session in sessions_seen:
                raise ValueError("Raw session blocks are not contiguous")
            sessions_seen.add(session)
            previous = session
    if not isinstance(groups, list) or not groups:
        raise ValueError("Nonempty groups required; partial banks are not evaluable")
    originals = {turn["source_turn_id"]: turn for turn in raw_turns}
    positions = {turn["source_turn_id"]: i for i, turn in enumerate(raw_turns)}
    labels, seen_groups = [None] * len(raw_turns), set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("Groups must be mappings")
        for field in ("conversation", "conversation_id"):
            if field in group and (type(group[field]) is not int or group[field] != conversation_id):
                raise ValueError("Cross-conversation group")
        group_id = group.get("group_id", group.get("segment_id"))
        if not isinstance(group_id, str) or not group_id or group_id in seen_groups:
            raise ValueError("Every group requires a unique nonempty group/segment ID")
        seen_groups.add(group_id)
        turns = group.get("turns")
        ids = group.get("source_turn_ids")
        if not isinstance(turns, list) or not turns or not isinstance(ids, list):
            raise ValueError("A group needs nonempty exact turns and source_turn_ids")
        if any(not isinstance(t, dict) or "source_turn_id" not in t for t in turns):
            raise ValueError("A group contains an invalid source turn")
        if ids != [t["source_turn_id"] for t in turns]:
            raise ValueError("Group source_turn_ids disagree with its turns")
        if any(t != originals.get(t["source_turn_id"]) for t in turns):
            raise ValueError("Unknown source or rewritten source text/date/speaker")
        order = [positions[source_id] for source_id in ids]
        if order != sorted(order) or len(set(order)) != len(order):
            raise ValueError("Group source order is invalid or duplicated")
        if "source_order_indices" in group and group["source_order_indices"] != order:
            raise ValueError("Group source_order_indices mismatch")
        if "text" in group and group["text"] != render_group(turns):
            raise ValueError("Group rendering differs from exact source")
        dates = sorted({turn["observation_date"] for turn in turns})
        sessions = list(dict.fromkeys(turn["session_id"] for turn in turns))
        expected_metadata = {"observation_dates": dates, "original_session_ids": sessions,
                             "observation_date": dates[0] if len(dates) == 1 else None,
                             "cross_date": len(dates) > 1, "cross_session": len(sessions) > 1}
        for field, expected in expected_metadata.items():
            if field in group and group[field] != expected:
                raise ValueError("Group " + field + " differs from source metadata")
        if contiguous and (order != list(range(order[0], order[-1] + 1)) or len(sessions) != 1):
            raise ValueError("Each segment must be contiguous within one original session")
        for index in order:
            if labels[index] is not None:
                raise ValueError("A source turn was assigned more than once")
            labels[index] = group_index
    if any(label is None for label in labels):
        raise ValueError("Partition dropped source turns")
    return labels


def _prf(matched, predicted, reference):
    """No-positive denominator is undefined, not a fabricated perfect score."""
    return {"matched_count": matched, "candidate_positive_count": predicted,
            "reference_positive_count": reference,
            "precision": matched / predicted if predicted else None,
            "recall": matched / reference if reference else None,
            "f1": 2 * matched / (predicted + reference) if predicted + reference else None}


def _boundaries(raw_turns, labels):
    result = {}
    for index in range(1, len(raw_turns)):
        session = raw_turns[index]["session_id"]
        if session != raw_turns[index - 1]["session_id"]:
            continue  # Original session edges are compulsory, never credited as semantic cuts.
        result.setdefault(session, [])
        if labels[index] != labels[index - 1]:
            result[session].append(index)
    return result


def _match_boundaries(candidate, reference, tolerance):
    """Maximum-cardinality one-to-one matching; ties minimize displacement.

    Sorted 1D positions admit a noncrossing optimum. Dynamic programming avoids
    nearest-first matching that can consume another candidate's only match.
    """
    p, r = len(candidate), len(reference)
    scores = [[(0, 0)] * (r + 1) for _ in range(p + 1)]
    moves = [[None] * (r + 1) for _ in range(p + 1)]
    for i in range(1, p + 1):
        moves[i][0] = "candidate"
    for j in range(1, r + 1):
        moves[0][j] = "reference"
    for i in range(1, p + 1):
        for j in range(1, r + 1):
            options = [(scores[i - 1][j], "candidate"), (scores[i][j - 1], "reference")]
            distance = abs(candidate[i - 1] - reference[j - 1])
            if distance <= tolerance:
                count, negative_distance = scores[i - 1][j - 1]
                options.append(((count + 1, negative_distance - distance), "match"))
            scores[i][j], moves[i][j] = max(options, key=lambda row: row[0])
    matched, i, j = [], p, r
    while i or j:
        move = moves[i][j]
        if move == "match":
            matched.append((candidate[i - 1], reference[j - 1]))
            i, j = i - 1, j - 1
        elif move == "candidate":
            i -= 1
        else:
            j -= 1
    return list(reversed(matched))


def boundary_agreement(raw_turns, candidate_labels, reference_labels, *, tolerance_turns=0):
    _nonnegative_int(tolerance_turns, "tolerance_turns")
    if len(candidate_labels) != len(raw_turns) or len(reference_labels) != len(raw_turns):
        raise ValueError("One partition label per source turn required")
    candidate, reference = _boundaries(raw_turns, candidate_labels), _boundaries(raw_turns, reference_labels)
    pairs, by_session = [], []
    for session in dict.fromkeys(turn["session_id"] for turn in raw_turns):
        left, right = candidate.get(session, []), reference.get(session, [])
        matched = _match_boundaries(left, right, tolerance_turns)
        for p, r in matched:
            pairs.append({"session_id": session, "candidate_right_source_index": p,
                          "reference_right_source_index": r, "displacement_turns": abs(p - r)})
        by_session.append({"session_id": session, "candidate_right_source_indices": left,
                           "reference_right_source_indices": right,
                           **_prf(len(matched), len(left), len(right))})
    return {"interpretation": INTERPRETATION, "tolerance_turns": tolerance_turns,
            "matching": "one_to_one_max_cardinality_then_minimum_displacement_within_session",
            "original_session_edges_excluded": True,
            **_prf(len(pairs), sum(map(len, candidate.values())), sum(map(len, reference.values()))),
            "matched_pairs": pairs, "by_session": by_session}


def clustering_agreement(candidate_labels, reference_labels):
    """Positive turn-pair agreement and B-cubed; no true-negative accuracy."""
    if not candidate_labels or len(candidate_labels) != len(reference_labels):
        raise ValueError("Nonempty equally sized complete partitions required")
    if any(label is None for label in candidate_labels + reference_labels):
        raise ValueError("Partial assignments are not evaluable")
    cells = Counter(zip(candidate_labels, reference_labels))
    candidate, reference = Counter(candidate_labels), Counter(reference_labels)
    choose2 = lambda count: count * (count - 1) // 2
    matched = sum(choose2(count) for count in cells.values())
    pairwise = _prf(matched, sum(map(choose2, candidate.values())), sum(map(choose2, reference.values())))
    precision_sum = sum(count * count / candidate[c] for (c, _), count in cells.items())
    recall_sum = sum(count * count / reference[r] for (_, r), count in cells.items())
    n = len(candidate_labels)
    precision, recall = precision_sum / n, recall_sum / n
    return {"interpretation": INTERPRETATION, "turn_count": n,
            "candidate_group_count": len(candidate), "reference_group_count": len(reference),
            "pairwise_positive": pairwise,
            "bcubed": {"precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall),
                        "precision_sum": precision_sum, "recall_sum": recall_sum},
            "negative_pair_accuracy_reported": False}


def _validate_atomic_membership(segment_labels, group_labels):
    memberships = {}
    for segment, group in zip(segment_labels, group_labels):
        if segment in memberships and memberships[segment] != group:
            raise ValueError("Final groups split an allegedly atomic segment")
        memberships[segment] = group


def compare_partitions(raw_turns, candidate_segments, candidate_groups, reference_segments,
                       reference_groups, *, conversation_id, reference_label="llm", tolerance_turns=1):
    """Compare one complete conversation, retaining hashes and audit denominators."""
    if not isinstance(reference_label, str) or not reference_label.strip():
        raise ValueError("An explicit LLM/reference label is required")
    cseg = validate_partition(raw_turns, candidate_segments, conversation_id=conversation_id, contiguous=True)
    cgroup = validate_partition(raw_turns, candidate_groups, conversation_id=conversation_id)
    rseg = validate_partition(raw_turns, reference_segments, conversation_id=conversation_id, contiguous=True)
    rgroup = validate_partition(raw_turns, reference_groups, conversation_id=conversation_id)
    _validate_atomic_membership(cseg, cgroup)
    _validate_atomic_membership(rseg, rgroup)
    return {"version": VERSION, "interpretation": INTERPRETATION,
            "reference_is_human_gold": False, "reference_label": reference_label,
            "conversation_id": conversation_id, "turn_count": len(raw_turns),
            "session_count": len({turn["session_id"] for turn in raw_turns}),
            "source_sha256": _hash(raw_turns),
            "partition_sha256": {"candidate_segments": _hash(candidate_segments),
                                 "candidate_groups": _hash(candidate_groups),
                                 "reference_segments": _hash(reference_segments),
                                 "reference_groups": _hash(reference_groups)},
            "complete_source_partition_verified": True,
            "boundary_exact": boundary_agreement(raw_turns, cseg, rseg),
            "boundary_tolerant": boundary_agreement(raw_turns, cseg, rseg, tolerance_turns=tolerance_turns),
            "event_clustering": clustering_agreement(cgroup, rgroup),
            "new_model_calls": 0,
            "limitations": ["LLM reference can be wrong; disagreement needs source adjudication.",
                            "Agreement does not establish downstream QA benefit.",
                            "No QA/gold answers enter this comparison."]}


def aggregate_comparisons(results):
    """Pool within-conversation counts only; never add easy cross-conversation pairs."""
    if not isinstance(results, list) or not results:
        raise ValueError("At least one audited conversation result required")
    seen = set()
    reference, tolerance = results[0]["reference_label"], results[0]["boundary_tolerant"]["tolerance_turns"]
    for row in results:
        if (row["version"] != VERSION or row["interpretation"] != INTERPRETATION
                or not row["complete_source_partition_verified"] or row["reference_label"] != reference
                or row["boundary_tolerant"]["tolerance_turns"] != tolerance):
            raise ValueError("Incompatible or unaudited comparison results")
        if row["conversation_id"] in seen:
            raise ValueError("Duplicate conversation would change weighting")
        seen.add(row["conversation_id"])
    result = {"version": VERSION, "interpretation": INTERPRETATION, "reference_label": reference,
              "conversation_ids": sorted(seen), "turn_count": sum(r["turn_count"] for r in results),
              "reference_is_human_gold": False, "cross_conversation_pairs_counted": 0,
              "aggregation": "micro_counts_and_turn_weighted_bcubed_within_conversation_only"}
    for name in ("boundary_exact", "boundary_tolerant"):
        result[name] = _prf(*(sum(r[name][key] for r in results)
                             for key in ("matched_count", "candidate_positive_count", "reference_positive_count")))
    result["boundary_tolerant"]["tolerance_turns"] = tolerance
    rows = [r["event_clustering"] for r in results]
    pairs = _prf(*(sum(r["pairwise_positive"][key] for r in rows)
                   for key in ("matched_count", "candidate_positive_count", "reference_positive_count")))
    p = sum(r["bcubed"]["precision_sum"] for r in rows) / result["turn_count"]
    r = sum(r["bcubed"]["recall_sum"] for r in rows) / result["turn_count"]
    result["event_clustering"] = {"pairwise_positive": pairs,
                                  "bcubed": {"precision": p, "recall": r, "f1": 2*p*r/(p+r)}}
    return result


def summarize_call_budget(trace, *, session_lengths, physical_usage=None):
    """Summarize an existing sparse trace without equating logical calls to HTTP.

    Optional physical_usage is a caller-audited ledger summary. Its fields remain
    None when unavailable, never inferred as zero from a cached/logical trace.
    No token estimates are derived from characters or logical criterion counts.
    """
    if not isinstance(session_lengths, list) or not session_lengths:
        raise ValueError("Session lengths required")
    for length in session_lengths:
        _nonnegative_int(length, "session length")
        if not length:
            raise ValueError("Empty sessions are not eligible units")
    requests = trace.get("requests")
    if not isinstance(requests, list):
        raise ValueError("Trace requires an explicit requests list")
    logical = len(requests)
    if "logical_judge_calls" in trace and trace["logical_judge_calls"] != logical:
        raise ValueError("Declared logical call count differs from trace")
    criteria = sum(len(row.get("criteria", {})) for row in requests)
    if "criterion_evaluations" in trace and trace["criterion_evaluations"] != criteria:
        raise ValueError("Declared criterion count differs from trace")
    eligible = sum(length - 1 for length in session_lengths)
    indices = {row["right_source_index"] for row in requests if "right_source_index" in row}
    valid_indices, offset = set(), 0
    for length in session_lengths:
        valid_indices.update(range(offset + 1, offset + length))
        offset += length
    if any(type(index) is not int or index not in valid_indices for index in indices):
        raise ValueError("A judged boundary is outside an original session")
    usage = {"physical_http_attempts": None, "cache_hits": None,
             "known_input_tokens": None, "unknown_usage_attempts": None}
    if physical_usage is not None:
        if not isinstance(physical_usage, dict) or set(physical_usage) - set(usage):
            raise ValueError("Unsupported physical ledger summary fields")
        for field, value in physical_usage.items():
            if value is not None:
                _nonnegative_int(value, field)
            usage[field] = value
    return {"logical_judge_calls": logical, "criterion_evaluations": criteria,
            "eligible_within_session_gaps": eligible, "distinct_queried_gaps": len(indices),
            "queried_gap_coverage": len(indices)/eligible if eligible else None,
            "unqueried_gaps": eligible-len(indices),
            "unqueried_means": "not_assessed_not_confirmed_same_topic",
            "request_status_counts": dict(Counter(row.get("status", "unspecified") for row in requests)),
            **usage, "new_model_calls": 0,
            "token_accounting": "ledger_only_unknown_not_zero_no_character_proxy"}


def synthetic_midpoint_counterexample():
    """Reuse the frozen synthetic probe: termination does not prove cut recall."""
    from tools.probe_midpoint_segmentation import CASES, run_case
    topics = deepcopy(CASES["short_B_then_return_A"])
    case = run_case("short_B_then_return_A", topics)
    rows = []
    for strategy in case["strategies"]:
        evaluation = strategy["with_pruning_assumptions"]["evaluation"]
        rows.append({"strategy": strategy["strategy"], "logical_comparisons": strategy["oracle_calls"],
                     "terminated": strategy["terminated"],
                     "unvisited_turn_count": strategy["unvisited_turn_count"],
                     "true_boundary_after_turns": evaluation["true_boundary_after_turns"],
                     "missed_boundary_after_turns": evaluation["missed_boundary_after_turns"],
                     "unknown_gap_after_turns": evaluation["unknown_gap_after_turns"],
                     "all_gaps_identified_correctly": evaluation["all_gaps_identified_correctly"]})
    return {"experiment_kind": "synthetic_perfect_oracle_not_real_JEV",
            "topics": topics, "model_api_calls": 0, "strategies": rows,
            "conclusion": "Finite midpoint recursion is not boundary convergence; same-topic pruning can miss A-B-A."}
