"""Offline, query-blind topic grouping of raw conversation turns.

This optional later-stage module neither extracts facts nor rewrites source text.
No work happens on import. Its full-conversation neighborhood context is offline;
the output must not be described as a causal online memory-writing experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from datetime import date, datetime

VERSION = "raw-turn-topics-v1"
JEV_THRESHOLD = .7
MAX_CANDIDATES = 3
MAX_ANCHORS = 3
EMBEDDING_MODEL = "gemini-embedding-001"
MAX_EMBEDDING_TOKENS = 2048
TEMPORAL_INSTRUCTION = (
    "These are unmodified raw turns regrouped offline. Each turn has its own source "
    "observation date, original session and speaker. Resolve relative time using that "
    "turn's date and local source context; do not apply one shared observation date "
    "to this entire group. Do not invent connections between juxtaposed turns."
)
SAME_TOPIC = (
    "The target raw turn belongs with the candidate group's anchors in the same "
    "concrete event, activity, problem, or conversational topic. The relationship "
    "must be supported by the supplied original text and its adjacent context. "
    "Sharing a person, place, broad category, or generic sentiment alone is not "
    "sufficient. It should fit the group's anchors jointly; one weak connecting "
    "anchor is not enough when the other anchors concern unrelated events. "
    "Brief replies, corrections and negations may belong if their adjacent context "
    "clearly identifies the same topic. Judge the target's topic; do not transfer "
    "a neighbor's unrelated content into the target."
)


class GroupCapacityError(ValueError):
    """A source unit or requested group exceeds the explicit token budget."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_units(raw_turns):
    if not isinstance(raw_turns, list):
        raise ValueError("raw_turns must be an ordered list")
    units = copy.deepcopy(raw_turns)
    allowed = {"source_turn_id", "session_id", "speaker", "observation_date", "text",
               "role", "caption", "image_query", "observation_timestamp"}
    required = {"source_turn_id", "session_id", "speaker", "observation_date", "text"}
    seen = set()
    for unit in units:
        if not isinstance(unit, dict) or not required <= set(unit) or set(unit) - allowed:
            raise ValueError("Provide raw-turn fields only; QA, gold and extracted facts are not inputs")
        if any(not isinstance(unit[key], str) or not unit[key].strip() for key in required):
            raise ValueError("Raw turns require nonempty IDs, speaker, date and exact original text")
        if date.fromisoformat(unit["observation_date"]).isoformat() != unit["observation_date"]:
            raise ValueError("observation_date must be YYYY-MM-DD")
        if unit["source_turn_id"] in seen:
            raise ValueError("Each raw turn ID must be unique")
        seen.add(unit["source_turn_id"])
        if any(not isinstance(unit[key], str) for key in set(unit) - required):
            raise ValueError("Optional raw-turn fields must be strings")
        if "role" in unit and unit["role"] not in {"user", "assistant"}:
            raise ValueError("Unsupported message role")
    return units


def raw_turns_from_locomo(conversation):
    """Accept only item['conversation'], never an item containing its QA array."""
    if not isinstance(conversation, dict) or any(key in conversation for key in ("qa", "question", "answer", "gold_evidence")):
        raise ValueError("Pass the raw conversation mapping, without any QA fields")
    keys = [key for key in conversation if re.fullmatch(r"session_\d+", key)]

    def timestamp(key):
        value = conversation[key + "_date_time"]
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
        raise ValueError("Unsupported original LoCoMo observation date")

    keys.sort(key=lambda key: (timestamp(key), int(key.split("_")[1])))
    units = []
    for session_id in keys:
        for source in conversation[session_id]:
            if any(key in source for key in ("qa", "question", "answer", "gold_evidence", "facts")):
                raise ValueError("Unexpected evaluation/fact fields in a raw source turn")
            unit = {"source_turn_id": source["dia_id"], "session_id": session_id,
                    "speaker": source["speaker"], "observation_date": timestamp(session_id).date().isoformat(),
                    "observation_timestamp": conversation[session_id + "_date_time"],
                    "text": source.get("text", ""),
                    "role": "user" if source["speaker"] == conversation.get("speaker_a") else "assistant"}
            if source.get("blip_caption"):
                unit["caption"] = source["blip_caption"]
            if source.get("query"):
                unit["image_query"] = source["query"]
            units.append(unit)
    return _validate_units(units)


def render_turn(unit):
    """Deterministic metadata prefix; the exact source text is not paraphrased."""
    header = {key: unit[key] for key in ("source_turn_id", "session_id", "speaker", "observation_date")}
    text = "[SOURCE " + json.dumps(header, ensure_ascii=False, sort_keys=True) + "]\n" + unit["text"]
    if "caption" in unit:
        text += "\n[ORIGINAL IMAGE CAPTION]\n" + unit["caption"]
    if "image_query" in unit:
        text += "\n[ORIGINAL IMAGE QUERY]\n" + unit["image_query"]
    return text


def render_group(units):
    return TEMPORAL_INSTRUCTION + "\n\n" + "\n\n".join(render_turn(unit) for unit in units)


def _unit_vector(value):
    if not isinstance(value, list) or len(value) != 768:
        raise ValueError("Topic embedding must contain 768 dimensions")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("Topic embedding values must be finite")
    norm = math.sqrt(sum(float(x) ** 2 for x in value))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Topic embedding norm must be positive and finite")
    return [float(x) / norm for x in value]


def _anchors(indices):
    return sorted(set([indices[0], indices[len(indices) // 2], indices[-1]]))[:MAX_ANCHORS]


def verify_partition(raw_turns, groups):
    """Verify exactly-once assignment and unchanged source fields/order."""
    units = _validate_units(raw_turns)
    originals = {unit["source_turn_id"]: unit for unit in units}
    positions = {unit["source_turn_id"]: index for index, unit in enumerate(units)}
    assigned = []
    for group in groups:
        turns = group.get("turns", [])
        ids = [unit["source_turn_id"] for unit in turns]
        if not turns or group.get("source_turn_ids") != ids:
            raise ValueError("Group IDs differ from its nonempty raw turns")
        if any(unit != originals.get(unit["source_turn_id"]) for unit in turns):
            raise ValueError("Grouping rewrote source text or metadata")
        order = [positions[source_id] for source_id in ids]
        if order != sorted(order):
            raise ValueError("A group's source turns are out of original order")
        if group.get("text") != render_group(turns):
            raise ValueError("Group text is not the deterministic source rendering")
        assigned.extend(ids)
    if len(assigned) != len(units) or len(set(assigned)) != len(assigned) or set(assigned) != set(originals):
        raise ValueError("Every source turn must be assigned exactly once")
    return True


def group_raw_turns(raw_turns, *, method="jev_topics", api=None, token_counter,
                    matched_group_sizes=None, max_group_tokens=2048, embedding_threshold=.7,
                    embedding_vectors=None):
    """Group raw turns with explicit exact token checks and an auditable trace.

    ``token_counter(text)`` must count with gemini-embedding-001's actual tokenizer;
    e.g. caller passes ``lambda s: api.count_tokens(s, model=..., stage=...)``.
    The module does not infer tokens from characters or silently split a turn.

    Methods: original_sessions, matched_contiguous, embedding_topics, jev_topics.
    The original-session control may explicitly use max_group_tokens=None because
    it does not embed groups. This exception is exposed in the output trace; topic
    methods always cap rendered groups at <=2048. matched_group_sizes matches turn
    counts, not necessarily token lengths. Full-conversation use is offline only.
    """
    start = time.perf_counter()
    methods = {"original_sessions", "matched_contiguous", "embedding_topics", "jev_topics"}
    if method not in methods:
        raise ValueError("Unknown grouping method")
    if not callable(token_counter):
        raise ValueError("Caller must provide an exact embedding-model token counter")
    if max_group_tokens is not None and (isinstance(max_group_tokens, bool) or not isinstance(max_group_tokens, int) or max_group_tokens < 1):
        raise ValueError("max_group_tokens must be a positive integer")
    if method in {"embedding_topics", "jev_topics"} and (max_group_tokens is None or max_group_tokens > MAX_EMBEDDING_TOKENS):
        raise ValueError("Topic groups must use a token cap no greater than 2048")
    if isinstance(embedding_threshold, bool) or not isinstance(embedding_threshold, (int, float)) or not 0 <= embedding_threshold <= 1:
        raise ValueError("embedding_threshold must be between zero and one")
    if method == "jev_topics" and api is None:
        raise ValueError("JEV topic grouping needs an explicit API instance")
    units = _validate_units(raw_turns)
    trace = {"version": VERSION, "method": method, "input_sha256": _hash(units),
             "input_turn_count": len(units), "input_kind": "raw_turns_not_extracted_facts",
             "offline_full_conversation": True, "online_causal_claim": False,
             "qa_visible": False, "jev_threshold": JEV_THRESHOLD, "embedding_threshold": embedding_threshold,
             "candidate_groups_max": MAX_CANDIDATES, "anchors_per_group_max": MAX_ANCHORS,
             "max_group_tokens": max_group_tokens, "exact_token_count_evaluations": 0,
             "embedding_method_calls": 0, "embedded_unit_count": 0, "embedded_input_tokens": 0,
             "jev_score_many_calls": 0, "jev_criterion_evaluations": 0,
             "no_generated_source_text": True, "decisions": [],
             "api_http_requests_and_billing": "Use provider logs; method calls exclude internal batches, cache and retries",
             "token_counter_contract": "Caller-supplied exact gemini-embedding-001 token counts"}
    counts = {}

    def count(text):
        key = _hash(text)
        if key not in counts:
            value = token_counter(text)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Exact token counter must return a positive integer for nonempty text")
            counts[key] = value
            trace["exact_token_count_evaluations"] += 1
        return counts[key]

    def group_count(indices):
        return count(render_group([units[index] for index in indices]))

    def fits(indices):
        size = group_count(indices)
        return max_group_tokens is None or size <= max_group_tokens

    def enforce(indices):
        if not fits(indices):
            raise GroupCapacityError("Requested group exceeds its token cap; no original turn was split or truncated")

    groups = []
    if method == "original_sessions":
        by_session = {}
        for index, unit in enumerate(units):
            by_session.setdefault(unit["session_id"], []).append(index)
        groups = list(by_session.values())
        for indices in groups:
            enforce(indices)
        trace["original_session_preserved"] = True
        trace["group_embedding_not_used"] = True
    elif method == "matched_contiguous":
        if (not isinstance(matched_group_sizes, list)
                or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in matched_group_sizes)
                or sum(matched_group_sizes) != len(units)):
            raise ValueError("matched_group_sizes must be positive counts summing to all turns")
        offset = 0
        for size in matched_group_sizes:
            indices = list(range(offset, offset + size))
            enforce(indices)
            groups.append(indices)
            offset += size
        trace["matching"] = "group_turn_counts_only_not_equal_tokens"
    elif units:
        for index in range(len(units)):
            enforce([index])
        embedding_texts = [render_turn(unit) for unit in units]
        embedding_counts = [count(text) for text in embedding_texts]
        if any(size > MAX_EMBEDDING_TOKENS for size in embedding_counts):
            raise GroupCapacityError("An intact raw turn exceeds embedding tokens; no silent split or truncation")
        if embedding_vectors is None:
            if api is None:
                raise ValueError("Embedding topics need an API instance or explicit precomputed vectors")
            embedding_vectors = api.embed(embedding_texts, query=False, stage="topic_unit_embedding")
            trace["embedding_method_calls"] = 1
            trace["embedded_unit_count"] = len(units)
            trace["embedded_input_tokens"] = sum(embedding_counts)
        else:
            trace["precomputed_embeddings_used"] = True
        if not isinstance(embedding_vectors, list) or len(embedding_vectors) != len(units):
            raise ValueError("One embedding per original raw turn is required")
        vectors = [_unit_vector(value) for value in embedding_vectors]
        sums = []
        for index, unit in enumerate(units):
            similarities = []
            for group_index, total in enumerate(sums):
                norm = math.sqrt(sum(value * value for value in total))
                similarity = sum(x * y for x, y in zip(vectors[index], total)) / norm if norm else -1.
                similarities.append((group_index, similarity))
            ranked = sorted(similarities, key=lambda row: (-row[1], row[0]))[:MAX_CANDIDATES]
            candidates = []
            for group_index, similarity in ranked:
                possible = groups[group_index] + [index]
                if fits(possible):
                    candidates.append({"group_index": group_index, "cosine": similarity,
                                       "anchor_indices": _anchors(groups[group_index])})
            decision = {"source_turn_id": unit["source_turn_id"],
                        "proposed_groups": [{"group_index": group_index, "cosine": similarity} for group_index, similarity in ranked],
                        "capacity_eligible_groups": [candidate["group_index"] for candidate in candidates]}
            assigned = None
            if method == "embedding_topics":
                if candidates and candidates[0]["cosine"] >= embedding_threshold:
                    assigned = candidates[0]["group_index"]
            elif candidates:
                context_indices = [neighbor for neighbor in (index - 1, index + 1) if 0 <= neighbor < len(units)]
                state = {"task": "query_blind_raw_turn_topic_assignment", "target": unit,
                         "adjacent_context_only_not_additional_assigned_content": [units[i] for i in context_indices],
                         "candidate_groups": [{"key": "group_" + str(candidate["group_index"]),
                                               "anchors": [units[i] for i in candidate["anchor_indices"]]}
                                              for candidate in candidates]}
                criteria = {"group_" + str(candidate["group_index"]): SAME_TOPIC +
                            " Evaluate only the candidate_groups entry whose key is group_" +
                            str(candidate["group_index"]) + "." for candidate in candidates}
                scores = api.score_many(json.dumps(state, ensure_ascii=False), criteria, stage="topic_same_concrete_event")
                trace["jev_score_many_calls"] += 1
                trace["jev_criterion_evaluations"] += len(criteria)
                if not isinstance(scores, dict):
                    raise ValueError("JEV must return a mapping of independent positive criterion scores")
                for key in criteria:
                    score = scores.get(key)
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid JEV topic criterion score")
                decision["criterion_scores"] = {key: scores[key] for key in criteria}
                decision["anchor_source_turn_ids"] = {"group_" + str(candidate["group_index"]):
                                                       [units[i]["source_turn_id"] for i in candidate["anchor_indices"]]
                                                       for candidate in candidates}
                best = max(candidates, key=lambda candidate: (scores["group_" + str(candidate["group_index"])],
                                                              candidate["cosine"], -candidate["group_index"]))
                if scores["group_" + str(best["group_index"])] >= JEV_THRESHOLD:
                    assigned = best["group_index"]
            if assigned is None:
                assigned = len(groups)
                groups.append([index])
                sums.append(list(vectors[index]))
                decision["action"] = "new_group"
            else:
                groups[assigned].append(index)
                sums[assigned] = [x + y for x, y in zip(sums[assigned], vectors[index])]
                decision["action"] = "append_to_group"
            decision["assigned_group_index"] = assigned
            trace["decisions"].append(decision)

    output = []
    for group_index, indices in enumerate(groups):
        turns = [copy.deepcopy(units[index]) for index in indices]
        dates = sorted({turn["observation_date"] for turn in turns})
        sessions = list(dict.fromkeys(turn["session_id"] for turn in turns))
        output.append({"group_id": f"topic_{group_index:04d}", "source_turn_ids": [turn["source_turn_id"] for turn in turns],
                       "source_order_indices": list(indices), "turns": turns, "text": render_group(turns),
                       "embedding_model_token_count": group_count(indices), "original_session_ids": sessions,
                       "observation_dates": dates, "observation_date": dates[0] if len(dates) == 1 else None,
                       "cross_session": len(sessions) > 1, "cross_date": len(dates) > 1,
                       "requires_per_turn_date_extraction": True})
    verify_partition(units, output)
    trace.update({"group_count": len(output), "group_turn_counts": [len(group["turns"]) for group in output],
                  "group_token_counts": [group["embedding_model_token_count"] for group in output],
                  "assigned_source_turn_ids": [source_id for group in output for source_id in group["source_turn_ids"]],
                  "assignment_sha256": _hash([group["source_turn_ids"] for group in output]),
                  "exactly_once_verified": True, "elapsed_seconds": time.perf_counter() - start})
    return {"groups": output, "trace": trace, "temporal_instruction": TEMPORAL_INSTRUCTION}
