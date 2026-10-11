"""Original-source local context for the frozen native Mem0 group writer.

No work or model I/O runs on import. Native Memory.add inputs and the extraction
system prompt remain unchanged. Only extraction user-prompt source boundaries
and its Last k Messages input change. This remains an offline memory protocol:
native existing-memory retrieval can see previously submitted nonlocal groups.
"""
from __future__ import annotations

import contextlib
import copy

from tools import run_mem0_regroup_sdk20 as frozen

VERSION = "source-local-group-context-v1"
CONTEXT_ONLY = (
    "CONTEXT ONLY: This original preceding turn is supplied solely to resolve "
    "references in the named target segment(s). Do not extract new standalone "
    "facts from this context turn. It is not a new member of this write group."
)
MEMBER_SCOPE = (
    "These are the new source members of the named segment. Preserve its speaker "
    "and individual source date. A segment boundary or juxtaposition does not "
    "establish a shared event, relationship, or shared time. " + CONTEXT_ONLY
)


def _source_index(source_sessions):
    if not isinstance(source_sessions, list) or not source_sessions:
        raise ValueError("Original source session envelopes are required")
    index = {}
    order = 0
    sessions_seen = set()
    for session in source_sessions:
        if set(session) != {"session_id", "observation_date", "source_turn_ids", "messages"}:
            raise ValueError("Pass original session envelopes only, without QA/gold/facts")
        if session["session_id"] in sessions_seen:
            raise ValueError("Duplicate source session")
        sessions_seen.add(session["session_id"])
        ids = session["source_turn_ids"]
        messages = session["messages"]
        if not ids or len(ids) != len(messages) or len(set(ids)) != len(ids):
            raise ValueError("Invalid original session membership")
        date = frozen.original._validate_date(session["observation_date"])
        for position, (tid, message) in enumerate(zip(ids, messages)):
            if tid in index:
                raise ValueError("Duplicate source turn")
            if (message.get("role") not in {"user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    or not message["content"] or set(message) - {"role", "content", "name"}):
                raise ValueError("Invalid original source message")
            index[tid] = {
                "source_turn_id": tid, "session_id": session["session_id"],
                "observation_date": date, "source_order_index": order,
                "session_position": position, "session_ids": list(ids),
                "message": copy.deepcopy(message),
            }
            order += 1
    return index


def build_context_plan(envelope, *, source_sessions, segments):
    """Pure, QA-blind plan; at most two original predecessors per segment.

    Predecessors are the two immediate source-session positions, without looking
    farther back when they are already owned by this group. Shared context is
    rendered once with all target segment IDs. Whole-conversation text is never
    sent as history. Exact native source message contents are checked.
    """
    source = _source_index(source_sessions)
    segment_map = {s["segment_id"]: s for s in segments}
    if len(segment_map) != len(segments):
        raise ValueError("Duplicate segment identity")
    sids = envelope["segment_ids"]
    ids = envelope["source_turn_ids"]
    if not sids or len(set(sids)) != len(sids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid group membership")
    if any(tid not in source for tid in ids):
        raise ValueError("Unknown source member")
    expected = []
    membership = {}
    contexts = {}
    boundaries = []
    owned = set(ids)
    for sid in sids:
        if sid not in segment_map:
            raise ValueError("Unknown segment")
        members = segment_map[sid]["source_turn_ids"]
        if not members or any(tid not in source for tid in members):
            raise ValueError("Unknown segment member")
        rows = [source[tid] for tid in members]
        if (len({r["session_id"] for r in rows}) != 1
                or [r["session_position"] for r in rows]
                != list(range(rows[0]["session_position"], rows[0]["session_position"] + len(rows)))):
            raise ValueError("Each source segment must be contiguous within one original session")
        for tid in members:
            if tid in membership:
                raise ValueError("Overlapping source segments")
            membership[tid] = sid
        expected.extend(members)
        first = rows[0]
        previous = first["session_ids"][max(0, first["session_position"] - 2):first["session_position"]]
        context_ids = [tid for tid in previous if tid not in owned]
        boundaries.append({"segment_id": sid, "source_turn_ids": list(members),
                           "context_turn_ids": context_ids})
        for tid in context_ids:
            contexts.setdefault(tid, []).append(sid)
    expected.sort(key=lambda tid: source[tid]["source_order_index"])
    if expected != ids:
        raise ValueError("Envelope does not equal the ordered complete segment union")
    lengths = [len(envelope[k]) for k in ("messages", "turn_sources", "source_order_indices")]
    if any(n != len(ids) for n in lengths):
        raise ValueError("Envelope source dimensions differ")
    for tid, message, stamp, position in zip(ids, envelope["messages"], envelope["turn_sources"], envelope["source_order_indices"]):
        row = source[tid]
        if message != row["message"] or position != row["source_order_index"]:
            raise ValueError("Envelope differs from original source content/order")
        if any(stamp.get(k) != row[k] for k in ("source_turn_id", "session_id", "observation_date", "source_order_index")):
            raise ValueError("Envelope source/date metadata differs")
    history = []
    for tid in sorted(contexts, key=lambda t: source[t]["source_order_index"]):
        row = source[tid]
        stamp = {k: row[k] for k in ("source_turn_id", "session_id", "observation_date", "source_order_index")}
        rendered = copy.deepcopy(row["message"])
        rendered["content"] = ("[CONTEXT_ONLY target=" + ",".join(contexts[tid])
                               + " turn=" + tid + " session=" + row["session_id"]
                               + " date=" + row["observation_date"] + "]\n"
                               + row["message"]["content"])
        if len(rendered["content"]) - len(row["message"]["content"]) >= 300:
            raise ValueError("Source context header exceeds the unchanged native history limit")
        history.append({**stamp, "message": copy.deepcopy(row["message"]),
                        "writer_message": rendered, "context_only": True,
                        "for_segment_ids": contexts[tid],
                        "native_history_character_limit": 300,
                        "native_visible_content": rendered["content"][:300] + ("..." if len(rendered["content"]) > 300 else ""),
                        "native_omitted_characters": max(0, len(rendered["content"]) - 300)})
    writer_messages = []
    for tid, message, stamp in zip(ids, envelope["messages"], envelope["turn_sources"]):
        rendered = frozen.render_source(message, stamp)
        sid = membership[tid]
        prefix = "[SEGMENT " + sid + "]\n"
        if tid == segment_map[sid]["source_turn_ids"][0]:
            prefix += MEMBER_SCOPE + "\n"
        rendered["content"] = prefix + rendered["content"]
        writer_messages.append(rendered)
    plan = {"version": VERSION, "group_id": envelope["group_id"],
            "predecessors_per_segment": 2, "boundaries": boundaries,
            "history_context": history, "writer_messages": writer_messages,
            "source_member_ids": list(ids), "context_member_overlap": [],
            "native_memory_inputs_changed": False, "extraction_system_prompt_changed": False,
            "existing_memory_retrieval_unchanged": True, "online_causal_claim": False}
    return {**plan, "sha256": frozen.digest(plan)}


@contextlib.contextmanager
def _local_hooks(plan):
    """Caller holds the native ingestion lock for the complete Memory.add call."""
    original_builder = frozen.adapt_group_builder
    original_publish = frozen.publish_json

    def annotate(record):
        if record.get("group_id") != plan["group_id"] or "request_sha256" not in record:
            return
        previous = record.get("source_local_context")
        if previous is not None and previous["sha256"] != plan["sha256"]:
            raise ValueError("Source-local context changed within a write")
        record["source_local_context"] = copy.deepcopy(plan)
        record["history_context"] = copy.deepcopy(plan["history_context"])
        record["effective_request_sha256"] = frozen.digest({
            "native_request_sha256": record["request_sha256"], "source_local_context_sha256": plan["sha256"]})
        record["adapter_protocol"] = VERSION
        for fact in record.get("fact_records", []):
            scope = fact["input_scope"]
            scope["historical_turn_ids"] = [r["source_turn_id"] for r in plan["history_context"]]
            scope["history_policy"] = VERSION
            scope["history_context_only"] = True

    def publish(path, value):
        annotate(value)
        return original_publish(path, value)

    @contextlib.contextmanager
    def builder(record):
        annotate(record)
        # The frozen native request fingerprint remains reproducible. A separate
        # effective fingerprint binds the actual extraction-only adaptation.
        effective = {**record, "writer_messages": copy.deepcopy(plan["writer_messages"]),
                     "history_context": copy.deepcopy(plan["history_context"])}
        try:
            with original_builder(effective):
                yield
        finally:
            record["builder_call_count"] = effective["builder_call_count"]

    frozen.publish_json = publish
    frozen.adapt_group_builder = builder
    try:
        yield
    finally:
        frozen.adapt_group_builder = original_builder
        frozen.publish_json = original_publish


class SourceLocalMem0GroupAdapter(frozen.Mem0GroupAdapter):
    """Thin subclass: all extraction/persistence code delegates to the old writer."""

    def __init__(self, memory, audit_dir, current_date=None, *, source_sessions, segments):
        super().__init__(memory, audit_dir, current_date=current_date)
        self.source_sessions = copy.deepcopy(source_sessions)
        self.segments = copy.deepcopy(segments)
        _source_index(self.source_sessions)

    def add_group(self, *, user_id, **envelope):
        plan = build_context_plan(envelope, source_sessions=self.source_sessions, segments=self.segments)
        with frozen.original._INGEST_LOCK:
            # Prevent accidental reuse of an old bank or stale source-context
            # checkpoint, including a successful no-model replay.
            for prior in self._records(user_id):
                if prior.get("source_local_context", {}).get("version") != VERSION:
                    raise frozen.original.SessionRecoveryRequired("Source-local writer requires its own fresh audit namespace")
                if prior["group_id"] == envelope["group_id"] and prior["source_local_context"]["sha256"] != plan["sha256"]:
                    raise frozen.original.SessionRecoveryRequired("Source-local context changed; no automatic replay")
            with _local_hooks(plan):
                record = super().add_group(user_id=user_id, **envelope)
            if record.get("source_local_context", {}).get("sha256") != plan["sha256"]:
                raise frozen.original.IncompleteSessionError("Effective source-local context was not persisted")
            return record
