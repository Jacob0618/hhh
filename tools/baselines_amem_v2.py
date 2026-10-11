"""A-Mem v2: narrow, lossless typing repair for generated connection indexes.

The frozen v1 algorithm runs in a private module namespace. Only canonical
decimal strings in evolution.suggested_connections may become integers.
Raw API responses and all scientific prompts/parameters remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import types
from pathlib import Path

from tools import baselines_amem_v1 as _v1

BASE_ADAPTER = Path(_v1.__file__).resolve()
BASE_ADAPTER_SHA256 = "3d371f0c2dccd8eba148f6f2965f63224706bf67f1a2803cd3b84aeb7a9bcf56"
SCHEMA = "amem-original-adapter-v2-lossless-connection-indices"
NORMALIZATION_POLICY = "canonical-nonnegative-decimal-string-to-actual-neighbor-index-v1"
UPSTREAM_REVISION = _v1.UPSTREAM_REVISION
SOURCE_SHA256 = _v1.SOURCE_SHA256
SEED_K = _v1.SEED_K
AMemError = _v1.AMemError
_SUMMARY_LOCKS = {}
_SUMMARY_LOCKS_GUARD = threading.Lock()


class AMemV2Error(AMemError):
    pass


def _verified_runtime():
    source_bytes = BASE_ADAPTER.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != BASE_ADAPTER_SHA256:
        raise AMemV2Error("Frozen v1 adapter hash mismatch; v2 must be reviewed against its exact base")
    runtime = types.ModuleType("_jev_amem_v2_frozen_v1_runtime")
    runtime.__file__ = str(BASE_ADAPTER)
    exec(compile(source_bytes, str(BASE_ADAPTER), "exec"), runtime.__dict__)
    runtime.SCHEMA = SCHEMA
    runtime.AMemError = _v1.AMemError
    # Reuse the process-shared native CPU encoder; never mutate v1 globals.
    runtime._get_encoder = _v1._get_encoder
    return runtime


_runtime = _verified_runtime()
_BASE_CONTROLLER = _runtime._Controller
_BASE_CLASSES = _runtime._classes


def normalize_evolution(value: dict, *, memory_count: int, neighbor_indices) -> tuple[dict, dict]:
    """Return a deep-copied response and a lossless conversion journal.

    Only [0-9] ASCII decimal, exactly '0' or [1-9][0-9]*, is accepted. Native
    integers and converted strings must refer to an actual retrieved neighbor.
    Noncanonical strings, bool/float, nonexistent/nonneighbor indexes and all
    non-list fields are rejected. Other fields are not changed or repaired.
    """
    if not isinstance(value, dict):
        raise AMemV2Error("Evolution response must be an object")
    if not isinstance(memory_count, int) or isinstance(memory_count, bool) or memory_count < 0:
        raise AMemV2Error("Invalid memory count for index validation")
    if neighbor_indices is None:
        raise AMemV2Error("Actual retrieved neighbor indexes were not captured")
    neighbors = list(neighbor_indices)
    if any(type(index) is not int or index < 0 or index >= memory_count for index in neighbors):
        raise AMemV2Error("Invalid actual-neighbor index metadata")
    connections = value.get("suggested_connections")
    if not isinstance(connections, list):
        raise AMemV2Error("suggested_connections must be an array")
    derived = copy.deepcopy(value)
    changes = []
    allowed = set(neighbors)
    for position, index in enumerate(connections):
        if type(index) is str:
            if re.fullmatch(r"(?:0|[1-9][0-9]*)", index, flags=re.ASCII) is None:
                raise AMemV2Error(f"suggested_connections[{position}] is not a canonical nonnegative decimal string")
            # Avoid giant integer parsing and reject nonexistent IDs losslessly.
            if len(index) > len(str(max(0, memory_count - 1))):
                raise AMemV2Error(f"suggested_connections[{position}] references a nonexistent note")
            numeric = int(index)
        elif type(index) is int:
            numeric = index
        else:
            raise AMemV2Error(f"suggested_connections[{position}] must be an integer or canonical decimal string")
        if numeric < 0 or numeric >= memory_count:
            raise AMemV2Error(f"suggested_connections[{position}] references a nonexistent note")
        if numeric not in allowed:
            raise AMemV2Error(f"suggested_connections[{position}] was not among actual retrieved neighbors")
        if type(index) is str:
            derived["suggested_connections"][position] = numeric
            changes.append({"json_pointer": f"/suggested_connections/{position}",
                            "from": index, "from_type": "string", "to": numeric, "to_type": "integer"})
    journal = {"policy": NORMALIZATION_POLICY, "memory_count": memory_count,
               "actual_neighbor_indices": neighbors, "changes": changes,
               "original_parsed_sha256": _v1._sha(value), "derived_parsed_sha256": _v1._sha(derived),
               "only_allowed_field_changed": all(key == "suggested_connections" or derived[key] == value[key] for key in value)}
    return derived, journal


def _transport_lineage(api):
    events = getattr(api, "events", None)
    if not events:
        return {"available": False, "owner": "parent transport exact-request admission gate"}
    event = events[-1]
    keys = ("request_hash", "cache_path", "cache_sha256", "cache_lineage", "context", "stage",
            "provider", "model", "operation", "cache_hit", "request_id", "status")
    return {"available": True, "event_sha256": _v1._sha(event),
            **{key: copy.deepcopy(event[key]) for key in keys if key in event}}


class _Controller(_BASE_CONTROLLER):
    def __init__(self, api, work_dir):
        super().__init__(api, work_dir)
        self.neighbor_indices = None

    def get_completion(self, prompt, response_format, temperature=None):
        schema = response_format["json_schema"]["schema"]
        properties = schema["properties"]
        stage = "amem_evolution" if "should_evolve" in properties else "amem_analysis" if "context" in properties else "amem_query_keywords"
        record_path = self.work_dir / "calls" / f"{self.operation}_{stage}.json"
        system = "You must respond with a JSON object."
        request = {"prompt": prompt, "system": system, "response_format": response_format,
                   "stage": stage, "max_output": 16384, "json_output": True}
        fingerprint = _v1._sha(request)
        record = {"schema": SCHEMA, "base_adapter_sha256": BASE_ADAPTER_SHA256,
                  "request_sha256": fingerprint, "request": request,
                  "status": "started", "raw_response": None}
        if record_path.exists():
            previous = _v1._read_json(record_path)
            if previous.get("schema") != SCHEMA or previous.get("request_sha256") != fingerprint:
                raise AMemV2Error(f"V2 cached call schema/request mismatch: {record_path}")
            if previous.get("status") != "success":
                raise AMemV2Error(f"Failed/unknown outcome requires explicit recovery: {record_path}")
            value, normalization = self._derive(previous["raw_response"], schema, stage)
            if value != previous.get("parsed_response") or normalization != previous.get("normalization"):
                raise AMemV2Error(f"V2 derived-response lineage mismatch: {record_path}")
            if _v1._sha(previous["raw_response"]) != previous.get("raw_response_payload_sha256"):
                raise AMemV2Error(f"V2 raw-response payload hash mismatch: {record_path}")
        else:
            _v1._write_json(record_path, record)
            try:
                raw = self.api.generate(prompt, system=system, json_output=True,
                                        max_output=16384, stage=stage)
                # A dict response aliases the provider's object; preserve before
                # any derived parsing. Only the copied parsed response can change.
                record["raw_response"] = copy.deepcopy(raw)
                record["raw_response_payload_sha256"] = _v1._sha(raw)
                record["transport_lineage"] = _transport_lineage(self.api)
                value, normalization = self._derive(raw, schema, stage)
                record.update(status="success", parsed_response=value,
                              normalized_parsed_response=copy.deepcopy(value), normalization=normalization)
                _v1._write_json(record_path, record)
            except BaseException as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                _v1._write_json(record_path, record)
                if not isinstance(exc, Exception):
                    raise
                raise AMemV2Error(f"{stage} failed; inspect {record_path}") from exc
        return json.dumps(value, ensure_ascii=False)

    def _derive(self, raw, schema, stage):
        parsed = copy.deepcopy(_v1._parse_response(raw))
        if stage == "amem_evolution":
            value, normalization = normalize_evolution(parsed, memory_count=self.memory_count,
                                                       neighbor_indices=self.neighbor_indices)
        else:
            value = parsed
            normalization = {"policy": "no-normalization", "changes": [],
                             "original_parsed_sha256": _v1._sha(parsed), "derived_parsed_sha256": _v1._sha(parsed)}
        # All original schema, action, empty-context and bounds gates still apply.
        super()._validate(value, schema, stage)
        return value, normalization


def _classes(encoder, controller):
    classes = _BASE_CLASSES(encoder, controller)
    system_class = classes["AgenticMemorySystem"]
    original = system_class.find_related_memories

    def find_related_memories(self, query, k=5):
        result = original(self, query, k)
        # Capture the native search result without parsing prompt text, encoding
        # twice, reordering, adding/removing candidates, or changing the result.
        controller.neighbor_indices = tuple(int(index) for index in result[1])
        controller.memory_count = len(self.memories)
        return result

    system_class.find_related_memories = find_related_memories
    return classes


_runtime._Controller = _Controller
_runtime._classes = _classes


def _summary_lock(work_dir):
    with _SUMMARY_LOCKS_GUARD:
        return _SUMMARY_LOCKS.setdefault(str(work_dir), threading.Lock())


def _call_summary(path, work_dir):
    record = _v1._read_json(path)
    if record.get("schema") != SCHEMA or record.get("status") != "success":
        raise AMemV2Error(f"Cannot export an unverified v2 call: {path}")
    return {"call_artifact": str(path.relative_to(work_dir)), "stage": record["request"]["stage"],
            "request_sha256": record["request_sha256"],
            "prompt_payload_sha256": _v1._sha(record["request"]["prompt"]),
            "raw_response_payload_sha256": record["raw_response_payload_sha256"],
            "derived_parsed_response_sha256": _v1._sha(record["parsed_response"]),
            "transport_lineage": record["transport_lineage"], "normalization": record["normalization"],
            "status": record["status"]}


def _export_summary(work_dir):
    output = work_dir / "generation_evidence_summary.json"
    with _summary_lock(work_dir):
        summary = {"schema": SCHEMA, "scope": "build_writer", "base_adapter_sha256": BASE_ADAPTER_SHA256,
                   "normalization_policy": NORMALIZATION_POLICY,
                   "raw_retention": "parent keeps original prompts and raw API responses once in generation evidence zip",
                   "query_retention": "parent per-question retrieval capture_stage.events and model_evidence.zip"}
        calls = {}
        for path in sorted((work_dir / "calls").glob("*.json")):
            if path.name.endswith("_amem_query_keywords.json"):
                continue
            calls[str(path.relative_to(work_dir))] = _call_summary(path, work_dir)
        summary["calls"] = [calls[key] for key in sorted(calls)]
        summary["call_count"] = len(calls)
        _v1._write_json(output, summary)


def _export_updates(work_dir, expected_notes):
    rows = [_v1._read_json(path) for path in (work_dir / "sessions").glob("session_*/note_*.json")]
    rows.sort(key=lambda row: row["ordinal"])
    if [row["ordinal"] for row in rows] != list(range(1, expected_notes + 1)):
        raise AMemV2Error("Per-note evolution journals are missing or duplicated")
    destination = work_dir / "memory_updates.jsonl"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(destination)


def build(conversation: dict, work_dir: Path, api) -> dict:
    state = _runtime.build(conversation, work_dir, api)
    state["base_adapter_sha256"] = BASE_ADAPTER_SHA256
    state["connection_index_normalization"] = NORMALIZATION_POLICY
    directory = Path(state["work_dir"])
    _export_summary(directory)
    _export_updates(directory, state["completed_notes"])
    state["export_paths"] = ["state.json", "generation_evidence_summary.json", "memory_updates.jsonl"]
    _v1._write_json(Path(state["work_dir"]) / "state.json", state)
    _v1._write_json(Path(state["work_dir"]) / "checkpoint.json", state)
    return state


def retrieve(question: str, state: dict, api) -> dict:
    return _runtime.retrieve(question, state, api)


def describe():
    return {**_runtime.describe(), "base_adapter_sha256": BASE_ADAPTER_SHA256,
            "base_adapter_path": str(BASE_ADAPTER), "connection_index_normalization": NORMALIZATION_POLICY,
            "normalization_scope": "only evolution.suggested_connections canonical nonnegative ASCII decimal strings",
            "connection_gate": "actual native retrieved neighbor indexes only", "raw_response_mutated": False,
            "historical_cache_admission": "owned by parent API exact-request source gate"}


def readiness():
    if hashlib.sha256(BASE_ADAPTER.read_bytes()).hexdigest() != BASE_ADAPTER_SHA256:
        raise AMemV2Error("Frozen v1 adapter changed")
    return {**_runtime.readiness(), "model_loaded": _v1._MODEL is not None,
            "base_adapter_verified": True, "normalization_policy": NORMALIZATION_POLICY}
