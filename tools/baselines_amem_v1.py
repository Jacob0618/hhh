"""Locked, original A-Mem experiment adapter (no Reader/Judge or API credentials).

Public contract: build(raw_conversation, work_dir, api), retrieve(question, state, api).
The original scientific classes/prompts are compiled from verified upstream source.
Only provider/model construction and silent exception fallbacks are adapted.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

UPSTREAM_REVISION = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "upstream/baselines_v1/amem"
SOURCE_SHA256 = {
    "memory_layer.py": "b9a5b5797881b25c5524b98a543b22a1bbbcc7f285437ca667a5673e7f293eba",
    "test_advanced.py": "7559e34cdd6a3993bf50bf4795342ec39bce91b1220d56ed206df021196c408a",
    "load_dataset.py": "b249429a6624e6e79e5e099c90bfcf7ab96c8fd40f3b1074b750dd0b2823661b",
}
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MODEL_DIR = Path("C:/Users/86139/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots") / MODEL_REVISION
SEED_K = 10
SCHEMA = "amem-original-adapter-v1"
_MODEL = None
_MODEL_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()
_QUERY_LOCKS = {}
_QUERY_LOCKS_GUARD = threading.Lock()


class AMemError(RuntimeError):
    """A failed or unverifiable operation must never become an empty memory."""


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sources() -> dict[str, str]:
    sources = {}
    for name, expected in SOURCE_SHA256.items():
        path = UPSTREAM_DIR / name
        if not path.is_file():
            raise AMemError(f"Missing pinned upstream source: {path}")
        source = path.read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise AMemError(f"Upstream source hash mismatch: {name}")
        sources[name] = source
    return sources


class _LocalEncoder:
    def __init__(self, model):
        self.model = model

    def encode(self, documents):
        # All conversations share one CPU model. No model/network work on import.
        with _ENCODE_LOCK:
            vectors = self.model.encode(documents, show_progress_bar=False)
        return _checked_vectors(vectors, len(documents))

    def get_config_dict(self):
        return {"model_name": "all-MiniLM-L6-v2"}


def _get_encoder():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            if not MODEL_DIR.is_dir():
                raise AMemError(f"Required local MiniLM revision is missing: {MODEL_DIR}")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise AMemError("A-Mem needs installed torch and sentence-transformers; no automatic installation/download is allowed") from exc
            torch.set_num_threads(1)
            _MODEL = _LocalEncoder(SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True))
        return _MODEL


def _checked_vectors(vectors, count):
    array = np.asarray(vectors)
    if array.shape != (count, 384) or not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise AMemError(f"Invalid MiniLM vectors: expected ({count}, 384), got {array.shape}")
    return array


def _cosine_similarity(x, y):
    # Keep the same implementation/float behavior as pinned upstream.
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(x, y)


def _validate_json(value, schema, location="response"):
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "integer": isinstance(value, int) and not isinstance(value, bool)}.get(kind, False)
    if not valid:
        raise AMemError(f"{location}: expected {kind}")
    if kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing or (schema.get("additionalProperties") is False and set(value) - set(properties)):
            raise AMemError(f"{location}: invalid object fields (missing={sorted(missing)})")
        for key, item in value.items():
            if key in properties:
                _validate_json(item, properties[key], f"{location}.{key}")
    elif kind == "array":
        for i, item in enumerate(value):
            _validate_json(item, schema["items"], f"{location}[{i}]")


def _parse_response(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise AMemError("Generation returned no JSON response")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AMemError("Generation returned malformed JSON") from exc
    if not isinstance(result, dict):
        raise AMemError("Generation response must be a JSON object")
    return result


class _Controller:
    def __init__(self, api, work_dir):
        self.api = api
        self.work_dir = Path(work_dir)
        self.operation = "unassigned"
        self.memory_count = 0

    def get_completion(self, prompt, response_format, temperature=None):
        # The upstream temperature argument is deliberately never forwarded.
        schema = response_format["json_schema"]["schema"]
        properties = schema["properties"]
        stage = "amem_evolution" if "should_evolve" in properties else "amem_analysis" if "context" in properties else "amem_query_keywords"
        record_path = self.work_dir / "calls" / f"{self.operation}_{stage}.json"
        system = "You must respond with a JSON object."
        request = {"prompt": prompt, "system": system, "response_format": response_format,
                   "stage": stage, "max_output": 16384, "json_output": True}
        fingerprint = _sha(request)
        record = {"schema": SCHEMA, "request_sha256": fingerprint, "request": request,
                  "status": "started", "raw_response": None}
        if record_path.exists():
            previous = _read_json(record_path)
            if previous.get("request_sha256") != fingerprint:
                raise AMemError(f"Cached call request mismatch: {record_path}")
            if previous.get("status") != "success":
                raise AMemError(f"Previous call failed or has unknown outcome; explicit recovery required: {record_path}")
            value = previous["parsed_response"]
        else:
            _write_json(record_path, record)
            try:
                raw = self.api.generate(prompt, system=system, json_output=True,
                                        max_output=16384, stage=stage)
                record["raw_response"] = raw
                value = _parse_response(raw)
                self._validate(value, schema, stage)
                record.update(status="success", parsed_response=value)
                _write_json(record_path, record)
            except BaseException as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                _write_json(record_path, record)
                if not isinstance(exc, Exception):
                    raise
                raise AMemError(f"{stage} failed; inspect {record_path}") from exc
        self._validate(value, schema, stage)
        return json.dumps(value, ensure_ascii=False)

    def _validate(self, value, schema, stage):
        _validate_json(value, schema)
        if stage == "amem_evolution":
            if any(action not in {"strengthen", "update_neighbor"} for action in value["actions"]):
                raise AMemError("Unknown A-Mem evolution action")
            if any(i < 0 or i >= self.memory_count for i in value["suggested_connections"]):
                raise AMemError("Evolution link references a nonexistent prior note")
        elif stage == "amem_analysis" and not value["context"].strip():
            raise AMemError("A-Mem analysis returned empty context")
        elif stage == "amem_query_keywords" and not value["keywords"].strip():
            raise AMemError("A-Mem query keywords are empty")


def _classes(encoder, controller):
    sources = _sources()
    nodes = []
    patched_handlers = 0
    for node in ast.parse(sources["memory_layer.py"]).body:
        if isinstance(node, ast.ClassDef) and node.name in {"MemoryNote", "SimpleEmbeddingRetriever", "AgenticMemorySystem"}:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in {"analyze_content", "process_memory"}:
                    for handler in ast.walk(method):
                        if isinstance(handler, ast.ExceptHandler):
                            handler.body = [ast.Raise()]
                            patched_handlers += 1
            nodes.append(node)
    for node in ast.parse(sources["test_advanced.py"]).body:
        if isinstance(node, ast.ClassDef) and node.name == "advancedMemAgent":
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "generate_query_llm"]
            for handler in ast.walk(node):
                if isinstance(handler, ast.ExceptHandler):
                    handler.body = [ast.Raise()]
                    patched_handlers += 1
            nodes.append(node)
    if len(nodes) != 4 or patched_handlers != 4:
        raise AMemError("Unexpected locked-source class/exception structure")
    namespace = {"__name__": "amem_pinned_runtime", "List": List, "Dict": Dict,
                 "Optional": Optional, "Literal": Literal, "Any": Any, "Union": Union,
                 "json": json, "re": re, "datetime": datetime, "uuid": uuid,
                 "np": np, "os": os, "Path": Path,
                 "cosine_similarity": _cosine_similarity,
                 "SentenceTransformer": lambda *args, **kwargs: encoder,
                 "LLMController": lambda *args, **kwargs: SimpleNamespace(llm=controller),
                 "print": lambda *args, **kwargs: None}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "<verified-amem-original>", "exec"), namespace)
    return namespace


def _sessions(conversation):
    if not isinstance(conversation, dict) or any(key in conversation for key in ("qa", "conversation", "event_summary", "session_summary", "observation")):
        raise AMemError("build accepts only the raw conversation mapping, without QA or summaries")
    allowed = {"speaker_a", "speaker_b"}
    sessions = []
    for key, turns in conversation.items():
        if re.fullmatch(r"session_\d+", key):
            allowed.update((key, key + "_date_time"))
            if not isinstance(turns, list):
                raise AMemError(f"Invalid turn list: {key}")
            timestamp = conversation.get(key + "_date_time")
            if not isinstance(timestamp, str) or not timestamp.strip():
                raise AMemError(f"Session timestamp missing: {key}")
            session_number = int(key.split("_")[1])
            try:
                date = datetime.strptime(timestamp, "%I:%M %p on %d %B, %Y")
            except ValueError as exc:
                raise AMemError(f"Unrecognized LoCoMo session date: {timestamp}") from exc
            sessions.append((date, session_number, key, timestamp, turns))
        elif re.fullmatch(r"session_\d+_date_time", key):
            allowed.add(key)  # LoCoMo includes dates of sessions with no turn list.
    if set(conversation) - allowed:
        raise AMemError(f"Unexpected conversation fields: {sorted(set(conversation) - allowed)}")
    sessions.sort(key=lambda item: (item[0], item[1]))
    if not sessions or not any(s[4] for s in sessions):
        raise AMemError("Conversation has no turns")
    for _, _, key, timestamp, turns in sessions:
        for position, turn in enumerate(turns):
            if not isinstance(turn, dict) or not all(isinstance(turn.get(k), str) for k in ("speaker", "dia_id", "text")):
                raise AMemError(f"Invalid turn: {key}[{position}]")
        yield key, timestamp, turns


def _note_record(note):
    return dict(vars(note))


def _snapshot(system, provenance, work_dir, input_sha, completed_notes, completed_sessions):
    state = {"schema": SCHEMA, "method": "A-Mem original", "upstream_revision": UPSTREAM_REVISION,
             "source_sha256_lf": SOURCE_SHA256, "model_revision": MODEL_REVISION,
             "seed_k": SEED_K, "evo_threshold": 100, "input_sha256": input_sha,
             "work_dir": str(work_dir), "completed_notes": completed_notes,
             "completed_sessions": list(completed_sessions), "evo_cnt": system.evo_cnt,
             "notes": [_note_record(note) for note in system.memories.values()],
             "provenance": list(provenance), "corpus": list(system.retriever.corpus),
             "caption_policy": "official loader: prefix [Image: blip_caption] when img_url and blip_caption exist"}
    state["memory_sha256"] = _sha({k: state[k] for k in ("notes", "provenance", "corpus", "evo_cnt")})
    if system.retriever.embeddings is not None:
        cache = work_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / "embeddings.tmp.npy"
        np.save(temporary, system.retriever.embeddings, allow_pickle=False)
        target = cache / "embeddings.npy"
        temporary.replace(target)
        state["embeddings_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write_json(work_dir / "checkpoint.json", state)
    return state


def _restore(state, api):
    if state.get("schema") != SCHEMA or state.get("upstream_revision") != UPSTREAM_REVISION or state.get("seed_k") != SEED_K:
        raise AMemError("A-Mem state protocol mismatch")
    expected = _sha({k: state[k] for k in ("notes", "provenance", "corpus", "evo_cnt")})
    if expected != state.get("memory_sha256"):
        raise AMemError("A-Mem state integrity mismatch")
    work_dir = Path(state["work_dir"])
    controller = _Controller(api, work_dir)
    encoder = _get_encoder()
    classes = _classes(encoder, controller)
    system = classes["AgenticMemorySystem"]()
    for record in state["notes"]:
        note = classes["MemoryNote"](**record)
        # Constructor uses `x or default`; restore exact persisted values afterward.
        note.__dict__.update(record)
        system.memories[note.id] = note
    if len(system.memories) != len(state["corpus"]) or len(system.memories) != len(state["provenance"]):
        raise AMemError("Notes, provenance and retrieval corpus are misaligned")
    system.evo_cnt = state["evo_cnt"]
    system.retriever.corpus = list(state["corpus"])
    system.retriever.document_ids = {doc: i for i, doc in enumerate(state["corpus"])}
    if state["corpus"]:
        cache_file = work_dir / "cache/embeddings.npy"
        if cache_file.is_file():
            if hashlib.sha256(cache_file.read_bytes()).hexdigest() != state.get("embeddings_sha256"):
                raise AMemError("A-Mem embedding cache integrity mismatch")
            system.retriever.embeddings = _checked_vectors(np.load(cache_file, allow_pickle=False), len(state["corpus"]))
        else:
            # Re-encode retained actual index text, not evolved note metadata: the
            # original deliberately keeps stale index rows between consolidations.
            system.retriever.embeddings = _checked_vectors(encoder.encode(state["corpus"]), len(state["corpus"]))
    return system, controller, classes


def build(conversation: dict, work_dir: Path, api) -> dict:
    """Build one independent conversation memory; safely resume completed notes."""
    sessions = list(_sessions(conversation))
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    input_sha = _sha(conversation)
    checkpoint = work_dir / "checkpoint.json"
    if checkpoint.exists():
        state = _read_json(checkpoint)
        if state.get("input_sha256") != input_sha or Path(state["work_dir"]).resolve() != work_dir:
            raise AMemError("Conversation/checkpoint mismatch")
        system, controller, classes = _restore(state, api)
        provenance = list(state["provenance"])
        completed_sessions = list(state["completed_sessions"])
    else:
        controller = _Controller(api, work_dir)
        classes = _classes(_get_encoder(), controller)
        system = classes["AgenticMemorySystem"]()
        provenance, completed_sessions = [], []
        state = _snapshot(system, provenance, work_dir, input_sha, 0, [])
    ordinal = 0
    for session_key, timestamp, turns in sessions:
        session_dir = work_dir / "sessions" / session_key
        _write_json(session_dir / "input.json", {"session": session_key, "timestamp": timestamp, "turns": turns})
        for position, turn in enumerate(turns):
            ordinal += 1
            if ordinal <= state["completed_notes"]:
                continue
            text = turn["text"]
            if "img_url" in turn and "blip_caption" in turn:
                if not isinstance(turn["blip_caption"], str):
                    raise AMemError("Invalid image caption")
                caption = f"[Image: {turn['blip_caption']}]"
                text = f"{caption} {text}" if text else caption
            content = "Speaker " + turn["speaker"] + "says : " + text
            controller.operation = f"note_{ordinal:06d}"
            controller.memory_count = len(system.memories)
            before = {key: _note_record(note) for key, note in system.memories.items()}
            # Stable IDs/last_accessed only affect persisted metadata, not prompts.
            note_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{input_sha}:{session_key}:{position}"))
            system.add_note(content, time=timestamp, id=note_id, last_accessed=timestamp)
            provenance.append({"note_id": note_id, "session": session_key, "turn_position": position,
                               "dia_id": turn["dia_id"], "speaker": turn["speaker"], "timestamp": timestamp})
            changed = [_note_record(note) for key, note in system.memories.items()
                       if key not in before or before[key] != _note_record(note)]
            _write_json(session_dir / f"note_{ordinal:06d}.json", {
                "ordinal": ordinal, "source": provenance[-1], "content": content,
                "analysis_call": f"calls/{controller.operation}_amem_analysis.json",
                "evolution_call": f"calls/{controller.operation}_amem_evolution.json",
                "changed_notes_after_evolution": changed, "evo_cnt": system.evo_cnt})
            state = _snapshot(system, provenance, work_dir, input_sha, ordinal, completed_sessions)
        if session_key not in completed_sessions:
            completed_sessions.append(session_key)
            state = _snapshot(system, provenance, work_dir, input_sha, ordinal, completed_sessions)
            _write_json(session_dir / "state.json", state)
    state["build_complete"] = True
    state["note_count"] = len(system.memories)
    state["export_paths"] = sorted(str(p.relative_to(work_dir)) for p in work_dir.rglob("*.json")
                                   if "cache" not in p.relative_to(work_dir).parts)
    if "state.json" not in state["export_paths"]:
        state["export_paths"].append("state.json")
    _write_json(checkpoint, state)
    _write_json(work_dir / "state.json", state)
    return state


def retrieve(question: str, state: dict, api) -> dict:
    """Original keyword -> cosine seed-k10 -> links expansion; no QA writeback."""
    if not isinstance(question, str) or not question.strip():
        raise AMemError("A nonempty question string is required")
    if not state.get("build_complete") or not state.get("notes"):
        raise AMemError("A-Mem build is incomplete or contains no notes")
    query_key = _sha({"question": question, "memory_sha256": state["memory_sha256"]})
    lock_key = str(Path(state["work_dir"]).resolve()) + ":" + query_key
    with _QUERY_LOCKS_GUARD:
        query_lock = _QUERY_LOCKS.setdefault(lock_key, threading.Lock())
    with query_lock:
        return _retrieve(question, state, api, query_key)


def _retrieve(question, state, api, query_key):
    path = Path(state["work_dir"]) / "retrieval" / f"{query_key}.json"
    if path.exists():
        cached = _read_json(path)
        if cached.get("question") != question or cached.get("memory_sha256") != state["memory_sha256"]:
            raise AMemError("Retrieval artifact identity mismatch")
        return cached
    system, controller, classes = _restore(state, api)
    controller.operation = f"query_{query_key}"
    controller.memory_count = len(system.memories)
    agent = classes["advancedMemAgent"]()
    agent.retriever_llm = SimpleNamespace(llm=controller)
    keywords = agent.generate_query_llm(question)
    # One native embedding search; intercept its results to recover exact evidence
    # without encoding/searching a second time (including tie ordering).
    search = system.retriever.search
    captured = []
    def capture_search(query, k):
        indices = search(query, k)
        captured.extend(int(i) for i in indices)
        return indices
    system.retriever.search = capture_search
    context = system.find_related_memories_raw(keywords, k=SEED_K)
    notes = list(system.memories.values())
    evidence = []
    for seed_rank, index in enumerate(captured):
        sequence = [(index, "seed")]
        # Upstream checks j >= k after appending, hence k + 1 links per seed.
        sequence.extend((linked, "link") for linked in notes[index].links[:SEED_K + 1])
        for note_index, via in sequence:
            if not isinstance(note_index, int) or not 0 <= note_index < len(notes):
                raise AMemError("Persisted A-Mem link is invalid")
            evidence.append({"rank": len(evidence), "seed_rank": seed_rank,
                             "seed_note_index": index, "note_index": note_index, "via": via,
                             "source": state["provenance"][note_index],
                             "note": _note_record(notes[note_index])})
    if not isinstance(context, str) or not context:
        raise AMemError("Nonempty A-Mem memory returned empty context")
    result = {"schema": SCHEMA, "question": question, "keywords": keywords, "seed_k": SEED_K,
              "seed_indices": captured, "context": context, "evidence": evidence,
              "memory_sha256": state["memory_sha256"], "upstream_revision": UPSTREAM_REVISION,
              "query_call": f"calls/{controller.operation}_amem_query_keywords.json"}
    _write_json(path, result)
    return result


def describe() -> dict:
    return {"schema": SCHEMA, "method": "A-Mem original", "upstream_revision": UPSTREAM_REVISION,
            "source_sha256_lf": SOURCE_SHA256, "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "encoder_revision": MODEL_REVISION, "model_path": str(MODEL_DIR), "device": "cpu",
            "seed_k": SEED_K, "evolution_neighbors": 5, "evolution_threshold": 100,
            "granularity": "one original speaker turn per note", "query_generation_calls": 1,
            "link_expansion": "up to k+1 links for each seed; no deduplication",
            "reader": "external common Reader", "generation_temperature": "omitted"}


def readiness() -> dict:
    import importlib.util
    _sources()
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("numpy", "sklearn", "torch", "sentence_transformers")}
    return {"ready": all(dependencies.values()) and MODEL_DIR.is_dir(),
            "dependencies": dependencies, "local_model_exists": MODEL_DIR.is_dir(),
            "source_verified": True, "model_loaded": _MODEL is not None}
