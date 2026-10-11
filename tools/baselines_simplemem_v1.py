"""Pinned SimpleMem text bridge. Importing this module makes no API calls.

Uses official classes/prompts, a private import namespace, and fail-closed exception
handlers. Generation, retries, metering and Gemini thinking control belong to api.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import sys
import threading
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from functools import wraps

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "upstream/baselines_v1/simplemem"
SOURCE_SHA = "db80b6a7c591e0ea730a058e9f5fc4eb06572299"
SOURCE_HASHES = {
    "simplemem/core/answer_generator.py": "ba8946134e9db0de63da00a9f223229800d2487c6fd34d22d7c5296093d0470c",
    "simplemem/core/config_default.py": "77b06780502e42a62ed308eee75dadd2a8192f20b127177bb14b5c614cc622cf",
    "simplemem/core/hybrid_retriever.py": "63d6a3522e8a4c5d3a073b5dbf8ac003844c37993bf16033bb5c9a3807db5381",
    "simplemem/core/memory_builder.py": "ade5d9c252158ede2d0bd4ca3da027e913d6e88dbe616e00f5962178053c9dc7",
    "simplemem/core/database/vector_store_backend.py": "2406f5c267f9013af0dc34c8668def8b20b3e66326827c57d975f6129f502c26",
    "simplemem/core/database/vector_store.py": "a7519fa27cfcfdceb4e146780c948c96e2840cea11965a94c296b786623fff29",
    "simplemem/core/models/memory_entry.py": "487209f5d6981a12568ec3f6f4597d2bd1437a82a0ef9af85c5dc755bc2947c8",
}
SETTINGS = {
    "WINDOW_SIZE": 40, "OVERLAP_SIZE": 2,
    "SEMANTIC_TOP_K": 25, "KEYWORD_TOP_K": 5, "STRUCTURED_TOP_K": 5,
    "ENABLE_PLANNING": True, "ENABLE_REFLECTION": True, "MAX_REFLECTION_ROUNDS": 2,
    "ENABLE_PARALLEL_PROCESSING": False, "MAX_PARALLEL_WORKERS": 1,
    "ENABLE_PARALLEL_RETRIEVAL": False, "MAX_RETRIEVAL_WORKERS": 1,
    "USE_JSON_FORMAT": False, "EMBEDDING_DIMENSION": 768,
    "MEMORY_TABLE_NAME": "memory_entries", "LANCEDB_PATH": "unused",
}
DEPENDENCIES = ("pydantic", "numpy", "lancedb", "pyarrow", "dateparser", "tantivy")
_PREFIX = "_jev_simplemem_pinned_v1"
_NATIVE = None
_LOAD_LOCK = threading.RLock()
_AUDIT_LOCK = threading.RLock()
_STAGES = {
    "_generate_memory_entries": "simplemem_write",
    "_generate_memory_entries_worker": "simplemem_write",
    "_analyze_information_requirements": "simplemem_requirements",
    "_generate_targeted_queries": "simplemem_query_plan",
    "_analyze_query": "simplemem_query_analysis",
    "_analyze_information_completeness": "simplemem_completeness",
    "_generate_missing_info_queries": "simplemem_missing_queries",
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def describe() -> dict:
    return {"method": "SimpleMem", "variant": "text-Gemini-embedding-adapted-v1",
            "source_sha": SOURCE_SHA, "source_hashes": dict(SOURCE_HASHES),
            "settings": dict(SETTINGS), "embedding": "gemini-embedding-001",
            "embedding_dimensions": 768, "generation": "gemini-2.5-flash",
            "thinking_budget": 0, "temperature_sent": False,
            "reader_judge": "external shared implementation",
            "writer_order": "session-order turns; cross-session windows; one conversation-final flush",
            "fts_finalization": "rebuild original Tantivy/en_stem index once after all writer windows",
            "state_integrity": "all native rows including vectors plus canonical exported entries and source/text exports",
            "error_policy": "hard stop; no adapter retries or empty fallback"}


def readiness() -> dict:
    mismatches = []
    for name, expected in SOURCE_HASHES.items():
        path = SOURCE_ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    return {"ready": not missing and not mismatches, "missing_dependencies": missing,
            "source_mismatches": mismatches, "source_sha": SOURCE_SHA}


def _preflight() -> None:
    status = readiness()
    if not status["ready"]:
        raise RuntimeError("SimpleMem preflight blocked: " + json.dumps(status))


class _LazyImport:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        return getattr(importlib.import_module(self.name), name)


class _SourceBridge(ast.NodeTransformer):
    """Only imports and exception policy change; algorithm/prompt bodies stay upstream."""
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("simplemem"):
            node.module = _PREFIX + node.module[len("simplemem"):]
        return node

    def visit_Import(self, node):
        aliases = [a for a in node.names if a.name not in ("lancedb", "pyarrow", "dateparser")]
        return ast.copy_location(ast.Import(names=aliases), node) if aliases else None

    def visit_ExceptHandler(self, node):
        # Upstream otherwise turns failed LLM/FTS/SQL/JSON operations into plausible
        # empty answers or retries. Never allow that in a paid auditable benchmark.
        node.body = [ast.copy_location(ast.Raise(exc=None, cause=None), node)]
        return node


def _load_native():
    global _NATIVE
    with _LOAD_LOCK:
        if _NATIVE is not None:
            return _NATIVE
        for path, expected in SOURCE_HASHES.items():
            if hashlib.sha256((SOURCE_ROOT / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Pinned SimpleMem source changed: {path}")
        for suffix in ("", ".core", ".core.models", ".core.database", ".core.utils"):
            module = types.ModuleType(_PREFIX + suffix)
            module.__path__ = []
            sys.modules[module.__name__] = module
        for suffix, fields in (
            (".core.settings", {"settings": types.SimpleNamespace(**SETTINGS)}),
            (".core.utils.llm_client", {"LLMClient": _LLMBridge}),
            (".core.utils.embedding", {"EmbeddingModel": _EmbeddingBridge}),
        ):
            module = types.ModuleType(_PREFIX + suffix)
            module.__dict__.update(fields)
            sys.modules[module.__name__] = module
        loaded = {}
        for relative in ("models/memory_entry", "database/vector_store_backend",
                         "database/vector_store", "memory_builder", "hybrid_retriever", "answer_generator"):
            path = SOURCE_ROOT / f"simplemem/core/{relative}.py"
            name = _PREFIX + ".core." + relative.replace("/", ".")
            module = types.ModuleType(name)
            module.__file__ = str(path)
            module.__dict__.update(lancedb=_LazyImport("lancedb"), pa=_LazyImport("pyarrow"),
                                   dateparser=_LazyImport("dateparser"))
            sys.modules[name] = module
            tree = ast.fix_missing_locations(_SourceBridge().visit(ast.parse(path.read_text(encoding="utf-8"))))
            exec(compile(tree, str(path), "exec"), module.__dict__)
            loaded[relative.rsplit("/", 1)[-1]] = module
        _NATIVE = types.SimpleNamespace(**loaded)
        return _NATIVE


class _Audit:
    def __init__(self, path: Path):
        self.path = path
        self.lock = _AUDIT_LOCK

    def record(self, payload):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **payload},
                                        ensure_ascii=False) + "\n")


class _LLMBridge:
    def __init__(self, api, audit: _Audit):
        self.api, self.audit = api, audit

    def chat_completion(self, messages, temperature=None, response_format=None, max_retries=None):
        caller = inspect.currentframe().f_back.f_code.co_name
        if caller not in _STAGES:
            raise RuntimeError(f"Unexpected SimpleMem generation stage: {caller}")
        stage = _STAGES[caller]
        if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
            raise RuntimeError("Unexpected native SimpleMem message structure")
        prompt, system = messages[1]["content"], messages[0]["content"]
        request_id = _hash({"stage": stage, "prompt": prompt, "system": system})
        self.audit.record({"event": "generation_request", "request_id": request_id,
                           "stage": stage, "prompt": prompt, "system": system,
                           "max_output": 16384, "temperature_sent": False})
        response = self.api.generate(prompt, system=system, json_output=False,
                                     max_output=16384, stage=stage)
        if isinstance(response, (dict, list)):
            response = json.dumps(response, ensure_ascii=False)
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"Empty/non-text SimpleMem generation: {stage}")
        self.audit.record({"event": "generation_response", "request_id": request_id,
                           "stage": stage, "response": response})
        return response

    def extract_json(self, response):
        text = response.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = text[:-3].strip()
        value = json.loads(text)
        caller = inspect.currentframe().f_back.f_code.co_name
        if caller == "_parse_llm_response":
            if not isinstance(value, list):
                raise ValueError("SimpleMem writer must return a JSON array")
            cls = _load_native().memory_entry.MemoryEntry
            for item in value:
                entry = cls.model_validate(item)
                if not entry.lossless_restatement.strip():
                    raise ValueError("Empty memory restatement")
        elif not isinstance(value, dict):
            raise ValueError(f"SimpleMem {caller} must return a JSON object")
        elif caller == "_analyze_information_requirements":
            if not isinstance(value.get("required_info"), list):
                raise ValueError("Missing required_info list")
        elif caller == "_analyze_query":
            for key in ("keywords", "persons", "entities"):
                if not isinstance(value.get(key), list) or any(not isinstance(x, str) for x in value[key]):
                    raise ValueError(f"Invalid query-analysis {key}")
            for key in ("time_expression", "location"):
                if key not in value or (value[key] is not None and not isinstance(value[key], str)):
                    raise ValueError(f"Invalid query-analysis {key}")
        elif caller in ("_generate_targeted_queries", "_generate_missing_info_queries"):
            key = "queries" if caller == "_generate_targeted_queries" else "targeted_queries"
            queries = value.get(key)
            if not isinstance(queries, list) or not queries or any(not isinstance(x, str) or not x.strip() for x in queries):
                raise ValueError(f"Invalid {key}")
        elif caller == "_analyze_information_completeness":
            if value.get("assessment") not in ("complete", "incomplete"):
                raise ValueError("Unknown completeness assessment")
        return value


class _EmbeddingBridge:
    dimension = 768

    def __init__(self, api, audit):
        self.api, self.audit = api, audit

    def _encode(self, texts, query):
        import numpy as np
        self.audit.record({"event": "embedding_request", "query": query, "texts": texts,
                           "model": "gemini-embedding-001", "dimensions": self.dimension})
        vectors = np.asarray(self.api.embed(texts, query=query), dtype=np.float64)
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Gemini embedding shape/finiteness contract failed")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Gemini returned a zero embedding")
        normalized = vectors / norms
        self.audit.record({"event": "embedding_response", "query": query,
                           "count": len(texts), "vectors_sha256": _hash(normalized.tolist())})
        return normalized

    def encode_documents(self, texts):
        return self._encode(texts, False)

    def encode_single(self, text, is_query=False):
        return self._encode([text], is_query)[0]


def _make_store(directory, api, audit):
    native = _load_native()
    return native.vector_store.VectorStore(db_path=str(directory), table_name="memory_entries",
                                           embedding_model=_EmbeddingBridge(api, audit))


def _finalize_fts(store, audit):
    """Refresh the original index after the last append, before any queries.

    Upstream builds Tantivy on the first insert only; its initialized flag would
    otherwise leave later writer windows out of lexical retrieval. Reuse the
    exact native local-index implementation and tokenizer, not another FTS.
    """
    count = store.backend.count()
    if count:
        store.backend._fts_initialized = False
        store.backend._init_fts_index()
    audit.record({"event": "final_fts_rebuild", "entry_count": count,
                  "backend": "Tantivy", "tokenizer_name": "en_stem",
                  "completed": True, "skipped_empty_bank": count == 0})


def _entry_digest(entries):
    return _hash(sorted(entries, key=lambda entry: entry["entry_id"]))


def _database_snapshot(store):
    """Bind one complete native snapshot, including the stored float32 vectors."""
    rows = store.table.to_arrow().to_pylist()
    if len(rows) != store.backend.count():
        raise RuntimeError("SimpleMem snapshot row count differs from native table")
    ids = [row["entry_id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("SimpleMem database has invalid or duplicate memory IDs")
    native = _load_native()
    results = native.vector_store_backend.LanceDBVectorStoreBackend._rows_to_results(rows)
    entries = store._results_to_entries(results)
    if len(entries) != len(rows):
        raise RuntimeError("SimpleMem native rows cannot be exported without loss")
    return entries, {"row_count": len(rows),
                     "rows_sha256": _hash(sorted(rows, key=lambda row: row["entry_id"])),
                     "entries_sha256": _entry_digest([entry.model_dump() for entry in entries])}


def _verify_database(store, state, export):
    _, actual = _database_snapshot(store)
    if actual != state.get("database_integrity") or actual["row_count"] != state["entry_count"]:
        raise RuntimeError("SimpleMem native database content differs from committed memory")
    if actual["entries_sha256"] != _entry_digest(export["entries"]):
        raise RuntimeError("SimpleMem native database does not match exported memory entries")


def _dialogues(conversation):
    if not isinstance(conversation, dict):
        raise ValueError("conversation must contain raw conversation only")
    allowed = {"speaker_a", "speaker_b"}
    for key in conversation:
        if key not in allowed and not re.fullmatch(r"session_\d+(?:_date_time)?", key):
            raise ValueError(f"Non-source conversation field is forbidden: {key}")
    native = _load_native()
    dialogues, sources = [], []
    sessions = sorted((int(k.split("_")[1]), k) for k, v in conversation.items()
                      if re.fullmatch(r"session_\d+", k))
    for session_id, key in sessions:
        turns = conversation[key]
        date = conversation.get(key + "_date_time")
        if not isinstance(turns, list) or not isinstance(date, str) or not date:
            raise ValueError(f"Invalid/missing session date or turns: {key}")
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("text", ""), str):
                raise ValueError("Invalid raw turn")
            if any(k in turn for k in ("qa", "answer", "evidence", "observation", "event_summary")):
                raise ValueError("Forbidden gold/derived field in turn")
            speaker, dia_id = turn.get("speaker"), turn.get("dia_id")
            if not isinstance(speaker, str) or not isinstance(dia_id, str) or not dia_id:
                raise ValueError("Each turn requires speaker and dia_id")
            text = turn.get("text", "")
            if "img_url" in turn and "blip_caption" in turn:
                text = f"[Image: {turn['blip_caption']}] {text}".rstrip()
            number = len(dialogues) + 1
            dialogues.append(native.memory_entry.Dialogue(dialogue_id=number, speaker=speaker,
                                                           content=text, timestamp=date))
            sources.append({"dialogue_id": number, "dia_id": dia_id, "session_id": session_id,
                            "date": date, "speaker": speaker, "content": text})
    if not dialogues or len({x["dia_id"] for x in sources}) != len(sources):
        raise ValueError("Empty conversation or duplicated source dia_id")
    return dialogues, sources


def _fatal_boundary(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from tools.baselines_common_v1 import BaselineFatal
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            raise BaselineFatal(f"SimpleMem {function.__name__}: {type(exc).__name__}: {exc}") from exc
    return wrapped


@_fatal_boundary
def build(conversation: dict, work_dir: Path, api) -> dict:
    """Build once, synchronously within a conversation. Parent parallelizes conversations."""
    _preflight()
    dialogues, sources = _dialogues(conversation)
    directory = Path(work_dir).resolve() / "simplemem"
    source_hash, protocol_hash = _hash(conversation), _hash(describe())
    state_path = directory / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_hash") != source_hash or state.get("protocol_hash") != protocol_hash:
            raise RuntimeError("Existing SimpleMem bank has different source/protocol")
        export = _verify_state(state)
        store = _make_store(directory / "lancedb", api, _Audit(directory / "requests.jsonl"))
        _verify_database(store, state, export)
        return state
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Partial SimpleMem build exists; explicit recovery required before more calls")
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "source.json", {"conversation": conversation, "sources": sources})
    audit = _Audit(directory / "requests.jsonl")
    store = _make_store(directory / "lancedb", api, audit)
    if store.backend.count() != 0:
        raise RuntimeError("Refusing to append to a nonempty uncommitted SimpleMem bank")
    native = _load_native()
    provenance = {}

    class Builder(native.memory_builder.MemoryBuilder):
        def _parse_llm_response(self, response, dialogue_ids):
            entries = super()._parse_llm_response(response, dialogue_ids)
            for position, entry in enumerate(entries):
                entry.entry_id = _hash({"source": source_hash, "window": dialogue_ids,
                                        "position": position, "entry": entry.model_dump(exclude={"entry_id"})})
                provenance[entry.entry_id] = list(dialogue_ids)
            _write_json(directory / "provenance.partial.json", provenance)
            return entries

    builder = Builder(llm_client=_LLMBridge(api, audit), vector_store=store,
                      window_size=40, enable_parallel_processing=False, max_parallel_workers=1)
    for dialogue in dialogues:
        builder.add_dialogue(dialogue)
    builder.process_remaining()
    _finalize_fts(store, audit)
    entries, database_integrity = _database_snapshot(store)
    if len(entries) != store.backend.count() or len(entries) != len(provenance):
        raise RuntimeError("Native index/export/provenance counts disagree")
    export = {"protocol": describe(), "source_hash": source_hash,
              "database_integrity": database_integrity,
              "entries": [entry.model_dump() for entry in entries],
              "window_provenance": provenance, "sources": sources}
    memory_path, text_path = directory / "memory.json", directory / "memory.txt"
    _write_json(memory_path, export)
    text_path.write_text(native.answer_generator.AnswerGenerator(None)._format_contexts(entries), encoding="utf-8")
    export_paths = [memory_path, text_path, directory / "source.json"]
    state = {"schema": "simplemem-adapter-state-v2", "method": "SimpleMem",
             "source_sha": SOURCE_SHA, "source_hash": source_hash, "protocol_hash": protocol_hash,
             "work_dir": str(directory), "memory_path": str(memory_path),
             "memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
             "database_integrity": database_integrity,
             "entry_count": len(entries), "turn_count": len(dialogues),
             "export_paths": [str(path) for path in export_paths],
             "export_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in export_paths},
             "request_log": str(audit.path), "embedding_dimensions": 768}
    _write_json(state_path, state)
    return state


def _verify_state(state):
    if (state.get("schema") != "simplemem-adapter-state-v2" or state.get("source_sha") != SOURCE_SHA
            or state.get("protocol_hash") != _hash(describe())):
        raise RuntimeError("SimpleMem state protocol/revision mismatch")
    path = Path(state["memory_path"])
    directory = Path(state["work_dir"]).resolve()
    if path.resolve() != directory / "memory.json":
        raise RuntimeError("SimpleMem memory export path differs from committed directory")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != state["memory_sha256"]:
        raise RuntimeError("SimpleMem memory export missing or changed")
    if set(state.get("export_sha256", {})) != {"memory.json", "memory.txt", "source.json"}:
        raise RuntimeError("SimpleMem required export integrity bindings are missing")
    for name, expected in state["export_sha256"].items():
        retained = directory / name
        if not retained.is_file() or hashlib.sha256(retained.read_bytes()).hexdigest() != expected:
            raise RuntimeError("SimpleMem source/text/memory export missing or changed: " + name)
    export = json.loads(path.read_text(encoding="utf-8"))
    if export.get("database_integrity") != state.get("database_integrity"):
        raise RuntimeError("SimpleMem export/database integrity bindings disagree")
    return export


@_fatal_boundary
def retrieve(question: str, state: dict, api) -> dict:
    """Return every native retrieved memory in native formatting, never a top-20 slice."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    _preflight()
    export = _verify_state(state)
    directory = Path(state["work_dir"])
    retrieval_id = _hash(question) + "-" + uuid.uuid4().hex
    audit = _Audit(directory / "retrieval" / (retrieval_id + ".requests.jsonl"))
    store = _make_store(directory / "lancedb", api, audit)
    _verify_database(store, state, export)
    native = _load_native()
    retriever = native.hybrid_retriever.HybridRetriever(
        llm_client=_LLMBridge(api, audit), vector_store=store, semantic_top_k=25,
        keyword_top_k=5, structured_top_k=5, enable_planning=True,
        enable_reflection=True, max_reflection_rounds=2, enable_parallel_retrieval=False,
        max_retrieval_workers=1)
    entries = retriever.retrieve(question)
    source_lookup = {x["dialogue_id"]: x for x in export["sources"]}
    evidence = []
    for entry in entries:
        ids = export["window_provenance"].get(entry.entry_id)
        if ids is None:
            raise RuntimeError("Retrieved memory is absent from source provenance")
        evidence.append({"memory_id": entry.entry_id, "memory": entry.model_dump(),
                         "source_dia_ids": [source_lookup[i]["dia_id"] for i in ids],
                         "attribution": "source window, not individually verified fact evidence"})
    result = {"context": native.answer_generator.AnswerGenerator(None)._format_contexts(entries),
              "evidence": evidence, "count": len(entries), "source_sha": SOURCE_SHA,
              "source_hash": state["source_hash"], "request_log": str(audit.path),
              "retrieval_id": retrieval_id}
    _write_json(directory / "retrieval" / (retrieval_id + ".result.json"), result)
    return result
