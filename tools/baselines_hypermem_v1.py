"""Pinned HyperMem stages 1--4, with explicit Gemini encoder/no-reranker adaptation."""
from __future__ import annotations

import ast
import asyncio
import contextvars
import copy
import hashlib
import inspect
import json
import pickle
import re
import sys
import threading
import types
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.baselines_common_v1 import BaselineFatal
from tools.baselines_amem_v1 import _sha, _write_json, _read_json, _sessions

UPSTREAM_REVISION = "15c700908f3ec64d6931f253695e42c679a4c958"
UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "upstream/baselines_v1/hypermem/hypermem"
SCHEMA = "hypermem-gemini-embedding-no-reranker-v1"
SOURCE_SHA256 = {
    "types.py": "59de4929c5a9c7f49aa1dbd073cfb401c3b2a7e023ed68c3062b4d34cc5f4a8c",
    "structure.py": "78f633e9ef7752bdb55ce88db451f0358711cfb4231af15678a3190dac0262c8",
    "utils/datetime_utils.py": "fe09b141433c0ca09c13f0473096b12d2674ffb9b7ce2a357ded0fb61f7a85ff",
    "utils/logger.py": "d02cb9ee69827f098e93f79c9251f230236683e99af4db1b053887781ce49cd7",
    "prompts/conv_prompts.py": "8fbb17e60337114d5ce5edc80205a131ab953d95f42d0ecb6f736a8936012d7d",
    "prompts/episode_prompts.py": "624b9966eb8b100781e86299a6c19448fb4b95b5799a07854a39a3f574bc6e88",
    "prompts/topic_prompts.py": "d1a5acaa386330bf64e254e9d0cb63875954d253e120b008eddb1c29e14dd390",
    "prompts/fact_prompts.py": "329acab01074a5d14d6e787856db34c6ce6177cb3888aa9665801143c6840b78",
    "extractors/episode_extractor.py": "cfe90dce4ab1d5bf3452b504cd2c57c6844c28dfb7a5660375a3cfb65bc238b3",
    "extractors/conv_episode_extractor.py": "43080b123a0c43b11fb24cd316c3ef7c9a4fe8714c9008635075f2568d94a247",
    "extractors/topic_extractor.py": "a1ba0ee79395259fbea9ef711174955175201e6ae580030741414c9cf03c424f",
    "extractors/fact_extractor.py": "1ad3b7a626af25792866ec63997bc3019cdcb9324a5bda0a7c30b1c375e16904",
    "extractors/hypergraph_extractor.py": "4b829d4dcbe64e885d14cdc8de7d8505c68746a25c29736f4fb4507481b39535",
    "main/stage1_memory_extraction.py": "177134a0ffb20d4a35b3cd035bceed327852b6a7a734bca003e70efca7dc9835",
    "main/stage2_hypergraph_extraction.py": "95f5ac4e67dac3db5a56c87d5aab74d5c03b44af458ed285bf5754bf488f98fe",
    "main/stage3_hypergraph_index.py": "1119393fc01430f9fd6d6375a5012edeb88692cdc915b87e234d9636c7d0d29d",
    "main/stage4_hypergraph_retrieval.py": "6352673ffc7a12eb5289299edca54e70a4e01b6ca0a13390fe208cd1e9ef98f0",
    "llm/embedding_provider.py": "6386b95e6e15a4b3b67db33ab05deeb8fd9bbd23e408a9a1bc2eb7e4498c6300",
}
CONFIG = {
    "retrieval_type": "rrf", "rerank_type": "fix", "hypergraph_retrieval_output_type": "011",
    "retrieval_config": {"initial_candidates": 100, "topic_top_k": 10, "episode_top_k": 10, "fact_top_k": 30},
    "temporal_enhancement": True, "use_reranker": False,
    "hyperedge_emb_aggregate_type": "sum", "node_emb_update_weight": 0.5,
}
_PACKAGE = "_jev_hypermem_pinned_v1"
_LOAD_LOCK = threading.Lock()
_MODULES = None
_UUID_CONTEXT = contextvars.ContextVar("hypermem_uuid", default=None)
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()
_INDEXES = {}
_EXPORT_ARTIFACTS = ("source_messages.json", "episodes.json", "topics/topics_conv_0.json",
                     "facts/facts_conv_0.json", "hypergraphs/hypergraph_conv_0.json",
                     "generation_evidence_summary.json", "embedding_evidence_summary.json")


class HyperMemError(BaselineFatal):
    pass


def _lock(key):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(key), threading.RLock())


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal_path(path):
    path = Path(path)
    return path.parent / ".integrity" / (path.name + ".sha256")


def _verify_seal(path):
    path = Path(path)
    try:
        if _seal_path(path).read_text(encoding="ascii") != _file_sha(path):
            raise HyperMemError(f"Artifact integrity mismatch: {path}")
    except OSError as exc:
        raise HyperMemError(f"Missing artifact or integrity seal: {path}") from exc


def _seal(path):
    seal = _seal_path(path)
    if seal.exists():
        _verify_seal(path)
        return
    seal.parent.mkdir(parents=True, exist_ok=True)
    with seal.open("x", encoding="ascii") as stream:
        stream.write(_file_sha(path))


def _verify_artifacts(work_dir, expected):
    for relative, digest in expected.items():
        path = Path(work_dir) / relative
        try:
            if _file_sha(path) != digest:
                raise HyperMemError(f"Artifact integrity mismatch: {relative}")
        except OSError as exc:
            raise HyperMemError(f"Required sealed artifact is missing: {relative}") from exc


def _uuid4():
    context = _UUID_CONTEXT.get()
    if context is None:
        raise HyperMemError("Deterministic HyperMem UUID context is missing")
    context["counter"] += 1
    return uuid.uuid5(uuid.NAMESPACE_URL, f"hypermem:{context['seed']}:{context['counter']}")


def _fatal_current():
    cause = sys.exception()
    raise HyperMemError(f"Upstream operation failed without retry: {type(cause).__name__}: {cause}") from cause


class _SourceAdapter(ast.NodeTransformer):
    def __init__(self, filename):
        self.filename = filename
        self.depth = 0
        self.handler_count = 0

    def visit_ImportFrom(self, node):
        if node.module and (node.module == "hypermem" or node.module.startswith("hypermem.")):
            node.module = _PACKAGE + node.module[len("hypermem"):]
        return node

    def visit_Import(self, node):
        # Keep generated IDs stable across independent conversations and replay.
        for alias in node.names:
            if alias.name == "uuid":
                alias.name, alias.asname = _PACKAGE + ".uuid", alias.asname or "uuid"
        return node

    def visit_FunctionDef(self, node):
        self.depth += 1
        node = self.generic_visit(node)
        self.depth -= 1
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ExceptHandler(self, node):
        # Imports stay optional. Algorithm failures never manufacture fallback
        # values or trigger unlimited internal retries/repair calls.
        if self.depth and (self.filename.startswith("extractors/") or self.filename.startswith("main/")):
            node.body = [ast.Expr(ast.Call(ast.Name("_fatal_current", ast.Load()), [], []))]
            self.handler_count += 1
            return node
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if self.filename == "main/stage1_memory_extraction.py" and node.arg == "timestamp" and isinstance(node.value, ast.IfExp):
            # Only the final Episode constructor uses previous-episode / now().
            node.value = ast.parse("from_iso_format(history_raw_data_list[-1].content['timestamp'])", mode="eval").body
        return self.generic_visit(node)


def _check_nltk():
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    try:
        stopwords.words("english")
        word_tokenize("Verify local tokenizer resources.")
    except LookupError as exc:
        raise HyperMemError("HyperMem needs preinstalled NLTK stopwords/punkt_tab resources; no download during an experiment") from exc


def _module(name, package=False):
    full_name = _PACKAGE + ("." + name if name else "")
    module = types.ModuleType(full_name)
    module.__package__ = full_name if package else full_name.rpartition(".")[0]
    if package:
        module.__path__ = []
    sys.modules[full_name] = module
    if "." in full_name:
        parent, _, leaf = full_name.rpartition(".")
        setattr(sys.modules[parent], leaf, module)
    return module


def _load():
    global _MODULES
    with _LOAD_LOCK:
        if _MODULES is not None:
            return _MODULES
        sources = {}
        for filename, expected in SOURCE_SHA256.items():
            source = (UPSTREAM_DIR / filename).read_text(encoding="utf-8")
            if hashlib.sha256(source.encode()).hexdigest() != expected:
                raise HyperMemError(f"Pinned HyperMem source mismatch: {filename}")
            sources[filename] = source
        for package in ("", "utils", "prompts", "llm", "extractors", "main"):
            _module(package, package=True)
        _module("uuid").uuid4 = _uuid4
        config = _module("config")
        config.ExperimentConfig = type("ExperimentConfig", (), copy.deepcopy(CONFIG))
        for name, typename in (("llm.llm_provider", "LLMProvider"), ("llm.embedding_provider", "EmbeddingProvider"), ("llm.reranker_provider", "RerankerProvider")):
            setattr(_module(name), typename, object)
        order = ["utils/datetime_utils.py", "utils/logger.py", "types.py", "structure.py",
                 "prompts/conv_prompts.py", "prompts/episode_prompts.py", "prompts/topic_prompts.py", "prompts/fact_prompts.py",
                 "extractors/episode_extractor.py", "extractors/conv_episode_extractor.py",
                 "extractors/topic_extractor.py", "extractors/fact_extractor.py", "extractors/hypergraph_extractor.py",
                 "main/stage1_memory_extraction.py", "main/stage2_hypergraph_extraction.py",
                 "main/stage3_hypergraph_index.py", "main/stage4_hypergraph_retrieval.py"]
        modules = {}
        for filename in order:
            tree = ast.parse(sources[filename])
            tree.body = [node for node in tree.body
                         if not (isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
                         and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                  and ast.unparse(node.value.func).startswith("sys.path."))]
            adapter = _SourceAdapter(filename)
            tree = ast.fix_missing_locations(adapter.visit(tree))
            module = _module(filename[:-3].replace("/", "."))
            module.__file__ = str(UPSTREAM_DIR / filename)
            module.__dict__.update(_fatal_current=_fatal_current, print=lambda *args, **kwargs: None)
            exec(compile(tree, module.__file__, "exec"), module.__dict__)
            if hasattr(module, "logger"):
                module.logger.disabled = True
            if hasattr(module, "console"):
                module.console = SimpleNamespace(print=lambda *args, **kwargs: None)
            if hasattr(module, "ensure_nltk_data"):
                module.ensure_nltk_data = _check_nltk
            module.adapter_exception_handlers = adapter.handler_count
            modules[filename] = module
        # Retain the original embedding provider cosine operation byte-for-byte.
        tree = ast.parse(sources["llm/embedding_provider.py"])
        method = next(method for node in tree.body if isinstance(node, ast.ClassDef)
                      for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "cosine_similarity")
        namespace = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "<original-hypermem-cosine>", "exec"), namespace)
        _Embedding.cosine_similarity = namespace["cosine_similarity"]
        _MODULES = modules
        return modules


_GENERATION_STAGES = {
    "_detect_boundary": "hypermem_boundary", "_generate_episode_memory": "hypermem_episode",
    "_extract_new_topic": "hypermem_topic_create", "_update_existing_topic": "hypermem_topic_update",
    "_assign_episode_roles_and_weights": "hypermem_episode_roles", "_llm_match_topics_batch": "hypermem_topic_match",
    "_extract_facts_stage": "hypermem_fact_extract", "_assign_fact_roles_stage": "hypermem_fact_roles",
}


class _Generation:
    def __init__(self, api, work_dir, group):
        self.api, self.work_dir, self.group = api, Path(work_dir), group
        self.count = 0

    async def generate(self, prompt, response_format=None, **kwargs):
        frame = inspect.currentframe().f_back
        method = frame.f_code.co_name
        stage = _GENERATION_STAGES.get(method)
        if stage is None:
            raise HyperMemError(f"Unregistered HyperMem generation caller: {method}")
        self.count += 1
        path = self.work_dir / "calls" / f"{self.group}_{self.count:06d}_{stage}.json"
        request = {"prompt": prompt, "response_format": response_format, "stage": stage,
                   "json_output": True, "max_output": 16384, "system": None}
        record = {"request": request, "request_sha256": _sha(request), "status": "started"}
        if path.exists():
            cached = _read_json(path)
            if cached.get("request_sha256") != record["request_sha256"] or cached.get("status") != "success":
                raise HyperMemError(f"Call mismatch/failed/unknown outcome requires explicit recovery: {path}")
            _verify_seal(path)
            value = cached["parsed_response"]
        else:
            if _seal_path(path).exists():
                raise HyperMemError(f"Sealed generation artifact is missing: {path}")
            _write_json(path, record)
            try:
                raw = self.api.generate(prompt, system=None, json_output=True, max_output=16384, stage=stage)
                record["raw_response"] = raw
                value = raw if isinstance(raw, dict) else json.loads(raw)
                self._validate(value, stage, frame.f_locals)
                record.update(status="success", parsed_response=value)
                _write_json(path, record)
                _seal(path)
            except BaseException as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                _write_json(path, record)
                if isinstance(exc, BaselineFatal):
                    raise
                raise HyperMemError(f"{stage} failed: {path}") from exc
        self._validate(value, stage, frame.f_locals)
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _validate(value, stage, local):
        if not isinstance(value, dict) or not value:
            raise HyperMemError(f"{stage} returned empty/nonobject JSON")
        if stage == "hypermem_boundary":
            if not all(isinstance(value.get(k), bool) for k in ("should_end", "should_wait")):
                raise HyperMemError("Boundary decision lacks boolean should_end/should_wait")
        elif stage == "hypermem_episode":
            if not all(isinstance(value.get(k), str) and value[k].strip() for k in ("title", "content", "summary")):
                raise HyperMemError("Episode response lacks nonempty title/content/summary")
        elif stage == "hypermem_topic_match":
            mapping = local["simple_to_real"]
            results = value.get("results")
            if not isinstance(results, list) or len(results) != len(mapping):
                raise HyperMemError("Topic-match response must decide every candidate")
            if {item.get("topic_id") for item in results if isinstance(item, dict)} != set(mapping):
                raise HyperMemError("Topic-match IDs are incomplete or invalid")
            if not all(isinstance(item.get("match"), bool) for item in results):
                raise HyperMemError("Topic-match decisions must be booleans")
        elif stage == "hypermem_episode_roles":
            rows = value.get("episode_roles")
            allowed = set(local["simple_to_real"])
            if not isinstance(rows, list) or len(rows) != len(allowed) or any(not isinstance(item, dict) for item in rows):
                raise HyperMemError("Episode role assignments are incomplete")
            if {item.get("episode_id") for item in rows} != allowed:
                raise HyperMemError("Episode role assignment IDs are invalid")
            roles = {"initiating", "developing", "climax", "concluding", "recurring", "background", "key_moment", "transition"}
            if any(item.get("role") not in roles or not isinstance(item.get("weight"), (int, float))
                   or isinstance(item.get("weight"), bool) or not 0 <= item["weight"] <= 1 for item in rows):
                raise HyperMemError("Episode role or weight is invalid")
        elif stage == "hypermem_fact_roles":
            facts = local["facts"]
            mapping = {f"fact_{i+1}": fact.fact_id for i, fact in enumerate(facts)}
            expected = {fact.fact_id for fact in facts}
            rows = value.get("fact_roles")
            if not isinstance(rows, list) or len(rows) != len(expected) or any(not isinstance(item, dict) for item in rows):
                raise HyperMemError("Fact role assignments are incomplete")
            if {mapping.get(item.get("fact_id"), item.get("fact_id")) for item in rows} != expected:
                raise HyperMemError("Fact role assignment IDs are invalid")
        # The pinned extractors supply their original validators for other stages;
        # AST hard errors prevent invalid outputs reaching their retry fallbacks.


class _Embedding:
    def __init__(self, api, work_dir, query=False):
        self.api, self.work_dir, self.query = api, Path(work_dir), query
        self.evidence = []

    def embed(self, texts):
        identity = {"texts": list(texts), "query": self.query, "model": "gemini-embedding-001", "dimensions": 768}
        path = self.work_dir / "embedding_evidence" / (("query_" if self.query else "document_") + _sha(identity) + ".json")
        with _lock(path):
            cache_hit = path.exists()
            if path.exists():
                record = _read_json(path)
                if record.get("identity") != identity or record.get("status") != "success":
                    raise HyperMemError(f"Embedding mismatch/failed/unknown outcome: {path}")
                _verify_seal(path)
                vectors = record["vectors"]
            else:
                if _seal_path(path).exists():
                    raise HyperMemError(f"Sealed embedding artifact is missing: {path}")
                record = {"identity": identity, "status": "started"}
                _write_json(path, record)
                try:
                    vectors = self.api.embed(list(texts), query=self.query)
                    vectors = np.asarray(vectors).tolist()
                    self._validate(vectors, len(texts))
                    record.update(status="success", vectors=vectors)
                    _write_json(path, record)
                    _seal(path)
                except BaseException as exc:
                    record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    _write_json(path, record)
                    if isinstance(exc, BaselineFatal):
                        raise
                    raise HyperMemError(f"Embedding failed: {path}") from exc
            self._validate(vectors, len(texts))
            self.evidence.append({"request_sha256": _sha(identity), "query": self.query,
                                  "text_count": len(texts), "texts_sha256": _sha(identity["texts"]),
                                  "model": identity["model"], "dimensions": 768,
                                  "vectors_sha256": _sha(vectors), "cache_hit": cache_hit,
                                  "vector_values_retained_in_results": False})
            return vectors

    @staticmethod
    def _validate(vectors, count):
        array = np.asarray(vectors)
        if array.shape != (count, 768) or not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
            raise HyperMemError("Gemini embedding must be finite, nonzero, 768-dimensional")
        if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-4):
            raise HyperMemError("Gemini embedding adapter contract requires L2-normalized vectors")


def _messages(conversation, modules):
    date_utils = modules["utils/datetime_utils.py"]
    messages = []
    for session, timestamp, turns in _sessions(conversation):
        real_date = datetime.strptime(timestamp, "%I:%M %p on %d %B, %Y")
        iso = date_utils.to_iso_format(real_date)
        for position, turn in enumerate(turns):
            speaker = turn["speaker"]
            content = turn["text"]
            if turn.get("img_url"):
                content = f"[{speaker} shared an image: {turn.get('blip_caption', 'an image')}] {content}"
            messages.append({"speaker_id": f"{speaker.lower().replace(' ', '_')}_0", "user_name": speaker,
                             "speaker_name": speaker, "content": content, "timestamp": iso,
                             "original_timestamp": timestamp, "dia_id": turn["dia_id"], "session": session,
                             "source_turn_position": position,
                             **{k: turn[k] for k in ("img_url", "blip_caption", "query") if k in turn}})
    return messages


def _stage_checkpoint(path, data):
    stages = ["episode_stage.json", "topic_stage.json", "fact_stage.json", "graph_stage.json"]
    artifacts = ["source_messages.json", "episodes.json", "topics/topics_conv_0.json",
                 "facts/facts_conv_0.json", "hypergraphs/hypergraph_conv_0.json"][:stages.index(Path(path).name) + 2]
    _write_json(path, {"data": data, "uuid_counter": _UUID_CONTEXT.get()["counter"],
                       "artifacts": {name: _file_sha(Path(path).parent / name) for name in artifacts}})
    _seal(path)


def _restore_counter(path):
    _verify_seal(path)
    checkpoint = _read_json(path)
    _verify_artifacts(Path(path).parent, checkpoint["artifacts"])
    _UUID_CONTEXT.get()["counter"] = checkpoint["uuid_counter"]
    return checkpoint["data"]


def _build_indexes(work_dir, api, modules, expected=None, expected_vectors=None):
    if expected_vectors is not None:
        if not expected_vectors:
            raise HyperMemError("Completed HyperMem state lacks document vector bindings")
        _verify_artifacts(work_dir, expected_vectors)
    _check_nltk()
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    cache = work_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    paths = {"bm25": cache / "hypergraph_bm25_index_conv_0.pkl",
             "vectors": cache / "hypergraph_embedding_index_conv_0.pkl",
             "enhanced": cache / "hypergraph_embedding_index_enhanced_conv_0.pkl"}
    if expected:
        for name, path in paths.items():
            if path.exists() and _file_sha(path) != expected.get(name):
                raise HyperMemError(f"Index cache hash mismatch: {path}")
    for path in paths.values():
        if path.exists():
            _verify_seal(path)
    stage3 = modules["main/stage3_hypergraph_index.py"]
    if not paths["bm25"].exists():
        result = stage3.build_bm25_index_for_single_conv(0, work_dir / "hypergraphs", cache,
                                                       PorterStemmer(), set(stopwords.words("english")), skip_existing=False)
        if result != (True, "completed"):
            raise HyperMemError("Native BM25 index did not complete")
    if not paths["vectors"].exists() or not paths["enhanced"].exists():
        result = stage3.build_embedding_index_for_single_conv(0, work_dir / "hypergraphs", cache,
                    _Embedding(api, work_dir), alpha=0.5, aggregate_type="sum", skip_existing=False)
        if result != (True, "completed"):
            raise HyperMemError("Native enhanced embedding index did not complete")
    hashes = {name: _file_sha(path) for name, path in paths.items()}
    if expected is not None and hashes != expected:
        raise HyperMemError("Rebuilt native index differs from the completed state")
    for path in paths.values():
        _seal(path)
    with paths["bm25"].open("rb") as stream:
        bm25 = pickle.load(stream)
    with paths["vectors"].open("rb") as stream:
        vectors = pickle.load(stream)
    if not bm25.get("docs") or not vectors:
        raise HyperMemError("Required native retrieval index is empty")
    return bm25, vectors, hashes


def _write_evidence_summaries(work_dir):
    generations = []
    for path in sorted((work_dir / "calls").glob("*.json")):
        _verify_seal(path)
        record = _read_json(path)
        generations.append({"call_artifact": str(path.relative_to(work_dir)),
                            "stage": record["request"]["stage"], "status": record["status"],
                            "request_sha256": record["request_sha256"],
                            "prompt_sha256": _sha(record["request"]["prompt"]),
                            "raw_response_sha256": _sha(record.get("raw_response")),
                            "parsed_response_sha256": _sha(record.get("parsed_response"))})
    _write_json(work_dir / "generation_evidence_summary.json", {
        "calls": generations, "full_raw_retention": "parent experiment retains one compressed raw API evidence copy"})
    embeddings = []
    for path in sorted((work_dir / "embedding_evidence").glob("document_*.json")):
        _verify_seal(path)
        record = _read_json(path)
        identity = record["identity"]
        embeddings.append({"artifact": str(path.relative_to(work_dir)), "artifact_sha256": _file_sha(path),
                           "request_sha256": _sha(identity), "query": identity["query"],
                           "text_count": len(identity["texts"]), "texts_sha256": _sha(identity["texts"]),
                           "model": identity["model"], "dimensions": identity["dimensions"],
                           "vectors_sha256": _sha(record.get("vectors")), "status": record["status"]})
    _write_json(work_dir / "embedding_evidence_summary.json", {
        "calls": embeddings, "vector_values_retained_in_results": False,
        "cleanup_policy": "complete vectors and indexes are disposable work data after parent audit/seal",
        "replay_limit": "after cache deletion, exact numeric replay requires re-encoding; summaries verify prior vectors only"})


async def _build_async(conversation, work_dir, api, modules):
    stage1 = modules["main/stage1_memory_extraction.py"]
    stage2 = modules["main/stage2_hypergraph_extraction.py"]
    date_utils = modules["utils/datetime_utils.py"]
    if (work_dir / "graph_stage.json").exists():
        _restore_counter(work_dir / "graph_stage.json")
        return _read_json(work_dir / "hypergraphs/hypergraph_conv_0.json")
    messages = _messages(conversation, modules)
    source_path = work_dir / "source_messages.json"
    if source_path.exists():
        _verify_seal(source_path)
        if _read_json(source_path) != messages:
            raise HyperMemError("Source message mapping differs from the requested conversation")
    else:
        if _seal_path(source_path).exists():
            raise HyperMemError("Sealed source message mapping is missing")
        _write_json(source_path, messages)
        _seal(source_path)
    for session in sorted({message["session"] for message in messages}):
        _write_json(work_dir / "sessions" / session / "source.json", [m for m in messages if m["session"] == session])
    episodes_file = work_dir / "episodes.json"
    checkpoint = work_dir / "episode_stage.json"
    if checkpoint.exists():
        _restore_counter(checkpoint)
        episodes = stage2.load_episodes_from_json(str(episodes_file))
    else:
        raw = stage1.convert_conversation_to_raw_data_list(messages)
        episodes = await stage1.episode_extraction_from_conversation(raw, llm_provider=_Generation(api, work_dir, "episode"), smart_mask=True, conv_id="0")
        if not episodes:
            raise HyperMemError("Episode extraction produced no episodes")
        for ep in episodes:
            if isinstance(ep.timestamp, (int, float)):
                ep.timestamp = date_utils.from_timestamp(ep.timestamp)
            elif isinstance(ep.timestamp, str):
                ep.timestamp = date_utils.from_iso_format(ep.timestamp)
            if not ep.subject or not ep.episode_description or not ep.original_data:
                raise HyperMemError("Episode lacks generated memory or source provenance")
        _write_json(episodes_file, [ep.to_dict() for ep in episodes])
        _stage_checkpoint(checkpoint, {"episode_count": len(episodes), "artifact_sha256": _file_sha(episodes_file)})
    checkpoint = work_dir / "topic_stage.json"
    if checkpoint.exists():
        _restore_counter(checkpoint)
        topic_result, episode_edges = stage2.load_topic_results("0", work_dir / "topics")
    else:
        topic_result, episode_edges = await stage2.extract_topics_for_episodes(episodes, _Generation(api, work_dir, "topic"))
        if not topic_result or not topic_result.topics or not episode_edges:
            raise HyperMemError("Topic/episode-hyperedge extraction is empty")
        stage2.save_topic_results("0", topic_result, episode_edges, work_dir / "topics")
        _stage_checkpoint(checkpoint, {"topic_count": len(topic_result.topics)})
    checkpoint = work_dir / "fact_stage.json"
    if checkpoint.exists():
        _restore_counter(checkpoint)
        fact_results, fact_edges = stage2.load_fact_results("0", work_dir / "facts")
    else:
        fact_results, fact_edges = await stage2.extract_facts_for_topics(topic_result.topics, episodes, _Generation(api, work_dir, "fact"))
        if len(fact_results) != len(topic_result.topics) or not fact_edges:
            raise HyperMemError("Fact extraction did not cover all topics")
        stage2.save_fact_results("0", fact_results, fact_edges, work_dir / "facts")
        _stage_checkpoint(checkpoint, {"fact_count": sum(len(result.facts) for result in fact_results)})
    graph_builder = modules["extractors/hypergraph_extractor.py"].HypergraphExtractor()
    graph = graph_builder.build_hypergraph(episodes, fact_results, fact_edges, topic_result, episode_edges)
    errors = graph.validate_bidirectional_links()
    if any(errors.values()):
        raise HyperMemError(f"Hypergraph bidirectional links are invalid: {errors}")
    graph_dict = graph.to_dict()
    _write_json(work_dir / "hypergraphs/hypergraph_conv_0.json", graph_dict)
    for session in sorted({message["session"] for message in messages}):
        containing = [ep.to_dict() for ep in episodes if any(m.get("session") == session for m in ep.original_data)]
        _write_json(work_dir / "sessions" / session / "episodes.json", containing)
    _stage_checkpoint(work_dir / "graph_stage.json", {"graph_complete": True})
    return graph_dict


def build(conversation: dict, work_dir: Path, api) -> dict:
    list(_sessions(conversation))  # reject QA/summary-bearing data before I/O/API
    modules = _load()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    identity = {"schema": SCHEMA, "input_sha256": _sha(conversation), "upstream_revision": UPSTREAM_REVISION, "config": CONFIG}
    with _lock(work_dir):
        identity_path = work_dir / "identity.json"
        if identity_path.exists() and _read_json(identity_path) != identity:
            raise HyperMemError("HyperMem work directory belongs to different input/config")
        _write_json(identity_path, identity)
        state_path = work_dir / "state.json"
        if state_path.exists():
            state = _read_json(state_path)
            _verify_state(state)
            return state
        token = _UUID_CONTEXT.set({"seed": identity["input_sha256"], "counter": 0})
        try:
            graph = asyncio.run(_build_async(conversation, work_dir, api, modules))
            bm25, vectors, cache_hashes = _build_indexes(work_dir, api, modules)
            state = {**identity, "work_dir": str(work_dir), "build_complete": True,
                     "speaker_a": conversation["speaker_a"], "speaker_b": conversation["speaker_b"],
                     "graph_sha256": _file_sha(work_dir / "hypergraphs/hypergraph_conv_0.json"),
                     "graph_counts": {key: len(graph[key]) for key in ("episodes", "topics", "facts", "episode_hyperedges", "fact_hyperedges")},
                     "cache_sha256": cache_hashes, "source_sha256_lf": SOURCE_SHA256}
            _write_evidence_summaries(work_dir)
            state["export_paths"] = ["state.json", *_EXPORT_ARTIFACTS]
            state["artifact_sha256"] = {name: _file_sha(work_dir / name) for name in _EXPORT_ARTIFACTS}
            state["document_embedding_sha256"] = {
                str(path.relative_to(work_dir)): _file_sha(path)
                for path in sorted((work_dir / "embedding_evidence").glob("document_*.json"))}
            _write_json(state_path, state)
            _seal(state_path)
            _INDEXES[str(work_dir)] = (state["graph_sha256"], graph, bm25, vectors)
            return state
        except BaseException as exc:
            _write_json(work_dir / "build_failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
            if isinstance(exc, BaselineFatal):
                raise
            raise HyperMemError("HyperMem build failed; see build_failure.json") from exc
        finally:
            _UUID_CONTEXT.reset(token)


def _verify_state(state):
    if not state.get("build_complete") or state.get("schema") != SCHEMA or state.get("config") != CONFIG or state.get("upstream_revision") != UPSTREAM_REVISION:
        raise HyperMemError("Incomplete/mismatched HyperMem state")
    work_dir = Path(state["work_dir"])
    _verify_seal(work_dir / "state.json")
    if _read_json(work_dir / "state.json") != state:
        raise HyperMemError("Returned HyperMem state differs from the sealed state file")
    if state.get("export_paths") != ["state.json", *_EXPORT_ARTIFACTS] or set(state.get("artifact_sha256", {})) != set(_EXPORT_ARTIFACTS):
        raise HyperMemError("HyperMem permanent export bindings are incomplete")
    _verify_artifacts(work_dir, state["artifact_sha256"])
    if state["artifact_sha256"]["hypergraphs/hypergraph_conv_0.json"] != state["graph_sha256"]:
        raise HyperMemError("HyperMem graph integrity mismatch")


def retrieve(question: str, state: dict, api) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise HyperMemError("Nonempty question string required")
    _verify_state(state)
    work_dir = Path(state["work_dir"])
    key = _sha({"question": question, "graph_sha256": state["graph_sha256"], "config": CONFIG})
    output = work_dir / "retrieval" / (key + ".json")
    with _lock(output):
        if output.exists():
            _verify_seal(output)
            cached_result = _read_json(output)
            if cached_result.get("question") != question or cached_result.get("graph_sha256") != state["graph_sha256"] or cached_result.get("config") != CONFIG:
                raise HyperMemError("Retrieval artifact identity mismatch")
            return cached_result
        if _seal_path(output).exists():
            raise HyperMemError("Sealed retrieval result is missing")
        modules = _load()
        with _lock(work_dir):
            cached = _INDEXES.get(str(work_dir))
            if cached is None or cached[0] != state["graph_sha256"]:
                graph = _read_json(work_dir / "hypergraphs/hypergraph_conv_0.json")
                bm25, vectors, _ = _build_indexes(work_dir, api, modules, state["cache_sha256"], state["document_embedding_sha256"])
                cached = (state["graph_sha256"], graph, bm25, vectors)
                _INDEXES[str(work_dir)] = cached
        _, graph, bm25, vectors = cached
        stage4 = modules["main/stage4_hypergraph_retrieval.py"]
        embedding_provider = _Embedding(api, work_dir, query=True)
        results = stage4.hierarchical_retrieval(question, graph, bm25=bm25["bm25"], docs=bm25["docs"],
                    emb_index=vectors, embedding_provider=embedding_provider,
                    reranker_provider=None, config=SimpleNamespace(**copy.deepcopy(CONFIG)))
        if not results.get("episodes") or not results.get("topics"):
            raise HyperMemError("Native retrieval returned no topic/episode from nonempty hypergraph")
        context = stage4.format_hierarchical_results(results, state["speaker_a"], state["speaker_b"], "011")
        evidence = []
        for layer in ("topics", "episodes", "facts"):
            for rank, (node, score) in enumerate(results.get(layer, [])):
                evidence.append({"layer": layer, "rank": rank, "score": float(score), "included_in_context": layer != "topics", "node": node})
        result = {"schema": SCHEMA, "question": question, "context": context, "evidence": evidence,
                  "retrieval_log": results.get("retrieval_log", {}), "graph_sha256": state["graph_sha256"],
                  "config": CONFIG, "upstream_revision": UPSTREAM_REVISION,
                  "embedding_evidence_summary": embedding_provider.evidence}
        result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
        _write_json(output, result)
        _seal(output)
        return result


def describe():
    return {"schema": SCHEMA, "method": "HyperMem adapted Gemini embedding/no reranker", "upstream_revision": UPSTREAM_REVISION,
            "source_sha256_lf": SOURCE_SHA256, "config": CONFIG, "encoder": "gemini-embedding-001", "dimensions": 768,
            "generation_temperature": "omitted", "max_output": 16384, "reader": "external common Reader",
            "granularity": "original smart-mask episode boundary algorithm over chronological turns"}


def readiness():
    modules = _load()
    _check_nltk()
    return {"ready": True, "source_verified": True, "loaded_modules": len(modules),
            "nltk_data_ready": True, "local_encoder_required": False}
