"""Pinned PRISM R5-1 adapter using native graph/search and shared Gemini transport.

No model loads, dotenv reads, API calls, or worker launches happen on import.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import threading

from tools import baselines_vmem_v1 as shared
from tools.baselines_common_v1 import BaselineFatal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "fdc09e6bfe31f9c2852eb830ee07de1d750472f4"
UPSTREAM = ROOT / "upstream/baselines_v1/prism"
_SOURCE = shared.IsolatedSource(UPSTREAM, "_jev_prism_pinned_v1", {"mmem", "scripts"}, disable_dotenv=True)
_RUNTIMES = {}
_RUNTIME_LOCK = threading.RLock()
_INTENTS = ("temporal", "causal", "multi_hop", "entity_centric")
_STATE_FILES = ("prism_manifest.json", "checkpoint_pointer.json", "source_chunks.json",
                "prism_memory.json", "prism_memory.txt")


def describe() -> dict:
    return {"method": "PRISM", "schema": "prism-r5-1-adapter-v1", "source_sha": SOURCE_SHA,
        "variant": "R5-1", "generation_model": "gemini-2.5-flash",
        "embedding": "all-MiniLM-L6-v2", "embedding_revision": shared.MINILM_REVISION,
        "embedding_dimension": 384, "embedding_device": "cpu", "chunk_max_chars": 1024,
        "chunking": "native session-bounded turn-boundary chunks", "caption_input": "append [Image: caption] to source turn text",
        "wide_search_top_k": 30, "bundle_top_k": 10, "rerank_top_n": 5,
        "display_mode": "detail", "intent_classifier_mode": "hybrid",
        "enable_relation_paths": True, "enable_query_sensitive_cost": True, "enable_reranking": True,
        "enable_intent_quota": False, "enable_tiered_discount": False,
        "causal_consolidation_interval": 5, "causal_confidence_threshold": 0.7,
        "entity_merge_threshold": 0.90, "semantic_similarity_threshold": 0.85,
        "key_sentences_embedding": False, "schema_errors": "hard_failure",
        "artifact_integrity": "SHA256 of completed checkpoint, source provenance and permanent exports before reuse/retrieval",
        "reader": "external shared reader", "judge": "external shared judge"}


def readiness() -> dict:
    names = ("numpy", "sentence_transformers", "torch", "networkx", "faiss", "pydantic", "openai", "dotenv")
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    return {"ready": not missing and shared.readiness()["local_model_present"] and (UPSTREAM / "mmem/config.py").is_file(),
        "missing_python_modules": missing, "local_model_present": shared.readiness()["local_model_present"],
        "local_model_path": str(shared.MINILM_DIR), "inference_verified": False}


def _config():
    native = _SOURCE.load("mmem.config")
    # Explicit values avoid every credential-reading default factory in LLMConfig.
    return native.MMemConfig(
        llm=native.LLMConfig(api_key="", base_url="adapter://shared-gemini-transport",
            extraction_model="gemini-2.5-flash", causal_model="gemini-2.5-flash",
            qa_model="gemini-2.5-flash", judge_model="gemini-2.5-flash"),
        embedding=native.EmbeddingConfig(model_name=str(shared.MINILM_DIR), dimension=384, device="cpu"),
        retrieval=native.RetrievalConfig(enable_reranking=True, rerank_top_n=5,
            intent_classifier_mode="hybrid", enable_intent_quota=False, enable_tiered_discount=False))


def _stage(prompt: str) -> str:
    if prompt.startswith("You are an information extraction engine."):
        return "prism_extract"
    if prompt.startswith("You are a lightweight information extraction engine."):
        return "prism_extract_fallback"
    if prompt.startswith("You are a causal reasoning engine."):
        return "prism_causal"
    if prompt.startswith("Rate the relevance of each memory snippet"):
        return "prism_rerank"
    if prompt.startswith("Analyze the intent of the following query."):
        return "prism_intent"
    raise BaselineFatal("PRISM attempted an undeclared generation prompt")


def _validate_json(stage: str, raw, prompt: str):
    """Prevent valid-but-wrong JSON from activating upstream silent fallbacks."""
    try:
        if stage.startswith("prism_extract"):
            if not isinstance(raw, dict) or not isinstance(raw.get("episode_summary"), str) or not raw["episode_summary"].strip():
                raise ValueError("Nonempty episode_summary required")
            required = ("episode_summary", "entities") if stage == "prism_extract_fallback" else (
                "episode_summary", "entities", "facet_points", "facets", "temporal_info")
            if any(key not in raw for key in required):
                raise ValueError("Extraction response omitted required graph fields")
            extractor = _SOURCE.load("mmem.ingestion.extractor")
            extractor.ExtractionResult.model_validate(raw)
        elif stage == "prism_causal":
            if not isinstance(raw, list) and not (isinstance(raw, dict) and isinstance(raw.get("causal_pairs"), list)):
                raise ValueError("causal_pairs list required")
            payload = {"causal_pairs": raw} if isinstance(raw, list) else raw
            _SOURCE.load("mmem.ingestion.extractor").CausalResponse.model_validate(payload)
        elif stage == "prism_rerank":
            expected = {int(value) for value in re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE)}
            if not isinstance(raw, list) or len(raw) != len(expected):
                raise ValueError("One score per candidate is required")
            indices = []
            for item in raw:
                if not isinstance(item, dict) or type(item.get("index")) is not int:
                    raise ValueError("Invalid rerank index")
                score = item.get("score")
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 10:
                    raise ValueError("Invalid rerank score")
                indices.append(item["index"])
            if set(indices) != expected or len(set(indices)) != len(indices):
                raise ValueError("Missing, duplicate, or foreign rerank index")
        elif stage == "prism_intent":
            if not isinstance(raw, dict) or any(name not in raw for name in _INTENTS):
                raise ValueError("All four intent scores are required")
            for name in _INTENTS:
                score = raw[name]
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid intent score")
        return raw
    except Exception as exc:
        raise BaselineFatal(f"PRISM invalid {stage} response ({type(exc).__name__})") from None


class NativeLLMBridge:
    """Keep upstream prompts unchanged; transport owns usage, retries, and parsing."""
    def __init__(self, api):
        self.api = api
        self.calls = []

    def chat_json(self, prompt, model=None, system="", temperature=None):
        stage = _stage(prompt)
        try:
            raw = self.api.generate(prompt, system=system or None, json_output=True,
                max_output=16384, stage=stage)
            if isinstance(raw, str):
                raw = json.loads(raw)
            raw = _validate_json(stage, raw, prompt)
        except Exception as exc:
            raise BaselineFatal(f"PRISM {stage} transport failed ({type(exc).__name__})") from None
        self.calls.append({"stage": stage, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        return raw


def _chunks(conversation: dict):
    clean = shared.clean_conversation(conversation)
    # Native PRISM text loader drops captions; expose the same original captions
    # as every comparator before calling the unchanged native 1024-char chunker.
    for key, turns in clean.items():
        if re.fullmatch(r"session_[0-9]+", key):
            for turn in turns:
                caption = turn["blip_caption"].strip()
                if caption:
                    turn["text"] = (turn["text"] + f"\n[Image: {caption}]").strip()
    loader = _SOURCE.load("scripts.eval.locomo_loader")
    native = loader._parse_conversation("conversation_" + shared.fingerprint(clean)[:12], clean)
    chunks = _SOURCE.load("scripts.eval.locomo_chunker").chunk_conversation(native, max_chunk_chars=1024)
    if not chunks:
        raise ValueError("PRISM native chunker returned no nonempty source chunks")
    return chunks


def _pipeline_class():
    native = _SOURCE.load("mmem.ingestion.pipeline")
    # Preserve the complete consolidation algorithm; remove its outer error-to-[]
    # wrapper so a local index/schema failure cannot be counted as success.
    native.run_consolidation = _SOURCE.load("mmem.ingestion.consolidation")._run
    return native.IngestionPipeline


def _checkpoint_save(pipeline, work_dir: Path, processed: int) -> None:
    checkpoint = work_dir / f"checkpoint_{processed % 2}"
    pipeline.save_checkpoint(checkpoint)
    files = {str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(checkpoint.rglob("*")) if path.is_file()}
    shared.write_json(work_dir / "checkpoint_pointer.json", {"directory": checkpoint.name,
        "processed_chunks": processed, "files_sha256": files})


def _checkpoint_load(work_dir: Path, config, api):
    pointer = json.loads((work_dir / "checkpoint_pointer.json").read_text(encoding="utf-8"))
    if pointer["directory"] not in ("checkpoint_0", "checkpoint_1"):
        raise RuntimeError("Invalid PRISM checkpoint directory")
    checkpoint = work_dir / pointer["directory"]
    for relative, expected in pointer["files_sha256"].items():
        file = (checkpoint / relative).resolve()
        if checkpoint.resolve() not in file.parents or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError("PRISM checkpoint file changed or incomplete")
    pipeline = _pipeline_class().load_checkpoint(checkpoint, config=config,
        embedder=shared._ENCODER, llm_client=NativeLLMBridge(api))
    return pipeline, int(pointer["processed_chunks"])


def _make_runtime(pipeline, config):
    # Lazy classifier keeps ingestion free of query-time work; its prototype
    # vectors reuse the locked shared encoder and its exact-text cache.
    return {"pipeline": pipeline, "config": config, "classifier": None, "lock": threading.RLock()}


def _completed_files(work_dir: Path):
    pointer = json.loads((work_dir / "checkpoint_pointer.json").read_text(encoding="utf-8"))
    if pointer["directory"] not in ("checkpoint_0", "checkpoint_1") or not pointer.get("files_sha256"):
        raise RuntimeError("Invalid PRISM completed checkpoint inventory")
    checkpoint_files = [(Path(pointer["directory"]) / name).as_posix()
                        for name in pointer["files_sha256"]]
    return pointer, [*_STATE_FILES, *checkpoint_files]


def _verify_state(state: dict, expected_work_dir: Path | None = None) -> dict:
    work_dir = Path(state["work_dir"]).resolve()
    if expected_work_dir is not None and work_dir != expected_work_dir.resolve():
        raise RuntimeError("PRISM completed state points to another bank")
    if state.get("method") != "prism" or state["manifest"]["protocol"] != describe():
        raise RuntimeError("PRISM retrieval protocol mismatch")
    inventory = state.get("files_sha256")
    if not isinstance(inventory, dict) or not set(_STATE_FILES).issubset(inventory):
        raise RuntimeError("PRISM completed state lacks its artifact inventory")
    if shared.file_inventory(work_dir, inventory) != inventory:
        raise RuntimeError("PRISM checkpoint/source/export artifact changed")
    pointer, required = _completed_files(work_dir)
    if set(inventory) != set(required) or pointer["processed_chunks"] != state["chunk_count"]:
        raise RuntimeError("PRISM completed artifact inventory/count mismatch")
    for relative, digest in pointer["files_sha256"].items():
        if inventory[(Path(pointer["directory"]) / relative).as_posix()] != digest:
            raise RuntimeError("PRISM completed state/checkpoint binding differs")
    if json.loads((work_dir / "prism_manifest.json").read_text(encoding="utf-8")) != state["manifest"]:
        raise RuntimeError("PRISM source manifest differs from the completed state")
    exports = [str(work_dir / name) for name in ("prism_memory.json", "prism_memory.txt")]
    if state.get("export_paths") != exports:
        raise RuntimeError("PRISM export paths differ from the committed bank")
    return json.loads((work_dir / "source_chunks.json").read_text(encoding="utf-8"))


def build(conversation: dict, work_dir: Path, api) -> dict:
    clean = shared.clean_conversation(conversation)
    work_dir = Path(work_dir).resolve()
    shared.verify_source(UPSTREAM, SOURCE_SHA)
    manifest = {"protocol": describe(), "conversation_sha256": shared.fingerprint(clean)}
    manifest_path, state_path = work_dir / "prism_manifest.json", work_dir / "prism_state.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("PRISM checkpoint input/protocol mismatch")
    else:
        if (work_dir / "checkpoint_pointer.json").exists():
            raise RuntimeError("Cannot adopt PRISM checkpoint without its input manifest")
        shared.write_json(manifest_path, manifest)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["manifest"] != manifest:
            raise RuntimeError("PRISM completed state manifest mismatch")
        _verify_state(state, work_dir)
        return state
    chunks = _chunks(clean)
    provenance = {}
    for chunk in chunks:
        chunk_id = hashlib.sha256(chunk.text.encode()).hexdigest()[:12]
        provenance.setdefault(chunk_id, {"text": chunk.text, "occurrences": []})["occurrences"].append(
            {"session_num": chunk.session_num, "dia_ids": chunk.dia_ids})
    shared.write_json(work_dir / "source_chunks.json", provenance)
    config = _config()
    if (work_dir / "checkpoint_pointer.json").exists():
        pipeline, processed = _checkpoint_load(work_dir, config, api)
    else:
        pipeline = _pipeline_class()(config=config, embedder=shared._ENCODER, llm_client=NativeLLMBridge(api))
        processed = 0
    if not 0 <= processed <= len(chunks):
        raise RuntimeError("PRISM checkpoint processed-chunk count is invalid")
    for index, chunk in enumerate(chunks[processed:], processed):
        scoped = api.scoped(chunk_index=index, session_id=chunk.session_num) if hasattr(api, "scoped") else api
        pipeline._extractor._llm = NativeLLMBridge(scoped)
        pipeline.ingest(chunk.text)
        _checkpoint_save(pipeline, work_dir, index + 1)
    export_json, export_text = work_dir / "prism_memory.json", work_dir / "prism_memory.txt"
    graph = pipeline.graph.to_dict()
    shared.write_json(export_json, {"manifest": manifest, "graph": graph, "source_chunks": provenance,
        "stats": pipeline.stats()})
    lines = []
    for node in graph["nodes"]:
        content = node.get("summary") or node.get("content") or node.get("theme") or node.get("name") or ""
        lines.append(f"[{node['__node_type__']} {node['id']}] {content}")
    for edge in graph["edges"]:
        lines.append(json.dumps(edge, ensure_ascii=False))
    export_text.write_text("\n".join(lines), encoding="utf-8")
    state = {"method": "prism", "work_dir": str(work_dir), "manifest": manifest,
        "chunk_count": len(chunks), "stats": pipeline.stats(), "export_paths": [str(export_json), str(export_text)],
        "files_sha256": shared.file_inventory(work_dir, _completed_files(work_dir)[1])}
    _verify_state(state, work_dir)
    shared.write_json(state_path, state)
    with _RUNTIME_LOCK:
        _RUNTIMES[str(work_dir)] = _make_runtime(pipeline, config)
    return state


def retrieve(question: str, state: dict, api) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")
    provenance = _verify_state(state)
    work_dir = Path(state["work_dir"]).resolve()
    key = str(work_dir)
    with _RUNTIME_LOCK:
        if key not in _RUNTIMES:
            config = _config()
            pipeline, processed = _checkpoint_load(work_dir, config, api)
            if processed != state["chunk_count"]:
                raise RuntimeError("PRISM retrieval requires the completed build")
            _RUNTIMES[key] = _make_runtime(pipeline, config)
        runtime = _RUNTIMES[key]
    # The graph/vector store are read-only here, but native hybrid N4 owns a
    # mutable LRU and a client reference. Serialize queries only within this conv.
    with runtime["lock"]:
        pipeline, config = runtime["pipeline"], runtime["config"]
        client = NativeLLMBridge(api)
        if runtime["classifier"] is None:
            classifier_type = _SOURCE.load("mmem.retrieval.intent_classifier").IntentClassifier
            runtime["classifier"] = classifier_type(embedder=shared._ENCODER, llm_client=client,
                high_conf_threshold=config.retrieval.intent_high_conf_threshold,
                margin_threshold=config.retrieval.intent_margin_threshold,
                low_conf_threshold=config.retrieval.intent_low_conf_threshold,
                secondary_margin=config.retrieval.intent_secondary_margin)
        classifier = runtime["classifier"]
        classifier.llm_client = client
        preprocessed = _SOURCE.load("mmem.retrieval.query_preprocessor").preprocess_query(
            question, config.retrieval, intent_classifier=classifier)
        result = _SOURCE.load("mmem.retrieval.search").bundle_search(question,
            graph=pipeline.graph, vector_store=pipeline.vector_store, embedder=shared._ENCODER,
            config=config, top_k=10, display_mode="detail", chunk_store=pipeline.chunk_store,
            llm_client=client, intent_classifier=classifier)
        if not result.bundles or not result.context_text.strip():
            raise BaselineFatal("PRISM unexpectedly returned no context from a nonempty memory")
        assembler = _SOURCE.load("mmem.retrieval.output_assembler")
        evidence = []
        for bundle in result.bundles:
            episode = pipeline.graph.get_node(bundle.episode_id)
            chunk_ids = list(episode.source_chunk_ids)
            dia_ids = list(dict.fromkeys(dia for cid in chunk_ids for occurrence in provenance[cid]["occurrences"]
                for dia in occurrence["dia_ids"]))
            text = assembler._assemble_detail([bundle], pipeline.graph, config.retrieval, chunk_store=pipeline.chunk_store)
            evidence.append({"id": bundle.episode_id, "text": text, "summary": episode.summary,
                "dia_ids": dia_ids, "chunk_ids": chunk_ids, "score": float(bundle.score),
                "best_path": bundle.best_path, "best_path_detail": bundle.best_path_detail,
                "observation_date": episode.timestamp_text or ""})
        return {"context": result.context_text, "evidence": evidence,
            "trace": {**result.debug_info, "internal_calls": client.calls, "intent_mode": "hybrid",
                "classifier_used": preprocessed.classifier_used,
                "intent_hints": sorted(preprocessed.query_type_hints),
                "intent_confidences": preprocessed.intent_confidences,
                "vector_query": preprocessed.vector_query},
            "source_sha": SOURCE_SHA}
