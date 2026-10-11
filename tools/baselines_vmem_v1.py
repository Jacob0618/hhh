"""Pinned V-Mem LoCoMo text adapter. Importing this module performs no inference.

The small source loader isolates upstream's flat module names from other baselines.
PRISM reuses only the loader, checked input adapter, and one locked CPU MiniLM.
"""
from __future__ import annotations

import ast
import builtins
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import types

from tools.baselines_common_v1 import BaselineFatal as AdapterFatal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "36916b14dc5241e04acbf5cd0f3c40799bc09550"
UPSTREAM = ROOT / "upstream/baselines_v1/vmem"
MINILM_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MINILM_DIR = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots" / MINILM_REVISION
_SOURCE_LOCK = threading.RLock()
_STATE_LOCK = threading.RLock()
_RUNTIMES: dict[str, object] = {}
_STATE_FILES = ("rounds.json", "vectors.npy", "vec_meta.json", "keyword_index.json",
                "vmem_memory.json", "vmem_memory.txt")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_inventory(directory: Path, names) -> dict:
    """Hash explicit native/result artifacts without trusting paths outside the bank."""
    directory = Path(directory).resolve()
    result = {}
    for name in sorted(names):
        relative = Path(name)
        path = (directory / relative).resolve()
        if relative.is_absolute() or directory not in path.parents or not path.is_file():
            raise RuntimeError("Memory artifact missing or outside its bank: " + str(name))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[relative.as_posix()] = digest.hexdigest()
    return result


def clean_conversation(conversation: dict) -> dict:
    """Accept only raw speaker/session/date/text/caption fields, in session order."""
    if not isinstance(conversation, dict) or "conversation" in conversation or "qa" in conversation:
        raise ValueError("build expects only the raw conversation dictionary, never the sample or QA")
    out = {key: str(conversation.get(key) or "") for key in ("speaker_a", "speaker_b")}
    keys = sorted((key for key in conversation if re.fullmatch(r"session_[0-9]+", key)), key=lambda key: int(key.split("_")[1]))
    seen_ids = set()
    for key in keys:
        sid = int(key.split("_")[1])
        turns = conversation[key]
        if not isinstance(turns, list):
            raise ValueError(f"{key} must contain a list of turns")
        out[key] = []
        out[key + "_date_time"] = str(conversation.get(key + "_date_time") or "")
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"Invalid turn in {key}")
            row = {field: str(turn.get(field) or "") for field in ("speaker", "text", "blip_caption")}
            row["dia_id"] = str(turn.get("dia_id") or f"D{sid}:{index}")
            if row["dia_id"] in seen_ids:
                raise ValueError(f"Duplicate dialogue id {row['dia_id']}")
            seen_ids.add(row["dia_id"])
            out[key].append(row)
    if not keys or not any(out[key] for key in keys):
        raise ValueError("Conversation contains no turns")
    return out


class IsolatedSource:
    """Load original Python source under a private namespace without sys.path edits.

    Absolute imports within the upstream repository are redirected to that namespace.
    Optionally remove the one module-level load_dotenv call; no credentials are read.
    Algorithms, constants, prompts, and relative imports are otherwise unchanged.
    """
    def __init__(self, path: Path, namespace: str, local_roots: set[str], disable_dotenv=False):
        self.path, self.namespace = path, namespace
        self.local_roots, self.disable_dotenv = local_roots, disable_dotenv

    def load(self, name: str):
        with _SOURCE_LOCK:
            return self._load(self.namespace + ("." + name if name else ""))

    def _exists(self, full: str) -> bool:
        relative = full.removeprefix(self.namespace).lstrip(".")
        path = self.path.joinpath(*relative.split(".")) if relative else self.path
        return path.is_dir() or path.with_suffix(".py").is_file()

    def _load(self, full: str):
        if full in sys.modules:
            return sys.modules[full]
        if full != self.namespace:
            parent, _, child = full.rpartition(".")
            self._load(parent)
            if full in sys.modules:
                return sys.modules[full]
        relative = full.removeprefix(self.namespace).lstrip(".")
        path = self.path.joinpath(*relative.split(".")) if relative else self.path
        package = path.is_dir()
        file = path / "__init__.py" if package else path.with_suffix(".py")
        if not package and not file.is_file():
            raise ImportError(f"Pinned source module missing: {full}")
        module = types.ModuleType(full)
        module.__file__ = str(file)
        module.__package__ = full if package else full.rpartition(".")[0]
        if package:
            module.__path__ = [str(path)]
        module.__spec__ = importlib.util.spec_from_file_location(full, file, submodule_search_locations=[str(path)] if package else None)
        namespace_builtins = vars(builtins).copy()
        namespace_builtins["__import__"] = self._import
        module.__dict__["__builtins__"] = namespace_builtins
        sys.modules[full] = module
        if full != self.namespace:
            setattr(sys.modules[parent], child, module)
        try:
            if file.is_file():
                tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
                if self.disable_dotenv:
                    tree.body = [node for node in tree.body if not (
                        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id == "load_dotenv"
                    )]
                exec(compile(tree, str(file), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(full, None)
            if full != self.namespace and getattr(sys.modules.get(parent), child, None) is module:
                delattr(sys.modules[parent], child)
            raise
        return module

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            full = importlib.util.resolve_name("." * level + name, globals["__package__"])
        elif name.split(".")[0] in self.local_roots:
            full = self.namespace + "." + name
        else:
            return builtins.__import__(name, globals, locals, fromlist, level)
        if not full.startswith(self.namespace + "."):
            return builtins.__import__(name, globals, locals, fromlist, level)
        with _SOURCE_LOCK:
            module = self._load(full)
            if fromlist:
                for item in fromlist:
                    if item != "*" and not hasattr(module, item) and self._exists(full + "." + item):
                        self._load(full + "." + item)
                return module
            first = full[len(self.namespace) + 1:].split(".")[0]
            return self._load(self.namespace + "." + first)


class SharedMiniLM:
    """One lazy local-only CPU model and a bounded exact-text embedding cache."""
    dimension = 384

    def __init__(self):
        self._lock = threading.RLock()
        self._model = None
        self._cache = OrderedDict()

    def encode(self, texts, batch_size=64):
        import numpy as np
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self._lock:
            missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
            if missing:
                if self._model is None:
                    if not (MINILM_DIR / "modules.json").is_file():
                        raise RuntimeError(f"Complete pinned local MiniLM snapshot missing: {MINILM_DIR}")
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(str(MINILM_DIR), device="cpu", local_files_only=True)
                vectors = np.asarray(self._model.encode(missing, batch_size=batch_size,
                    show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
                if vectors.shape != (len(missing), self.dimension) or not np.isfinite(vectors).all():
                    raise RuntimeError("MiniLM returned invalid shape or nonfinite embeddings")
                self._cache.update(zip(missing, vectors))
            result = np.stack([self._cache[text] for text in texts]).astype(np.float32)
            for text in texts:
                self._cache.move_to_end(text)
            while len(self._cache) > 20000:
                self._cache.popitem(last=False)
            return result

    def embed(self, text):
        return self.encode([text])[0]

    def embed_batch(self, texts):
        return self.encode(texts)


_ENCODER = SharedMiniLM()
_SOURCE = IsolatedSource(UPSTREAM, "_jev_vmem_pinned_v1", {"model", "indexes", "memory", "embeddings", "retrieve", "answer", "visual_embeddings"})


def verify_source(path: Path, sha: str) -> None:
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "-C", str(path), "diff", "--name-only", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != sha or changed:
        raise RuntimeError(f"Pinned source mismatch or tracked edits: {path}")


def describe() -> dict:
    return {"method": "V-Mem", "schema": "vmem-native-text-adapter-v1", "source_sha": SOURCE_SHA,
        "branch": "LoCoMo text", "embedding": "all-MiniLM-L6-v2", "embedding_revision": MINILM_REVISION,
        "embedding_dimension": 384, "embedding_device": "cpu", "unit": "turn", "fold_caption": True,
        "dense_k": 20, "sparse_k": 20, "sparse_method": "unigram_bigram_term_count",
        "fusion": "RRF", "top_k": 12, "neighbor_window": 1, "aggregation_top_k": 30,
        "aggregation_window": 0, "merge_runs": True, "hyde": False, "infer_retry": False,
        "artifact_integrity": "SHA256 of all native text indexes and permanent exports before reuse/retrieval",
        "reader": "external shared reader", "judge": "external shared judge"}


def readiness() -> dict:
    modules = ("numpy", "sentence_transformers", "torch")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    local_model = (MINILM_DIR / "model.safetensors").is_file() and (MINILM_DIR / "modules.json").is_file()
    return {"ready": not missing and local_model and (UPSTREAM / "retrieve.py").is_file(),
        "missing_python_modules": missing, "local_model_present": local_model,
        "local_model_path": str(MINILM_DIR), "inference_verified": False}


def _modules():
    model = _SOURCE.load("model")
    indexes = _SOURCE.load("indexes")
    memory = _SOURCE.load("memory")
    retrieval = _SOURCE.load("retrieve")
    retrieval.embeddings.encode = _ENCODER.encode
    answer = _SOURCE.load("answer")
    return model, indexes, memory, retrieval, answer


def _verify_state(state: dict, expected_work_dir: Path | None = None) -> None:
    work_dir = Path(state["work_dir"]).resolve()
    if expected_work_dir is not None and work_dir != expected_work_dir.resolve():
        raise RuntimeError("V-Mem completed state points to another bank")
    if state.get("method") != "vmem" or state["manifest"]["protocol"] != describe():
        raise RuntimeError("V-Mem retrieval protocol mismatch")
    inventory = state.get("files_sha256")
    if not isinstance(inventory, dict) or set(inventory) != set(_STATE_FILES):
        raise RuntimeError("V-Mem completed state lacks its exact artifact inventory")
    if file_inventory(work_dir, _STATE_FILES) != inventory:
        raise RuntimeError("V-Mem memory/index/export artifact changed")
    if any((work_dir / name).exists() for name in ("visual.npy", "visual_meta.json")):
        raise RuntimeError("V-Mem text bank contains an undeclared visual index")
    exports = [str(work_dir / name) for name in ("vmem_memory.json", "vmem_memory.txt")]
    if state.get("export_paths") != exports:
        raise RuntimeError("V-Mem export paths differ from the committed bank")


def build(conversation: dict, work_dir: Path, api) -> dict:
    clean = clean_conversation(conversation)
    work_dir = Path(work_dir).resolve()
    manifest = {"protocol": describe(), "conversation_sha256": fingerprint(clean)}
    state_file = work_dir / "vmem_state.json"
    verify_source(UPSTREAM, SOURCE_SHA)
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["manifest"] != manifest:
            raise RuntimeError("V-Mem checkpoint input/protocol mismatch")
        _verify_state(state, work_dir)
        return state
    model, indexes, memory_module, _, answer = _modules()
    rounds = []
    # Exact native load_locomo_rounds mapping, with session iteration made numeric.
    for key, turns in clean.items():
        if not re.fullmatch(r"session_[0-9]+", key):
            continue
        sid = int(key.split("_")[1])
        for index, turn in enumerate(turns):
            dia, speaker = turn["dia_id"], turn["speaker"]
            text, cap = turn["text"].strip(), turn["blip_caption"].strip()
            turn_text = (text + f"\n[Image: {cap}]").strip() if cap and text else text
            units = []
            if turn_text:
                units.append(model.Unit(unit_id=f"{dia}::turn", round_id=dia, kind="user", speaker=speaker, text=turn_text))
            if cap:
                units.append(model.Unit(unit_id=f"{dia}::img", round_id=dia, kind="image", speaker=speaker, text=cap, image_id=dia))
            if units:
                rounds.append(model.Round(round_id=dia, session_id=sid, index=index,
                    date=clean[key + "_date_time"], units=units))
    if not rounds:
        raise RuntimeError("No nonempty V-Mem units after native loading")
    vec_ids, texts, metas = [], [], []
    keyword_index = indexes.KeywordIndex()
    for item in rounds:
        for unit in item.units:
            vec_ids.append(unit.unit_id)
            texts.append(unit.index_text)
            metas.append({"round_id": unit.round_id, "kind": unit.kind, "image_id": unit.image_id})
            keyword_index.add(unit.unit_id, unit.index_text)
    vector_index = indexes.VectorIndex()
    vector_index.build(vec_ids, _ENCODER.encode(texts), metas)
    memory = memory_module.Memory({item.round_id: item for item in rounds}, vector_index, keyword_index)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(work_dir / "rounds.json", {item.round_id: item.to_dict() for item in rounds})
    vector_index.save(work_dir)
    keyword_index.save(work_dir)
    export_json, export_text = work_dir / "vmem_memory.json", work_dir / "vmem_memory.txt"
    write_json(export_json, {"manifest": manifest, "rounds": [item.to_dict() for item in rounds]})
    export_text.write_text(answer.format_context(rounds), encoding="utf-8")
    state = {"method": "vmem", "work_dir": str(work_dir), "manifest": manifest,
        "round_count": len(rounds), "unit_count": len(vec_ids),
        "export_paths": [str(export_json), str(export_text)],
        "files_sha256": file_inventory(work_dir, _STATE_FILES)}
    _verify_state(state, work_dir)
    write_json(state_file, state)
    with _STATE_LOCK:
        _RUNTIMES[str(work_dir)] = memory
    return state


def retrieve(question: str, state: dict, api) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")
    _verify_state(state)
    _, _, memory_module, retrieval, answer = _modules()
    work_dir = str(Path(state["work_dir"]).resolve())
    with _STATE_LOCK:
        if work_dir not in _RUNTIMES:
            _RUNTIMES[work_dir] = memory_module.Memory.load(work_dir)
        memory = _RUNTIMES[work_dir]
    trace = {}
    rounds = retrieval.retrieve(memory, question, top_k=12, dense_k=20, sparse_k=20,
        window=1, agg_top_k=30, agg_window=0, hypotheticals=None, trace=trace)
    if not rounds:
        raise RuntimeError("V-Mem retrieval unexpectedly returned no rounds from a nonempty index")
    evidence = [{"id": item.round_id, "dia_ids": [item.round_id], "session_id": item.session_id,
        "observation_date": item.date, "text": answer.format_context([item]),
        "units": [unit.__dict__.copy() for unit in item.units]} for item in rounds]
    return {"context": answer.format_context(rounds), "evidence": evidence, "trace": trace,
        "source_sha": SOURCE_SHA}
