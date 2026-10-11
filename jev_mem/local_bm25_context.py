"""Process-local loading of an admitted BM25 snapshot, with unchanged scoring.

The context changes only Qdrant/bm25 resource resolution. It neither invents
missing files nor changes BM25 defaults. Other sparse models are delegated.
No download, model construction or environment change occurs on import.
"""
from __future__ import annotations

import contextlib
import hashlib
import threading
from pathlib import Path

PROTOCOL = "qdrant-bm25-existing-local-snapshot-v1"
MODEL = "Qdrant/bm25"
_LOCK = threading.RLock()
_ACTIVE = False


def validate_resources(snapshot, expected_files_sha256):
    snapshot = Path(snapshot).resolve()
    expected = {Path(p).resolve(): h for p, h in expected_files_sha256.items()}
    if not snapshot.is_dir() or not expected or any(p.parent != snapshot for p in expected):
        raise ValueError("BM25 resources must be a pinned flat snapshot")
    actual = {p.resolve() for p in snapshot.iterdir() if p.is_file()}
    if actual != set(expected):
        raise ValueError("Pinned BM25 snapshot inventory changed")
    for path, sha in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Pinned BM25 resource changed: " + path.name)
    english = snapshot / "english.txt"
    if english not in expected or not english.read_text(encoding="utf-8").splitlines():
        raise ValueError("Pinned English stopword resource is missing or empty")
    return snapshot


@contextlib.contextmanager
def local_bm25_context(snapshot, expected_files_sha256):
    """Use existing resources for every BM25 store created in this process.

    Enter around one worker's entire writer/search operation. Do not share a
    process with independent concurrent experiments while the context is active.
    The SDK imports SparseTextEmbedding lazily, so no SDK source patch is needed.
    """
    global _ACTIVE
    path = validate_resources(snapshot, expected_files_sha256)
    import fastembed
    with _LOCK:
        if _ACTIVE:
            raise RuntimeError("A BM25 resource context is already active")
        original = fastembed.SparseTextEmbedding

        def constructor(*args, **kwargs):
            model = kwargs.get("model_name", args[0] if args else None)
            if model != MODEL:
                return original(*args, **kwargs)
            if "specific_model_path" in kwargs and Path(kwargs["specific_model_path"]).resolve() != path:
                raise ValueError("BM25 caller requested another resource snapshot")
            if kwargs.get("local_files_only", True) is not True:
                raise ValueError("BM25 caller requested network resource loading")
            options = {**kwargs, "specific_model_path": str(path), "local_files_only": True}
            return original(*args, **options)

        _ACTIVE = True
        fastembed.SparseTextEmbedding = constructor
        try:
            yield {"protocol": PROTOCOL, "snapshot": str(path),
                   "resource_loading_only": True, "algorithm_defaults_changed": False}
        finally:
            fastembed.SparseTextEmbedding = original
            _ACTIVE = False


@contextlib.contextmanager
def forbid_network():
    """For explicit offline admission tests only; never wrap a live API writer."""
    from unittest.mock import patch
    targets = ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
               "requests.Session.request", "httpx.Client.send", "httpx.AsyncClient.send")
    with contextlib.ExitStack() as stack:
        guards = [stack.enter_context(patch(name, side_effect=AssertionError("Offline BM25 validation forbids network"))) for name in targets]
        yield guards


def compare_persisted_fact_vectors(native_dirs, snapshot, expected_files_sha256):
    """Read SQLite consistently and reproduce stored fact sparse vectors offline.

    This validates algorithm/resource equivalence only. It is not a source/date
    audit and does not approve downstream QA or an incomplete bank.
    """
    import pickle
    import sqlite3
    points = []
    inventory = []
    for native in native_dirs:
        path = Path(native) / "qdrant/collection/facts/storage.sqlite"
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Fact SQLite snapshot is inconsistent")
            stored = [pickle.loads(row[0]) for row in connection.execute("SELECT point FROM points").fetchall()]
            connection.rollback()
        finally:
            connection.close()
        inventory.append({"native": str(Path(native).resolve()), "facts": len(stored)})
        points.extend(stored)
    if not points:
        raise ValueError("No persisted fact vectors were supplied for equivalence")
    with forbid_network() as guards, local_bm25_context(snapshot, expected_files_sha256):
        import fastembed
        encoder = fastembed.SparseTextEmbedding(model_name=MODEL)
        outputs = list(encoder.embed([p.payload.get("text_lemmatized") or p.payload.get("data", "") for p in points]))
        if len(outputs) != len(points):
            raise ValueError("Offline BM25 produced the wrong number of vectors")
        for point, vector in zip(points, outputs):
            before = point.vector.get("bm25") if isinstance(point.vector, dict) else None
            if before is None or dict(zip(before.indices, before.values)) != dict(zip(vector.indices.tolist(), vector.values.tolist())):
                raise ValueError("Offline BM25 differs from a persisted fact vector")
        if any(g.call_count for g in guards):
            raise AssertionError("Offline validation attempted network access")
    validate_resources(snapshot, expected_files_sha256)
    return {"protocol": PROTOCOL, "facts_compared": len(points), "exact_matches": len(points),
            "mismatches": 0, "network_calls": 0, "model_api_calls": 0,
            "source_native_writes": 0, "banks": inventory}
