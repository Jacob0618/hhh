"""Initialize real Mem0 with empty local stores; forbid model/network work.

The deliberately invalid credential/model names cannot validate API access.
No LoCoMo data is opened, no add/search runs, and no local NLP inference runs.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["MEM0_TELEMETRY"] = "false"
os.environ["MEM0_DIR"] = str(ROOT / "runtime/mem0")
os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
os.environ["FASTEMBED_CACHE_PATH"] = str(ROOT / "cache/fastembed")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
# Discard inherited provider credentials in this check's process only.
for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ.pop(name, None)

blocked_events = []
# Python 3.11 on Windows uses a local `ver` subprocess for OS discovery.
# Cache that read-only inventory before forbidding child processes below.
platform.uname()


def forbid_external_work(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
        blocked_events.append(event)
        raise RuntimeError("Offline startup check forbids network/subprocess work: " + event)


sys.addaudithook(forbid_external_work)


def main():
    import mem0
    from mem0 import Memory
    from mem0.memory import main as memory_module
    from mem0.memory.telemetry import MEM0_TELEMETRY

    assert importlib.metadata.version("mem0ai") == "2.2.1"
    assert MEM0_TELEMETRY is False
    actual_source = Path(memory_module.__file__)
    expected_source = ROOT / "upstream/mem0/mem0/memory/main.py"
    actual_hash = hashlib.sha256(actual_source.read_bytes()).hexdigest()
    assert actual_hash == hashlib.sha256(expected_source.read_bytes()).hexdigest(), "Wrong Mem0 source installed"
    config_path = ROOT / "configs/mem0_offline_startup.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("llm", "embedder"):
        assert config[key]["config"]["api_key"] == "offline-placeholder-not-a-credential"
    store_path = Path(config["vector_store"]["config"]["path"]).resolve()
    assert store_path.is_relative_to(ROOT / "runtime/startup")
    Path(config["history_db_path"]).parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as warning_rows:
        warnings.simplefilter("always")
        memory = Memory.from_config(config)
        try:
            entity_store = memory.entity_store
            shared_client = entity_store.client is memory.vector_store.client
            assert shared_client
            collections = memory.vector_store.client.get_collections().collections
            counts = {c.name: memory.vector_store.client.count(collection_name=c.name, exact=True).count
                      for c in collections}
            assert all(count == 0 for count in counts.values()), "Startup store must remain empty"
            tables = [row[0] for row in memory.db.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            assert "history" in tables and "messages" in tables
            for table in ("history", "messages"):
                assert memory.db.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            assert memory.vector_store._has_bm25_slot
            assert memory.vector_store._bm25_encoder is None, "No encoder should have run"
            provider_names = {"llm": type(memory.llm).__name__, "embedder": type(memory.embedding_model).__name__}
        finally:
            memory.llm.client.close()
            memory.embedding_model.client.close()
            memory.db.connection.close()
            memory.vector_store.client.close()
    assert not blocked_events, blocked_events
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    required = ["mem0ai", "qdrant-client", "google-genai", "fastembed", "spacy", "en-core-web-sm"]
    normalized = {k.lower().replace("_", "-"): v for k, v in packages.items()}
    assert all(name in normalized for name in required)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_offline_sdk_initialization_only",
        "python": sys.executable,
        "mem0_version": importlib.metadata.version("mem0ai"),
        "mem0_import": mem0.__file__,
        "installed_main_sha256": actual_hash,
        "installed_source_matches_checkout": True,
        "providers": provider_names,
        "collections_and_point_counts": counts,
        "sqlite_tables": tables,
        "entity_and_memory_share_local_client": shared_client,
        "bm25_collection_slot_present": True,
        "bm25_encoder_initialized": False,
        "spacy_inference_run": False,
        "model_requests": 0,
        "network_attempts": blocked_events,
        "locomo_opened": False,
        "telemetry_enabled": False,
        "placeholder_credentials_only": True,
        "api_authentication_or_model_availability_verified": False,
        "benchmark_ready": False,
        "remaining_before_live_run": ["select and verify actual model IDs/API access",
            "implement session adapter/date and provenance handling",
            "verify BM25 assets and real hybrid retrieval at separate authorized smoke stage",
            "freeze JEV criteria and evaluation protocol"],
        "warnings": sorted(set(str(w.message) for w in warning_rows)),
        "packages": dict(sorted(packages.items(), key=lambda x: x[0].lower())),
    }
    report_path = ROOT / "research/local_startup_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda x: x[0].lower())]
    (ROOT / "configs/installed_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "mem0_version", "providers", "collections_and_point_counts",
                       "network_attempts", "model_requests", "api_authentication_or_model_availability_verified"]}, indent=2))


if __name__ == "__main__":
    main()
