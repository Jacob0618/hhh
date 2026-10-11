"""Offline-only creation of the v2 manifest and byte-bound v1 cache seed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "runs/pilot400_modules_v1"
DESTINATION = ROOT / "runs/pilot400_modules_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from tools import run_pilot400_experiments_v2 as driver
    from tools.recover_pilot400_writer_v1 import seal_tree, http_hash
    old_lineage_path = ROOT / "runs/pilot400_writer_recovery_v1/recovery_control/lineage.json"
    old_lineage = json.loads(old_lineage_path.read_text(encoding="utf-8"))
    if seal_tree(SOURCE) != old_lineage["source_files"]:
        raise ValueError("Frozen v1 source changed")
    if DESTINATION.exists():
        raise ValueError("v2 directory already exists; never reseed or overwrite it")
    gate = driver.main_batch_gate(driver.CLOSURE_PATH)
    DESTINATION.mkdir()
    manifest = driver.ensure_manifest(DESTINATION, create=True)
    cache_dir = DESTINATION / "api_cache"
    cache_dir.mkdir()
    copied = []
    for source in sorted((SOURCE / "api_cache").glob("*.json")):
        record = json.loads(source.read_text(encoding="utf-8"))
        if http_hash(record["request"]) != source.stem:
            raise ValueError("Source cache request/hash mismatch")
        expected = old_lineage["source_files"][source.relative_to(SOURCE).as_posix()]
        if sha(source) != expected["sha256"]:
            raise ValueError("Source cache bytes changed")
        target = cache_dir / source.name
        shutil.copyfile(source, target)
        if sha(target) != expected["sha256"]:
            raise ValueError("Cache copy changed bytes")
        copied.append({"request_hash": source.stem, "sha256": expected["sha256"], "bytes": expected["bytes"]})
    if seal_tree(SOURCE) != old_lineage["source_files"]:
        raise ValueError("Source mutated during cache preparation")
    diagnostic = ROOT / "runs/pilot400_writer_recovery_v1/api_calls.jsonl"
    record = {"schema": "jev-mem-pilot400-v2-cache-seed-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_run": str(SOURCE), "destination_run": str(DESTINATION),
              "source_ledger_path": str(SOURCE / "api_calls.jsonl"),
              "source_ledger_sha256": sha(SOURCE / "api_calls.jsonl"),
              "source_seal_path": str(old_lineage_path), "source_seal_sha256": sha(old_lineage_path),
              "protocol_sha256": manifest["protocol_sha256"], "main_batch_gate": gate,
              "copied_cache_files": copied, "model_calls_made": 0,
              "unused_diagnostic_ledgers": [{"path": str(diagnostic), "sha256": sha(diagnostic)}] if diagnostic.exists() else [],
              "policy": "Only v1 HTTP caches copied, including the original invalid extraction. No banks, QA, tags, statuses or physical logs copied. Validity is checked by the frozen v2 writer; failed exact retry response is not substituted. Source cost and diagnostic cost remain separate; cache hits add no real usage."}
    (DESTINATION / "cache_seed_lineage.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared_offline", "destination": str(DESTINATION),
                      "cache_files": len(copied), "protocol_sha256": manifest["protocol_sha256"], "model_calls": 0}))


if __name__ == "__main__":
    main()
