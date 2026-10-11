"""One reviewed writer retry in a separate namespace; never launch a driver.

prepare/check are offline. retry makes at most one physical attempt for the pinned
failed extraction, with provider retries=0. finalize only consumes its persisted
response. The original run is sealed and never edited. A driver.lock guard blocks
the frozen driver until the replacement response passes the original parser.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pilot400_experiments as driver
from jev_mem.group_writer import GroupFactWriter, MODEL, SYSTEM_PROMPT, prepare_group
from jev_mem.providers import ModelAPI

SOURCE = ROOT / "runs/pilot400_modules_v1"
DESTINATION = ROOT / "runs/pilot400_writer_recovery_v1"
SCHEMA = "jev-mem-pilot400-writer-recovery-v1"
TARGET = "conversations/conv_1/banks/regroup"
FAILED_HASH = "74fae466ec1160dfa6597ec03b7ab9b1d1be871b8631b0ba2955bb94c256677a"
GROUP_HASH = "611079a96ca670af887c88b90ec3c476547943ef18aca0633f1fb6bea6793415"
ERROR = "Context citation is not associated with any cited member"
REQUEST_FIELDS = ("user_id", "group_id", "grouping_method", "source_sha256", "members",
    "read_only_context", "member_context_ids", "member_source_turn_ids", "context_source_turn_ids",
    "member_source_indices", "current_date", "protocol", "model", "system_prompt_sha256", "embedding_dimensions")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def http_hash(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seal_tree(directory):
    rows = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Source links are not allowed")
        if path.is_file():
            rows[path.relative_to(directory).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return rows


def decode(response):
    candidates = response.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Replacement generation is missing or incomplete")
    text = "".join(x.get("text", "") for x in candidates[0].get("content", {}).get("parts", [])
                   if not x.get("thought"))
    return json.loads(text)


def signature_from_pending(pending):
    return {"provider": "gemini", "model": MODEL, "operation": "generateContent", "body": {
        "contents": [{"role": "user", "parts": [{"text": pending["prompt"]}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 16384,
                             "thinkingConfig": {"thinkingBudget": 0}, "responseMimeType": "application/json"},
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}}}


def inspect_source(source=SOURCE):
    """Full source/bank/cache audit, with no API construction or file writes."""
    source = Path(source).resolve()
    before = seal_tree(source)
    if (source / "driver.lock").exists():
        raise ValueError("Original pilot still has driver.lock")
    state = read(source / "write_status.json")
    if state.get("status") != "requires_review" or state.get("done") != 2 or set(state.get("conversations", {})) != {"0", "1"}:
        raise ValueError("Expected the terminal two-conversation development write")
    forbidden = ("conversations/conv_*/qa/*/*.json", "conversations/conv_*/tags/*.json",
                 "conversations/conv_*/tag_summary.json", "conversations/conv_*/qa_summary.json",
                 "conversations/conv_*/qa_status.json",
                 "tag_status.json", "qa_status.json", "orchestration_pipeline_v1/status.json")
    if any(any(source.glob(pattern)) for pattern in forbidden):
        raise ValueError("This recovery only covers the pre-QA/pre-tag development write")
    manifest = driver.ensure_manifest(source, create=False)
    data = driver.load_data()
    for ci, name in ((0, "session"), (0, "regroup"), (1, "session")):
        raw = driver.raw_turns_from_locomo(data[ci]["conversation"])
        driver.load_bank(source, ci, name, raw, manifest)
    raw = driver.raw_turns_from_locomo(data[1]["conversation"])
    folder = source / TARGET
    bound = driver._bank_binding(manifest, 1, raw, "regroup")
    failure = driver._verified(folder / "bank_summary.json", driver.ARTIFACT_SCHEMA, bound)
    if failure["status"] != "failed" or failure.get("error", {}).get("message") != ERROR:
        raise ValueError("The pinned failure is not present")
    grouping = driver.require(driver._verified(folder / "grouping.json", driver.ARTIFACT_SCHEMA,
                                               {**bound, "operation": "grouping"}))
    driver.verify_partition(raw, grouping["groups"])
    pending = read(folder / "pending_group.json")
    bank = read(folder / "group_fact_bank.json")
    if (pending.get("group_id") != "sparse_topic_0004" or pending.get("sequence") != 4
            or pending.get("request_sha256") != GROUP_HASH or len(bank["groups"]) != 4
            or len(grouping["groups"]) != 26 or "embeddings" in pending):
        raise ValueError("Pinned partial-commit scope changed")
    if (read(folder / "raw_source.json") != {"user_id": "pilot400_conv_1",
            "source_sha256": driver.fingerprint(raw), "raw_turns": raw}
            or bank.get("source_sha256") != driver.fingerprint(raw)):
        raise ValueError("Partial bank source binding differs")
    flattened = [fact for group in bank["groups"] for fact in group["fact_records"]]
    if flattened != bank["facts"] or len({f["id"] for f in flattened}) != len(flattened):
        raise ValueError("Partial committed facts differ from group records")
    driver._vectors([f["embedding"] for f in flattened], len(flattened))
    positions = {t["source_turn_id"]: i for i, t in enumerate(raw)}
    groups = sorted(grouping["groups"], key=lambda g: min(positions[t["source_turn_id"]] for t in g["turns"]))
    for index, group in enumerate(bank["groups"] + [pending]):
        prepared = prepare_group(members=groups[index]["turns"], raw_turns=raw,
                                 current_date=manifest["protocol"]["parameters"]["current_date"])
        if (group["members"] != prepared["payload"]["MEMBERS"]
                or group["read_only_context"] != prepared["payload"]["READ_ONLY_CONTEXT"]
                or group["member_context_ids"] != prepared["payload"]["member_context_ids"]
                or group["prompt"] != json.dumps(prepared["payload"], ensure_ascii=False, indent=2)
                or group["system_prompt"] != SYSTEM_PROMPT
                or driver.fingerprint({k: group[k] for k in REQUEST_FIELDS}) != group["request_sha256"]):
            raise ValueError("Group request does not match frozen ordered raw sources")
        if index < 4:
            parsed = GroupFactWriter._parse(group["raw_response"], group, group["request_sha256"])
            if parsed != [{k:v for k,v in f.items() if k != "embedding"} for f in group["fact_records"]]:
                raise ValueError("Partial committed extraction no longer validates")
            if read(folder / "groups" / (group["request_sha256"] + ".json")) != group:
                raise ValueError("Partial group audit differs")
    signature = signature_from_pending(pending)
    request_hash = http_hash(signature)
    if request_hash != FAILED_HASH:
        raise ValueError("Pinned HTTP payload changed")
    cache = read(source / "api_cache" / (request_hash + ".json"))
    if cache["request"] != signature or decode(cache["response"]) != pending["raw_response"]:
        raise ValueError("Invalid extraction cache and pending response differ")
    try:
        GroupFactWriter._parse(pending["raw_response"], pending, pending["request_sha256"])
    except Exception as error:
        if str(error) != ERROR:
            raise ValueError("Different parse failure requires a separate review") from error
    else:
        raise ValueError("Original extraction is no longer a failed response")
    logs = [json.loads(line) for line in (source / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    success = {row["request_hash"] for row in logs if isinstance(row.get("status"), int) and 200 <= row["status"] < 300}
    for path in (source / "api_cache").glob("*.json"):
        record = read(path)
        if (http_hash(record["request"]) != path.stem or path.stem not in success
                or set(record["request"]) != {"provider", "model", "operation", "body"}
                or not isinstance(record.get("response"), dict)):
            raise ValueError("Cache does not bind a successful original physical request: " + path.name)
    if seal_tree(source) != before:
        raise ValueError("Original source changed during audit")
    return {"source": str(source), "files": before, "protocol_sha256": manifest["protocol_sha256"],
            "request_hash": request_hash, "group_request_hash": pending["request_sha256"], "signature": signature}


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def namespace_lock(destination):
    path = Path(destination) / "recovery_control/operation.lock"
    # The operating system releases the byte lock on process death. The file may
    # remain; its existence is never interpreted as ownership or an API intent.
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another recovery operation owns this namespace") from None
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def prepare(source=SOURCE, destination=DESTINATION, *, source_auditor=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Recovery namespace must be separate from original")
    audit = (source_auditor or inspect_source)(source)
    destination.mkdir(parents=True, exist_ok=False)
    guard = {"schema": SCHEMA, "purpose": "Block frozen driver until one exact response passes original parser",
             "source": str(source), "created_utc": now()}
    _write_new(destination / "driver.lock", guard)
    excluded, copied = {}, {}
    invalid = {TARGET + "/bank_summary.json", TARGET + "/pending_group.json",
               "api_cache/" + audit["request_hash"] + ".json"}
    for relative, row in audit["files"].items():
        reusable = relative == "run_manifest.json" or relative.startswith(("api_cache/", "conversations/"))
        if relative in invalid or not reusable:
            excluded[relative] = row
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        if sha(target) != row["sha256"]:
            raise ValueError("Copied bytes differ: " + relative)
        copied[relative] = row
    if seal_tree(source) != audit["files"]:
        raise ValueError("Original run changed during copying")
    lineage = {"schema": SCHEMA, "created_utc": now(), "source": str(source), "destination": str(destination),
               "source_files": audit["files"], "copied_files": copied, "excluded_files": excluded,
               "protocol_sha256": audit["protocol_sha256"], "request_hash": audit["request_hash"],
               "group_request_hash": audit["group_request_hash"], "signature": audit["signature"],
               "guard": guard, "policy": "One physical retry; no fact/citation repair; no driver auto-launch",
               "accounting": "Original HTTP ledger stays at source. New ledger contains only new namespace events; copied cache reuse is not billed again."}
    _write_new(destination / "recovery_control/lineage.json", lineage)
    return {"status": "prepared_blocked", "copied_files": len(copied), "excluded_files": len(excluded),
            "request_hash": audit["request_hash"], "model_calls": 0}


def validate_namespace(destination, *, allow_released=False):
    destination = Path(destination).resolve()
    lineage = read(destination / "recovery_control/lineage.json")
    if lineage["schema"] != SCHEMA or lineage["destination"] != str(destination):
        raise ValueError("Namespace lineage identity differs")
    if seal_tree(Path(lineage["source"])) != lineage["source_files"]:
        raise ValueError("Original frozen bytes changed")
    for relative, expected in lineage["copied_files"].items():
        if sha(destination / relative) != expected["sha256"]:
            raise ValueError("Copied artifact changed before recovery: " + relative)
    if http_hash(lineage["signature"]) != lineage["request_hash"]:
        raise ValueError("Recovery signature binding differs")
    guard = destination / "driver.lock"
    if guard.exists():
        if read(guard) != lineage["guard"]:
            raise ValueError("Recovery no longer owns its driver guard")
    elif not (allow_released and (destination / "recovery_control/outcome.json").exists()
              and read(destination / "recovery_control/outcome.json").get("status") == "ready_for_frozen_driver"):
        raise ValueError("Recovery no longer owns its driver guard")
    return lineage


def _logs(destination, request_hash):
    path = Path(destination) / "api_calls.jsonl"
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    physical = [row for row in rows if not row.get("cache_hit")]
    if any(row.get("request_hash") != request_hash for row in physical) or len(physical) > 1:
        raise ValueError("Recovery must have at most one physical request of the exact hash")
    return physical


def _finalize(destination, lineage):
    control = Path(destination) / "recovery_control"
    started = read(control / "attempt_started.json")
    if started.get("request_hash") != lineage["request_hash"] or started.get("max_http_attempts") != 1:
        raise ValueError("Attempt marker binding differs")
    existing_outcome = read(control / "outcome.json") if (control / "outcome.json").exists() else None
    if existing_outcome and existing_outcome.get("status") not in {"ready_for_frozen_driver", "failed_validation"}:
        raise ValueError("Terminal unknown/HTTP failure cannot be replaced or retried")
    records = _logs(destination, lineage["request_hash"])
    if len(records) != 1 or records[0].get("status") != 200 or records[0].get("attempt") != 1:
        raise ValueError("No single successful HTTP response to validate")
    path = Path(destination) / "api_cache" / (lineage["request_hash"] + ".json")
    record = read(path)
    if record.get("request") != lineage["signature"] or record.get("stage") != "group_fact_extraction":
        raise ValueError("Replacement cache is not the exact extraction request")
    pending = read(Path(lineage["source"]) / TARGET / "pending_group.json")
    try:
        response = decode(record["response"])
        facts = GroupFactWriter._parse(response, pending, lineage["group_request_hash"])
    except Exception as error:
        outcome = {"schema": SCHEMA, "status": "failed_validation", "request_hash": lineage["request_hash"],
                   "error_type": type(error).__name__, "error": str(error),
                   "retry_allowed": False, "driver_resume_allowed": False}
        return _publish_outcome(control, outcome, existing_outcome, started)
    updated = copy.deepcopy(pending)
    updated["raw_response"] = response
    updated["status"] = "extracted"
    updated.pop("error_type", None)
    pending_path = Path(destination) / TARGET / "pending_group.json"
    expected_bytes = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if pending_path.exists():
        if pending_path.read_bytes() != expected_bytes:
            raise ValueError("Interrupted replacement pending does not exactly match the validated response")
    else:
        _write_new(pending_path, updated)
    outcome = {"schema": SCHEMA, "status": "ready_for_frozen_driver", "request_hash": lineage["request_hash"],
               "group_request_hash": lineage["group_request_hash"], "replacement_cache_sha256": sha(path),
               "new_pending_sha256": sha(Path(destination) / TARGET / "pending_group.json"),
               "original_pending_sha256": lineage["source_files"][TARGET + "/pending_group.json"]["sha256"],
               "decoded_response_sha256": driver.fingerprint(response), "validated_fact_count": len(facts),
               "retry_allowed": False, "driver_resume_allowed": True,
               "next_step": "Reviewer may invoke frozen write driver in this new namespace; it resumes group 4 and remaining groups under the frozen policy."}
    outcome = _publish_outcome(control, outcome, existing_outcome, started)
    guard = Path(destination) / "driver.lock"
    if guard.exists():
        if read(guard) != lineage["guard"]:
            raise ValueError("Driver guard owner changed during finalization")
        guard.unlink()
    return outcome


def _publish_outcome(control, expected, existing, started):
    if existing is not None:
        if {k:v for k,v in existing.items() if k != "finished_utc"} != expected:
            raise ValueError("Interrupted terminal outcome differs from newly validated evidence")
        stamp = datetime.fromisoformat(existing["finished_utc"])
        began = datetime.fromisoformat(started["started_utc"])
        if stamp.tzinfo is None or began.tzinfo is None or not began <= stamp <= datetime.now(timezone.utc):
            raise ValueError("Terminal outcome timestamp is not bound to this attempt")
        return existing
    outcome = {**expected, "finished_utc": now()}
    _write_new(control / "outcome.json", outcome)
    return outcome


def retry(destination=DESTINATION, *, api_factory=None):
    destination = Path(destination).resolve()
    with namespace_lock(destination):
        lineage = validate_namespace(destination)
        marker = destination / "recovery_control/attempt_started.json"
        if marker.exists():
            raise ValueError("Attempt allowance already consumed; only offline finalize is permitted")
        if (destination / "api_cache" / (lineage["request_hash"] + ".json")).exists() or _logs(destination, lineage["request_hash"]):
            raise ValueError("Unexpected replacement cache/log before the sole attempt")
        api = (api_factory or ModelAPI)(destination, retries=0, timeout=150,
            context={"architecture": "pilot400", "conversation": 1, "method": "regroup",
                     "phase": SCHEMA, "group_id": "sparse_topic_0004"})
        _write_new(marker, {"schema": SCHEMA, "request_hash": lineage["request_hash"],
                           "started_utc": now(), "pid": os.getpid(), "max_http_attempts": 1,
                           "signature_sha256": http_hash(lineage["signature"])})
        signature = lineage["signature"]
        try:
            api.request(signature["provider"], signature["model"], signature["body"],
                        operation=signature["operation"], stage="group_fact_extraction")
        except BaseException as error:
            records = _logs(destination, lineage["request_hash"])
            known_http_failure = bool(records and isinstance(records[0].get("status"), int)
                                      and records[0]["status"] >= 400)
            outcome = {"schema": SCHEMA, "status": "failed_http" if known_http_failure else "unknown",
                       "request_hash": lineage["request_hash"], "error_type": type(error).__name__,
                       "physical_log_count": len(records), "finished_utc": now(),
                       "usage_unknown": not known_http_failure, "cost_total_usd": None,
                       "retry_allowed": False, "driver_resume_allowed": False}
            _write_new(destination / "recovery_control/outcome.json", outcome)
            return outcome
        return _finalize(destination, lineage)


def finalize(destination=DESTINATION):
    """Offline recovery of a persisted response after interruption; never retries."""
    with namespace_lock(destination):
        return _finalize(Path(destination), validate_namespace(destination, allow_released=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "retry", "finalize"))
    args = parser.parse_args(argv)
    if args.action == "check":
        audit = inspect_source()
        result = {"status": "source_validated", "source_files": len(audit["files"]),
                  "request_hash": audit["request_hash"], "model_calls": 0}
    elif args.action == "prepare":
        result = prepare()
    elif args.action == "retry":
        result = retry()
    else:
        result = finalize()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"source_validated", "prepared_blocked", "ready_for_frozen_driver"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
