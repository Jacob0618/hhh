"""Version 2 shared-writer, gated three-arm pilot driver. Import and `check` never read credentials.

Stages: check | write | tag | qa. The common writer permits at most one
validator-feedback re-extraction after the original strict parser rejects its
first response. Terminal artifacts are immutable: no implicit next stage, later
retry, or question replacement. A run-wide lock prevents concurrent ownership.

Required research/main_batch_closure.json (created by the reviewer, not this tool):
  schema: jev-mem-main-batch-closure-v1
  status: completed | recovery_reviewed
  source_dataset_sha256: the pinned SOURCE_HASH
  reviewed_utc: timezone-aware ISO timestamp; reviewed_by: nonempty reviewer label
  review_path: workspace-relative reviewed report; review_sha256: its byte SHA256
  live_qa_running: false
  closed_runs: exactly [{run_dir: "runs/session_top20_v1", requested_questions:1540,
                        status:"complete"|"reviewed_with_failures"},
                       {run_dir: "runs/independent_top20_v1", requested_questions:1540,
                        status:"complete"|"reviewed_with_failures"}]
`completed` requires both runs complete. `recovery_reviewed` permits explicitly
reviewed remaining failures. Actual Windows process inventory AND queue states
must also show no main-batch running work. No closure is fabricated or auto-signed.
The check stage reports the gate without creating a run or constructing ModelAPI.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import SOURCE_HASH, load_data
from jev_mem.evaluation_pilot import READER_PROTOCOL, answer_and_judge, prepare_reader_input
from jev_mem.evaluation import lexical_metrics
from jev_mem.group_writer_v2 import (GroupFactWriter, SCHEMA as BANK_SCHEMA, SYSTEM_PROMPT, prepare_group,
                                     INPUT_PAYLOAD_SCHEMA, REEXTRACTION_POLICY,
                                     BASE_EXTRACTION_PROVIDER_RETRIES, FEEDBACK_PROVIDER_RETRIES)
from jev_mem.pilot400 import ARMS, COHORT_SHA256, fingerprint, load_cohort, query_type_scores, retrieve_rank20, type_one_fact, validate_tags
from jev_mem.providers import ModelAPI
from jev_mem.sparse_topic_pipeline import run_sparse_topic_pipeline
from jev_mem.topic_grouping import raw_turns_from_locomo, render_group, render_turn, verify_partition
from tools.check_experiment_health import shared_read

RUN_SCHEMA = "jev-mem-pilot400-run-v2"
QA_SCHEMA = "jev-mem-pilot400-qa-v2"
ARTIFACT_SCHEMA = "jev-mem-pilot400-stage-artifact-v2"
CLOSURE_SCHEMA = "jev-mem-main-batch-closure-v1"
DEFAULT_RUN = ROOT / "runs/pilot400_modules_v2"
COHORT_PATH = ROOT / "configs/pilot400_manifest_v1.json"
CLOSURE_PATH = ROOT / "research/main_batch_closure.json"
MAIN_RUNS = ("runs/session_top20_v1", "runs/independent_top20_v1")
PROTECTED_PILOT_RUNS = ("runs/pilot400_modules_v1", "runs/pilot400_writer_recovery_v1")
SOURCE_FILES = (
    "tools/run_pilot400_experiments_v2.py", "jev_mem/pilot400.py", "jev_mem/evaluation_pilot.py",
    "jev_mem/evaluation.py", "jev_mem/group_writer.py", "jev_mem/group_writer_v2.py", "jev_mem/sparse_topic_pipeline.py",
    "jev_mem/topic_boundaries.py", "jev_mem/topic_boundary_budget.py", "jev_mem/topic_grouping.py",
    "jev_mem/routing.py", "jev_mem/selection.py", "jev_mem/providers.py", "jev_mem/data.py",
    "tools/check_experiment_health.py",
)
PARAMETERS = {"cut_threshold": .7, "keep_threshold": .3, "dependency_threshold": .5,
              "same_topic_threshold": .7, "max_group_tokens": 2048,
              "current_date": "2026-09-27", "writer_prompt_capacity_guard": 1_000_000,
              "calibrated": False, "threshold_status": "uncalibrated_engineering_initial_values",
              "initial_boundary_budget": 4, "refinement_budget": 2, "initial_window": 2, "expanded_window": 4,
              "final_fact_count": 20, "candidate_cap": 60, "top_per_channel": 20,
              "max_conversation_workers": 8, "query_workers": 4, "fact_tag_workers": 4,
              "provider_retries": 2, "provider_timeout": 150}


class ProtocolMismatch(RuntimeError):
    pass


class RecordedFailure(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(shared_read(path).decode("utf-8-sig"))


def file_sha(path):
    return hashlib.sha256(shared_read(path)).hexdigest()


def publish_json(path, value):
    """Bounded atomic publication retry for new artifacts/status, never API retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                temporary.replace(path)
                return
            except OSError as error:
                sharing = isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32, 33}
                if not sharing or attempt == 7:
                    raise
                time.sleep(min(.05 * 2 ** attempt, .5))
    finally:
        if temporary.exists():
            temporary.unlink()


def _error(error, stage, api=None):
    message = str(error) or type(error).__name__
    if api is not None and hasattr(api, "_redact"):
        message = api._redact(message)
    return {"type": type(error).__name__, "stage": stage, "message": message[:1800]}


def _seal(row):
    return {**row, "content_sha256": fingerprint(row)}


def _verified(path, schema, binding):
    row = read(path)
    if (row.get("schema") != schema or row.get("request_fingerprint") != fingerprint(binding)
            or row.get("status") not in ("complete", "failed")):
        raise ProtocolMismatch("Cached artifact identity/protocol differs: " + str(path))
    if row.get("content_sha256") != fingerprint({k: v for k, v in row.items() if k != "content_sha256"}):
        raise ProtocolMismatch("Cached artifact content changed: " + str(path))
    return row


def artifact(path, binding, work, *, stage, api=None, base=None, schema=ARTIFACT_SCHEMA):
    """Immutable first terminal outcome. Failed artifacts never rerun automatically."""
    path = Path(path)
    if path.exists():
        return _verified(path, schema, binding)
    header = {**(base or {}), "schema": schema, "request_fingerprint": fingerprint(binding),
              "binding": binding, "started_utc": now()}
    try:
        row = {**header, **work(), "status": "complete", "error": None}
    except ProtocolMismatch:
        raise  # A stale success must not be overwritten or disguised as a new failure.
    except Exception as error:
        row = {**header, "status": "failed", "result": None, "error": _error(error, stage, api)}
        if hasattr(error, "trace"):
            row["partial_trace"] = error.trace
    row = _seal({**row, "finished_utc": now()})
    if path.exists():
        raise ProtocolMismatch("Another owner created an artifact during this stage: " + str(path))
    publish_json(path, row)
    return row


def require(row):
    if row["status"] != "complete":
        err = row.get("error") or {}
        raise RecordedFailure(err.get("stage", "earlier stage") + ": " + err.get("message", "recorded failure"))
    return row


def protocol():
    return {"arms": list(ARMS), "source_dataset_sha256": SOURCE_HASH, "cohort_sha256": COHORT_SHA256,
            "cohort_file_sha256": file_sha(COHORT_PATH), "parameters": copy.deepcopy(PARAMETERS),
            "reader_protocol": READER_PROTOCOL,
            "writer_schema": BANK_SCHEMA, "input_payload_schema": INPUT_PAYLOAD_SCHEMA,
            "reextraction_policy": REEXTRACTION_POLICY,
            "base_extraction_provider_retries": BASE_EXTRACTION_PROVIDER_RETRIES,
            "feedback_provider_retries": FEEDBACK_PROVIDER_RETRIES,
            "models": {"writer": "gemini-2.5-flash", "reader": "gemini-2.5-flash",
                       "judge": "gemini-3.5-flash-lite", "jev": "jev-1.13.0",
                       "embedding": "gemini-embedding-001", "embedding_dimensions": 768},
            "grouping": "sparse_boundary_then_indivisible_segment_topics",
            "raw_embedding_text": "topic_grouping.render_turn_full_no_truncation",
            "failure_policy": "common_writer_one_validator_feedback_reextraction_then_immutable_terminal_no_retry_or_question_replacement",
            "code_sha256": {name: file_sha(ROOT / name) for name in SOURCE_FILES}}


def _safe_run_dir(path):
    path = Path(path).resolve()
    if path == ROOT or path in ROOT.parents:
        raise ValueError("Use a new, dedicated pilot run directory")
    for relative in MAIN_RUNS:
        live = (ROOT / relative).resolve()
        if path == live or path in live.parents or live in path.parents:
            raise ValueError("Pilot output cannot contain or overlap a live main run")
    for relative in PROTECTED_PILOT_RUNS:
        old = (ROOT / relative).resolve()
        if path == old or path in old.parents or old in path.parents:
            raise ValueError("Version 2 output cannot overlap preserved version 1 pilot/recovery artifacts")
    return path


def ensure_manifest(run_dir, *, create=False):
    run_dir = _safe_run_dir(run_dir)
    current = protocol()
    path = run_dir / "run_manifest.json"
    if path.exists():
        saved = read(path)
        if (saved.get("schema") != RUN_SCHEMA or saved.get("protocol") != current
                or saved.get("protocol_sha256") != fingerprint(current)):
            raise ProtocolMismatch("Pilot protocol/code changed; use a separately versioned run")
        return saved
    if run_dir.exists() and any(p.name != "driver.lock" for p in run_dir.iterdir()):
        raise ProtocolMismatch("Nonempty pilot directory has no matching manifest")
    saved = {"schema": RUN_SCHEMA, "created_utc": now(), "protocol": current,
             "protocol_sha256": fingerprint(current), "cohort_sha256": COHORT_SHA256,
             "source_dataset_sha256": SOURCE_HASH}
    if create:
        publish_json(path, saved)
    return saved


def main_process_inventory():
    """Read-only local Windows inventory. Unknown inventory fails closed."""
    if os.name != "nt":
        raise RuntimeError("Live-main process guard requires Windows; inject a mock only in offline tests")
    script = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
              "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'py.exe'\" "
              "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, encoding="utf-8", timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    running = []
    for row in rows:
        command = row.get("CommandLine")
        if not isinstance(command, str):
            raise RuntimeError("Cannot establish identity of a running Python process")
        if re.search(r"run_(session_experiments|independent_experiments|eight_lane_supervisor(?:_v\d+)?)\.py", command, re.I):
            running.append({"pid": row["ProcessId"], "command_line": command})
    return running


def _running_state(value):
    if isinstance(value, dict):
        if str(value.get("status", "")).lower() in {"running", "active", "starting"}:
            return True
        return any(_running_state(v) for v in value.values())
    return isinstance(value, list) and any(_running_state(v) for v in value)


def main_batch_gate(path=CLOSURE_PATH, *, process_probe=None):
    closure = read(path)
    if (closure.get("schema") != CLOSURE_SCHEMA or closure.get("status") not in {"completed", "recovery_reviewed"}
            or closure.get("source_dataset_sha256") != SOURCE_HASH or closure.get("live_qa_running") is not False
            or not isinstance(closure.get("reviewed_by"), str) or not closure["reviewed_by"].strip()):
        raise ValueError("Main-batch closure has not passed explicit review")
    reviewed = datetime.fromisoformat(closure["reviewed_utc"].replace("Z", "+00:00"))
    if reviewed.tzinfo is None:
        raise ValueError("Closure reviewed_utc must carry a timezone")
    report = (ROOT / closure["review_path"]).resolve()
    if not report.is_relative_to(ROOT) or file_sha(report) != closure["review_sha256"]:
        raise ValueError("Main-batch review report path/hash is invalid")
    entries = closure.get("closed_runs")
    if (not isinstance(entries, list) or len(entries) != 2
            or {row.get("run_dir") for row in entries} != set(MAIN_RUNS)):
        raise ValueError("Closure must account for both original main runs")
    for row in entries:
        if row.get("requested_questions") != 1540 or row.get("status") not in {"complete", "reviewed_with_failures"}:
            raise ValueError("Closure must retain each full 1540 denominator and reviewed state")
        if closure["status"] == "completed" and row["status"] != "complete":
            raise ValueError("A completed closure cannot hide remaining failures")
        queue_path = ROOT / row["run_dir"] / "queue_status.json"
        if not queue_path.exists() or _running_state(read(queue_path)):
            raise ValueError("Main queue is missing or still reports running work")
    running = (process_probe or main_process_inventory)()
    if running:
        raise ValueError("Main-batch Python workers/scheduler are still running")
    return {"closure_sha256": file_sha(path), "review_sha256": closure["review_sha256"],
            "reviewed_status": closure["status"], "checked_utc": now()}


@contextmanager
def run_lock(run_dir):
    path = Path(run_dir) / "driver.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    token = {"pid": os.getpid(), "token": uuid.uuid4().hex, "created_utc": now()}
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError("Pilot stage already owns driver.lock; review a stale lock before removing it") from None
    with handle:
        json.dump(token, handle)
    try:
        yield
    finally:
        if path.exists() and read(path) == token:
            path.unlink()


class Monitor:
    def __init__(self, run_dir, stage, conversations, manifest, gate):
        self.lock = threading.Lock()
        self.path = Path(run_dir) / (stage + "_status.json")
        self.log = Path(run_dir) / "logs" / (stage + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.state = {"stage": stage, "status": "running", "pid": os.getpid(), "started_utc": now(),
                      "protocol_sha256": manifest["protocol_sha256"], "main_batch_gate": gate,
                      "done": 0, "expected": len(conversations), "count_unit": "conversations",
                      "api_log_path": "api_calls.jsonl", "qa_path_template": "conversations/conv_{ci}/qa/{arm}/{qid}.json",
                      "api_log_attribution": "Shared run-root ledger; filter context.conversation, method, query_id, phase",
                      "conversations": {str(ci): {"status": "pending"} for ci in conversations}}
        publish_json(self.path, self.state)

    def update(self, ci=None, **values):
        with self.lock:
            target = self.state if ci is None else self.state["conversations"][str(ci)]
            target.update(values)
            target["updated_utc"] = now()
            self.state["updated_utc"] = now()
            self.state["done"] = sum(row.get("status") in {"complete", "failed"}
                                     for row in self.state["conversations"].values())
            with self.log.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"timestamp_utc": now(), "conversation": ci, **values}, ensure_ascii=False) + "\n")
            publish_json(self.path, self.state)


def conv_dir(run_dir, ci):
    return Path(run_dir) / "conversations" / ("conv_" + str(ci))


def binding(manifest, ci, raw_turns, **extra):
    return {"protocol_sha256": manifest["protocol_sha256"], "source_dataset_sha256": SOURCE_HASH,
            "cohort_sha256": COHORT_SHA256, "conversation": ci,
            "raw_source_sha256": fingerprint(raw_turns), **extra}


def original_groups(raw_turns):
    members = {}
    for turn in raw_turns:
        members.setdefault(turn["session_id"], []).append(turn)
    groups = [{"group_id": "original_" + sid, "turns": copy.deepcopy(turns),
               "source_turn_ids": [t["source_turn_id"] for t in turns], "text": render_group(turns)}
              for sid, turns in members.items()]
    verify_partition(raw_turns, groups)
    return groups


def _vectors(values, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Embedding count differs from complete inputs")
    for row in values:
        if (not isinstance(row, list) or len(row) != 768 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)
                or not math.isfinite(sum(v*v for v in row)) or sum(v*v for v in row) <= 0):
            raise ValueError("Embeddings require 768 finite nonzero dimensions")
    return values


def _bank_binding(manifest, ci, raw, name):
    return binding(manifest, ci, raw, bank=name)


def load_bank(run_dir, ci, name, raw, manifest):
    folder = conv_dir(run_dir, ci) / "banks" / name
    terminal = require(_verified(folder / "bank_summary.json", ARTIFACT_SCHEMA, _bank_binding(manifest, ci, raw, name)))
    path = folder / "group_fact_bank.json"
    if terminal.get("bank_sha256") != file_sha(path):
        raise ProtocolMismatch("Completed fact bank bytes changed")
    bank = read(path)
    if bank.get("schema") != BANK_SCHEMA or bank.get("source_sha256") != fingerprint(raw):
        raise ProtocolMismatch("Bank schema or complete source binding differs")
    if bank.get("user_id") != "pilot400_conv_" + str(ci):
        raise ProtocolMismatch("Bank conversation identity differs")
    expected_method = "original_sessions" if name == "session" else "jev_topics"
    if bank.get("grouping_method") != expected_method:
        raise ProtocolMismatch("Bank grouping method differs")
    archive = read(folder / "raw_source.json")
    if archive != {"user_id": bank["user_id"], "source_sha256": fingerprint(raw), "raw_turns": raw}:
        raise ProtocolMismatch("Original raw-source archive changed")
    groups = [{"turns": row["members"], "source_turn_ids": row["member_source_turn_ids"],
               "text": render_group(row["members"])} for row in bank["groups"]]
    verify_partition(raw, groups)
    flattened = [f for row in bank["groups"] for f in row["fact_records"]]
    if flattened != bank.get("facts") or len({f["id"] for f in flattened}) != len(flattened):
        raise ProtocolMismatch("Committed group records differ from the fact bank")
    if (terminal.get("coverage", {}).get("complete") is not True
            or terminal["coverage"].get("source_turns") != len(raw)
            or terminal.get("fact_count") != len(flattened)):
        raise ProtocolMismatch("Bank coverage summary is incomplete")
    _vectors([f["embedding"] for f in flattened], len(flattened))
    for fact in flattened:
        prepare_reader_input("Validate source-date schema.", [fact], reference_date=None)
    return {"facts": flattened, "sha256": file_sha(path), "path": path.relative_to(Path(run_dir)).as_posix(),
            "summary_sha256": file_sha(folder / "bank_summary.json")}


def write_conversation(run_dir, ci, raw, manifest, api, monitor):
    folder, params = conv_dir(run_dir, ci), manifest["protocol"]["parameters"]
    outcomes = {}
    monitor.update(ci, done=0, expected=2, count_unit="banks")
    for name in ("session", "regroup"):
        bank_api = api.scoped(conversation=ci, method=name, phase="pilot_write")
        destination = folder / "banks" / name
        bound = _bank_binding(manifest, ci, raw, name)

        def write_bank():
            group_binding = {**bound, "operation": "grouping"}

            def make_groups():
                if name == "session":
                    return {"groups": original_groups(raw), "writer_grouping_method": "original_sessions",
                            "trace": {"original_sessions_preserved": True, "qa_visible": False}}
                vectors = require(artifact(folder / "raw_embeddings.json", binding(manifest, ci, raw, operation="raw_embedding"),
                    lambda: {"vectors": _vectors(bank_api.embed([render_turn(t) for t in raw], query=False,
                                                                stage="pilot_raw_turn_embedding"), len(raw))},
                    stage="raw_embedding", api=bank_api))["vectors"]
                return run_sparse_topic_pipeline(raw, judge=bank_api, embeddings=vectors,
                    token_counter=lambda text: bank_api.count_tokens(text, model="gemini-embedding-001", stage="pilot_group_token_count"),
                    **{k: params[k] for k in ("cut_threshold", "keep_threshold", "dependency_threshold", "same_topic_threshold", "max_group_tokens")})

            grouping = require(artifact(destination / "grouping.json", group_binding, make_groups, stage="grouping", api=bank_api))
            groups = grouping["groups"]
            verify_partition(raw, groups)
            positions = {t["source_turn_id"]: i for i, t in enumerate(raw)}
            groups = sorted(groups, key=lambda g: min(positions[t["source_turn_id"]] for t in g["turns"]))
            writer = GroupFactWriter(bank_api, destination)
            try:
                for index, group in enumerate(groups):
                    # Capacity applies equally to A/C. Count full system+payload text;
                    # the conservative guard reserves room for provider framing.
                    prepared = prepare_group(members=group["turns"], raw_turns=raw, current_date=params["current_date"])
                    complete_prompt = SYSTEM_PROMPT + "\n" + json.dumps(prepared["payload"], ensure_ascii=False, indent=2)
                    count = bank_api.count_tokens(complete_prompt, model="gemini-2.5-flash", stage="pilot_writer_capacity")
                    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= params["writer_prompt_capacity_guard"]:
                        raise ValueError("Full writer prompt exceeds the frozen conservative capacity guard")
                    writer.add_group(user_id="pilot400_conv_" + str(ci), group_id=group["group_id"], members=group["turns"],
                                     raw_turns=raw, grouping_method=grouping["writer_grouping_method"], current_date=params["current_date"])
                    monitor.update(ci, status="running", bank=name, groups_completed=index + 1, groups_total=len(groups))
                coverage = writer.coverage()
                if not coverage["complete"] or coverage["source_turns"] != len(raw):
                    raise ValueError("Writer did not cover the complete raw conversation exactly once")
                return {"bank_sha256": file_sha(destination / "group_fact_bank.json"), "coverage": coverage,
                        "fact_count": len(writer.get_facts()), "grouping_sha256": file_sha(destination / "grouping.json")}
            finally:
                writer.close()

        try:
            row = artifact(destination / "bank_summary.json", bound, write_bank, stage="write", api=bank_api)
            if row["status"] == "complete":
                load_bank(run_dir, ci, name, raw, manifest)
            outcomes[name] = {"status": row["status"], "error": row.get("error")}
        except Exception as error:
            outcomes[name] = {"status": "failed", "error": _error(error, "bank_validation", bank_api)}
        monitor.update(ci, banks=outcomes, done=len(outcomes), expected=2, count_unit="banks")
    return {"status": "complete" if all(r["status"] == "complete" for r in outcomes.values()) else "failed", "banks": outcomes}


def tag_binding(manifest, ci, raw, bank):
    return binding(manifest, ci, raw, operation="all_fact_typing", bank_sha256=bank["sha256"])


def load_tags(run_dir, ci, raw, manifest, bank):
    path = conv_dir(run_dir, ci) / "tag_summary.json"
    row = require(_verified(path, ARTIFACT_SCHEMA, tag_binding(manifest, ci, raw, bank)))
    validate_tags(bank["facts"], row["tags"])
    return row["tags"], file_sha(path)


def tag_conversation(run_dir, ci, raw, manifest, api, monitor):
    bank = load_bank(run_dir, ci, "session", raw, manifest)
    api = api.scoped(conversation=ci, method="shared_session", phase="pilot_tag")
    bound = tag_binding(manifest, ci, raw, bank)

    def tag_all():
        tags, errors = {}, []

        def one(fact):
            path = conv_dir(run_dir, ci) / "tags" / (fingerprint(fact["id"]) + ".json")
            return fact["id"], artifact(path, {**bound, "fact_sha256": fingerprint(fact), "fact_id": fact["id"]},
                    lambda: {"tag": type_one_fact(fact, api.scoped(fact_id=fact["id"]))}, stage="tag", api=api)

        with ThreadPoolExecutor(max_workers=manifest["protocol"]["parameters"]["fact_tag_workers"]) as executor:
            futures = [executor.submit(one, fact) for fact in bank["facts"]]
            for future in as_completed(futures):
                fid, row = future.result()
                if row["status"] == "complete":
                    tags[fid] = row["tag"]
                else:
                    errors.append({"fact_id": fid, "error": row["error"]})
                monitor.update(ci, status="running", facts_completed=len(tags), facts_failed=len(errors), facts_total=len(bank["facts"]),
                               done=len(tags) + len(errors), expected=len(bank["facts"]), count_unit="facts")
        if errors:
            raise RecordedFailure(str(len(errors)) + " fact tags failed; first-pass per-fact records retained")
        validate_tags(bank["facts"], tags)
        return {"tags": tags, "fact_count": len(tags), "bank_sha256": bank["sha256"]}

    row = artifact(conv_dir(run_dir, ci) / "tag_summary.json", bound, tag_all, stage="tag_all", api=api)
    monitor.update(ci, done=len(bank["facts"]) if row["status"] == "complete" else None,
                   expected=len(bank["facts"]), count_unit="facts")
    return {"status": row["status"], "error": row.get("error"), "fact_count": row.get("fact_count")}


def validate_qa_cache(row, q, bank, tags, arm, raw, folder, query_bound, route_bound, run_dir):
    """Rebuild deterministic retrieval and the actual date-aware reader request."""
    if row["status"] != "complete":
        return
    vector = require(_verified(folder / "query_embeddings" / (q["id"] + ".json"), ARTIFACT_SCHEMA, query_bound))["vector"]
    _vectors([vector], 1)
    needs = (require(_verified(folder / "query_types" / (q["id"] + ".json"), ARTIFACT_SCHEMA, query_bound))["scores"]
             if arm == ARMS[1] else None)
    expected = retrieve_rank20(q["question"], bank["facts"], vector,
                              tags=tags if arm == ARMS[1] else None, needs=needs)
    path = folder / "retrieval" / arm / (q["id"] + ".json")
    retrieved = require(_verified(path, ARTIFACT_SCHEMA, route_bound))
    if any(retrieved.get(key) != expected[key] for key in ("pool", "selected", "trace")):
        raise ProtocolMismatch("Frozen retrieval differs from the current bound fact bank")
    ids = [f["id"] for f in expected["selected"]]
    prepared = prepare_reader_input(q["question"], expected["selected"],
                                    reference_date=max(t["observation_date"] for t in raw))
    result = row.get("result")
    if (row.get("selected_ids") != ids or len(ids) != 20 or len(set(ids)) != 20
            or row.get("retrieval_path") != path.relative_to(Path(run_dir)).as_posix()
            or row.get("retrieval_sha256") != file_sha(path)
            or not isinstance(result, dict) or result.get("selected_count") != 20
            or result.get("reader_input") != prepared["payload"]
            or result.get("input_hash") != prepared["input_hash"]
            or result.get("reader_request_hash") != prepared["reader_request_hash"]
            or result.get("reader_protocol") != READER_PROTOCOL
            or result.get("reader_model") != "gemini-2.5-flash" or result.get("judge_model") != "gemini-3.5-flash-lite"
            or not isinstance(result.get("answer"), str)):
        raise ProtocolMismatch("Cached QA input/selection/date protocol differs")
    used = result.get("used_fact_ids")
    judge = result.get("judge")
    if (not isinstance(used, list) or any(not isinstance(fid, str) for fid in used)
            or len(set(used)) != len(used) or not set(used) <= set(ids)
            or not isinstance(judge, dict) or type(judge.get("correct")) is not bool
            or not isinstance(judge.get("reason"), str)
            or type(judge.get("completeness")) not in (int, float)
            or not math.isfinite(judge["completeness"]) or not 0 <= judge["completeness"] <= 1):
        raise ProtocolMismatch("Cached QA response/citation schema differs")
    metrics = lexical_metrics(result["answer"], q["reference"])
    if any(result.get(key) != value for key, value in metrics.items()):
        raise ProtocolMismatch("Cached lexical metrics differ from the fixed reference")


def qa_conversation(run_dir, ci, raw, questions, manifest, api, monitor):
    folder, banks, failures = conv_dir(run_dir, ci), {}, {}
    for name in ("session", "regroup"):
        try:
            banks[name] = load_bank(run_dir, ci, name, raw, manifest)
        except Exception as error:
            failures[name] = error
    tags, tags_sha = None, None
    if "session" in banks:
        try:
            tags, tags_sha = load_tags(run_dir, ci, raw, manifest, banks["session"])
        except Exception as error:
            failures["tags"] = error
    requested = [q["id"] for q in questions]
    counts = {arm: {"requested": len(questions), "completed": 0, "failed": 0} for arm in ARMS}
    monitor.update(ci, done=0, expected=len(questions) * 3, count_unit="arm_questions")

    def one(q):
        outcomes = {}
        shared_api = api.scoped(conversation=ci, query_id=q["id"], method="shared_query", phase="pilot_qa_prepare")
        query_bound = binding(manifest, ci, raw, query_id=q["id"], question_sha256=fingerprint(q["question"]))
        for arm in ARMS:
            name = "regroup" if arm == ARMS[2] else "session"
            bank = banks.get(name)
            arm_api = api.scoped(conversation=ci, query_id=q["id"], method=arm, phase="pilot_qa")
            route_bound = {**query_bound, "arm": arm, "bank_sha256": bank["sha256"] if bank else None,
                           "tag_summary_sha256": tags_sha if arm == ARMS[1] else None}
            bound = {**route_bound, "reference_sha256": fingerprint(q["reference"])}
            base = {"arm": arm, "query_id": q["id"], "conversation": ci, "category": q["category"],
                    "cohort_sha256": COHORT_SHA256, "protocol_sha256": manifest["protocol_sha256"],
                    "bank_path": f"conversations/conv_{ci}/banks/{name}/group_fact_bank.json",
                    "bank_sha256": bank["sha256"] if bank else None, "selected_ids": [], "result": None}

            def evaluate():
                if name in failures:
                    raise failures[name]
                if arm == ARMS[1] and "tags" in failures:
                    raise failures["tags"]
                vector = require(artifact(folder / "query_embeddings" / (q["id"] + ".json"), query_bound,
                    lambda: {"vector": _vectors(shared_api.embed([q["question"]], query=True, stage="pilot_shared_query_embedding"), 1)[0]},
                    stage="query_embedding", api=shared_api))["vector"]
                needs = None
                if arm == ARMS[1]:
                    def type_query():
                        values, trace = query_type_scores(q["question"], shared_api)
                        return {"scores": values, "trace": trace}
                    needs = require(artifact(folder / "query_types" / (q["id"] + ".json"), query_bound,
                                            type_query, stage="query_type", api=shared_api))["scores"]
                retrieve_path = folder / "retrieval" / arm / (q["id"] + ".json")
                retrieved = require(artifact(retrieve_path, route_bound,
                    lambda: retrieve_rank20(q["question"], bank["facts"], vector,
                                           tags=tags if arm == ARMS[1] else None, needs=needs),
                    stage="retrieval", api=arm_api))
                selected = retrieved["selected"]
                if len(selected) != 20 or len({f["id"] for f in selected}) != 20:
                    raise ValueError("Every successful pilot answer requires exactly 20 unique facts")
                result = answer_and_judge(arm_api, q["question"], q["reference"], selected,
                                         reference_date=max(t["observation_date"] for t in raw))
                return {"selected_ids": [f["id"] for f in selected], "result": result,
                        "retrieval_path": retrieve_path.relative_to(Path(run_dir)).as_posix(),
                        "retrieval_sha256": file_sha(retrieve_path)}

            try:
                row = artifact(folder / "qa" / arm / (q["id"] + ".json"), bound, evaluate,
                               stage="qa", api=arm_api, base=base, schema=QA_SCHEMA)
                validate_qa_cache(row, q, bank, tags, arm, raw, folder, query_bound, route_bound, run_dir)
                outcomes[arm] = {"status": row["status"], "error": row.get("error")}
            except Exception as error:
                # Preserve invalid/stale first-pass files; report the problem separately.
                outcomes[arm] = {"status": "failed", "error": _error(error, "qa_cache_validation", arm_api)}
        return q["id"], outcomes

    errors = []
    with ThreadPoolExecutor(max_workers=manifest["protocol"]["parameters"]["query_workers"]) as executor:
        futures = [executor.submit(one, q) for q in questions]
        for future in as_completed(futures):
            qid, outcomes = future.result()
            for arm, outcome in outcomes.items():
                counts[arm]["completed" if outcome["status"] == "complete" else "failed"] += 1
                if outcome["status"] != "complete":
                    errors.append({"query_id": qid, "arm": arm, "error": outcome["error"]})
            done = sum(v["completed"] + v["failed"] for v in counts.values())
            monitor.update(ci, status="running", methods=counts, queries_observed=done // 3,
                           done=done, expected=len(questions) * 3, count_unit="arm_questions")
    summary = {"schema": "jev-mem-pilot400-conversation-qa-v2", "conversation": ci,
               "cohort_sha256": COHORT_SHA256, "protocol_sha256": manifest["protocol_sha256"],
               "requested_query_ids": requested, "methods": counts, "errors": errors,
               "status": "complete" if not errors else "failed", "updated_utc": now()}
    publish_json(folder / "qa_status.json", summary)
    return summary


def offline_check(run_dir=DEFAULT_RUN, *, closure_path=CLOSURE_PATH, process_probe=None):
    data = load_data()
    _, questions = load_cohort(COHORT_PATH, data)
    manifest = ensure_manifest(run_dir, create=False)
    errors = []
    for ci, item in enumerate(data):
        raw = raw_turns_from_locomo(item["conversation"])
        for name in ("session", "regroup"):
            if (conv_dir(run_dir, ci) / "banks" / name / "bank_summary.json").exists():
                try:
                    load_bank(run_dir, ci, name, raw, manifest)
                except Exception as error:
                    errors.append({"conversation": ci, "bank": name, "error": _error(error, "offline_bank_check")})
    try:
        gate = main_batch_gate(closure_path, process_probe=process_probe)
        ready = not errors
    except Exception as error:
        gate, ready = {"error": _error(error, "main_batch_gate")}, False
    return {"status": "offline_check_passed" if not errors else "offline_artifacts_need_review",
            "network_calls": 0, "credentials_loaded": False, "run_created": False,
            "requested_questions_per_arm": len(questions), "arms": list(ARMS),
            "protocol_sha256": manifest["protocol_sha256"], "ready_for_network": ready,
            "development_raw_source_review": "not_certified_by_this_offline_check",
            "scientific_live_green": False,
            "main_batch_gate": gate, "artifact_errors": errors}


def run_stage(stage, run_dir=DEFAULT_RUN, *, conversations=None, workers=8, closure_path=CLOSURE_PATH,
              api_factory=None, gate_checker=None):
    if stage not in {"write", "tag", "qa"}:
        raise ValueError("Only write/tag/qa are network stages")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Conversation workers must be between 1 and 8")
    cis = list(range(10)) if conversations is None else list(conversations)
    if (not cis or len(set(cis)) != len(cis) or any(type(ci) is not int or ci not in range(10) for ci in cis)):
        raise ValueError("Conversations must be unique pinned indices 0..9")
    data = load_data()
    _, questions = load_cohort(COHORT_PATH, data)
    gate = (gate_checker or main_batch_gate)(closure_path)  # Before credentials or any API instance.
    run_dir = _safe_run_dir(run_dir)
    with run_lock(run_dir):
        manifest = ensure_manifest(run_dir, create=True)
        monitor = Monitor(run_dir, stage, cis, manifest, gate)
        params = manifest["protocol"]["parameters"]
        try:
            api = (api_factory or ModelAPI)(run_dir, retries=params["provider_retries"], timeout=params["provider_timeout"],
                                           context={"architecture": "pilot400", "driver_stage": stage})
        except Exception as error:
            failure = {"type": type(error).__name__, "stage": "api_initialization",
                       "message": "API initialization failed before stage work; no credentials included in this record"}
            for ci in cis:
                monitor.update(ci, status="failed", error=failure)
            monitor.update(status="failed", error=failure, finished_utc=now())
            return {"stage": stage, "status": "requires_review", "error": failure, "cohort_sha256": COHORT_SHA256}

        def run_one(ci):
            monitor.update(ci, status="running")
            try:
                raw = raw_turns_from_locomo(data[ci]["conversation"])
                if stage == "write":
                    result = write_conversation(run_dir, ci, raw, manifest, api, monitor)
                elif stage == "tag":
                    result = tag_conversation(run_dir, ci, raw, manifest, api, monitor)
                else:
                    selected = [q for q in questions if q["conversation"] == ci]
                    result = qa_conversation(run_dir, ci, raw, selected, manifest, api, monitor)
            except Exception as error:
                result = {"status": "failed", "error": _error(error, stage, api)}
            monitor.update(ci, **result)
            return ci, result

        results = {}
        try:
            with ThreadPoolExecutor(max_workers=min(workers, len(cis))) as executor:
                futures = [executor.submit(run_one, ci) for ci in cis]
                for future in as_completed(futures):
                    ci, result = future.result()
                    results[str(ci)] = result
            status = "complete" if all(row["status"] == "complete" for row in results.values()) else "requires_review"
            monitor.update(status=status, finished_utc=now())
            return {"stage": stage, "status": status, "conversations": results, "cohort_sha256": COHORT_SHA256}
        except BaseException:
            monitor.update(status="interrupted_requires_review", finished_utc=now())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "write", "tag", "qa"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--closure", type=Path, default=CLOSURE_PATH)
    args = parser.parse_args(argv)
    if args.stage == "check":
        result = offline_check(args.run_dir, closure_path=args.closure)
    else:
        result = run_stage(args.stage, args.run_dir, conversations=[int(x) for x in args.conversations.split(",")],
                           workers=args.workers, closure_path=args.closure)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"complete", "offline_check_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
