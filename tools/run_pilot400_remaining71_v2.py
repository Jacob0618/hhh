"""Recover exactly the 71 reviewed C bank dependencies in a new namespace.

--check is read-only and loads no credentials. --run admits two reviewed complete
bank overlays, reuses sealed query vectors, freezes new deterministic rank20
selections, and then performs one structured Reader/Judge attempt per question.
Frozen original results and the prior eleven-question recovery are never edited.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import SOURCE_HASH, load_data
from jev_mem.evaluation_pilot import prepare_reader_input
from jev_mem.pilot400 import ARMS, COHORT_SHA256, fingerprint, load_cohort, reader_fact
from jev_mem.providers import ModelAPI
from jev_mem.topic_grouping import raw_turns_from_locomo
from tools.pilot400_recovery_qa_v1 import (
    ResponseSchemaAPI, evaluate_new, _sealed, _verify_sealed,
    READER_SCHEMA, JUDGE_SCHEMA, ORIGINAL_PROTOCOL, TERMINAL_SEAL)
from tools.pilot400_recovery_rank_v1 import retrieve, VERSION as RANK_VERSION
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now, _vectors

POLICY = "pilot400-remaining71-recovery-v2"
DEFAULT_OUTPUT = ROOT / "runs/pilot400_recovery_v2"
ORIGINAL_RUN = ROOT / "runs/pilot400_modules_v2"
PREVIOUS_RECOVERY = ROOT / "runs/pilot400_recovery_v1"
COHORT_PATH = ROOT / "configs/pilot400_manifest_v1.json"
ARM = ARMS[2]
COUNTS = {1: 21, 7: 50}
SOURCE_FILES = (
    "tools/run_pilot400_remaining71_v2.py", "tools/pilot400_recovery_rank_v1.py",
    "tools/pilot400_recovery_qa_v1.py", "tools/run_pilot400_experiments_v2.py",
    "jev_mem/evaluation_pilot.py", "jev_mem/evaluation.py", "jev_mem/pilot400.py",
    "jev_mem/providers.py", "jev_mem/routing.py", "jev_mem/data.py")


def _output(path):
    value = Path(path).resolve()
    if value != DEFAULT_OUTPUT.resolve():
        raise ValueError("This finite recovery is pinned to runs/pilot400_recovery_v2")
    return value


def fixed_questions(data):
    _, all_questions = load_cohort(COHORT_PATH, data)
    rows = [q for q in all_questions if q["conversation"] in COUNTS]
    if (len(rows) != 71 or len({q["id"] for q in rows}) != 71 or
            {ci: sum(q["conversation"] == ci for q in rows) for ci in COUNTS} != COUNTS):
        raise ValueError("The fixed 21 + 50 recovery population changed")
    return rows


def validate_admission(output, raw_by_conv):
    """Hard gate before constructing any ModelAPI or dispatching any request."""
    path = Path(output) / "bank_admission.json"
    admission = read(path)
    if (admission.get("schema") != "pilot400-recovery-v2-bank-admission"
            or admission.get("source_dataset_sha256") != SOURCE_HASH
            or admission.get("cohort_sha256") != COHORT_SHA256
            or set(admission.get("banks", {})) != {"1", "7"}):
        raise ValueError("Bank admission schema/source/cohort differs")
    review_path = (ROOT / admission["review_path"]).resolve()
    if not review_path.is_relative_to(ROOT) or file_sha(review_path) != admission["review_sha256"]:
        raise ValueError("Bank review bytes differ")
    banks = {}
    for ci in COUNTS:
        gate = admission["banks"][str(ci)]
        bank_root = (Path(output) / "banks" / ("conv_" + str(ci))).resolve()
        relative = gate.get("bank_path", "banks/conv_" + str(ci) + "/bank_overlay.json")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Admitted bank path must be relative to this recovery run")
        path = (Path(output) / relative).resolve()
        if not path.is_relative_to(bank_root) or path == bank_root:
            raise ValueError("Admitted bank path leaves its recovery conversation directory")
        if gate.get("admitted") is not True or file_sha(path) != gate.get("bank_sha256"):
            raise ValueError("Bank not admitted or bank hash differs: " + str(ci))
        bank = read(path)
        facts = bank.get("facts")
        if (not bank.get("schema") or bank.get("conversation") != ci
                or bank.get("source_sha256") != fingerprint(raw_by_conv[ci])
                or not isinstance(facts, list) or len(facts) < 20
                or len({f["id"] for f in facts}) != len(facts)):
            raise ValueError("Overlay bank identity/source/unique facts differ")
        _vectors([f["embedding"] for f in facts], len(facts))
        for fact in facts:
            prepare_reader_input("Validate recovered source dates.", [fact], reference_date=None)
        banks[ci] = {"facts": facts, "path": str(path), "sha256": file_sha(path)}
    return admission, banks


def collect(output=DEFAULT_OUTPUT):
    """Bind only previously unanswered C questions; never recompute original329."""
    output = _output(output)
    data = load_data()
    questions = fixed_questions(data)
    raw = {ci: raw_turns_from_locomo(data[ci]["conversation"]) for ci in COUNTS}
    admission, banks = validate_admission(output, raw)
    original_manifest = read(ORIGINAL_RUN / "run_manifest.json")
    seal = read(TERMINAL_SEAL)
    if (original_manifest["protocol_sha256"] != ORIGINAL_PROTOCOL or
            seal["protocol_sha256"] != ORIGINAL_PROTOCOL or
            Path(seal["run_dir"]).resolve() != ORIGINAL_RUN.resolve()):
        raise ValueError("Original run manifest or seal differs")
    sealed = {item["path"]: item["sha256"] for item in seal["files"]}
    common = {"policy": POLICY, "source_dataset_sha256": SOURCE_HASH,
        "cohort_sha256": COHORT_SHA256, "cohort_file_sha256": file_sha(COHORT_PATH),
        "original_manifest_sha256": file_sha(ORIGINAL_RUN / "run_manifest.json"),
        "original_terminal_seal_sha256": file_sha(TERMINAL_SEAL),
        "admission_sha256": file_sha(output / "bank_admission.json"),
        "bank_review_sha256": admission["review_sha256"], "rank_version": RANK_VERSION,
        "source_code": {name: file_sha(ROOT / name) for name in SOURCE_FILES}}
    records = []
    for q in questions:
        ci = q["conversation"]
        folder = ORIGINAL_RUN / "conversations" / ("conv_" + str(ci))
        old_path = folder / "qa" / ARM / (q["id"] + ".json")
        old = _verify_sealed(read(old_path))
        if (file_sha(old_path) != sealed.get(old_path.relative_to(ORIGINAL_RUN).as_posix())
                or old.get("status") != "failed" or old.get("result") is not None
                or old.get("selected_ids") != [] or old.get("conversation") != ci
                or old.get("arm") != ARM or old.get("query_id") != q["id"]
                or old["binding"]["question_sha256"] != fingerprint(q["question"])
                or old["binding"]["reference_sha256"] != fingerprint(q["reference"])):
            raise ValueError("Question is not an original sealed bank-dependency failure")
        if (folder / "retrieval" / ARM / old_path.name).exists():
            raise ValueError("Recovery must not recompute an original retrieval")
        if (PREVIOUS_RECOVERY / "conversations" / ("conv_" + str(ci)) / "qa" / ARM / old_path.name).exists():
            raise ValueError("Question has already been handled by recovery v1")
        vector_path = folder / "query_embeddings" / old_path.name
        vector_row = _verify_sealed(read(vector_path))
        expected = {"protocol_sha256": ORIGINAL_PROTOCOL, "source_dataset_sha256": SOURCE_HASH,
            "cohort_sha256": COHORT_SHA256, "conversation": ci, "raw_source_sha256": fingerprint(raw[ci]),
            "query_id": q["id"], "question_sha256": fingerprint(q["question"])}
        if (file_sha(vector_path) != sealed.get(vector_path.relative_to(ORIGINAL_RUN).as_posix())
                or vector_row.get("status") != "complete" or vector_row.get("binding") != expected
                or vector_row.get("request_fingerprint") != fingerprint(expected)):
            raise ValueError("Original query embedding or identity seal differs")
        _vectors([vector_row["vector"]], 1)
        raw_path = folder / "banks/regroup/raw_source.json"
        archive = read(raw_path)
        if archive != {"user_id": "pilot400_conv_" + str(ci), "source_sha256": fingerprint(raw[ci]), "raw_turns": raw[ci]}:
            raise ValueError("Original source/date archive differs")
        binding = {**common, "conversation": ci, "query_id": q["id"], "arm": ARM,
            "question_sha256": fingerprint(q["question"]), "reference_sha256": fingerprint(q["reference"]),
            "original_qa_path": str(old_path), "original_qa_sha256": file_sha(old_path),
            "query_embedding_path": str(vector_path), "query_embedding_sha256": file_sha(vector_path),
            "raw_source_path": str(raw_path), "raw_source_sha256": file_sha(raw_path),
            "bank_path": banks[ci]["path"], "bank_sha256": banks[ci]["sha256"],
            "reference_date": max(t["observation_date"] for t in raw[ci])}
        records.append({"q": q, "facts": banks[ci]["facts"], "vector": vector_row["vector"], "binding": binding})
    return records


def retrieval_for(record):
    result = retrieve(record["q"]["question"], record["facts"], record["vector"])
    ids = [f["id"] for f in result["selected"]]
    if len(ids) != 20 or len(set(ids)) != 20 or len(result["pool"]) > 60:
        raise ValueError("Selection must contain exactly 20 unique facts from at most 60")
    if result["selected"] != [reader_fact(f) for f in result["pool"][:20]]:
        raise ValueError("Selected evidence differs from exact rank20")
    result["trace"]["raw_member_excerpt_selected"] = sum(
        f.get("memory_kind") == "raw_member_excerpt" for f in result["selected"])
    prepare_reader_input(record["q"]["question"], result["selected"], reference_date=record["binding"]["reference_date"])
    return _sealed({"schema": POLICY + "-retrieval", "status": "complete", "binding": record["binding"], **result})


@contextmanager
def run_lock(output):
    path = Path(output) / "qa_remaining71.lock"
    token = {"pid": os.getpid(), "created_utc": now()}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(token, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        if path.exists() and read(path) == token:
            path.unlink()


def _paths(output, record):
    root = Path(output) / "conversations" / ("conv_" + str(record["q"]["conversation"]))
    name = record["q"]["id"] + ".json"
    return root / "retrieval" / ARM / name, root / "qa" / ARM / name


def one(record, output, base):
    retrieval_path, path = _paths(output, record)
    retrieval_row = _verify_sealed(read(retrieval_path))
    if retrieval_row["binding"] != record["binding"]:
        raise ValueError("Frozen retrieval binding differs")
    selected = retrieval_row["selected"]
    binding = {**record["binding"], "retrieval_sha256": file_sha(retrieval_path),
        "selected_sha256": fingerprint(selected)}
    if path.exists():
        saved = _verify_sealed(read(path))
        if (saved.get("binding") != binding or saved.get("status") not in ("complete", "failed")
                or saved.get("selected_ids") != [f["id"] for f in selected]):
            raise ValueError("Terminal QA identity differs")
        if saved["status"] == "complete":
            prepared = prepare_reader_input(record["q"]["question"], selected,
                                            reference_date=binding["reference_date"])
            if (saved["result"].get("reader_input") != prepared["payload"]
                    or saved["result"].get("input_hash") != prepared["input_hash"]):
                raise ValueError("Terminal successful QA input differs")
        elif saved.get("result") is not None:
            raise ValueError("Terminal failed QA must retain null result")
        for event in saved.get("api_events", []):
            if event.get("cache_path") and file_sha(event["cache_path"]) != event["cache_sha256"]:
                raise ValueError("Terminal QA raw response changed")
        return saved
    q = record["q"]
    api = ResponseSchemaAPI(base.scoped(conversation=q["conversation"], query_id=q["id"],
        method=ARM, phase="pilot_remaining71_qa"))
    started = now()
    result, error = None, None
    try:
        result = evaluate_new(q, selected, binding["reference_date"], api)
    except Exception as exc:
        error = {"type": type(exc).__name__, "stage": api.events[-1]["stage"] if api.events else "qa",
                 "message": api._redact(str(exc))[:1800]}
    row = _sealed({"schema": POLICY, "query_id": q["id"], "conversation": q["conversation"],
        "category": q["category"], "arm": ARM, "binding": binding, "status": "complete" if result is not None else "failed",
        "result": result, "error": error, "selected_ids": [f["id"] for f in selected],
        "retrieval_path": str(retrieval_path), "api_events": api.events,
        "started_utc": started, "finished_utc": now()})
    if path.exists():
        raise RuntimeError("Another owner created a terminal QA artifact")
    publish_json(path, row)
    return row


def execute(output=DEFAULT_OUTPUT):
    output = _output(output)
    records = collect(output)  # Entire admission is checked before credentials or mutations.
    with run_lock(output):
        # Publish every selection before constructing ModelAPI. Existing selections must match exactly.
        for record in records:
            path, _ = _paths(output, record)
            row = retrieval_for(record)
            if path.exists():
                if _verify_sealed(read(path)) != row:
                    raise ValueError("Existing deterministic retrieval changed")
            else:
                publish_json(path, row)
        plan = {"policy": POLICY, "expected": 71, "counts": {str(k): v for k, v in COUNTS.items()},
            "arm": ARM, "max_conversation_workers": 2, "query_workers": 4, "provider_retries": 0,
            "reader_schema": READER_SCHEMA, "judge_schema": JUDGE_SCHEMA,
            "entries": [{"binding": r["binding"], "retrieval_sha256": file_sha(_paths(output, r)[0])} for r in records]}
        plan_path = output / "qa_remaining71_plan.json"
        if plan_path.exists():
            if read(plan_path) != plan:
                raise ValueError("Frozen recovery plan changed")
        else:
            publish_json(plan_path, plan)
        base = ModelAPI(output, retries=0, context={"architecture": POLICY, "driver_stage": "qa"})
        state = {"policy": POLICY, "status": "running", "pid": os.getpid(), "started_utc": now(),
            "expected": 71, "done": 0, "completed": 0, "failed": 0, "max_active_queries": 8,
            "plan_sha256": file_sha(plan_path), "conversations": {str(ci): {"expected": count,
                "done": 0, "completed": 0, "failed": 0} for ci, count in COUNTS.items()}}
        status_path = output / "qa_remaining71_status.json"
        publish_json(status_path, state)
        lock = threading.Lock()
        def conversation(ci):
            rows = [r for r in records if r["q"]["conversation"] == ci]
            with ThreadPoolExecutor(max_workers=4) as executor:
                for future in as_completed([executor.submit(one, r, output, base) for r in rows]):
                    result = future.result()
                    with lock:
                        key = "completed" if result["status"] == "complete" else "failed"
                        for target in (state, state["conversations"][str(ci)]):
                            target["done"] += 1
                            target[key] += 1
                        state["updated_utc"] = now()
                        publish_json(status_path, state)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for future in as_completed([executor.submit(conversation, ci) for ci in COUNTS]):
                    future.result()
        except Exception as exc:
            state.update(status="blocked", updated_utc=now(), error=base._redact(str(exc))[:1800])
            publish_json(status_path, state)
            raise
        state.update(status="complete" if state["failed"] == 0 else "requires_review", finished_utc=now())
        publish_json(status_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.check:
        records = collect(args.output)
        selections = [retrieval_for(r) for r in records]
        print(json.dumps({"policy": POLICY, "questions": len(records), "counts": COUNTS,
            "query_ids": [r["q"]["id"] for r in records], "selections_validated": len(selections),
            "model_calls": 0, "credentials_loaded": False}, indent=2))
    else:
        print(json.dumps(execute(args.output), indent=2))


if __name__ == "__main__":
    main()
