"""Frozen, durable identity-only and bounded-refinement grouping; no writer or QA."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import refined_event_grouping_v1 as core
from jev_mem import global_segment_grouping as grouping
from jev_mem.providers import load_credentials
from jev_mem.sdk20_cached_api import CachedLegacyAPI
from tools.run_global_segment_grouping import (
    read, publish_json, pointer, checked_pointer, save_once, verify_seal, sha, now,
    exclusive_driver, load_plan, frozen,
)

CODE = ("jev_mem/refined_event_grouping_v1.py", "tools/run_refined_event_grouping_v1.py",
        "jev_mem/global_segment_grouping.py", "jev_mem/sparse_topic_pipeline.py",
        "jev_mem/topic_boundaries.py", "jev_mem/topic_grouping.py",
        "jev_mem/sdk20_cached_api.py", "jev_mem/providers.py",
        "tools/run_global_segment_grouping.py", "tools/run_pilot400_experiments_v2.py")
ARMS = ("identity_only", "identity_and_bounded_refinement")
PHASE = "refined_event_grouping_v1"


def prepare(run, source):
    run, source = Path(run).resolve(), Path(source).resolve()
    if run.exists():
        raise FileExistsError("A new run directory is required")
    parent = frozen(source)
    jobs = []
    for ci in range(10):
        plan = load_plan(parent, ci)
        embeddings = checked_pointer(plan["source_embeddings"])
        if (embeddings.get("status") != "complete"
                or grouping.fingerprint(embeddings["vectors"]) != plan["raw_turn_embedding_sha256"]):
            raise ValueError("The frozen source embeddings do not match the plan")
        jobs.append({"conversation": ci, "source_plan": parent["plans"][str(ci)],
                     "source_embeddings": plan["source_embeddings"],
                     "source_segments": len(plan["segments"]),
                     "initial_candidate_pairs": len(plan["candidate_pairs"])})
    run.mkdir(parents=True)
    save_once(run / "manifest.json", {
        "schema": core.VERSION, "created_utc": now(), "source_manifest": pointer(source / "manifest.json"),
        "jobs": jobs, "parameters": core.DEFAULTS, "pair_criteria": core.PAIR_CRITERIA,
        "arms": ARMS, "resources": {"conversation_workers": 8, "pairs_sequential_per_conversation": True,
                                     "jev_shared_physical_requests_per_second": 15, "transport_retries": 2},
        "model": core.MODEL, "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "source_policy": "Only frozen raw turns/segments/embeddings; no QA, gold or extracted facts",
        "cache_policy": "Exact complete protocol/model/stage/state/criteria event replay, or same-run full provider request cache",
        "failure_policy": "Retain failed/unknown conversation intent; no implicit logical redispatch",
        "identity_arm": "Same top5 graph, new three-criterion scoring; partial overlap diagnostic only",
        "refined_arm": "At most two extra gap checks per original session, then new top5 graph and identical identity scoring",
        "writer_invoked": False, "model_calls_at_prepare": 0,
    })
    return check(run)


def _manifest(run):
    manifest = verify_seal(read(run / "manifest.json"))
    if (manifest["schema"] != core.VERSION or manifest["parameters"] != core.DEFAULTS
            or manifest["pair_criteria"] != core.PAIR_CRITERIA
            or manifest["arms"] != list(ARMS) or manifest["model"] != core.MODEL):
        raise ValueError("Frozen refinement protocol changed")
    if manifest["code_sha256"] != {name: sha(ROOT / name) for name in CODE}:
        raise ValueError("Frozen code changed; use a new version")
    source = checked_pointer(manifest["source_manifest"])
    if frozen(Path(manifest["source_manifest"]["path"]).parent) != source:
        raise ValueError("Frozen parent protocol changed")
    if [job["conversation"] for job in manifest["jobs"]] != list(range(10)):
        raise ValueError("Exactly the ten original conversations are required")
    return manifest, source


def check(run):
    """Read-only source/config checks. Does not construct any API or read a vault."""
    run = Path(run).resolve()
    manifest, source = _manifest(run)
    for job in manifest["jobs"]:
        plan = load_plan(source, job["conversation"])
        if job["source_plan"] != source["plans"][str(job["conversation"])]:
            raise ValueError("Job source pointer differs")
        embeddings = checked_pointer(job["source_embeddings"])
        if grouping.fingerprint(embeddings["vectors"]) != plan["raw_turn_embedding_sha256"]:
            raise ValueError("Source embedding signature changed")
    return {"status": "prepared", "run_dir": str(run), "manifest": pointer(run / "manifest.json"),
            "conversations": 10, "arms": list(ARMS), "source_segments": sum(j["source_segments"] for j in manifest["jobs"]),
            "initial_candidate_pairs": sum(j["initial_candidate_pairs"] for j in manifest["jobs"]),
            "model_calls_during_check": 0}


def _provider_signature(event):
    return {"provider": "jev", "model": core.MODEL, "operation": "systemone", "body": {
        "model": core.MODEL, "state": event["state"],
        "questions": {key: {"type": "noul", "instructions": value} for key, value in event["criteria"].items()}}}


def validate_event(path, folder, manifest_sha, ci):
    """Bind a completed core event to its exact persisted provider raw response."""
    value = verify_seal(read(path))
    event = value["event"]
    if value["manifest_sha256"] != manifest_sha or value["conversation"] != ci:
        raise ValueError("Event manifest/conversation binding differs")
    signature = {key: event[key] for key in ("protocol", "model", "stage", "state", "criteria")}
    if (event["status"] != "complete" or event["protocol"] != core.VERSION or event["model"] != core.MODEL
            or event["request_sha256"] != grouping.fingerprint(signature)
            or Path(path).name != event["request_sha256"] + ".json"):
        raise ValueError("Incomplete or changed core event cannot be replayed")
    expected = _provider_signature(event)
    request_hash = hashlib.sha256(json.dumps(expected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    events = value["api_events"]
    if len(events) != 1:
        raise ValueError("One provider logical event is required for each persisted judgment")
    api_event = events[0]
    expected_cache = (folder / "api/api_cache" / (request_hash + ".json")).resolve()
    if (api_event["request_hash"] != request_hash or api_event["error_type"] is not None
            or Path(api_event["cache_path"]).resolve() != expected_cache
            or api_event["provider"] != "jev" or api_event["model"] != core.MODEL
            or api_event["operation"] != "systemone" or api_event["stage"] != event["stage"]
            or api_event["context"].get("conversation") != ci
            or api_event["context"].get("phase") != PHASE
            or api_event["context"].get("logical_request_sha256") != event["request_sha256"]):
        raise ValueError("Provider event binding differs")
    raw = checked_pointer({"path": api_event["cache_path"], "sha256": api_event["cache_sha256"]})
    if raw["request"] != expected:
        raise ValueError("Provider raw request changed")
    response = raw["response"]
    for key in ("model", "modelVersion"):
        if response.get(key) is not None and response[key] != core.MODEL:
            raise ValueError("Resolved JEV model differs")
    scores = {key: grouping.score_value(response["answers"][key]["noul"]) for key in event["criteria"]}
    if scores != event["scores"]:
        raise ValueError("Persisted scores differ from original raw response")
    return event


def run_conversation(run, ci, manifest, parent, keys, update):
    folder = run / f"conv_{ci}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha(run / "manifest.json")
    terminal = folder / "result.json"
    if terminal.exists():
        value = verify_seal(read(terminal))
        if value.get("status") != "complete" or value["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Existing failed/changed outcome is retained")
        for ref in value["artifacts"].values():
            verify_seal(checked_pointer(ref))
        for ref in value["events"]:
            checked_pointer(ref)
            validate_event(ref["path"], folder, manifest_sha, ci)
        update(ci, status="complete", reused_terminal=True, **value["counts"])
        return value
    intent = folder / "conversation_intent.json"
    if intent.exists():
        raise RuntimeError("Prior failed or unknown conversation is preserved; reviewed recovery must use a new version")
    plan = load_plan(parent, ci)
    embeddings = checked_pointer(plan["source_embeddings"])["vectors"]
    save_once(intent, {"manifest_sha256": manifest_sha, "conversation": ci, "started_utc": now(),
                       "source_plan": parent["plans"][str(ci)]})
    api = CachedLegacyAPI(folder / "api", keys=keys, retries=2, cache_roots=(),
                          context={"phase": PHASE, "conversation": ci})
    active = {"arm": "identity_only", "event": None, "api_events": [], "completed_requests": 0,
              "replay_events": 0, "boundary_requests": 0}

    class Hook:
        def score_many(self, state, criteria, *, stage):
            event = active["event"]
            if event is None or (state, criteria, stage) != (event["state"], event["criteria"], event["stage"]):
                raise ValueError("Unbound judge hook request")
            scoped = api.scoped(arm=active["arm"], logical_request_sha256=event["request_sha256"],
                                **event["identity"])
            try:
                return scoped.score_many(state, criteria, stage=stage)
            finally:
                active["api_events"] = scoped.events

    def progress(event):
        key = event["request_sha256"]
        path = folder / "events" / (key + ".json")
        request_intent = folder / "event_intents" / (key + ".json")
        if event["status"] == "started":
            active["event"], active["api_events"] = event, []
            if path.exists():
                saved = validate_event(path, folder, manifest_sha, ci)
                if any(saved[k] != event[k] for k in ("protocol", "model", "stage", "state", "criteria")):
                    raise ValueError("Completed event request changed")
            elif request_intent.exists():
                raise RuntimeError("Unknown prior request retained without implicit redispatch")
            else:
                save_once(request_intent, {"manifest_sha256": manifest_sha, "conversation": ci,
                                          "started_utc": now(), "event": event})
        elif event["status"] == "complete":
            if path.exists():
                saved = validate_event(path, folder, manifest_sha, ci)
                if event["resolution"] != "exact_completed_event_replay" or saved["scores"] != event["scores"]:
                    raise ValueError("A completed judgment cannot be overwritten or replaced")
                active["replay_events"] += 1
            else:
                if event["resolution"] != "judge_hook":
                    raise ValueError("Replay without a local source raw record is forbidden")
                save_once(path, {"manifest_sha256": manifest_sha, "conversation": ci,
                                 "finished_utc": now(), "event": event, "api_events": active["api_events"]})
                validate_event(path, folder, manifest_sha, ci)
                active["completed_requests"] += 1
                if event["stage"] == core.BOUNDARY_STAGE:
                    active["boundary_requests"] += 1
        else:
            failure = folder / "failed_events" / (key + ".json")
            if not failure.exists():
                save_once(failure, {"manifest_sha256": manifest_sha, "conversation": ci,
                                   "finished_utc": now(), "event": event, "api_events": active["api_events"]})
        update(ci, status="running", arm=active["arm"], last_stage=event["stage"],
               last_request_sha256=key, last_event_status=event["status"],
               completed_requests=active["completed_requests"], exact_event_replays=active["replay_events"],
               boundary_requests=active["boundary_requests"], last_activity_utc=now())

    try:
        identity = core.run_identity_only(plan, judge=Hook(), on_progress=progress)
        identity_path = folder / "identity_only/grouping.json"
        save_once(identity_path, identity)
        active["arm"] = ARMS[1]
        refined = core.run_refined_grouping(plan, identity, embeddings, judge=Hook(), on_progress=progress)
        refined_path = folder / "identity_and_bounded_refinement/grouping.json"
        save_once(refined_path, refined)
        events = [pointer(p) for p in sorted((folder / "events").glob("*.json"))]
        for ref in events:
            validate_event(ref["path"], folder, manifest_sha, ci)
        counts = {"source_segments": len(plan["segments"]), "identity_groups": len(identity["groups"]),
                  "refined_segments": len(refined["segments"]), "refined_groups": len(refined["groups"]),
                  "added_cuts": len(refined["refinement"]["trace"]["added_cut_indices"]),
                  "unique_completed_requests": len(events), "exact_event_replays": active["replay_events"],
                  "boundary_requests": active["boundary_requests"]}
        result = {"schema": core.VERSION, "status": "complete", "conversation": ci,
                  "manifest_sha256": manifest_sha, "finished_utc": now(), "counts": counts,
                  "artifacts": {ARMS[0]: pointer(identity_path), ARMS[1]: pointer(refined_path)}, "events": events,
                  "writer_invoked": False, "qa_visible": False}
        save_once(terminal, result)
        update(ci, status="complete", **counts)
        return result
    except Exception as error:
        failed = {"schema": core.VERSION, "status": "failed", "conversation": ci,
                  "manifest_sha256": manifest_sha, "finished_utc": now(), "arm": active["arm"],
                  "error_type": type(error).__name__, "error_message": api._redact(error)[:1000],
                  "partial_trace": error.trace if isinstance(error, core.RefinedGroupingError) else None}
        if not terminal.exists():
            save_once(terminal, failed)
        update(ci, status="failed", arm=active["arm"], error_type=type(error).__name__)
        raise


def run_experiment(run, conversations):
    run = Path(run).resolve()
    manifest, parent = _manifest(run)
    if not conversations or len(set(conversations)) != len(conversations) or any(ci not in range(10) for ci in conversations):
        raise ValueError("Unique original conversation IDs 0 through 9 are required")
    state = {"schema": core.VERSION, "status": "running", "pid": os.getpid(),
             "started_utc": now(), "updated_utc": now(), "manifest_sha256": sha(run / "manifest.json"),
             "conversations": {str(ci): {"status": "pending"} for ci in conversations}}
    lock = threading.Lock()
    def update(ci=None, **values):
        with lock:
            target = state if ci is None else state["conversations"][str(ci)]
            target.update(values)
            state["updated_utc"] = now()
            publish_json(run / "run_status.json", state)
    with exclusive_driver(run):
        vault = Path(r"C:\Users\86139\Desktop\jev\experiments\live\credential_store.py")
        if not vault.is_file():
            raise RuntimeError("Existing DPAPI vault is required; no environment fallback")
        keys = load_credentials()
        update()
        failures = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(run_conversation, run, ci, manifest, parent, keys, update): ci for ci in conversations}
            for future in as_completed(futures):
                ci = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append({"conversation": ci, "error_type": type(error).__name__})
                    update(ci, status="failed", error_type=type(error).__name__)
        update(status="failed" if failures else "complete", finished_utc=now(), failures=failures)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "run"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "runs/global_segment_grouping_v1")
    parser.add_argument("--conv", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        result = prepare(args.run_dir, args.source_dir)
    elif args.stage == "check":
        result = check(args.run_dir)
    else:
        result = run_experiment(args.run_dir, args.conv)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
