"""Prepare global within-conversation candidates offline, then explicitly score.

prepare/export-review never construct an API client. score is a separate command.
No writer, QA, gold, thresholds learned from labels, or source-run edits are used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import global_segment_grouping as grouping
from jev_mem.data import SOURCE_HASH
from jev_mem.topic_grouping import raw_turns_from_locomo
from jev_mem.sdk20_cached_api import CachedLegacyAPI
from jev_mem.providers import load_credentials
from tools.run_pilot400_experiments_v2 import read, publish_json

FILES = ("jev_mem/global_segment_grouping.py", "tools/run_global_segment_grouping.py",
         "jev_mem/sdk20_cached_api.py", "jev_mem/providers.py", "jev_mem/sparse_topic_pipeline.py",
         "jev_mem/topic_grouping.py", "jev_mem/data.py", "tools/run_pilot400_experiments_v2.py")
PARAMETERS = {"top_k": 5, "same_event_threshold": .7, "supplemental_pairs_per_segment": 2,
              "max_pair_characters": 200000, "conversation_workers": 8, "pair_workers": 4,
              "jev_model": "jev-1.13.0", "transport_retries": 2,
              "pair_input": "full_unmodified_segments_no_truncation", "writer_capacity_gate": "not_in_this_stage"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def checked_pointer(value):
    if sha(value["path"]) != value["sha256"]:
        raise ValueError("Immutable source pointer changed: " + value["path"])
    return read(value["path"])


def seal(value):
    return {**value, "content_sha256": grouping.fingerprint(value)}


def verify_seal(value):
    if grouping.fingerprint({k:v for k,v in value.items() if k != "content_sha256"}) != value.get("content_sha256"):
        raise ValueError("Content seal mismatch")
    return value


def save_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(seal(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare(run, source):
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Prepare requires a new empty run directory")
    dataset = ROOT/"data/locomo10.json"
    if sha(dataset) != SOURCE_HASH:
        raise ValueError("Frozen LoCoMo source changed")
    conversations = [item["conversation"] for item in read(dataset)]
    plans = []
    for ci in range(10):
        folder = source / f"conv_{ci}"
        gp, ep = folder/"banks/regroup/grouping.json", folder/"raw_embeddings.json"
        original, embeddings = verify_seal(read(gp)), verify_seal(read(ep))
        if original["status"] != "complete" or embeddings["status"] != "complete":
            raise ValueError("Only completed source grouping/embedding artifacts may be reused")
        if original["binding"]["conversation"] != ci or embeddings["binding"]["conversation"] != ci:
            raise ValueError("Source conversation binding mismatch")
        plan = grouping.prepare_candidates(ci, original["segments"], embeddings["vectors"],
                    top_k=PARAMETERS["top_k"], max_pair_characters=PARAMETERS["max_pair_characters"])
        if [turn for s in plan["segments"] for turn in s["turns"]] != raw_turns_from_locomo(conversations[ci]):
            raise ValueError("Segments must preserve the exact original conversation fields and source order")
        if any(v["binding"]["raw_source_sha256"] != plan["raw_source_sha256"] for v in (original, embeddings)):
            raise ValueError("Segment/embedding raw turn provenance mismatch")
        plan.update(source_grouping=pointer(gp), source_embeddings=pointer(ep),
                    source_dataset=pointer(dataset), source_order_checked_against_raw_conversation=True,
                    initial_judgments_started=False, qa_or_gold_used_or_forwarded=False)
        plans.append(plan)
    if sum(len(p["segments"]) for p in plans) != 385:
        raise ValueError("This frozen source cohort must contain exactly385 original segments")
    run.mkdir(parents=True, exist_ok=True)
    pointers = {}
    for plan in plans:
        path = run/"conversations"/f"conv_{plan['conversation']}"/"plan.json"
        save_once(path, plan)
        pointers[str(plan["conversation"])] = pointer(path)
    manifest = {"schema": grouping.VERSION, "created_utc": now(), "parameters": PARAMETERS,
                "criterion": grouping.CRITERION, "plans": pointers,
                "code_sha256": {name: sha(ROOT/name) for name in FILES},
                "all_segments": 385, "cross_conversation_pairs": 0,
                "new_embedding_calls": 0, "model_calls_at_prepare": 0,
                "manual_labels_used_to_select_or_score": False}
    save_once(run/"manifest.json", manifest)
    export_reviews(run, range(10))
    return {"prepared": True, "model_calls": 0,
            "conversations": [{"conversation": p["conversation"], **p["coverage"]} for p in plans]}


def frozen(run):
    manifest = verify_seal(read(run/"manifest.json"))
    if manifest["schema"] != grouping.VERSION or manifest["parameters"] != PARAMETERS or manifest["criterion"] != grouping.CRITERION:
        raise ValueError("Frozen global-grouping parameters changed")
    if manifest["code_sha256"] != {name:sha(ROOT/name) for name in FILES}:
        raise ValueError("Frozen code changed; do not resume this version")
    return manifest


def load_plan(manifest, ci):
    plan = verify_seal(checked_pointer(manifest["plans"][str(ci)]))
    if plan["conversation"] != ci:
        raise ValueError("Plan conversation differs")
    checked_pointer(plan["source_grouping"])
    checked_pointer(plan["source_embeddings"])
    return plan


@contextmanager
def exclusive_driver(run):
    path = run/"driver.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_score(row, plan, manifest, left, right):
    verify_seal(row)
    if row["plan_sha256"] != manifest["plans"][str(plan["conversation"])]["sha256"] or row["pair_id"] != grouping.pair_key(left,right):
        raise ValueError("Pair result binding differs")
    if row["status"] != "complete":
        raise RuntimeError("A prior failed pair is preserved; review before a separate recovery version")
    state = grouping.pair_state(plan,left,right)
    expected = {"provider":"jev", "model":grouping.MODEL, "operation":"systemone", "body":{
        "model":grouping.MODEL, "state":state, "questions":{"same_concrete_event":{"type":"noul","instructions":grouping.CRITERION}}}}
    events = row["api_events"]
    if len(events) != 1:
        raise ValueError("Exactly one logical pair request event expected")
    event = events[0]
    raw = checked_pointer({"path":event["cache_path"], "sha256":event["cache_sha256"]})
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if raw["request"] != expected or event["request_hash"] != digest or event["error_type"] is not None:
        raise ValueError("Pair raw request/event changed")
    expected_cache = Path(manifest["plans"][str(plan["conversation"])]["path"]).parents[2]/"api/api_cache"/(digest+".json")
    expected_context = {"phase":"global_segment_grouping", "conversation":plan["conversation"],
                        "pair_id":row["pair_id"], "pair_kind":row["kind"]}
    if (Path(event["cache_path"]).resolve() != expected_cache.resolve() or event["context"] != expected_context
            or event["provider"] != "jev" or event["model"] != grouping.MODEL
            or event["operation"] != "systemone" or event["stage"] != "global_segment_same_event"):
        raise ValueError("Pair API event context or namespace differs")
    for name in ("model", "modelVersion"):
        if raw["response"].get(name) is not None and raw["response"][name] != grouping.MODEL:
            raise ValueError("Resolved JEV model differs from the frozen version")
    score = grouping.score_value(raw["response"]["answers"]["same_concrete_event"]["noul"])
    if score != row["score"] or row["state_sha256"] != hashlib.sha256(state.encode()).hexdigest():
        raise ValueError("Pair score differs from its raw response")
    return score


def score_run(run, conversations):
    manifest = frozen(run)
    lock = threading.Lock()
    state = {"stage":"global_segment_pairs", "pid":os.getpid(), "started_utc":now(), "updated_utc":now(),
             "status":"running", "requested_conversations":conversations,
             "conversations":{str(c):{"status":"pending"} for c in conversations},
             "manifest_sha256":sha(run/"manifest.json")}
    def update(ci=None, **values):
        with lock:
            target = state if ci is None else state["conversations"][str(ci)]
            target.update(values)
            state["updated_utc"] = now()
            publish_json(run/"score_status.json",state)
    with exclusive_driver(run):
        if not Path(r"C:\Users\86139\Desktop\jev\experiments\live\credential_store.py").is_file():
            raise RuntimeError("Existing DPAPI credential loader required; environment fallback is disabled")
        keys = load_credentials()
        api = CachedLegacyAPI(run/"api", keys=keys, cache_roots=(), retries=2, context={"phase":"global_segment_grouping"})
        update()
        def conv(ci):
            plan = load_plan(manifest,ci)
            folder = run/"conversations"/f"conv_{ci}"
            terminal = folder/"grouping.json"
            if terminal.exists():
                old = verify_seal(read(terminal))
                if old["status"] != "complete" or old["plan_sha256"] != manifest["plans"][str(ci)]["sha256"]:
                    raise ValueError("Existing terminal grouping binding differs")
                grouping.verify_partition([t for s in plan["segments"] for t in s["turns"]],old["groups"])
                expected_segments={s["segment_id"]:s for s in plan["segments"]}
                member_ids=[sid for g in old["groups"] for sid in g["segment_ids"]]
                if len(member_ids)!=len(expected_segments) or set(member_ids)!=set(expected_segments):
                    raise ValueError("Terminal grouping does not assign every atomic segment exactly once")
                for key,value in old["trace"]["all_observed_pair_scores"].items():
                    row = verify_seal(read(folder/"pairs"/(key+".json")))
                    if validate_score(row,plan,manifest,row["left_index"],row["right_index"]) != value:
                        raise ValueError("Terminal grouping score differs")
                update(ci,status="complete",reused_terminal=True,groups=len(old["groups"]))
                return
            progress = {"initial_done":0,"initial_expected":len(plan["candidate_pairs"]),"supplemental_done":0}
            progress_lock = threading.Lock()
            update(ci,status="running",**progress)
            def pair(left,right,kind):
                key = grouping.pair_key(left,right)
                path = folder/"pairs"/(key+".json")
                if path.exists():
                    value = validate_score(read(path),plan,manifest,left,right)
                else:
                    scoped = api.scoped(conversation=ci,pair_id=key,pair_kind=kind)
                    row = {"schema":grouping.VERSION, "conversation":ci, "pair_id":key,
                           "left_index":min(left,right),"right_index":max(left,right),
                           "plan_sha256":manifest["plans"][str(ci)]["sha256"],"kind":kind,"started_utc":now()}
                    try:
                        result = grouping.judge_pair(plan,left,right,scoped)
                        row.update(result,status="complete",error=None)
                    except Exception as exc:
                        row.update(status="failed",score=None,error={"type":type(exc).__name__,"message":scoped._redact(exc)})
                    row.update(api_events=scoped.events,finished_utc=now())
                    save_once(path,row)
                    value = validate_score(read(path),plan,manifest,left,right)
                with progress_lock:
                    progress["initial_done" if kind=="initial" else "supplemental_done"] += 1
                    update(ci,**progress)
                return value
            initial = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                tasks = {pool.submit(pair,e["left_index"],e["right_index"],"initial"):e["pair_id"] for e in plan["candidate_pairs"]}
                errors=[]
                for future in as_completed(tasks):
                    try: initial[tasks[future]]=future.result()
                    except Exception as exc: errors.append(exc)
                if errors:
                    raise RuntimeError(f"{len(errors)} initial pair failures preserved; no clustering") from errors[0]
            result = grouping.complete_link_groups(plan,initial,
                supplemental=lambda left,right:pair(left,right,"complete_link_supplemental"),
                threshold=.7,supplemental_budget=2*len(plan["segments"]))
            result.update(status="complete",plan_sha256=manifest["plans"][str(ci)]["sha256"],finished_utc=now())
            save_once(terminal,result)
            update(ci,status="complete",groups=len(result["groups"]),merged_segment_reduction=result["trace"]["merged_segment_reduction"],**progress)
        errors=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks={pool.submit(conv,ci):ci for ci in conversations}
            for future in as_completed(tasks):
                ci=tasks[future]
                try: future.result()
                except Exception as exc:
                    error={"type":type(exc).__name__,"message":api._redact(exc)}
                    errors.append({"conversation":ci,**error})
                    update(ci,status="failed",error=error)
        update(status="failed" if errors else "complete",finished_utc=now(),errors=errors)
    export_reviews(run,conversations)
    return state


def export_reviews(run, conversations):
    manifest=frozen(run)
    output=[]
    for ci in conversations:
        plan=load_plan(manifest,ci)
        folder=run/"conversations"/f"conv_{ci}"
        pairs={e["pair_id"]:dict(e) for e in plan["candidate_pairs"]}
        for path in sorted((folder/"pairs").glob("*.json")):
            row=verify_seal(read(path));key=row["pair_id"]
            pairs.setdefault(key,{"pair_id":key,"left_index":row["left_index"],"right_index":row["right_index"],
                                  "cosine":plan["cosine_matrix"][row["left_index"]][row["right_index"]],"proposed_by":[]})
            pairs[key].update(status=row["status"],score=row.get("score"),pair_result=pointer(path),kind=row["kind"])
        index=[]
        for key,pair in sorted(pairs.items(),key=lambda kv:(-(kv[1].get("score") or -1),-kv[1]["cosine"],kv[0])):
            left,right=pair["left_index"],pair["right_index"]
            path=folder/"review_pairs"/(key+".md")
            path.parent.mkdir(parents=True,exist_ok=True)
            # Generated source display only; human decisions go in a separate file.
            segments=[plan["segments"][i] for i in (left,right)]
            lines=[f"# Conversation {ci}: {key}","",f"Cosine: {pair['cosine']:.8f}; JEV score: {pair.get('score','not scored')}",
                   "", "人工核查：是否同一具体事件/明确持续线索；指出支持或冲突的原turn ID。不要查看QA/gold。",
                   "全部原turn如下。评语/人工标签请存manual_review.json，避免生成式导出覆盖批注。",""]
            for seg in segments:
                content=json.dumps(seg["turns"],ensure_ascii=False,indent=2)
                fence="`"*max(3,max((len(s) for s in content.split() if s and set(s)=={'`'}),default=0)+1)
                lines += [f"## {seg['segment_id']}","",fence+"json",content,fence,""]
            path.write_text("\n".join(lines),encoding="utf-8")
            index.append({**pair,"source_review":str(path.resolve()),"manual_label":None,
                          "same_event_threshold":.7,"full_original_turns_present":True})
        publish_json(folder/"review_index.json",{"conversation":ci,"plan":manifest["plans"][str(ci)],
                     "manual_labels_are_external_not_training_or_threshold_tuning":True,"pairs":index})
        output.append({"conversation":ci,"review_pairs":len(index)})
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","score","export-review"))
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"runs/pilot400_modules_v2/conversations")
    parser.add_argument("--conv",nargs="+",type=int,default=list(range(10)))
    args=parser.parse_args(argv)
    if len(set(args.conv))!=len(args.conv) or any(ci not in range(10) for ci in args.conv):
        parser.error("Unique conversation IDs0–9 required")
    run=args.run_dir.resolve()
    if args.stage=="prepare": result=prepare(run,args.source_dir.resolve())
    elif args.stage=="score": result=score_run(run,args.conv)
    else: result=export_reviews(run,args.conv)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if isinstance(result,dict) and result.get("status")=="failed" else 0


if __name__=="__main__":
    raise SystemExit(main())
