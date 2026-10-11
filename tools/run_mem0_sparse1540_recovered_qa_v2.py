"""QA over an admitted, explicit recovery input map; never redirects the v1 run.

`check` is read-only and never constructs a provider. `qa` writes only this new
namespace. An existing coordinator/worker state is never restarted implicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_sparse1540 as base
from jev_mem.mem0_transfer_qa import StructuredMem0QAAPI

OUT = ROOT / "runs/mem0_sparse1540_index_recovery_v2"
POLICY = "mem0-native-sparse1540-orphan-recovery-v2"
ARMS = base.ARMS
REQUIRED_CODE = ("tools/run_mem0_sparse1540_recovered_qa_v2.py", *base.CODE_FILES)


def conv_path(ci):
    return OUT / ("conv_" + str(ci))


def contained(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Input/output pointer leaves designated namespace")
    return path


def signed_read(path, digest):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if base.hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Signed input bytes changed: " + str(path))
    return json.loads(raw)


def input_binding(manifest, input_hash, row, q, retrieval, reference_date):
    return {"input_manifest_sha256": input_hash,
            "source_manifest_sha256": manifest["source_manifest"]["sha256"],
            "bank_sha256": row["bank_sha256"], "retrieval_sha256": row["retrieval_files"][q["id"]],
            "selected_sha256": base.fingerprint(retrieval["selected"]),
            "question_sha256": base.fingerprint(q["question"]),
            "reference_sha256": base.fingerprint(q["reference"]), "reference_date": reference_date}


def validate_row(row, qs):
    ci, arm = row["conversation"], row["arm"]
    recovered = row.get("index_recovery") is not None
    expected_folder = OUT / "recovered" / arm / ("conv_" + str(ci))
    if recovered:
        if ci != 3:
            raise ValueError("Only the two predeclared conv3 orphan recoveries are admitted")
        pointer = row["index_recovery"]
        contained(pointer["path"], expected_folder)
        signed_read(pointer["path"], pointer["sha256"])
        contained(row["bank_path"], expected_folder if arm == ARMS[1] else base.OUT / "conv_3")
        contained(row["retrieval_dir"], expected_folder)
        contained(row["native_ledger_path"], expected_folder / "native")
        contained(row["native_cache_dir"], expected_folder / "native")
    else:
        if ci == 3:
            raise ValueError("Both conv3 arms require declared recovery")
        expected_bank = base.conv_path(ci) / ("session_bank.json" if arm == ARMS[0] else "regroup_bank.json")
        if Path(row["bank_path"]).resolve() != expected_bank.resolve():
            raise ValueError("Unchanged bank must reference its original v1 path")
        if Path(row["retrieval_dir"]).resolve() != (base.conv_path(ci) / "retrieval" / arm).resolve():
            raise ValueError("Unchanged retrieval must reference its original v1 path")
        native_folder = (base.BASE / ("conv_" + str(ci))) if arm == ARMS[0] else base.conv_path(ci) / "native"
        if (Path(row["native_ledger_path"]).resolve() != (native_folder / "api_calls.jsonl").resolve()
                or Path(row["native_cache_dir"]).resolve() != (native_folder / "api_cache").resolve()):
            raise ValueError("Original native provenance paths changed")
    bank = signed_read(row["bank_path"], row["bank_sha256"])
    if (bank.get("conversation") != ci or bank.get("arm") != arm or bank.get("complete") is not True
            or bank.get("schema") != base.POLICY + "-bank" or bank.get("source_dataset_sha256") != base.SOURCE_HASH):
        raise ValueError("Mapped bank identity/completeness differs")
    base.vector_ok(bank["facts"])
    for fact in bank["facts"]:
        if (not isinstance(fact.get("text"), str) or not fact["text"].strip()
                or date.fromisoformat(fact["observation_date"]).isoformat() != fact["observation_date"]):
            raise ValueError("Invalid canonical fact text/date")
    expected_ids = {q["id"] for q in qs}
    if set(row["retrieval_files"]) != expected_ids:
        raise ValueError("Mapped retrieval IDs differ from complete conversation cohort")
    directory = Path(row["retrieval_dir"]).resolve()
    if {p.stem for p in directory.glob("*.json")} != expected_ids:
        raise ValueError("Missing/extra retrieval file")
    retrievals = {}
    for q in qs:
        retrieval = signed_read(directory / (q["id"] + ".json"), row["retrieval_files"][q["id"]])
        base.validate_retrieval(retrieval, q, arm, bank, row["bank_sha256"])
        selected = retrieval["selected"]
        if (len(selected) != 20 or len({f["id"] for f in selected}) != 20
                or len({" ".join(f["text"].split()) for f in selected}) != 20):
            raise ValueError("Both arms require twenty unique canonical evidence facts")
        retrievals[q["id"]] = retrieval
    return {"row": row, "bank": bank, "retrievals": retrievals}


def verify_inputs(ci=None):
    path = OUT / "input_manifest.json"
    digest = base.sha(path)
    manifest = signed_read(path, digest)
    if manifest.get("policy") != POLICY:
        raise ValueError("Wrong recovery policy")
    source_pointer = manifest["source_manifest"]
    if Path(source_pointer["path"]).resolve() != (base.OUT / "manifest.json").resolve():
        raise ValueError("Wrong frozen source manifest path")
    source = signed_read(source_pointer["path"], source_pointer["sha256"])
    # This verifies the old code/dependencies/source population without opening
    # a native store or claiming that the failed original write gate succeeded.
    if source != base.verify_manifest(ci):
        raise ValueError("Frozen source protocol differs")
    policy_pointer = manifest["recovery_policy"]
    signed_read(policy_pointer["path"], policy_pointer["sha256"])
    if manifest.get("models") != source["models"]:
        raise ValueError("Recovery cannot change models")
    if any(manifest.get(k) != v for k, v in {"provider_retries": 2, "conversation_workers": 8,
                                            "query_workers": 4, "final_unique_facts": 20}.items()):
        raise ValueError("Frozen QA resource/protocol settings changed")
    codes = manifest["code_sha256"]
    if not set(REQUIRED_CODE) <= set(codes):
        raise ValueError("Missing frozen QA dependencies")
    for relative, expected in codes.items():
        code = contained(ROOT / relative, ROOT)
        if base.sha(code) != expected:
            raise ValueError("Frozen recovery code changed: " + relative)
    qs = base.questions()
    if manifest.get("question_ids") != [q["id"] for q in qs] or len(qs) != 1540:
        raise ValueError("Fixed1540 cohort changed")
    rows = manifest["rows"]
    if (len(rows) != 20 or {(r["conversation"], r["arm"]) for r in rows}
            != {(i, a) for i in range(10) for a in ARMS}
            or any(type(r["conversation"]) is not int for r in rows)):
        raise ValueError("Input map must contain exactly twenty unique arm/conversation rows")
    audit_pointer = manifest["input_audit"]
    audit = signed_read(audit_pointer["path"], audit_pointer["sha256"])
    expected_audit = {"policy": POLICY, "admitted": True,
        "source_manifest_sha256": source_pointer["sha256"], "recovery_policy_sha256": policy_pointer["sha256"],
        "rows_sha256": base.fingerprint(rows), "question_ids_sha256": base.fingerprint(manifest["question_ids"]),
        "model_calls": 0}
    if any(type(audit.get(k)) is not type(v) or audit.get(k) != v for k, v in expected_audit.items()):
        raise ValueError("Independent complete input audit did not admit this exact map")
    audited_files = audit.get("audited_files")
    if not isinstance(audited_files, dict) or not audited_files:
        raise ValueError("Independent audit must bind its actual evidence files")
    for evidence_path, evidence_sha in audited_files.items():
        if base.sha(evidence_path) != evidence_sha:
            raise ValueError("Independent audit evidence changed: " + evidence_path)
    loaded = {}
    for row in rows:
        if ci is None or row["conversation"] == ci:
            loaded[(row["conversation"], row["arm"])] = validate_row(row, [q for q in qs if q["conversation"] == row["conversation"]])
    for evidence_path, evidence_sha in audited_files.items():
        if base.sha(evidence_path) != evidence_sha:
            raise ValueError("Independent audit evidence changed during validation: " + evidence_path)
    if base.sha(path) != digest:
        raise ValueError("Input manifest changed during validation")
    return manifest, digest, loaded


@contextmanager
def namespace_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def pristine(folder, state_path):
    if state_path.exists() or any((folder / name).exists() for name in ("qa", "qa_api")):
        raise ValueError("Prior QA state/output/attempt exists; no automatic restart or redispatch")


def qa_conv(ci, input_hash):
    folder = conv_path(ci)
    with namespace_lock(folder / "qa_worker.lock"):
        manifest, digest, loaded = verify_inputs(ci)
        if digest != input_hash:
            raise ValueError("Worker input manifest differs from admitted coordinator")
        state_path = folder / "qa_status.json"
        pristine(folder, state_path)
        qs = base.questions(ci)
        reference_date = list(base.session_envelopes(base.load_data()[ci]))[-1]["observation_date"]
        state = {"policy": POLICY, "stage": "qa", "conversation": ci, "pid": os.getpid(),
            "input_manifest_sha256": digest, "status": "running", "started_utc": base.now(),
            "expected": len(qs) * 2, "done": 0, "completed": 0, "failed": 0,
            "arms": {a: {"completed": 0, "failed": 0} for a in ARMS}}
        def save_state():
            state["updated_utc"] = base.now()
            base.publish_json(state_path, state)
        save_state()  # Durable before constructing any credential-bearing client.
        provider = None
        try:
            provider = base.ModelAPI(folder / "qa_api", retries=2,
                context={"architecture": POLICY, "conversation": ci, "phase": "qa"})
            def one(q, arm):
                entry = loaded[(ci, arm)]
                selected = entry["retrievals"][q["id"]]["selected"]
                binding = input_binding(manifest, digest, entry["row"], q, entry["retrievals"][q["id"]], reference_date)
                api = StructuredMem0QAAPI(provider.scoped(query_id=q["id"], method=arm))
                started = base.now(); result = error = None
                try:
                    result = base.native.answer_and_judge(api, q["question"], q["reference"], selected, reference_date=reference_date)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": provider._redact(str(exc))[:1500]}
                value = base._sealed({"schema": POLICY + "-qa", "query_id": q["id"], "conversation": ci,
                    "category": q["category"], "arm": arm, "binding": binding,
                    "status": "complete" if result is not None else "failed", "result": result, "error": error,
                    "selected_ids": [f["id"] for f in selected], "api_events": api.events,
                    "started_utc": started, "finished_utc": base.now()})
                base.save_once(folder / "qa" / arm / (q["id"] + ".json"), value)
                return value
            with ThreadPoolExecutor(max_workers=4) as pool:
                tasks = [pool.submit(one, q, arm) for q in qs for arm in ARMS]
                for future in as_completed(tasks):
                    row = future.result()
                    key = "completed" if row["status"] == "complete" else "failed"
                    state[key] += 1; state["arms"][row["arm"]][key] += 1; state["done"] += 1
                    state["last_query_id"] = row["query_id"]; save_state()
            if base.sha(OUT / "input_manifest.json") != digest:
                raise ValueError("Input manifest changed during QA")
            state["status"] = "complete" if not state["failed"] else "requires_review"
        except BaseException as exc:
            state.update(status="blocked", error={"type": type(exc).__name__,
                "message": provider._redact(str(exc))[:1500] if provider else "QA provider setup/worker interrupted"})
        finally:
            state["finished_utc"] = base.now(); save_state()
        return state


def launch():
    with namespace_lock(OUT / "qa_coordinator.lock"):
        status_path = OUT / "qa_coordinator_status.json"
        if status_path.exists():
            raise ValueError("Prior coordinator state exists; no implicit restart")
        manifest, digest, _ = verify_inputs()
        for ci in range(10):
            pristine(conv_path(ci), conv_path(ci) / "qa_status.json")
        state = {"policy": POLICY, "stage": "qa", "pid": os.getpid(), "status": "running",
            "input_manifest_sha256": digest, "expected": 3080, "workers": 8, "query_workers": 4,
            "started_utc": base.now(), "conversations": list(range(10)), "finished": [], "failed": []}
        base.publish_json(status_path, state)
        try:
            with ProcessPoolExecutor(max_workers=8) as pool:
                tasks = {pool.submit(qa_conv, ci, digest): ci for ci in range(10)}
                for future in as_completed(tasks):
                    ci = tasks[future]
                    try:
                        result = future.result()
                        expected = 2 * len(base.questions(ci))
                        complete = result["status"] == "complete" and result["done"] == result["expected"] == expected
                    except Exception as exc:
                        complete = False
                        state.setdefault("errors", []).append({"conversation": ci, "type": type(exc).__name__})
                    state["finished"].append(ci)
                    if not complete:
                        state["failed"].append(ci)
                    state["updated_utc"] = base.now(); base.publish_json(status_path, state)
            state["status"] = "complete" if not state["failed"] else "requires_review"
        except BaseException as exc:
            state.update(status="interrupted", error_type=type(exc).__name__)
            raise
        finally:
            state["finished_utc"] = base.now(); state["updated_utc"] = base.now()
            base.publish_json(status_path, state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "qa"))
    args = parser.parse_args(argv)
    if args.stage == "check":
        manifest, digest, loaded = verify_inputs()
        print(json.dumps({"policy": POLICY, "input_manifest_sha256": digest,
            "rows": len(loaded), "questions_per_arm": len(manifest["question_ids"]), "model_calls": 0}))
        return 0
    state = launch()
    print(json.dumps(state, indent=2))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
