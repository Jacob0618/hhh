"""Offline, terminal-sealed compatibility view of existing plain Judge outputs.

No generation, provider construction, credential access or native client. The
original strict artifacts and scores remain immutable and separately reported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_mem0_regroup_plain_qa_v1 as strict
from tools import seal_mem0_plain_qa_firstpass_v1 as sealing
from tools.replay_plain_judge_compat_v1 import replay_source
from tools.run_sdk20_modules import digest, sha, now, save_new, verify
from jev_mem.judge_terminal_label_compat_v1 import PROTOCOL

SCHEMA = "mem0-plain-judge-label-compat-overlay-v1"
POLICY = {
    "interpretation_protocol": PROTOCOL,
    "planned_total": 3080, "per_arm": 1540,
    "arms": list(strict.ARMS), "original_strict_success_unchanged": True,
    "only_extend_error_type": "InvalidJudgeLabel",
    "require_original_http_200_stop_one_ordinary_text_part": True,
    "allow_string_thoughtSignature_metadata": True,
    "terminal_nonempty_line_exact_lowercase_label": True,
    "all_exact_lowercase_decision_tokens_must_agree": True,
    "opposite_label_in_quoted_explanation_still_rejected": True,
    "reader_and_f1_unchanged": True, "model_calls": 0,
    "post_failure_compatibility_analysis_not_original_strict_success": True,
}
CODE = (
    "tools/recover_mem0_plain_judge_labels_v1.py",
    "tools/replay_plain_judge_compat_v1.py",
    "jev_mem/judge_terminal_label_compat_v1.py",
    "tools/summarize_mem0_regroup_plain_qa_v1.py",
    "tools/seal_mem0_plain_qa_firstpass_v1.py",
    "tools/summarize_mem0_sparse1540.py", "tools/summarize_sdk20_modules.py",
    "tools/summarize_pilot400.py", "tools/run_sdk20_modules.py",
)


def pointer(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def check_pointer(ref):
    path = Path(ref["path"]).resolve()
    if not path.is_file() or sha(path) != ref["sha256"]:
        raise ValueError("Bound source file changed: " + str(path))
    if "bytes" in ref and path.stat().st_size != ref["bytes"]:
        raise ValueError("Bound source file size changed: " + str(path))
    return path


def read_pointer(ref, *, sealed=False):
    path = check_pointer(ref)
    return verify(path) if sealed else json.loads(path.read_text(encoding="utf-8-sig"))


@contextmanager
def offline():
    def forbidden(*args, **kwargs):
        raise AssertionError("Compatibility analysis forbids network and credential access")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                       "requests.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                       "jev_mem.providers.load_credentials"):
            stack.enter_context(patch(target, side_effect=forbidden))
        yield


def validate_namespace(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.parent != ROOT / "runs" or output == source or output.exists():
        raise ValueError("Compatibility output must be a new direct ROOT/runs namespace")
    return output


def validate_population(manifest, seal, report):
    """Bind every fixed slot and strict status before deriving another view."""
    expected = {(ci, arm, qid) for ci in range(10) for arm in strict.ARMS
                for qid in manifest["query_ids"][str(ci)]}
    if len(expected) != 3080 or any(len(manifest["query_ids"][str(ci)]) != len(set(manifest["query_ids"][str(ci)])) for ci in range(10)):
        raise ValueError("Manifest does not define the complete two-arm1540 cohort")
    def indexed(values, arm_key):
        result = {(r["conversation"], r[arm_key], r["query_id"]): r for r in values}
        if len(result) != len(values) or set(result) != expected:
            raise ValueError("Every planned slot must appear exactly once")
        return result
    sealed_slots, strict_slots = indexed(seal["slots"], "arm"), indexed(report["slots"], "method")
    for key, row in strict_slots.items():
        if row.get("errors"):
            raise ValueError("Invalid strict source artifacts cannot be repaired by format compatibility")
        for stage in sealing.STAGES:
            if row[stage + "_status"] != sealed_slots[key]["statuses"][stage]:
                raise ValueError("Strict summary and terminal seal disagree on a stage")
        score = row["scores"]["judge_correct"]
        if (row["judge_status"] == "complete") != (score is not None):
            raise ValueError("Strict Judge validity and score disagree")
    return sealed_slots, strict_slots


def source_admission(seal_path, summary_path):
    """Read-only admission. Full source hashes are checked once per invocation."""
    seal_ref, summary_ref = pointer(seal_path), pointer(summary_path)
    seal, report = read_pointer(seal_ref, sealed=True), read_pointer(summary_ref, sealed=True)
    checked = {}
    def check_once(ref):
        path = Path(ref["path"]).resolve()
        if path not in checked:
            check_pointer(ref)
            checked[path] = {"sha256": ref["sha256"], "bytes": path.stat().st_size}
        elif (checked[path]["sha256"] != ref["sha256"]
              or ("bytes" in ref and checked[path]["bytes"] != ref["bytes"])):
            raise ValueError("Different bindings refer to the same source file")
    run = Path(seal["run_dir"]).resolve()
    if (seal.get("schema") != sealing.SCHEMA or seal.get("status") != "all10_terminal_strict_firstpass_frozen"
            or seal.get("expected_slots") != 3080 or seal.get("slot_count") != 3080
            or not all(seal.get(k) is True for k in ("all_read_file_hashes_unchanged", "original_source_native_unchanged", "strict_judge_failures_preserved"))):
        raise ValueError("A complete immutable strict first-pass seal is required")
    manifest = read_pointer(seal["source_manifest"])
    protocol = manifest["protocol_sha256"]
    if (Path(seal["source_manifest"]["path"]).resolve() != run / "run_manifest.json"
            or protocol != digest({k: v for k, v in manifest.items() if k != "protocol_sha256"})
            or seal["protocol_sha256"] != protocol):
        raise ValueError("Source manifest and terminal seal protocol differ")
    if (report.get("schema") != strict.SCHEMA or Path(report["run"]).resolve() != run
            or report.get("protocol_sha256") != protocol or report.get("planned_total") != 3080
            or report.get("audit_status") != "no_invalid_artifacts" or report.get("invalid_slot_count") != 0):
        raise ValueError("A fully validated3080-slot strict summary is required")
    snapshots = [read_pointer(p) for p in seal["process_snapshots"]]
    if len(snapshots) != 2:
        raise ValueError("Two terminal snapshots required")
    elapsed, stable = sealing.check_snapshot_pair(*snapshots, run, protocol)
    if elapsed != seal["snapshot_gap_seconds"] or stable["controller"] != seal["controller"]:
        raise ValueError("Terminal evidence was changed")
    checks = seal["process_checks"]
    if checks["matching_live_processes_before"] or checks["matching_live_processes_after"]:
        raise ValueError("Seal is not a stopped source")
    expected_files = {Path(p["path"]).resolve(): p for p in seal["file_inventory"]}
    actual_files = {p.resolve() for p in run.rglob("*") if p.is_file() and not p.name.endswith(".lock")}
    if len(expected_files) != seal["file_count"] or actual_files != set(expected_files):
        raise ValueError("Frozen QA namespace file population differs")
    for ref in expected_files.values():
        check_once(ref)
    natives = seal["original_source_native_inventories"]
    if len(natives) != 20 or {(r["conversation"], r["arm"]) for r in natives} != {(ci, arm) for ci in range(10) for arm in strict.ARMS}:
        raise ValueError("The twenty original native sources must all remain bound")
    for row in seal["original_source_native_inventories"]:
        native = Path(row["native_source"])
        inventory = {p.relative_to(native).as_posix(): sha(p)
                     for p in [*list((native / "qdrant").rglob("*")), native / "history.sqlite"]
                     if p.is_file() and p.name != ".lock"}
        if inventory != row["inventory"]:
            raise ValueError("Original native source inventory changed")
        check_once(row["source_bank"])
    for ref in [*seal["frozen_code"], seal["seal_tool"], seal["gate"], seal["plain_text_config"]]:
        check_once(ref)
    for path, expected in report["input_sha256"].items():
        check_once({"path": path, "sha256": expected})
    sealed_slots, strict_slots = validate_population(manifest, seal, report)
    for key, slot in sealed_slots.items():
        ci, arm, qid = key
        for stage in sealing.STAGES:
            ref = slot["artifacts"][stage]
            expected = run / f"conv_{ci}" / stage / arm / (qid + ".json")
            if Path(ref["path"]).resolve() != expected or expected_files.get(expected) != ref:
                raise ValueError("Slot artifact pointer is not in the immutable run inventory")
    return {"run": run, "manifest": manifest, "seal": seal, "strict_summary": report,
            "sealed_slots": sealed_slots, "strict_slots": strict_slots,
            "binding": {"source_manifest": seal["source_manifest"], "terminal_seal": seal_ref, "strict_summary": summary_ref}}


def derive_slot(source_slot, strict_slot, reader, judge, *, api_dir, ledger, question, reference):
    """Pure derivation; only Judge validity/score may differ from the strict slot."""
    for stage, row in (("reader", reader), ("judge", judge)):
        if row["status"] != source_slot["statuses"][stage]:
            raise ValueError("Stage status differs from sealed source")
        if any(row.get(k) != strict_slot[v] for k, v in (("conversation", "conversation"), ("method", "method"), ("query_id", "query_id"))):
            raise ValueError("Stage identity differs")
        if row.get("question") != question or row.get("reference") != reference:
            raise ValueError("Original question/reference differs")
    proof, correct, label = None, None, None
    status = "retained_failure"
    if judge["status"] == "complete" or (judge["status"] == "failed" and (judge.get("error") or {}).get("type") == "InvalidJudgeLabel"):
        if reader["status"] != "complete":
            raise ValueError("A generated Judge requires its original valid Reader")
        proof = replay_source(api_dir, judge, strict.stage_context(strict_slot["conversation"], strict_slot["method"], strict_slot["query_id"], "judge"),
                              ledger, question=question, reference=reference, reader_text=reader["result"]["prediction"]["answer"], snapshot=strict.Snapshot())
        parsed = proof["interpretation"]
        if judge["status"] == "complete":
            status = "strict_inherited"
        elif parsed["status"] == "compat_valid":
            status = "compat_recovered"
        if status in ("strict_inherited", "compat_recovered"):
            correct, label = parsed["correct"], parsed["label"]
    changed = deepcopy(strict_slot)
    if status == "compat_recovered":
        if strict_slot["scores"]["judge_correct"] is not None:
            raise ValueError("An original valid score cannot be replaced")
        changed["judge_status"] = changed["qa_status"] = "complete"
        changed["scores"]["judge_correct"] = float(correct)
    elif status == "strict_inherited" and strict_slot["scores"]["judge_correct"] != float(correct):
        raise ValueError("Strict score differs from original result")
    if (changed["reader_status"] != strict_slot["reader_status"]
            or any(changed["scores"][m] != strict_slot["scores"][m] for m in strict.METRICS if m != "judge_correct")):
        raise ValueError("Compatibility cannot change Reader or F1")
    return {"schema": SCHEMA, "conversation": strict_slot["conversation"], "method": strict_slot["method"],
            "query_id": strict_slot["query_id"], "status": status,
            "original": {"artifacts": source_slot["artifacts"], "statuses": source_slot["statuses"],
                         "strict_summary_slot_sha256": digest(strict_slot)},
            "judge_result": {"correct": correct, "label": label} if correct is not None else None,
            "source_replay": proof, "original_judge_error": judge.get("error"),
            "strict_slot": strict_slot, "compat_slot": changed,
            "reader_and_f1_unchanged": True, "model_calls": 0}, changed


def compare_views(report, changed, *, bootstrap_samples, denominator=1540):
    strict_view = strict.aggregate(report["slots"], denominator=denominator, bootstrap_samples=bootstrap_samples)
    for key in ("arms", "paired_merged_minus_unmerged"):
        if strict_view[key] != report[key]:
            raise ValueError("Strict aggregate differs; use its original bootstrap sample count")
    compat_view = strict.aggregate(changed, denominator=denominator, bootstrap_samples=bootstrap_samples)
    for arm in strict.ARMS:
        for metric in strict.METRICS:
            if metric != "judge_correct" and compat_view["arms"][arm]["metrics"][metric] != report["arms"][arm]["metrics"][metric]:
                raise ValueError("F1 aggregate changed")
    for metric in strict.METRICS:
        if metric != "judge_correct" and compat_view["paired_merged_minus_unmerged"][metric] != report["paired_merged_minus_unmerged"][metric]:
            raise ValueError("F1 paired population changed")
    return strict_view, compat_view


def execute(seal_path, summary_path, output, *, bootstrap_samples=2000):
    output = validate_namespace(output, Path(verify(seal_path)["run_dir"]))
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("Positive bootstrap sample count required")
    with offline():
        admitted = source_admission(seal_path, summary_path)
        run, original = admitted["run"], admitted["strict_summary"]
        cohort = strict.driver.sources.cohort()[0]
        snap = strict.Snapshot()
        _, ledger, _, missing = strict.ledger_index(snap, run, cohort)
        if missing:
            raise ValueError("Sealed complete run is missing an API ledger")
        manifest = {"schema": SCHEMA, "created_utc": now(), "policy": POLICY,
                    "sources": admitted["binding"], "source_protocol_sha256": admitted["manifest"]["protocol_sha256"],
                    "code_sha256": {name: sha(ROOT / name) for name in CODE},
                    "bootstrap_samples": bootstrap_samples, "query_ids": admitted["manifest"]["query_ids"],
                    "limitations": "Post-failure offline format interpretation, not a new model judgment or an original strict success. All source responses remain unchanged."}
        manifest["protocol_sha256"] = digest(manifest)
        save_new(output / "run_manifest.json", manifest)
        artifacts, changed, counts = [], [], Counter()
        recovered_labels = Counter()
        by_arm = {arm: {"statuses": Counter(), "recovered_labels": Counter()} for arm in strict.ARMS}
        retained = []
        for ci in range(10):
            for arm in strict.ARMS:
                for q in cohort[str(ci)]:
                    key = ci, arm, q["id"]
                    slot, original_slot = admitted["sealed_slots"][key], admitted["strict_slots"][key]
                    reader = read_pointer(slot["artifacts"]["reader"], sealed=True)
                    judge = read_pointer(slot["artifacts"]["judge"], sealed=True)
                    value, changed_slot = derive_slot(slot, original_slot, reader, judge,
                        api_dir=run / f"conv_{ci}/api", ledger=ledger, question=q["question"], reference=q["reference"])
                    value["protocol_sha256"] = manifest["protocol_sha256"]
                    path = output / "slots" / arm / (q["id"] + ".json")
                    save_new(path, value); artifacts.append(pointer(path)); changed.append(changed_slot); counts[value["status"]] += 1
                    by_arm[arm]["statuses"][value["status"]] += 1
                    if value["status"] == "compat_recovered":
                        recovered_labels[value["judge_result"]["label"]] += 1
                        by_arm[arm]["recovered_labels"][value["judge_result"]["label"]] += 1
                    elif changed_slot["scores"]["judge_correct"] is None:
                        retained.append({"conversation": ci, "method": arm, "query_id": q["id"],
                            "original_error": judge.get("error"),
                            "reason": (value["source_replay"] or {}).get("interpretation", {}).get("rejection_reason"),
                            "sidecar": pointer(path)})
        strict_view, compat_view = compare_views(original, changed, bootstrap_samples=bootstrap_samples)
        # One final full-source pass; per-slot replay has only its own raw/intent Snapshot.
        final = source_admission(seal_path, summary_path)
        if final["binding"] != admitted["binding"] or snap.unchanged():
            raise ValueError("Source changed while deriving the compatibility view")
        if any(sha(ROOT / name) != expected for name, expected in manifest["code_sha256"].items()):
            raise ValueError("Compatibility source changed during execution")
        report = {"schema": SCHEMA, "created_utc": now(), "protocol_sha256": manifest["protocol_sha256"],
                  "sources": admitted["binding"], "planned_total": 3080, "statuses": dict(counts),
                  "recovered_labels": dict(recovered_labels),
                  "recovery_by_arm": {arm: {k: dict(v) for k, v in values.items()} for arm, values in by_arm.items()},
                  "strict": strict_view, "compatibility": compat_view, "slots": changed,
                  "sidecars": artifacts, "retained_failures": retained,
                  "reader_and_f1_unchanged": True, "source_hashes_unchanged": True,
                  "new_model_calls": 0, "new_model_cost_usd": 0, "native_clients": 0,
                  "original_usage": {"strict_summary": admitted["binding"]["strict_summary"], "not_rebilled_or_free_cold_start": True},
                  "interpretation": manifest["limitations"]}
        save_new(output / "summary.json", report)
        (output / "summary.md").write_text(render(report), encoding="utf-8")
        return {"output": str(output), "status": "complete", "counts": dict(counts), "new_model_calls": 0}


def render(report):
    lines = ["# 纯文本 Judge 离线兼容结果", "", "原严格结果保留；末行一致标签兼容视图单列。Reader 与 F1 逐题未改，模型新增调用为0。", "",
             "|臂|strict Judge有效/1540|strict有效题均分|兼容Judge有效/1540|兼容有效题均分|", "|---|---:|---:|---:|---:|"]
    def fmt(x): return "—" if x is None else f"{100*x:.3f}"
    for arm in strict.ARMS:
        a = report["strict"]["arms"][arm]["metrics"]["judge_correct"]
        b = report["compatibility"]["arms"][arm]["metrics"]["judge_correct"]
        lines.append(f"|{arm}|{a['valid_denominator']}/1540|{fmt(a['complete_only_mean'])}|{b['valid_denominator']}/1540|{fmt(b['complete_only_mean'])}|")
    lines += ["", "状态计数：" + json.dumps(report["statuses"], ensure_ascii=False), "", report["interpretation"], "",
              "失败仍在1540计划分母内；有效题均分与缺失计零敏感度分开。标签冲突保持失败。兼容规则为观察到格式失败后的分析，不能据此声称记忆能力提高。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "execute"))
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--strict-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.phase == "check":
        with offline():
            value = source_admission(args.seal, args.strict_summary)
        result = {"status": "admitted", "planned_total": len(value["strict_slots"]), "sources": value["binding"], "model_calls": 0}
    else:
        if args.output_dir is None: parser.error("execute requires --output-dir")
        result = execute(args.seal, args.strict_summary, args.output_dir, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
