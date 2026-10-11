"""Read-only source-bound partition agreement; never calls a model or reads QA."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
from tools.run_global_segment_grouping import verify_seal
from jev_mem.data import load_data, SOURCE_HASH
from jev_mem.topic_grouping import raw_turns_from_locomo
from jev_mem.segmentation_comparison_v1 import compare_partitions, aggregate_comparisons, validate_partition
from jev_mem.fullconv_llm_segmentation_v1 import response_text, parse_partition

SCHEMA = "regroup-refinement-partition-comparison-v1"
CANDIDATES = ("original_global", "identity_only", "identity_and_bounded_refinement")
REFERENCE = "gemini-3.8-flash_fullconv_firstpass_plus_uniform_budget_recovery"


def snapshot(path, *, sealed=False):
    path = Path(path).resolve()
    data = shared_read(path)
    value = json.loads(data.decode("utf-8-sig"))
    if sealed:
        verify_seal(value)
    return value, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def bound(pointer, *, sealed=False):
    value, actual = snapshot(pointer["path"], sealed=sealed)
    if actual != pointer:
        raise ValueError("Source pointer/hash changed")
    return value


def original_turns():
    return [raw_turns_from_locomo(item["conversation"]) for item in load_data()]


def controller_terminal(name, row):
    """Historical scoring used separate dev and remaining dispatches.

    Its last controller legitimately covers only requested_conversations=2..9.
    Validate that explicit dispatch population; all ten immutable grouping files
    are independently checked below. New fullconv controllers require all ten.
    """
    if row.get("status") not in ("complete", "failed"):
        return False
    children = row.get("conversations")
    if not isinstance(children, dict):
        return False
    requested = row.get("requested_conversations", list(range(10))) if name == "original_global" else list(range(10))
    if (not isinstance(requested, list) or not requested or len(set(requested)) != len(requested)
            or any(type(ci) is not int or ci not in range(10) for ci in requested)):
        return False
    return (set(children) == {str(ci) for ci in requested}
            and all(v.get("status") in ("complete", "failed") for v in children.values()))


def ledger_accounting(path):
    """Count all physical attempts, including failed/non-text/truncated requests."""
    path = Path(path)
    if not path.exists():
        return {"status": "missing_ledger", "physical_http_attempts": None, "usd_total": None}
    data = shared_read(path)
    if data and not data.endswith(b"\n"):
        raise ValueError("Incomplete API ledger line; not a terminal accounting snapshot")
    rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    physical = [row for row in rows if row.get("cache_hit") is not True]
    known = [row for row in physical if isinstance(row.get("usage"), dict)]
    by_conversation = {}
    for row in physical:
        ci = str(row.get("context", {}).get("conversation", "unknown"))
        state = by_conversation.setdefault(ci, {"physical_http_attempts": 0, "http200": 0, "errors": 0})
        state["physical_http_attempts"] += 1
        state["http200" if row.get("status") == 200 else "errors"] += 1
    return {"status": "accounted", "ledger": {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()},
            "ledger_events": len(rows), "physical_http_attempts": len(physical),
            "cache_events": len(rows)-len(physical), "http_status_counts": dict(Counter(str(r.get("status")) for r in physical)),
            "logical_request_count": len({(r.get("stage"), r.get("request_hash"), json.dumps(r.get("context"), sort_keys=True)) for r in physical}),
            "attempts_with_reported_usage": len(known), "attempts_with_unknown_usage": len(physical)-len(known),
            "reported_usage": {key: sum(r["usage"].get(key, 0) for r in known)
                               for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")},
            "by_conversation": by_conversation, "usd_total": None,
            "cost_status": "No verified model-price/invoice binding here; unknown failures are not free"}


def llm_lineage(result, manifest, review, ci, raw_turns):
    """Replay the actual response, respecting inherited first-pass input, not prospective64K input."""
    input_ = bound(result["input"], sealed=True)
    if input_["conversation"] != ci or input_["raw_turns"] != raw_turns:
        raise ValueError("LLM source turns differ from original conversation")
    inherited = result.get("inherited_firstpass") is True
    if inherited:
        row = next((r for r in review["inherited_complete"] if r["conversation"] == ci), None)
        if row is None or result.get("source_result") != row["result"]:
            raise ValueError("Inherited LLM row is not in recovery review")
        original = bound(row["result"], sealed=True)
        if any(original.get(k) != result.get(k) for k in ("segments", "groups", "input", "response_text", "api_events")):
            raise ValueError("Inherited LLM partition differs from original first-pass result")
    else:
        expected = next(j["input"] for j in manifest["jobs"] if j["conversation"] == ci)
        if result["input"] != expected or ci not in {r["conversation"] for r in review["failure_population"]}:
            raise ValueError("Recovered LLM row is outside the reviewed one-attempt population")
    events = result.get("api_events", [])
    if len(events) != 1 or events[0].get("error_type") is not None:
        raise ValueError("Expected one completed original LLM response event")
    event = events[0]
    pointer = {"path": event["cache_path"], "sha256": event["cache_sha256"]}
    saved = bound(pointer)
    signature = saved["request"]
    request_hash = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (signature.get("provider") != "gemini" or signature.get("operation") != "generateContent"
            or signature.get("model") != manifest["model"] or signature.get("body") != input_["request_body"]
            or request_hash != event.get("request_hash") or event.get("context", {}).get("conversation") != ci):
        raise ValueError("LLM raw request identity differs")
    text = response_text(saved["response"])
    replay = parse_partition(text, raw_turns)
    if text != result.get("response_text") or any(replay[k] != result[k] for k in ("segments", "groups", "trace")):
        raise ValueError("LLM parsed partition differs from original model response")
    return {"kind": "inherited_firstpass" if inherited else "single_reviewed_recovery",
            "model": signature["model"], "effective_generation": signature["body"]["generationConfig"],
            "effective_tool_config": signature["body"].get("toolConfig"),
            "input": result["input"], "response": pointer,
            "source_result": result.get("source_result"), "llm_is_human_gold": False}


def _load_partition(path, ci, raw, *, extra_check=None):
    if not Path(path).exists():
        return {"status": "missing", "artifact": None, "partition": None}
    pointer = None
    try:
        value, pointer = snapshot(path, sealed=True)
        if value.get("conversation") != ci:
            raise ValueError("Partition conversation identity differs")
        if value.get("status") != "complete":
            return {"status": value.get("status", "unknown"), "artifact": pointer, "partition": None,
                    "error": value.get("error"), "usable": False}
        validate_partition(raw, value["segments"], conversation_id=ci, contiguous=True)
        validate_partition(raw, value["groups"], conversation_id=ci)
        lineage = extra_check(value) if extra_check else None
        return {"status": "complete", "artifact": pointer, "partition": value, "lineage": lineage,
                "source_turns": len(raw), "segments": len(value["segments"]), "groups": len(value["groups"]),
                "cross_date_groups": sum(g.get("cross_date", False) for g in value["groups"])}
    except Exception as exc:
        return {"status": "invalid_artifact", "artifact": pointer, "partition": None,
                "error": {"type": type(exc).__name__, "message": str(exc)}, "usable": False}


def build_report(old_root, refined_root, llm_root, *, allow_running_snapshot=False):
    old_root, refined_root, llm_root = map(lambda p: Path(p).resolve(), (old_root, refined_root, llm_root))
    roots = {"original_global": old_root, "refined": refined_root, "llm": llm_root}
    manifests, manifest_ptrs, statuses, status_ptrs = {}, {}, {}, {}
    names = {"original_global": "score_status.json", "refined": "run_status.json", "llm": "status.json"}
    for name, root in roots.items():
        manifests[name], manifest_ptrs[name] = snapshot(root / "manifest.json", sealed=True)
        statuses[name], status_ptrs[name] = snapshot(root / names[name])
    terminal = all(controller_terminal(name, row) for name, row in statuses.items())
    if not terminal and not allow_running_snapshot:
        raise RuntimeError("At least one grouping controller is still active; use an explicit provisional snapshot or wait")
    review, review_ptr = snapshot(llm_root / "recovery_review.json", sealed=True)
    raw_sources = original_turns()
    if len(raw_sources) != 10:
        raise ValueError("The fixed ten original conversations are required")
    states = {name: [] for name in (*CANDIDATES, "llm_reference")}
    comparisons = {name: [] for name in CANDIDATES}
    for ci, raw in enumerate(raw_sources):
        llm = _load_partition(llm_root / f"conv_{ci}/grouping.json", ci, raw,
                              extra_check=lambda value: llm_lineage(value, manifests["llm"], review, ci, raw))
        states["llm_reference"].append({"conversation": ci, **{k: v for k, v in llm.items() if k != "partition"}})
        for arm in CANDIDATES:
            path = (old_root / f"conversations/conv_{ci}/grouping.json" if arm == "original_global" else
                    refined_root / f"conv_{ci}/{arm}/grouping.json")
            def candidate_check(value):
                if arm != "original_global":
                    result, _ = snapshot(refined_root / f"conv_{ci}/result.json", sealed=True)
                    if (result.get("status") != "complete" or result.get("conversation") != ci
                            or result.get("manifest_sha256") != manifest_ptrs["refined"]["sha256"]):
                        raise ValueError("Refined conversation terminal/manifest binding differs")
                    expected = result["artifacts"][arm]
                    _, current = snapshot(path, sealed=True)
                    if expected != current:
                        raise ValueError("Refined artifact differs from terminal result")
                return {"model": "jev-1.13.0", "kind": "frozen_historical" if arm == "original_global" else "new_firstpass"}
            candidate = _load_partition(path, ci, raw, extra_check=candidate_check)
            state = {"conversation": ci, **{k: v for k, v in candidate.items() if k != "partition"}}
            state["comparison_status"] = "not_evaluable"
            if candidate["status"] == llm["status"] == "complete":
                try:
                    c, r = candidate["partition"], llm["partition"]
                    comparison = compare_partitions(raw, c["segments"], c["groups"], r["segments"], r["groups"],
                                                    conversation_id=ci, reference_label=REFERENCE, tolerance_turns=1)
                    comparisons[arm].append(comparison)
                    state["comparison_status"] = "complete"
                except Exception as exc:
                    state.update(comparison_status="invalid_partition_relationship", comparison_error=str(exc))
            states[arm].append(state)
    aggregates = {}
    for arm, rows in comparisons.items():
        valid = {r["conversation_id"] for r in rows}
        aggregates[arm] = {"planned_conversations": 10, "compared_conversations": len(rows),
                           "missing_or_invalid_conversations": sorted(set(range(10)) - valid),
                           "compared_source_turns": sum(r["turn_count"] for r in rows),
                           "planned_source_turns": sum(map(len, raw_sources)),
                           "metrics": aggregate_comparisons(rows) if rows else None}
    common = sorted(set.intersection(*({r["conversation_id"] for r in rows} for rows in comparisons.values())))
    common_metrics = {arm: aggregate_comparisons([r for r in rows if r["conversation_id"] in common]) if common else None
                      for arm, rows in comparisons.items()}
    first_path = review.get("source_manifest", {}).get("path")
    accounting = {"recovery": ledger_accounting(llm_root / "api/api_calls.jsonl")}
    if first_path:
        accounting["firstpass"] = ledger_accounting(Path(first_path).parent / "api/api_calls.jsonl")
    accounted = [v for v in accounting.values() if v["status"] == "accounted"]
    accounting["combined"] = {"all_phase_ledgers_available": len(accounted) == 2,
        "physical_http_attempts": sum(v["physical_http_attempts"] for v in accounted) if len(accounted) == 2 else None,
        "attempts_with_unknown_usage": sum(v["attempts_with_unknown_usage"] for v in accounted) if len(accounted) == 2 else None,
        "reported_usage": {key: sum(v["reported_usage"][key] for v in accounted)
                           for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")},
        "usd_total": None, "scope": "All firstpass and recovery physical attempts, not only usable partitions"}
    report = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "agreement_to_llm_not_accuracy", "reference_is_human_gold": False,
              "snapshot_kind": "terminal" if terminal else "provisional_running_not_final",
              "all_controllers_terminal": terminal, "dataset_sha256": SOURCE_HASH,
              "planned_conversations": list(range(10)), "planned_source_turns": sum(map(len, raw_sources)),
              "source_manifests": manifest_ptrs, "controller_snapshots": status_ptrs,
              "controller_statuses": {k: v.get("status") for k, v in statuses.items()}, "recovery_review": review_ptr,
              "reference_model": manifests["llm"].get("model"), "source_states": states,
              "aggregates": aggregates, "common_conversations": common,
              "common_conversation_metrics": common_metrics, "per_conversation": comparisons,
              "lineage_counts": dict(Counter(r.get("lineage", {}).get("kind", r["status"]) for r in states["llm_reference"])),
              "llm_physical_accounting": accounting,
              "new_model_calls": 0, "qa_or_gold_used": False, "thresholds_tuned_from_agreement": False,
              "limitations": ["LLM reference may be incorrect; source adjudication is required.",
                              "Boundary/grouping agreement is not downstream QA quality.",
                              "Incomplete partitions retain their ten-conversation denominator and are not scored as zeros.",
                              "One LLM first-pass success and nine budget/tool-config recovery candidates have different effective generation settings."]}
    # Grouping artifacts and manifests are immutable. Hot controller snapshots are
    # point-in-time evidence and are not mistaken for a seal on future progress.
    for pointer in [*manifest_ptrs.values(), review_ptr,
                    *[r["artifact"] for rows in states.values() for r in rows if r.get("artifact")]]:
        bound(pointer)
    return report


def markdown(report):
    lines = ["# 分区对照：与全文 Gemini 的一致性", "", "这不是人工gold准确率；不使用QA/gold，不据此调整阈值。", "",
             f"状态：{report['snapshot_kind']}；固定10个conversation，共{report['planned_source_turns']}个原turn。", "",
             "| 方案 | 可比较conv/10 | Exact边界F1 | ±1边界F1 | 正turn对F1 | B-cubed F1 |", "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{100*value:.3f}"
    for arm, value in report["aggregates"].items():
        m = value["metrics"]
        scores = [None] * 4 if m is None else [m["boundary_exact"]["f1"], m["boundary_tolerant"]["f1"],
                    m["event_clustering"]["pairwise_positive"]["f1"], m["event_clustering"]["bcubed"]["f1"]]
        lines.append(f"| {arm} | {value['compared_conversations']}/10 | " + " | ".join(map(fmt, scores)) + " |")
    lines.extend(["", "边界只计原session内部；±1为一对一匹配。聚类只计conversation内部正turn对，不报告负对accuracy。", "",
                  "各方案与LLM不可比较conv（具体哪一端失败见source_states；不代表候选分区本身失败）：",
                  *[f"- {arm}: {value['missing_or_invalid_conversations']}" for arm, value in report["aggregates"].items()],
                  "", "LLM谱系：" + json.dumps(report["lineage_counts"], ensure_ascii=False),
                  "", "LLM两阶段全部物理调用（含失败）：" + str(report["llm_physical_accounting"]["combined"]["physical_http_attempts"])
                  + "；无usage尝试：" + str(report["llm_physical_accounting"]["combined"]["attempts_with_unknown_usage"])
                  + "。费用未核价，不将未知视为免费。",
                  "", "LLM源中conv9继承首遍16K上限的成功结果；其余9conv各允许一次64K上限、function calling NONE的技术恢复。原失败保留。",
                  "", "每conv来源hash、原始响应重放与完整性、共同可比较集合、有效源turn数及缺失状态详见同名JSON。"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-root", type=Path, default=ROOT / "runs/global_segment_grouping_v1")
    p.add_argument("--refined-root", type=Path, default=ROOT / "runs/refined_event_grouping_v1")
    p.add_argument("--llm-root", type=Path, default=ROOT / "runs/fullconv_llm_segmentation_recovery_v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-running-snapshot", action="store_true")
    a = p.parse_args(); output = a.output.resolve(); md = output.with_suffix(".md")
    if output.exists() or md.exists():
        raise FileExistsError("New report paths required; no report overwrite")
    result = build_report(a.old_root, a.refined_root, a.llm_root, allow_running_snapshot=a.allow_running_snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as h:
        json.dump(result, h, ensure_ascii=False, indent=2); h.write("\n")
    with md.open("x", encoding="utf-8") as h:
        h.write(markdown(result))
    print(json.dumps({"output": str(output), "status": result["snapshot_kind"],
                      "counts": {arm: value["compared_conversations"] for arm, value in result["aggregates"].items()},
                      "new_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__": main()
