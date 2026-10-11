"""Offline descriptive routing diagnostics; no outcome-guided tuning or API use.

Run after QA is terminal and the independent QA integrity gate passes. This
report describes the actual selections, not semantic relevance or causal gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ARMS = ("session_untyped_rank", "session_typed_rank", "regroup_untyped_rank")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(run, manifest):
    run, manifest = Path(run), Path(manifest)
    state = read(run / "qa_status.json")
    if state["status"] not in {"complete", "failed", "requires_review"}:
        raise ValueError("QA must be terminal before descriptive analysis")
    if any(v["status"] not in {"complete", "failed"} for v in state["conversations"].values()):
        raise ValueError("All ten conversation outcomes must be terminal")
    questions = read(manifest)["questions"]
    if len(questions) != 400 or len({q["query_id"] for q in questions}) != 400:
        raise ValueError("Fixed 400 cohort required")
    rows, missing, arm_rows, needs_rows = [], [], {arm: [] for arm in ARMS}, []
    for q in questions:
        ci, qid = q["conversation"], q["query_id"]
        folder = run / "conversations" / f"conv_{ci}"
        found = {}
        for arm in ARMS:
            path = folder / "retrieval" / arm / f"{qid}.json"
            if not path.exists():
                missing.append({"conversation": ci, "query_id": qid, "arm": arm, "reason": "no_retrieval_artifact"})
                continue
            artifact = read(path)
            if artifact["status"] != "complete":
                missing.append({"conversation": ci, "query_id": qid, "arm": arm, "reason": "retrieval_not_complete"})
                continue
            facts, pool = artifact["selected"], artifact["pool"]
            ids = [f["id"] for f in facts]
            if len(ids) != 20 or len(set(ids)) != 20 or len(pool) > 60:
                raise ValueError("Unexpected frozen selection budget")
            found[arm] = ids
            arm_rows[arm].append({
                "conversation": ci, "query_id": qid,
                "pool_count": len(pool), "selected_text_characters": sum(len(f["text"]) for f in facts),
                "channel_memberships": dict(Counter(c for f in facts for c in f["routing"]["channels"])),
                "typed_only_selected": sum(set(f["routing"]["channels"]) == {"typed"} for f in facts),
            })
        if ARMS[0] in found and ARMS[1] in found:
            a, b = found[ARMS[0]], found[ARMS[1]]
            rows.append({"conversation": ci, "query_id": qid,
                         "shared_selected_facts": len(set(a) & set(b)),
                         "identical_set": set(a) == set(b), "identical_order": a == b})
        path = folder / "query_types" / f"{qid}.json"
        if path.exists():
            item = read(path)
            if item["status"] == "complete":
                needs_rows.append({"query_id": qid, "scores": item["scores"]})
    result = {
        "schema": "jev-mem-pilot400-routing-description-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_calls_made": 0,
        "run_dir": str(run.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "fixed_questions": 400,
        "A_B_paired_retrieval_count": len(rows),
        "A_B_mean_shared_facts_of_20": mean([r["shared_selected_facts"] for r in rows]) if rows else None,
        "A_B_identical_set_questions": sum(r["identical_set"] for r in rows),
        "A_B_identical_order_questions": sum(r["identical_order"] for r in rows),
        "A_B_shared_count_histogram": dict(sorted(Counter(r["shared_selected_facts"] for r in rows).items())),
        "missing_retrievals": missing, "per_query_AB_overlap": rows,
        "arms": {},
        "query_needs_count": len(needs_rows),
        "query_needs_mean": {key: mean(r["scores"][key] for r in needs_rows)
                             for key in needs_rows[0]["scores"]} if needs_rows else {},
        "limitations": [
            "Retrieval artifacts include reader-format failures; this is not successful-QA conditioning.",
            "Channel memberships overlap and do not form exclusive quotas.",
            "Text character count is not model token count or provider usage.",
            "Set overlap and score distributions are not evidence recall, calibration, or semantic accuracy.",
            "This descriptive script relies on the separate seal/raw/retrieval integrity gate; it does not replace it.",
        ],
    }
    for arm, records in arm_rows.items():
        total = Counter()
        for record in records:
            total.update(record["channel_memberships"])
        result["arms"][arm] = {
            "completed_retrievals": len(records),
            "mean_pool_count": mean(r["pool_count"] for r in records) if records else None,
            "mean_selected_text_characters": mean(r["selected_text_characters"] for r in records) if records else None,
            "channel_membership_totals": dict(total),
            "typed_only_selected_total": sum(r["typed_only_selected"] for r in records),
            "per_query": records,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = describe(args.run_dir, args.manifest)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("A_B_paired_retrieval_count", "A_B_mean_shared_facts_of_20", "A_B_identical_set_questions", "query_needs_count")}))


if __name__ == "__main__":
    main()
