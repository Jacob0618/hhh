"""Check completed retrieval artifacts and frozen code without loading API keys."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, queries
from tools import run_session_experiments as mem0_runner
from tools import run_independent_experiments as independent_runner


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    data = load_data()
    output = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "read_only_retrieval_integrity_not_final_QA_audit", "runs": []}
    for architecture, name in [("mem0", "session_top20_v1"), ("independent", "independent_top20_v1")]:
        run = ROOT / "runs" / name
        manifest = read(run / "run_manifest.json") if architecture == "independent" else None
        if manifest:
            mismatches = [] if manifest["protocol"] == independent_runner._protocol() else ["independent_protocol"]
        else:
            source = read(ROOT / "research/protocol_snapshot_reader_v2/manifest.json")["source_sha256"]
            mismatches = [relative for relative, expected in source.items()
                          if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected]
        errors, expected_ids, actual_ids = [], set(), set()
        hist = Counter()
        for ci, item in enumerate(data):
            conv = run / f"conv_{ci}"
            summary = read(conv / "ingestion_summary.json")
            facts_file = conv / "facts.json"
            facts_hash = hashlib.sha256(facts_file.read_bytes()).hexdigest()
            facts = {fact["id"]: fact for fact in read(facts_file)}
            if not summary["complete"]:
                errors.append(f"conv_{ci}: incomplete ingestion")
            if manifest and facts_hash != summary["facts_sha256"]:
                errors.append(f"conv_{ci}: changed facts snapshot")
            actual_ids.update(p.stem for p in (conv / "retrieval").glob("*.json"))
            for q in queries(item, ci):
                expected_ids.add(q["id"])
                path = conv / "retrieval" / (q["id"] + ".json")
                if not path.exists():
                    continue
                try:
                    row = read(path)
                    if manifest:
                        independent_runner._verify_retrieval(row, independent_runner._retrieval_binding(q, summary, manifest))
                    else:
                        mem0_runner._verify_retrieval(row, q, summary, facts_hash)
                    for key in (["pool60"] if manifest else ["pool60", "sdk20"]):
                        candidates = row[key]
                        if len({f["id"] for f in candidates}) != len(candidates):
                            raise ValueError("duplicate candidate IDs")
                        if any(f["id"] not in facts or f["text"] != facts[f["id"]]["text"] for f in candidates):
                            raise ValueError("candidate identity/text differs from frozen facts")
                        hist[f"{key}:{len(candidates)}"] += 1
                except Exception as error:
                    errors.append(f"{q['id']}: {type(error).__name__}: {error}")
        output["runs"].append({"architecture": architecture, "expected_queries": len(expected_ids),
            "retrieval_queries": len(actual_ids), "missing_ids": sorted(expected_ids - actual_ids),
            "unexpected_ids": sorted(actual_ids - expected_ids), "code_mismatches": mismatches,
            "errors": errors, "candidate_count_histogram": dict(hist),
            "passed": not errors and not mismatches and actual_ids == expected_ids})
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
