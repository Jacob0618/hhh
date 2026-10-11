"""Freeze an outcome-independent, paired LoCoMo pilot cohort without API calls.

Selection reads only source identity, conversation index and category. Question
text is hashed for auditing after allocation; answers and live results are never
used. This file intentionally imports no experiment runner or provider.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DEFAULT_SEED = "jev-mem-pilot400-v1-20260928"
TARGET = 400
ELIGIBLE_CATEGORIES = (1, 2, 3, 4)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    """Hash IDs in supplied order, joined by LF with no trailing LF."""
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def sampling_key(seed: str, query_id: str) -> str:
    # JSON encodes the two components unambiguously, including Unicode seeds.
    payload = json.dumps([seed, query_id], ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def eligible_rows(data: list[dict]) -> list[dict]:
    rows = []
    for ci, conversation in enumerate(data):
        for qi, qa in enumerate(conversation["qa"]):
            category = int(qa["category"])
            if category not in ELIGIBLE_CATEGORIES:
                continue
            rows.append({
                "query_id": f"c{ci}_q{qi}",
                "conversation": ci,
                "source_qa_index": qi,
                "sample_id": conversation["sample_id"],
                "stable_source_id": f"{conversation['sample_id']}/qa/{qi}",
                "category": category,
                "question_utf8_sha256": sha256_bytes(qa["question"].encode("utf-8")),
            })
    return rows


def hamilton_allocations(counts: dict[tuple[int, int], int], target: int) -> dict[tuple[int, int], int]:
    """Use integer remainders to avoid floating point tie ambiguity."""
    if any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError("Stratum counts must be nonnegative integers")
    total = sum(counts.values())
    if not isinstance(target, int) or not 0 <= target <= total:
        raise ValueError("Target must be an integer between zero and population size")
    if total == 0:
        return {key: 0 for key in counts}
    allocation = {key: target * size // total for key, size in counts.items()}
    priority = sorted(counts, key=lambda key: (-(target * counts[key] % total), key[0], key[1]))
    for key in priority[:target - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def select_rows(rows: list[dict], target: int = TARGET, seed: str = DEFAULT_SEED) -> tuple[list[dict], dict]:
    """Only query_id, conversation and category can affect membership."""
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate query IDs in population")
    strata: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(int(row["conversation"]), int(row["category"]))].append(row)
    counts = {key: len(values) for key, values in strata.items()}
    allocation = hamilton_allocations(counts, target)
    selected = []
    for key in sorted(strata):
        ranked = sorted(strata[key], key=lambda row: (sampling_key(seed, row["query_id"]), row["query_id"]))
        selected.extend(ranked[:allocation[key]])
    selected.sort(key=lambda row: (row["conversation"], row["source_qa_index"]))
    return selected, allocation


def _distribution(rows: list[dict]) -> dict:
    return {
        "by_conversation": dict(sorted(Counter(str(row["conversation"]) for row in rows).items(), key=lambda pair: int(pair[0]))),
        "by_category": dict(sorted(Counter(str(row["category"]) for row in rows).items(), key=lambda pair: int(pair[0]))),
    }


def build_manifest(data_path: Path = ROOT / "data" / "locomo10.json") -> dict:
    raw = data_path.read_bytes()
    source_sha = sha256_bytes(raw)
    if source_sha != DATASET_SHA256:
        raise ValueError(f"Pinned dataset SHA256 mismatch: {source_sha}")
    data = json.loads(raw)
    rows = eligible_rows(data)
    if len(data) != 10 or len(rows) != 1540:
        raise ValueError("Expected ten conversations and exactly 1540 eligible questions")
    selected, allocations = select_rows(rows)
    if {row["conversation"] for row in selected} != set(range(10)):
        raise ValueError("Every source conversation must be retained")
    selected_ids = [row["query_id"] for row in selected]
    source_ids = [row["query_id"] for row in rows]
    population_counts = Counter((row["conversation"], row["category"]) for row in rows)
    strata = []
    for ci in range(10):
        for category in ELIGIBLE_CATEGORIES:
            count = population_counts.get((ci, category), 0)
            chosen = allocations.get((ci, category), 0)
            strata.append({
                "conversation": ci, "category": category,
                "population_count": count, "sample_count": chosen,
                "quota_numerator": TARGET * count, "quota_denominator": len(rows),
                "ideal_quota": TARGET * count / len(rows),
                "allocation_error": chosen - TARGET * count / len(rows),
            })
    enriched = [{**row, "sampling_sha256": sampling_key(DEFAULT_SEED, row["query_id"]),
                 "development_accessible": row["conversation"] in (0, 1)} for row in selected]
    return {
        "schema": "jev-mem-pilot400-manifest-v1",
        "protocol_date": "2026-09-28",
        "source": {"path_relative_to_workspace": "data/locomo10.json", "sha256": source_sha,
                   "bytes": len(raw), "conversations": len(data), "raw_questions": sum(len(c["qa"]) for c in data)},
        "population": {"eligible_categories": list(ELIGIBLE_CATEGORIES), "count": len(rows),
                       "additional_filters": [], "deduplicate_question_text": False,
                       "query_ids_sha256": ids_sha256(source_ids), **_distribution(rows)},
        "sampling": {
            "target_count": TARGET, "seed": DEFAULT_SEED,
            "stratification": "conversation_x_category",
            "allocation": "Hamilton largest remainder using exact integer remainders",
            "allocation_tiebreak": "conversation ascending, then category ascending",
            "within_stratum": "ascending SHA256(UTF8(compact JSON [seed, query_id])), then query_id",
            "selection_features": ["query_id", "conversation", "category"],
            "forbidden_selection_features": ["answer", "evidence", "question text", "model output", "score", "success/failure"],
            "output_order": "original conversation index, then original unfiltered QA index",
            "hash_contract": "SHA256 of UTF-8 newline-joined IDs, no trailing newline",
        },
        "pairing": {
            "required_query_count_per_arm": TARGET,
            "required_query_ids_sha256": ids_sha256(selected_ids),
            "all_arms_use_identical_query_ids": True,
            "policy": "No resampling by arm or architecture; failures retain original IDs and denominator 400; do not replace missing questions.",
        },
        "exposure": {
            "development_conversations": [0, 1],
            "development_selected_count": sum(row["conversation"] in (0, 1) for row in selected),
            "is_blind_holdout": False,
            "note": "Conversations 0/1 are available for development; prior experiment execution or inspection prevents claiming an untouched blind test. Report this as a fixed paired pilot.",
        },
        "sample": {"count": len(selected), "query_ids_sha256": ids_sha256(selected_ids), **_distribution(selected)},
        "strata": strata,
        "query_ids": selected_ids,
        "questions": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "locomo10.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "pilot400_manifest_v1.json")
    parser.add_argument("--check", action="store_true", help="Verify existing manifest byte-for-byte without writing")
    args = parser.parse_args()
    manifest = build_manifest(args.data)
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.check:
        if args.output.read_bytes() != encoded:
            raise SystemExit("Manifest differs from deterministic reconstruction")
        status = "verified"
    elif args.output.exists():
        if args.output.read_bytes() != encoded:
            raise SystemExit("Refusing to overwrite an existing different manifest; version the new protocol")
        status = "unchanged"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(encoded)
        status = "created"
    print(json.dumps({"status": status, "output": str(args.output), "sample": manifest["sample"],
                      "development_selected_count": manifest["exposure"]["development_selected_count"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
