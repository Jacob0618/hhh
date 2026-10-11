"""Query-blind, stdlib-only source/time audit. No models, databases or Reader writes.

Input scope gives candidate source references, never verified assertion support.
Calendar candidates are explicit interpretations, never automatic event dates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

VERSION = "source-time-audit-sidecar-v1"
DATASET_SHA = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
TIME_RE = re.compile(
    r"\b(?:the day before yesterday|day before yesterday|yesterday|today|tomorrow|"
    r"(?:last|next|this)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year)|"
    r"recently|lately|(?:\d+|one|two|three|a|an)\s+(?:days?|weeks?|months?|years?)\s+ago)\b", re.I)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def raw_turns(conversation):
    """Only raw conversation fields allowed; QA/gold is not passed into this API."""
    forbidden = {"qa", "question", "answer", "gold", "gold_evidence", "facts"}
    if forbidden & conversation.keys():
        raise ValueError("QA/gold/facts forbidden in raw conversation")
    turns = []
    seen = set()
    for session in sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                          key=lambda k: int(k.split("_")[1])):
        stamp = conversation[session + "_date_time"]
        observed = None
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
            try:
                observed = datetime.strptime(stamp, fmt).date().isoformat()
                break
            except ValueError:
                continue
        if observed is None:
            raise ValueError("Unknown source observation timestamp")
        for turn in conversation[session]:
            if forbidden & turn.keys():
                raise ValueError("QA/gold/facts forbidden in raw turn")
            tid = turn["dia_id"]
            if tid in seen:
                raise ValueError("Duplicate source turn ID")
            seen.add(tid)
            text = turn.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Source text must be a string")
            row = {"source_turn_id": tid, "session_id": session,
                   "source_observation_date": observed, "source_timestamp": stamp,
                   "speaker": turn["speaker"], "text": text, "text_sha256": text_sha(text)}
            # Captions remain distinct source fields; no merging into textual evidence.
            for field in ("blip_caption", "query"):
                if field in turn:
                    row[field] = turn[field]
            turns.append(row)
    return turns


def calendar_candidate(expression, anchor):
    """Only simple calendar interpretations; not semantic/event attachment."""
    if anchor is None:
        return None
    current = date.fromisoformat(anchor)
    expression = expression.lower()
    offsets = {"today": 0, "yesterday": -1, "tomorrow": 1,
               "day before yesterday": -2, "the day before yesterday": -2}
    if expression in offsets:
        return {"value": (current + timedelta(days=offsets[expression])).isoformat(),
                "granularity": "day", "rule": "source-observation-calendar-day-offset"}
    if expression in {"next month", "last month", "this month"}:
        offset = {"next month": 1, "last month": -1, "this month": 0}[expression]
        month_index = current.year * 12 + current.month - 1 + offset
        year, month = divmod(month_index, 12)
        return {"value": f"{year:04d}-{month + 1:02d}", "granularity": "month",
                "rule": "source-observation-calendar-month-offset"}
    return None


def expressions(text, anchor=None):
    results = []
    for match in TIME_RE.finditer(text):
        candidate = calendar_candidate(match.group(), anchor)
        results.append({"expression": match.group(), "start": match.start(), "end": match.end(),
                        "expression_sha256": text_sha(match.group()),
                        "anchor_source_observation_date": anchor,
                        "calendar_candidate": candidate,
                        "resolution_status": "candidate_only" if candidate else "unresolved",
                        "event_attachment_status": "unknown", "alternative_anchor_possible": True,
                        "rule_version": VERSION})
    return results


def fact_record(fact, bank_sha, conversation, source_by_id):
    fact_id, text = fact["id"], fact["text"]
    if not isinstance(fact_id, str) or not isinstance(text, str):
        raise ValueError("Fact ID/text must be strings")
    scope = fact.get("input_scope") or {}
    if not isinstance(scope, dict):
        raise ValueError("input_scope must be a mapping")
    source_ids = scope.get("source_turn_ids", [])
    if not isinstance(source_ids, list) or len(set(source_ids)) != len(source_ids):
        raise ValueError("Invalid or duplicate input_scope IDs")
    missing = [tid for tid in source_ids if tid not in source_by_id]
    if missing:
        raise ValueError(f"Input scope has unknown source IDs: {missing}")
    links = [{"source_turn_id": tid, "source_text_sha256": source_by_id[tid]["text_sha256"],
              "source_observation_date": source_by_id[tid]["source_observation_date"],
              "relative_expression_indexes": list(range(len(source_by_id[tid]["relative_time_expressions"])))}
             for tid in source_ids]
    return {"schema": VERSION, "conversation": conversation, "bank_sha256": bank_sha,
            "fact_id": fact_id, "fact_sha256": digest(fact), "fact_text_sha256": text_sha(text),
            "fact_observation_date": fact.get("observation_date"),
            "input_scope": scope, "input_scope_sha256": digest(scope),
            "candidate_source_links": links, "candidate_scope_status": "unverified_not_assertion_support",
            "assertion_spans": [], "context_spans": [], "support_status": "unknown",
            "source_available_at": None, "source_available_at_status": "unknown_without_support_alignment",
            "existing_verified_support": fact.get("verified_support", []),
            "existing_support_status": fact.get("support_status"),
            # Fact observation_date is not a proven relative-expression anchor.
            "fact_relative_time_expressions": expressions(text),
            "event_status": "unknown", "event_date": None,
            "reader_visible": False, "qa_or_gold_used": False}


def build(root):
    root = Path(root)
    dataset_path = root / "data/locomo10.json"
    before = {str(dataset_path): file_sha(dataset_path)}
    if before[str(dataset_path)] != DATASET_SHA:
        raise ValueError("Frozen source dataset SHA mismatch")
    # Only item['conversation'] enters all downstream functions; QA is never consulted.
    conversations = [item["conversation"] for item in read_json(dataset_path)]
    if len(conversations) != 10:
        raise ValueError("Expected ten original conversations")
    ledger, records, banks = [], [], []
    for ci, conversation in enumerate(conversations):
        source = raw_turns(conversation)
        for row in source:
            row.update({"schema": VERSION, "conversation": ci, "source_dataset_sha256": DATASET_SHA,
                        "relative_time_expressions": expressions(row["text"], row["source_observation_date"]),
                        "semantic_support_status": "not_assessed"})
        source_by_id = {r["source_turn_id"]: r for r in source}
        path = root / f"runs/sdk20_modules_v1/conv_{ci}/session/bank.json"
        bank_sha = file_sha(path)
        before[str(path)] = bank_sha
        bank = read_json(path)
        if bank["conversation"] != ci or bank["source_dataset_sha256"] != DATASET_SHA:
            raise ValueError("Bank source/conversation binding mismatch")
        ids = [f["id"] for f in bank["facts"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate fact IDs in bank")
        banks.append({"conversation": ci, "path": str(path), "sha256": bank_sha, "facts": len(ids)})
        records.extend(fact_record(f, bank_sha, ci, source_by_id) for f in bank["facts"])
        ledger.extend(source)
    if len(records) != 2376:
        raise ValueError("Expected original baseline 2376 facts")
    if any(file_sha(p) != sha for p, sha in before.items()):
        raise ValueError("Frozen input changed during read-only audit")
    manifest = {"schema": VERSION, "code_sha256": file_sha(__file__), "model_calls": 0,
                "database_opens": 0, "qa_or_gold_used": False, "reader_modified": False,
                "frozen_inputs_unchanged": True, "inputs": before, "banks": banks,
                "fact_records": len(records), "raw_turns": len(ledger),
                "verified_assertion_spans": 0, "support_status_counts": dict(Counter(r["support_status"] for r in records)),
                "source_relative_expressions": sum(len(r["relative_time_expressions"]) for r in ledger),
                "source_calendar_candidates": sum(bool(e["calendar_candidate"]) for r in ledger for e in r["relative_time_expressions"]),
                "limits": ["input_scope is candidate provenance, not semantic support",
                           "calendar candidates assume source observation anchor and are not event dates",
                           "regex coverage is deliberately incomplete; absence is not no temporal content",
                           "no automatic assertion/context alignment or event-status prediction",
                           "baseline only; prior manually reviewed cases remain in linked research reports"]}
    return records, ledger, manifest


def write_outputs(root, out):
    out = Path(out).resolve()
    root = Path(root).resolve()
    allowed = (root / "research/planned_execution_2026-10-03").resolve()
    if out != allowed:
        raise ValueError("Audit output must be the new planned_execution research namespace")
    names = ["source_time_sidecar_facts.jsonl", "source_time_sidecar_raw_turns.jsonl",
             "source_time_sidecar_manifest.json", "source_time_sidecar_readout.md"]
    if any((out / n).exists() for n in names):
        raise FileExistsError("Refuse to overwrite existing sidecar artifacts")
    facts, ledger, manifest = build(root)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(names[:2], (facts, ledger)):
        with (out / name).open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["outputs"] = {n: file_sha(out / n) for n in names[:2]}
    with (out / names[2]).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    readout = (f"# Source/time只读旁表\n\n原baseline {len(facts)} facts，{len(ledger)} raw turns；"
               f"识别{manifest['source_relative_expressions']}个源相对时间表达，"
               f"其中{manifest['source_calendar_candidates']}个有显式日/月历候选。\n\n"
               "自动确定：bank/fact/text/source SHA、input_scope与原turn存在性、原session观察日期、"
               "正则命中的原表达与字符offset。仅today/yesterday/tomorrow/day-before-yesterday及last/next/this month"
               "在假定source观察日为锚时给calendar candidate；它不是已解析event date。\n\n"
               "所有2376条assertion_spans/context_spans为空、support_status=unknown；"
               "整组input_scope仅unverified候选出处，source_available_at、事件状态及日期不猜。"
               "weekday/week/ago/recently等不强行消歧，正则未命中也不等于没有时间内容。"
               "现有关键案例保留在design_resolution/track_a报告，不按QA答案补写旁表。\n\n"
               "0模型调用、0数据库打开、不使用QA/gold、不改旧facts或Reader；输入11个文件SHA前后相同。"
               "此为审计基础设施，无新增QA分数，不提供全库missing-bank比例。\n")
    with (out / names[3]).open("x", encoding="utf-8") as stream:
        stream.write(readout)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = write_outputs(args.root, args.root / "research/planned_execution_2026-10-03")
    print(json.dumps({k: manifest[k] for k in ("fact_records", "raw_turns", "source_relative_expressions", "source_calendar_candidates", "frozen_inputs_unchanged")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
