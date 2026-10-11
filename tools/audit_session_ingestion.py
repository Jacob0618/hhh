"""Static session-batch audit. Does not import Mem0 or submit any messages."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_date(value):
    for pattern in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    raise ValueError("Unsupported session date; do not silently reorder")


def main():
    data_path = ROOT / "data/locomo10.json"
    sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert sha == "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    source = json.loads(data_path.read_text(encoding="utf-8"))
    rows = []
    question_count = 0
    for ci, item in enumerate(source):
        conversation = item["conversation"]
        sessions = [k for k in conversation if re.fullmatch(r"session_\d+", k)]
        sessions.sort(key=lambda k: (parse_date(conversation[k + "_date_time"]), int(k.split("_")[1])))
        question_count += sum(int(q["category"]) in (1, 2, 3, 4) for q in item["qa"])
        prior_turn_ids = []
        for order, session in enumerate(sessions):
            turns = conversation[session]
            turn_ids = [t["dia_id"] for t in turns]
            assert len(turn_ids) == len(set(turn_ids))
            assert not set(turn_ids).intersection(prior_turn_ids)
            assert all(t["speaker"] in (conversation["speaker_a"], conversation["speaker_b"]) for t in turns)
            # Source-character estimate only. It is NOT a tokenizer/context-window check.
            text_characters = sum(len(str(t.get(k, ""))) for t in turns for k in ("text", "blip_caption", "query"))
            rows.append({"conversation_index": ci, "sample_id": item.get("sample_id"),
                         "session_id": session, "session_order": order,
                         "observation_datetime_naive": parse_date(conversation[session + "_date_time"]).isoformat(),
                         "new_turn_ids": turn_ids, "new_turns": len(turns),
                         "previous_tail_turn_ids": prior_turn_ids[-10:],
                         "source_text_characters": text_characters,
                         "planned_add_submissions": 1})
            prior_turn_ids.extend(turn_ids)
    totals = {"conversations": len(source), "sessions": len(rows), "turns": sum(r["new_turns"] for r in rows),
              "questions_category_1_4": question_count, "planned_session_add_submissions": len(rows),
              "max_session_turns": max(r["new_turns"] for r in rows),
              "max_session_source_characters": max(r["source_text_characters"] for r in rows)}
    assert totals["sessions"] == 272 and totals["turns"] == 5882 and question_count == 1540
    report = {"status": "static_preflight_only_no_ingestion", "source_sha256": sha,
              "protocol": "one_complete_session_per_add", "final_read_top_k": 20,
              "write_fact_cap": None, "totals": totals, "sessions": rows,
              "tokenizer_check": "pending_model_selection_no_character_to_token_conversion_claim",
              "timezone": "Dataset supplies no timezone; naive original date/time retained. Do not infer one.",
              "tail_scope": "Planned source-order tail only; live sidecar must advance only after successful commit."}
    (ROOT / "research/session_ingestion_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
