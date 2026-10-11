"""Pinned LoCoMo identities; separate write-time messages from evaluation QA."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def load_data():
    path = ROOT / "data/locomo10.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASH
    return json.loads(path.read_text(encoding="utf-8"))


def date_value(text):
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError("Invalid observation date")


def session_envelopes(item):
    conv = item["conversation"]
    keys = [k for k in conv if re.fullmatch(r"session_\d+", k)]
    keys.sort(key=lambda k: (date_value(conv[k + "_date_time"]), int(k.split("_")[1])))
    for key in keys:
        messages, turn_ids = [], []
        for turn in conv[key]:
            speaker, text = turn["speaker"], turn.get("text", "")
            caption, image_query = turn.get("blip_caption", ""), turn.get("query", "")
            if caption and image_query:
                text += f" [Sharing image - query: {image_query}. The image shows: {caption}]"
            elif image_query:
                text += f" [Sharing image - query for: {image_query}]"
            elif caption:
                text += f" [Sharing image that shows: {caption}]"
            if not text:
                raise ValueError("Unexpected empty source turn")
            messages.append({"role": "user" if speaker == conv["speaker_a"] else "assistant",
                             "content": f"{speaker}: {text}"})
            turn_ids.append(turn["dia_id"])
        yield {"session_id": key, "observation_date": date_value(conv[key + "_date_time"]).date().isoformat(),
               "messages": messages, "source_turn_ids": turn_ids}


def queries(item, ci):
    for qi, qa in enumerate(item["qa"]):
        if int(qa["category"]) in (1, 2, 3, 4):
            yield {"id": f"c{ci}_q{qi}", "source_qa_index": qi, "question": qa["question"],
                   "reference": qa["answer"], "category": int(qa["category"]),
                   "gold_evidence": qa.get("evidence", [])}
