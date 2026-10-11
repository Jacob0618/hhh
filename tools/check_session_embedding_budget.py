"""Tokenize all complete session retrieval strings with the selected model."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, session_envelopes
from jev_mem.providers import ModelAPI, dump_json

api = ModelAPI(ROOT / "runs/20260927_session_preflight", retries=2)
tasks = [(ci, s) for ci, item in enumerate(load_data()) for s in session_envelopes(item)]


def count(task):
    ci, session = task
    # This mirrors mem0.memory.utils.parse_messages for these two roles.
    text = "\n".join(m["role"] + ": " + m["content"] for m in session["messages"]) + "\n"
    n = api.count_tokens(text, stage="full_session_embedding_length")
    return {"conversation": ci, "session": session["session_id"], "tokens": n, "within_2048": n <= 2048}


with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(count, tasks))
report = {"model": "gemini-embedding-001", "sessions": len(rows), "maximum_tokens": max(r["tokens"] for r in rows),
          "over_limit": [r for r in rows if not r["within_2048"]], "rows": rows}
dump_json(ROOT / "runs/20260927_session_preflight/embedding_budget.json", report)
print(json.dumps({k:v for k,v in report.items() if k != "rows"}, ensure_ascii=False), flush=True)
