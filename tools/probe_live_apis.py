"""Four minimal authorized API checks; no LoCoMo data is sent."""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.providers import ModelAPI, dump_json

run_dir = ROOT / "runs/20260927_api_probe"
api = ModelAPI(run_dir, retries=0, timeout=60)


def check(name):
    if name == "jev":
        output = api.score_many("The box is blue.", {"blue": "Is the box explicitly described as blue?"}, stage="auth_probe")
        return {"ok": True, "model": "jev-1.13.0", "result": output}
    if name == "embedding":
        vector = api.embed(["The box is blue."], stage="auth_probe")[0]
        return {"ok": True, "model": "gemini-embedding-001", "dimensions": len(vector),
                "normalized_norm": math.sqrt(sum(x*x for x in vector))}
    output = api.generate('Return the JSON object {"ok": true}.', model=name, json_output=True,
                          max_output=1024, stage="auth_probe")
    return {"ok": output == {"ok": True}, "model": name, "result": output}


results = {}
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(check, name): name for name in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "embedding", "jev"]}
    for future in as_completed(futures):
        name = futures[future]
        try:
            results[name] = future.result()
        except Exception as error:
            results[name] = {"ok": False, "error": api._redact(error)[:1600]}
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
dump_json(run_dir / "summary.json", {"utc": datetime.now(timezone.utc).isoformat(), "checks": results,
                                    "locomo_sent": False, "all_ok": all(r["ok"] for r in results.values())})
