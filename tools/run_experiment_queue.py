"""Resumable, bounded stage queue. Each conversation owns its Qdrant process."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.providers import dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["mem0", "independent"], default="mem0")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--stages", default="ingest,retrieve,qa")
    parser.add_argument("--parallel-conversations", type=int, default=3)
    parser.add_argument("--methods", default="sdk20,rank,mmr,jev_single,conditional")
    parser.add_argument("--query-workers", type=int, default=4)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    if not all(s in {"ingest", "retrieve", "qa", "tag"} for s in stages):
        raise ValueError("Unknown stage")
    cis = [int(x) for x in args.conversations.split(",")]
    runner = ROOT / "tools" / ("run_session_experiments.py" if args.architecture == "mem0" else "run_independent_experiments.py")
    files = list((ROOT / "jev_mem").glob("*.py")) + [runner]
    snapshot = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_dir = run_dir / "queues" / stamp
    queue_dir.mkdir(parents=True, exist_ok=True)
    state = {"architecture":args.architecture,"started_utc":stamp,"pid":os.getpid(),
             "stages":stages,"conversations":cis,"methods":args.methods.split(","),
             "status":"running","source_hashes":snapshot,"jobs":{}}
    lock = threading.Lock()
    def update(key, value):
        with lock:
            state["jobs"][key] = value
            dump_json(queue_dir / "status.json",state)
            dump_json(run_dir / "queue_status.json",state)
    dump_json(queue_dir / "status.json",state)
    failed = False
    for stage in stages:
        def job(ci):
            key = f"{stage}/conv_{ci}"
            command = [sys.executable,"-u",str(runner),stage,"--run-dir",str(run_dir),
                       "--conversations",str(ci),"--methods",args.methods,"--workers",str(args.query_workers)]
            log_path = queue_dir / f"{stage}_conv_{ci}.log"
            update(key,{"status":"running","log":str(log_path)})
            with log_path.open("w",encoding="utf-8") as log:
                result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            success = result.returncode == 0
            if stage == "qa" and success:
                summary_path = run_dir / f"conv_{ci}" / "qa_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                success = all(m.get("completed") == m.get("requested") for m in summary["methods"].values())
            update(key,{"status":"complete" if success else "failed","exit_code":result.returncode,"log":str(log_path)})
            print(json.dumps({"stage":stage,"conversation":ci,"success":success}),flush=True)
            return success
        with ThreadPoolExecutor(max_workers=args.parallel_conversations) as pool:
            futures = [pool.submit(job,ci) for ci in cis]
            for future in as_completed(futures):
                if not future.result():
                    failed = True
        if failed:
            break
    state["status"] = "requires_review" if failed else "complete"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(queue_dir / "status.json",state)
    dump_json(run_dir / "queue_status.json",state)
    print(json.dumps({"queue_status":state["status"],"run_dir":str(run_dir)}),flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
