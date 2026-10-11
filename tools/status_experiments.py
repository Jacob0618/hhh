"""Read-only compact progress, never an experiment or a final integrity audit."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",action="append")
    args=parser.parse_args()
    directories=args.run_dir or [str(ROOT/"runs/session_top20_v1"),str(ROOT/"runs/independent_top20_v1"),str(ROOT/"runs/routing_top20_v1")]
    output={"time_utc":datetime.now(timezone.utc).isoformat(),"progress_only_not_final_audit":True,"runs":[]}
    for directory in directories:
        path=Path(directory)
        if not path.exists():
            continue
        row={"run_dir":str(path),"ingested_sessions":0,"facts":0,"complete_conversations":[],
             "qa_by_method":{},"retrieval_files":0,"tag_files":0}
        for conv in path.glob("conv_*"):
            summary=conv/"ingestion_summary.json"
            if summary.exists():
                data=json.loads(summary.read_text(encoding="utf-8"))
                row["ingested_sessions"]+=data["sessions_completed"]
                row["facts"]+=data["facts"]
                if data["complete"]:
                    row["complete_conversations"].append(data["conversation"])
            row["retrieval_files"]+=len(list((conv/"retrieval").glob("*.json")))
            row["tag_files"]+=len(list((conv/"tags").glob("*.json")))
            for method in (conv/"qa").glob("*"):
                if method.is_dir():
                    row["qa_by_method"][method.name]=row["qa_by_method"].get(method.name,0)+len(list(method.glob("*.json")))
        if (path/"queue_status.json").exists():
            queue=json.loads((path/"queue_status.json").read_text(encoding="utf-8"))
            row["queue_status"]=queue["status"]
            row["queue_pid"]=queue["pid"]
            row["active_jobs"]=[name for name,j in queue["jobs"].items() if j["status"]=="running"]
            row["failed_jobs"]=[name for name,j in queue["jobs"].items() if j["status"]=="failed"]
        output["runs"].append(row)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
