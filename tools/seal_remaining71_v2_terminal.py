"""Read and seal all terminal recovery-v2 files; does not modify the run."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now
from tools.check_experiment_health import shared_read
import hashlib

if __name__ == "__main__":
    run = ROOT / "runs/pilot400_recovery_v2"
    out = ROOT / "research/pilot400_remaining71_v2_terminal_seal.json"
    if out.exists():
        raise ValueError("Terminal seal already exists")
    status = read(run / "qa_remaining71_status.json")
    if status["status"] != "requires_review" or status["done"] != 71 or status["completed"] != 70 or status["failed"] != 1:
        raise ValueError("Unexpected terminal population")
    entries = []
    for path in sorted(run.rglob("*")):
        if path.is_file():
            data = shared_read(path)
            entries.append({"path": path.relative_to(run).as_posix(), "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})
    for item in entries:
        if file_sha(run / item["path"]) != item["sha256"]:
            raise ValueError("Run changed during sealing")
    publish_json(out, {"schema": "remaining71-v2-terminal-seal", "run_dir": str(run), "created_utc": now(),
        "completed": 70, "failed": 1, "failed_query_id": "c1_q43", "files": entries})
    print({"files": len(entries), "seal_path": str(out), "seal_sha256": file_sha(out)})
