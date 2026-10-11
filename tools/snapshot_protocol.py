"""Save runnable source and package versions before the main QA stages."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.providers import dump_json

target=ROOT/"research/protocol_snapshot_reader_v2"
target.mkdir(parents=True,exist_ok=True)
sources=list((ROOT/"jev_mem").glob("*.py"))+list((ROOT/"tools").glob("run*exper*.py"))
sources += [ROOT/"configs/experiment_plan.json",ROOT/"pyproject.toml"]
hashes={}
for source in sources:
    relative=source.relative_to(ROOT)
    body=source.read_bytes()
    path=target/relative
    if path.exists() and path.read_bytes()!=body:
        raise RuntimeError("Snapshot exists with different content; create a new version")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(body)
    hashes[str(relative)]=hashlib.sha256(body).hexdigest()
packages={dist.metadata["Name"]:dist.version for dist in importlib.metadata.distributions()}
dump_json(target/"manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
    "source_sha256":hashes,"installed_packages":packages,"python":sys.version,
    "qa_protocol":"brief_answer_local_evidence_ids_v2",
    "development_seen_before_freeze":["c0_q"+str(i) for i in range(8)],
    "old_research_history":"not claimed clean holdout"})
print(json.dumps({"snapshot":str(target),"source_files":len(hashes),"packages":len(packages)}))
