"""Offline strict source/raw/date/index audit for closed C1/C2 writer banks.

Development [0,1] audits unlock expansion only. Only a complete ten-bank strict
audit can produce a QA gate. Failed/partial banks are retained and never exported
or relabelled complete. Historical pinned metadata exceptions do not transfer.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import os
import threading
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_regroup_refinement_banks_v1 as strict
from tools import run_mem0_regroup_sdk20 as writer
from tools import run_boundary_variants_v1 as boundary
from tools import run_boundary_variants_parallel_v1 as parallel
from tools import mem0_regroup_firstpass_review_v2 as index
from tools.run_sdk20_modules import digest, sha, verify, save_new, now
from jev_mem.local_bm25_context_v2 import forbid_network

SCHEMA = "boundary-variants-terminal-bank-audit-v1"
GATE_SCHEMA = "boundary-variants-full1540-qa-gate-v1"
ARM_METHODS = {"c1_multiscale": "boundary_multiscale", "c2_neighbour": "boundary_neighbor"}
CODE = ("tools/audit_boundary_variants_banks_v1.py", "tools/audit_regroup_refinement_banks_v1.py",
        "tools/mem0_regroup_firstpass_review_v2.py", "tools/review_regroup_refinement_banks_v2.py",
        "tools/run_boundary_variants_v1.py", "tools/run_boundary_variants_parallel_v1.py", "jev_mem/local_bm25_context_v2.py")
_HELD_LOCKS=threading.local()


@contextmanager
def source_locks(run):
    """Probe and hold existing source locks without changing any lock bytes."""
    run=Path(run).resolve();handles=[]
    if not hasattr(_HELD_LOCKS,"paths"):_HELD_LOCKS.paths=set()
    try:
        for path in (run.parents[1]/"driver.lock",run/"writer_execution.lock"):
            if not path.exists():continue
            if path in _HELD_LOCKS.paths:continue
            handle=path.open("r+b");handle.seek(0)
            try:
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                handle.close();raise ValueError("Writer/controller lock is held; no concurrent source audit/copy") from None
            handles.append((path,handle));_HELD_LOCKS.paths.add(path)
        yield
    finally:
        for path,handle in reversed(handles):
            handle.seek(0)
            if os.name=="nt":msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)
            handle.close()
            _HELD_LOCKS.paths.remove(path)


def closed_controllers(run, arm, conversations, *, locks_held=False):
    """Bind completed controller/worker evidence, or explicit serial closure."""
    if not locks_held:
        with source_locks(run):return closed_controllers(run,arm,conversations,locks_held=True)
    run=Path(run).resolve();parent=run.parents[1]
    evidence=[]
    for folder in sorted((parent/"parallel_controller_v1").glob("write_*")):
        mpath=folder/"manifest.json"
        cm=index.verify(mpath);policy=cm["policy"]
        selected=set(conversations)&set(policy["conversations"])
        if arm not in policy["arms"] or not selected:continue
        if (cm.get("schema")!=parallel.VERSION or policy.get("stage")!="write" or
            cm.get("parent")!=boundary.pointer(parent/"manifest.json") or
            cm.get("code_sha256")!={p:sha(ROOT/p) for p in parallel.FILES}):
            raise ValueError("Controller source/policy/code binding differs")
        state=index.read(folder/"parent_state.json")
        manifest_pointer=index.pointer(mpath)
        if (state.get("status")!="complete" or not state.get("finished_utc") or
            state.get("controller_manifest")!=manifest_pointer or state.get("stage")!="write" or
            state.get("requested_conversations")!=policy["conversations"] or state.get("arms")!=policy["arms"] or
            set(state.get("outcomes",{}))!={str(ci) for ci in policy["conversations"]} or
            any(x.get("status")!="complete" for x in state["outcomes"].values())):
            raise ValueError("Active/interrupted/failed controller requires independent review")
        try: live=parallel.process_identity(state["pid"])
        except RuntimeError: live=None
        if live==state.get("parent_process_identity"):
            raise ValueError("Matching controller process remains live")
        workers=[]
        for ci in sorted(selected):
            job=folder/"conversations"/f"conv_{ci}"/arm
            terminal=index.verify(job/"terminal.json");worker=index.read(job/"worker_state.json")
            expected={"write_status":index.pointer(run/f"conv_{ci}"/arm/"write_status.json"),
                      "bank":index.pointer(run/f"conv_{ci}"/arm/"bank.json")}
            if (terminal.get("status")!="complete" or not terminal.get("finished_utc") or
                terminal.get("controller_manifest")!=manifest_pointer or
                (terminal.get("stage"),terminal.get("conversation"),terminal.get("arm"))!=("write",ci,arm) or
                terminal.get("outputs")!=expected or worker.get("status")!="complete" or
                worker.get("terminal")!=index.pointer(job/"terminal.json")):
                raise ValueError("Controller worker terminal/source binding differs")
            workers.append({"conversation":ci,"terminal":index.pointer(job/"terminal.json"),
                            "worker_state":index.pointer(job/"worker_state.json")})
        evidence.append({"manifest":manifest_pointer,"parent_state":index.pointer(folder/"parent_state.json"),"workers":workers})
    return {"mode":"parallel_closed" if evidence else "serial_closed","controllers":evidence}


def preflight(run, conversations, *, locks_held=False):
    run = Path(run).resolve(); m = writer.manifest(run)
    meta = m.get("boundary_variant", {})
    arm = meta.get("arm")
    if arm not in ARM_METHODS or m["arms"] != [arm] or conversations not in ([0,1], list(range(10))):
        raise ValueError("Only development01 or full10 boundary source audits supported")
    boundary.manifest(Path(meta["parent"]["path"]).parent)
    if boundary.pointer(meta["parent"]["path"]) != meta["parent"] or boundary.pointer(meta["global"]["path"]) != meta["global"]:
        raise ValueError("Boundary writer parent/global binding changed")
    closed_controllers(run,arm,conversations,locks_held=locks_held)
    targets, failures = [], []
    # Preflight ALL selected statuses before opening any SQLite.
    for ci in conversations:
        folder = run/f"conv_{ci}"/arm
        state = index.read(folder/"write_status.json")
        if (state.get("status") not in ("complete","failed") or not state.get("finished_utc") or
            (state.get("conversation"),state.get("arm"),state.get("protocol_sha256")) != (ci,arm,m["protocol_sha256"])):
            raise ValueError("Every selected bank must be closed terminal before audit")
        if state["status"] == "failed" or state.get("completed_groups") != state.get("expected_groups"):
            failures.append({"conversation":ci,"arm":arm,"original_write_status":state["status"],
                             "write_status":index.pointer(folder/"write_status.json"),
                             "completed_groups":state.get("completed_groups"),"expected_groups":state.get("expected_groups"),
                             "admission":"blocked_requires_independent_new_failure_proof; never partial admission"})
            continue
        bank = index.bound(state["bank"])
        if (bank.get("complete") is not True or bank.get("source_dataset_sha256") != writer.SOURCE_HASH or
            (bank.get("conversation"),bank.get("arm"),bank.get("protocol_sha256")) != (ci,arm,m["protocol_sha256"]) or
            index.verify(folder/"bank.json") != bank or state["facts"] != len(bank["facts"])):
            raise ValueError("Closed bank/status/seal identity mismatch")
        for p in (folder/"native").rglob("*"):
            if p.is_symlink() or (p.is_file() and p.name.endswith(("-wal","-journal")) and p.stat().st_size):
                raise ValueError("Symlink or live uncheckpointed database refused")
        targets.append((ci,arm,folder,state,state,bank))
    if failures:
        return m, targets, failures
    # Legacy audit body expects refinement.resources, only for BM25 comparison.
    # This is a private in-memory view; no manifest/source bytes are rewritten.
    view = copy.deepcopy(m); view["refinement"] = {"resources":meta["resources"]}
    return view, targets, []


def audit(run, conversations):
    run=Path(run).resolve()
    index.HASHES.clear()
    with source_locks(run),forbid_network() as guards:
        m, targets, failures = preflight(run, conversations,locks_held=True)
        if failures:
            return {"schema":SCHEMA,"arm":m["boundary_variant"]["arm"],"conversations":conversations,
                    "writer_manifest":index.pointer(run/"run_manifest.json"),"strict_passed":False,
                    "status":"blocked_original_writer_failure","failures":failures,"rows":[],
                    "qa_gate_passed":False,"model_calls":0,"source_mutations":0,
                    "limitations":["Original failed status stays failed; old pinned c3 exception cannot admit new C identities."]}
        # Use the original complete inspection BODY, with private globals only.
        # Its old outcome-report pointer is transparently rebound to the real
        # boundary terminal status; no nonexistent write_outcome is fabricated.
        facade = SimpleNamespace(**vars(index))
        def terminal_pointer(path):
            path=Path(path)
            return index.pointer(path.with_name("write_status.json") if path.name=="write_outcome.json" else path)
        facade.pointer = terminal_pointer
        env=dict(strict.inspect_bank.__globals__,index_audit=facade)
        inspect=FunctionType(strict.inspect_bank.__code__,env,strict.inspect_bank.__name__,strict.inspect_bank.__defaults__)
        dataset=ROOT/"data/locomo10.json"
        if index.sha(dataset) != m["dataset_sha256"]:
            raise ValueError("Frozen dataset drift")
        data=index.read(dataset); prompt_env=strict.prompt_environment(); extractor,nlp=index.load_extractor()
        rows=[]
        for target in targets:
            ci=target[0]; row=inspect(target,m,data[ci]["conversation"],prompt_env,extractor,nlp)
            row["writer_terminal"]=row.pop("outcome")
            row.update(write_status=index.pointer(target[2]/"write_status.json"),
                       group_inputs=index.pointer(target[2]/"group_inputs.json"),
                       original_write_status="complete",writer_terminal_kind="write_status_only")
            rows.append(row)
        resource=m["boundary_variant"]["resources"]
        bm25=strict.compare_persisted_fact_vectors([Path(r["native_dir"]) for r in rows],resource["snapshot"],resource["files_sha256"])
        if bm25["facts_compared"] != bm25["exact_matches"] or bm25["facts_compared"] != sum(r["facts"] for r in rows):
            raise ValueError("Complete BM25 source equivalence failed")
        if any(g.call_count for g in guards):
            raise ValueError("Audit network attempt")
        scheduling=closed_controllers(run,m["boundary_variant"]["arm"],conversations,locks_held=True)
        for p,h in list(index.HASHES.items()):
            if index.sha(p) != h:
                raise ValueError("Audit source bytes drift")
    return {"schema":SCHEMA,"created_utc":now(),"arm":m["boundary_variant"]["arm"],
            "writer_manifest":index.pointer(run/"run_manifest.json"),"conversations":conversations,
            "rows":rows,"banks":rows,"strict_passed":all(r["strict_passed"] for r in rows),
            "writer_scheduling":scheduling,
            "status":"strict_source_index_reviewed" if all(r["strict_passed"] for r in rows) else "blocked_entity_links",
            "offline_bm25_equivalence":bm25,"source_file_hashes":dict(index.HASHES),
            "code_sha256":{p:sha(ROOT/p) for p in CODE},"audit_tool":index.pointer(__file__),
            "qa_gate_passed":False,"model_calls":0,"source_mutations":0,"qa_gold_used":False,
            "limitations":["Raw/group input coverage is not per-fact semantic truth; semantic targets disclosed in every row.",
                            "Unused entity sparse omissions disclosed; partial entity-edge completeness not proved.",
                            "No original failed status rewritten; reviewed metadata exceptions require a new identity-pinned independent proof."]}


def write_gate(audit_path, output):
    report=verify(audit_path)
    if report.get("schema") != SCHEMA or report.get("conversations") != list(range(10)) or report.get("strict_passed") is not True:
        raise ValueError("Only a full ten-bank strict report admits QA")
    arm=report["arm"]; rows=report["rows"]
    if len(rows)!=10 or {(r["conversation"],r["arm"]) for r in rows}!={(ci,arm) for ci in range(10)}:
        raise ValueError("Whole distinct ten-bank cohort required")
    if report["code_sha256"] != {p:sha(ROOT/p) for p in CODE}:
        raise ValueError("Audit implementation binding drift")
    for p,h in report["source_file_hashes"].items():
        if sha(p)!=h: raise ValueError("Audited source drift")
    value={"schema":GATE_SCHEMA,"status":"passed","arm":ARM_METHODS[arm],"source_arm":arm,
           "source_manifest":report["writer_manifest"],"audit":index.pointer(audit_path),
           "checks":{"ten_complete_banks":True,"source_dates_and_generation_chain_verified":True,
                     "current_sdk20_index_dependencies_valid":True,"content_quality_findings_disclosed":True},
           "banks":[{"conversation":r["conversation"],"arm":ARM_METHODS[arm],"source_arm":arm,
                     "bank":r["bank"],"write_status":r["write_status"],"source_manifest":report["writer_manifest"],
                     "group_inputs":r["group_inputs"],"native_inventory":r["native_inventory"],
                     "source_kind":"strict_complete","original_write_status":"complete",
                     "audits":{"source_date":[index.pointer(audit_path)],"index":[index.pointer(audit_path)]}} for r in rows],
           "limitations":report["limitations"],"model_calls":0,"source_mutations":0}
    save_new(output,value)
    return index.pointer(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--writer-run",type=Path,required=True);p.add_argument("--conversations",nargs="+",type=int,default=list(range(10)))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--gate",type=Path)
    a=p.parse_args(); result=audit(a.writer_run,a.conversations);save_new(a.output,result)
    if a.gate: write_gate(a.output,a.gate)
    print({"audit":str(a.output),"strict_passed":result["strict_passed"],"banks":len(result["rows"]),"model_calls":0})


if __name__=="__main__":main()
