"""Class-preserving local BM25 recovery of the zero-call conv1 merged failure.

No old file or store is changed. Admission includes a fresh-process native
Memory construction test inside the new context, not only an encoder probe.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_mem0_regroup_bm25_v1 as prior
from tools import recover_mem0_regroup_sdk20 as legacy
from tools import run_mem0_regroup_sdk20 as frozen
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.run_sdk20_modules import digest, sha, now, verify
from jev_mem.local_bm25_context_v2 import PROTOCOL as BM25_PROTOCOL, local_bm25_context, compare_persisted_fact_vectors

SCHEMA = "dated-mem0-c1-merged-class-bm25-recovery-v2"
TARGET = (1, "dated_merged")
FAILURE_AUDIT = ROOT / "research/mem0_regroup_bm25_v1_class_api_failure_audit_2026-09-28.json"
CODE = ("tools/recover_mem0_regroup_bm25_v2.py", "jev_mem/local_bm25_context_v2.py")
POLICY = {"schema": SCHEMA, "target": list(TARGET), "resource_policy": BM25_PROTOCOL,
          "class_api": "subclass_original_SparseTextEmbedding_override_init_only",
          "changes": "resource_resolution_only_same_snapshot_and_scoring_defaults",
          "previous_failure": "zero_group_zero_model_calls_class_method_lookup",
          "success_banks": "three_existing_banks_reference_only",
          "bank_attempt": "new_namespace_once_failed_or_unknown_never_reenter",
          "native_smoke": "fresh_child_context_before_mem0_qdrant_import_network_forbidden",
          "writer_api": "unchanged_frozen_write_arm_and_legacy_PinnedRecoveryAPI"}

NATIVE_SMOKE = r'''
import sys,json,tempfile
from pathlib import Path
resources=json.load(sys.stdin)
from jev_mem.local_bm25_context_v2 import local_bm25_context,forbid_network
assert 'qdrant_client' not in sys.modules and 'mem0' not in sys.modules
class DummyAPI:
    keys={'GEMINI_API_KEY':'offline-only-noncredential','TYPESAFE_API_KEY':'offline-only-noncredential'}
    model_calls=0
    def request(self,*a,**k):
        self.model_calls+=1
        raise AssertionError('Model calls forbidden in native dependency smoke')
    generate=request
    embed=request
api=DummyAPI()
with forbid_network() as guards,local_bm25_context(resources['snapshot'],resources['files_sha256']):
    import fastembed
    assert isinstance(fastembed.SparseTextEmbedding,type)
    supported=fastembed.SparseTextEmbedding.list_supported_models()
    assert any(x['model']=='Qdrant/bm25' for x in supported)
    from tools import run_session_experiments as native
    with tempfile.TemporaryDirectory(prefix='bm25_v2_fresh_native_') as d:
        memory=native.open_memory(Path(d)/'native',api)
        try:
            from mem0.utils import spacy_models
            assert spacy_models.get_nlp_full() is not None and spacy_models.get_nlp_lemma() is not None
            facts=memory.vector_store._get_bm25_encoder()
            entities=memory.entity_store._get_bm25_encoder()
            assert facts is not None and entities is not None
            assert len(list(facts.embed(['local memory validation']))[0].indices)>0
            assert len(list(entities.embed(['memory validation']))[0].indices)>0
        finally:
            native.close_memory(memory)
    assert sum(g.call_count for g in guards)==0 and api.model_calls==0
print(json.dumps({'fresh_import_order_verified':True,'native_open_memory':True,'class_model_discovery':True,
    'facts_bm25':True,'entity_bm25':True,'nlp_full_and_lemma':True,'network_calls':0,'model_api_calls':0,
    'temporary_store_closed_and_removed':True}))
'''


def fresh_native_smoke(resources):
    result = subprocess.run([sys.executable, "-c", NATIVE_SMOKE], input=json.dumps(resources),
                            text=True, capture_output=True, cwd=ROOT, timeout=60)
    if result.returncode != 0:
        raise RuntimeError("Fresh-process native BM25 initialization failed; no live writer may start: " + result.stderr[-1800:])
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    if proof != {"fresh_import_order_verified": True, "native_open_memory": True,
            "class_model_discovery": True, "facts_bm25": True, "entity_bm25": True,
            "nlp_full_and_lemma": True, "network_calls": 0, "model_api_calls": 0,
            "temporary_store_closed_and_removed": True}:
        raise ValueError("Fresh native dependency proof is incomplete")
    return proof


def source_admission(source_run):
    source = Path(source_run).resolve()
    parent = prior.manifest(source)
    audit = read(FAILURE_AUDIT)
    if (audit.get("schema") != "dated-mem0-bm25-v1-class-api-failure-audit-v1"
            or audit.get("source_run") != str(source)
            or audit.get("source_manifest") != frozen.pointer(source / "run_manifest.json")
            or audit.get("zero_model_calls") is not True):
        raise ValueError("Reviewed zero-call class-API failure audit is required")
    folder = source / "conv_1/dated_merged"
    state = frozen.check_pointer(audit["status"])
    frozen.check_pointer(audit["outcome"])
    frozen.check_pointer(audit["bank_intent"])
    frozen.check_pointer(audit["input"])
    if (state.get("status") != "failed" or state.get("completed_groups") != 0 or state.get("facts") != 0
            or state.get("error") != {"type": "AttributeError", "message": "'function' object has no attribute 'list_supported_models'"}):
        raise ValueError("Unexpected failure requires a separate review")
    native = folder / "native"
    inventory = {str(p.relative_to(native)): sha(p) for p in native.rglob("*") if p.is_file()}
    if inventory != audit["native_inventory"]:
        raise ValueError("Frozen failed native namespace changed")
    if ((native / "api_calls.jsonl").exists() or list((native / "api_cache").glob("*.json"))
            or list((native / "request_intents").glob("*.json")) or list((native / "group_audit").rglob("*.json"))):
        raise ValueError("Source had model/group activity; not the zero-call failure")
    info = parent["local_bm25_recovery"]
    return parent, info["reused_complete_banks"], info["resources"]


def acceptance(reused, resources):
    vectors = compare_persisted_fact_vectors([r["native_dir"] for r in reused], resources["snapshot"], resources["files_sha256"])
    if vectors["facts_compared"] != 518 or vectors["exact_matches"] != 518:
        raise ValueError("Fixed 518-fact BM25 equivalence failed")
    return {"vectors": vectors, "fresh_native": fresh_native_smoke(resources)}


def prepare(run_dir, source_run):
    run = Path(run_dir).resolve()
    source = Path(source_run).resolve()
    if run.exists() or run == source or source in run.parents:
        raise FileExistsError("Class-preserving recovery requires another new namespace")
    parent, reused, resources = source_admission(source)
    proof = acceptance(reused, resources)
    target = next(x for x in parent["inputs"] if (x["conversation"], x["arm"]) == TARGET)
    src = Path(target["input"]["path"])
    dst = run / "conv_1/dated_merged/group_inputs.json"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(src.read_bytes())
    if sha(src) != sha(dst):
        raise ValueError("Frozen target input changed while copying")
    # Keep inherited evidence recognizable to the frozen parent validator.
    (run / "offline_bm25_acceptance.json").write_bytes((source / "offline_bm25_acceptance.json").read_bytes())
    value = copy.deepcopy(parent)
    value.pop("protocol_sha256")
    value["created_utc"] = now()
    value["inputs"] = [({**x, "input": frozen.pointer(dst)} if (x["conversation"], x["arm"]) == TARGET else x) for x in parent["inputs"]]
    value["local_bm25_class_recovery"] = {"schema": SCHEMA, "policy": POLICY,
        "source_run": str(source), "source_manifest": frozen.pointer(source / "run_manifest.json"),
        "failure_audit": frozen.pointer(FAILURE_AUDIT), "acceptance": proof,
        "resources": resources, "reused_complete_banks": reused,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "class_consumer_sha256": {p: sha(ROOT / p) for p in (
            ".venv/Lib/site-packages/qdrant_client/fastembed_common.py",
            ".venv/Lib/site-packages/qdrant_client/qdrant_fastembed.py")},
        "model_calls_during_prepare": 0, "source_date_index_gate_passed": False}
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    frozen.save_new(run / "class_preserving_bm25_acceptance.json", proof)
    return {"prepared": str(run), "protocol_sha256": digest(value), "reused_banks": 3,
            "new_writer_banks": 1, "fresh_native_smoke": True, "offline_fact_vectors": 518, "model_calls": 0}


def manifest(run):
    value = prior.manifest(run)
    info = value.get("local_bm25_class_recovery", {})
    if info.get("schema") != SCHEMA or info.get("policy") != POLICY:
        raise ValueError("Class-preserving BM25 recovery policy changed")
    for path, expected in {**info["code_sha256"], **info["class_consumer_sha256"]}.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Class-preserving runtime code changed: " + path)
    frozen.check_pointer(info["source_manifest"])
    frozen.check_pointer(info["failure_audit"])
    _, reused, resources = source_admission(info["source_run"])
    if (reused, resources) != (info["reused_complete_banks"], info["resources"]):
        raise ValueError("Class-preserving source references changed")
    proof = verify(Path(run) / "class_preserving_bm25_acceptance.json")
    if proof != {**info["acceptance"], "content_sha256": digest(info["acceptance"])}:
        raise ValueError("Class-preserving acceptance artifact changed")
    return value


def write(run):
    value = manifest(run)
    info = value["local_bm25_class_recovery"]
    acceptance(info["reused_complete_banks"], info["resources"])
    with frozen.writer_process_lock(run), local_bm25_context(info["resources"]["snapshot"], info["resources"]["files_sha256"]):
        return legacy.run_arm(run, TARGET[0], TARGET[1], value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--source-run", type=Path, required=True)
    for command in ("check", "write"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_dir, args.source_run)
    elif args.command == "check":
        value = manifest(args.run_dir)
        info = value["local_bm25_class_recovery"]
        result = {"protocol_sha256": value["protocol_sha256"], "acceptance": acceptance(info["reused_complete_banks"], info["resources"]), "model_calls": 0}
    else:
        result = write(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
