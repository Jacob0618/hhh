"""Independent C1/C2 full1540 thin adapter; original plain.one is unchanged.

Prepare/check/retrieve/seal/seal-firstpass/recover are offline. Only qa dispatches Reader/Judge.
Retrieval uses exact QUERY cache-only embeddings and newly copied native stores.
"""
from __future__ import annotations
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from types import FunctionType as _FunctionType
import ast as _ast
import inspect as _inspect
import json as _json
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from tools import run_regroup_refinement_qa_v1 as _template
from tools import boundary_variants_qa_sources_v1 as _sources
from jev_mem.providers import ModelAPI as _ModelAPI
from jev_mem.sdk20_cached_api import digest as _request_digest

# Copy bindings, never monkeypatch old modules or any active H implementation.
for _name,_value in vars(_template).items():
    if not _name.startswith("__"):
        globals()[_name]=_value
sources=_sources
SCHEMA="boundary-variants-plain-qa-full1540-v1"
POLICY=_deepcopy(_template.POLICY)
POLICY.update(slots_per_arm=1540,
              cache_policy="exact_QUERY_pack_cache_only_no_historical_Reader_or_Judge",
              reference_comparison="frozen_native_baseline1540_original_scores_not_redefined",
              source_gate="one_complete_strict10_bank_boundary_source_review",
              declared_source_arms=dict(_sources.audit.ARM_METHODS))
CODE=tuple(dict.fromkeys((*_template.CODE,
    *_sources.audit.CODE,
    "tools/run_boundary_variants_qa_v1.py","tools/boundary_variants_qa_sources_v1.py",
    "tools/audit_boundary_variants_banks_v1.py","tools/run_boundary_variants_v1.py",
    "tools/freeze_sdk20_full1540_v1.py","jev_mem/qa_embedding_cache_pack_v1.py")))


CARDINALITY_FIELDS={"prepare":"qa_total","retrieve":"expected_slots","qa":"planned_per_arm"}


def _cardinality_code(original,field):
    """Allow exactly one named reporting dict field400→1540; no other edits."""
    tree=_ast.parse(_inspect.getsource(original));matches=[]
    for node in _ast.walk(tree):
        if isinstance(node,_ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,_ast.Constant) and key.value==field:
                    if not isinstance(value,_ast.Constant) or type(value.value)is not int or value.value!=400:
                        raise ValueError("Legacy cardinality field changed; explicit review required")
                    matches.append(value)
    if len(matches)!=1:raise ValueError("Exactly one named legacy reporting cardinality required")
    matches[0].value=1540;namespace={}
    exec(compile(_ast.fix_missing_locations(tree),__file__+"::"+original.__name__,"exec"),original.__globals__,namespace)
    return namespace[original.__name__].__code__


REBOUND_FUNCTIONS=tuple(name for name,value in vars(_template).items()
                        if isinstance(value,_FunctionType) and value.__module__==_template.__name__)
for _name in REBOUND_FUNCTIONS:
    _original=getattr(_template,_name)
    _code=(_cardinality_code(_original,CARDINALITY_FIELDS[_name]) if _name in CARDINALITY_FIELDS else _original.__code__)
    _fn=_FunctionType(_code,globals(),_name,
                      _original.__defaults__,_original.__closure__)
    _fn.__kwdefaults__=_original.__kwdefaults__
    _fn.__module__=__name__;_fn.__qualname__=_name;_fn.__doc__=_original.__doc__
    globals()[_name]=_fn


class CacheOnlyEmbeddings:
    """Constructor placeholder with no credentials or transport implementation."""
    embed=_ModelAPI.embed
    keys={"GEMINI_API_KEY":"offline-cache-only-placeholder"}

    def __init__(self,entries,*,context=None):
        self.entries,self.context,self.events=entries,context or {},[]

    def scoped(self,**context):
        return type(self)(self.entries,context={**self.context,**context})

    def _redact(self,error):
        return str(error)[:1500]

    def count_tokens(self,*args,**kwargs):
        raise RuntimeError("New countTokens prohibited in C cache-only retrieval")

    def request(self,provider,model,body,*,operation,stage):
        signature={"provider":provider,"model":model,"operation":operation,"body":body}
        key=_request_digest(signature);row=self.entries.get(key)
        if row is None:raise RuntimeError("Missing exact frozen QUERY embedding; no network fallback: "+key)
        path=sources.checked_pointer(row["target"]);raw=path.read_bytes()
        if embedding_pack.validate_raw(raw,key)!=signature:
            raise ValueError("Frozen QUERY request signature changed")
        self.events.append({"request_hash":key,"stage":stage,"context":self.context,"cache_hit":True,
                            "model_calls":0,"source":row["source"],"target":row["target"],"timestamp_utc":now()})
        return _json.loads(raw)["response"]


def retrieve_worker(run_dir,ci):
    """Same original SDK20 retrieve_arm body, exact cache-only API, no DPAPI."""
    from jev_mem.local_bm25_context_v2 import forbid_network
    run=Path(run_dir);m=manifest(run,conversation=ci);folder=run/f"conv_{ci}"
    native_qa.durable_intent(folder/"retrieval_execution_intent.json",
                            {"protocol_sha256":m["protocol_sha256"],"conversation":ci,"pid":os.getpid(),"created_utc":now()})
    pack=embedding_pack.verify_pack(m["embedding_cache_pack"],conversation=ci)
    entries={e["request_hash"]:e for e in pack["entries"] if e["conversation"]==ci}
    api=CacheOnlyEmbeddings(entries,context={"conversation":ci,"architecture":SCHEMA,"phase":"retrieval"})
    rows,_=sources.cohort();resource=m["resources"]
    with forbid_network() as guards,local_bm25_context(resource["snapshot"],resource["files_sha256"]):
        native_qa.retrieve_arm(run,ci,m["arm"],rows[str(ci)],m,api)
        if any(g.call_count for g in guards):raise RuntimeError("Unexpected retrieval network attempt")
    selections=[verify(folder/"selection"/m["arm"]/(q["id"]+".json")) for q in rows[str(ci)]]
    return {"conversation":ci,"terminal":True,"total":len(selections),
            "counts":dict(Counter(s["status"] for s in selections)),"model_calls":0}


_manifest_core=manifest
_terminal_core=terminal
_qa_core=qa
_prepare_core=prepare


def prepare(run_dir,gate_path):
    gate=sources.verify(gate_path)
    writer_run=sources.checked_pointer(gate["source_manifest"]).parent
    # Keep original Writer locks throughout re-admission and the byte copy.
    with sources.audit.source_locks(writer_run):return _prepare_core(run_dir,gate_path)


def manifest(run_dir,*,conversation=None,recheck_sources=False):
    m=_manifest_core(run_dir,conversation=conversation,recheck_sources=recheck_sources)
    if m["phase_kind"]=="technical_recovery":
        seal=verify(sources.checked_pointer(m["parent"]["firstpass_seal"]))
        if (seal.get("schema")!=SCHEMA+"-firstpass-seal" or seal.get("total")!=1540 or
            seal["manifest"]!=m["parent"]["manifest"] or seal["outcome"]!=m["parent"]["outcome"] or
            seal["source_jobs"]!=m["source_manifest"]):
            raise ValueError("Recovery must bind the sealed whole firstpass")
    return m


def terminal(run_dir):
    m,out=_terminal_core(run_dir)
    if out.get("planned_per_arm")!=1540 or (m["phase_kind"]=="firstpass" and out.get("total")!=1540):
        raise ValueError("Firstpass must contain all1540 slots")
    return m,out


def seal_firstpass(run_dir):
    run=Path(run_dir).resolve();m,out=terminal(run)
    if m["phase_kind"]!="firstpass":raise ValueError("Only firstpass can be sealed")
    value={"schema":SCHEMA+"-firstpass-seal","protocol_sha256":m["protocol_sha256"],"total":1540,
           "manifest":sources.pointer(run/"run_manifest.json"),"outcome":sources.pointer(run/"run_outcome.json"),
           "source_jobs":m["source_manifest"],"stage_inventory":out["results"],"model_calls":0,
           "unknown_policy":"unknown artifacts retained; never implicit redispatch"}
    path=run/"firstpass_seal.json"
    if path.exists():
        if {k:v for k,v in verify(path).items() if k!="content_sha256"}!=value:
            raise ValueError("Sealed firstpass changed")
    else:save_new(path,value)
    return verify(path)


def qa(run_dir,*,api_factory=None):
    # Admission establishes full cohort. This guard also catches partial jobs
    # before constructing clients if any future adapter changes that binding.
    m=execution_manifest(run_dir)
    if m["phase_kind"]=="firstpass" and len(m["jobs"])!=1540:
        raise ValueError("No partial firstpass QA dispatch")
    out=_qa_core(run_dir,api_factory=api_factory)
    if m["phase_kind"]=="firstpass":seal_firstpass(run_dir)
    return out


def prepare_recovery(firstpass_run):
    source_run=Path(firstpass_run).resolve();sealed=seal_firstpass(source_run);m,out=terminal(source_run)
    jobs=recovery_jobs(m,out);run=source_run.with_name(source_run.name+"_recovery_v1")
    if run.exists():raise FileExistsError("One fixed recovery namespace; never overwrite or retry again")
    value={k:copy.deepcopy(v) for k,v in m.items() if k not in {"protocol_sha256","jobs"}}
    value.update(created_utc=now(),phase_kind="technical_recovery",
                 phase_id=SCHEMA+"_"+m["arm"]+"_technical_recovery",jobs=jobs,cache_roots=[],
                 parent={"run_dir":str(source_run),"manifest":sealed["manifest"],"outcome":sealed["outcome"],
                         "firstpass_seal":sources.pointer(source_run/"firstpass_seal.json")})
    run.mkdir(parents=True,exist_ok=False)
    publish_json(run/"run_manifest.json",{**value,"protocol_sha256":digest(value)})
    return {"run_dir":str(run),"jobs":len(jobs),"model_calls":0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","check","retrieve","seal","qa","seal-firstpass","recover"))
    p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--gate",type=Path)
    a=p.parse_args();run=a.run_dir.resolve()
    if a.stage=="prepare":
        if a.gate is None:p.error("prepare requires --gate")
        result=prepare(run,a.gate)
    elif a.stage=="check":
        m=manifest(run,recheck_sources=True);result={"valid":True,"arm":m["arm"],"phase_kind":m["phase_kind"],"model_calls":0}
    elif a.stage=="retrieve":result=retrieve(run)
    elif a.stage=="seal":result={"jobs":len(seal_jobs(run)["jobs"]),"model_calls":0}
    elif a.stage=="qa":result=qa(run)
    elif a.stage=="seal-firstpass":result={"sealed_slots":seal_firstpass(run)["total"],"model_calls":0}
    else:result=prepare_recovery(run)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
