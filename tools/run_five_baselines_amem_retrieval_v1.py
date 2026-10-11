"""One immutable, independently admitted retrieval continuation; no bank rebuild.

The original failed firstpass namespace remains unchanged. This entry has no
resume path and no historical cache roots. The only permitted method request is
the qualified complete-transport, truncated c2_q42 keyword request, exactly once.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tools import audit_amem_retrieval_admission_v1 as admission
from tools import baselines_amem_retrieval_v1 as adapter
from tools import baseline_controller_guard_v3 as guard
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, read, sha, save_new, now
from jev_mem.gemini25_ablation_provider_v1 import generation_body, MODEL
from jev_mem.sdk20_cached_api import digest

ROOT=admission.ROOT; SOURCE=admission.SOURCE; FINAL=admission.FINAL; WORK_OWNER=admission.WORK_OWNER
QID=admission.QID; KEY=admission.KEY; CONTEXT=admission.CONTEXT
base=admission.base; amem=admission.amem
SCHEMA='amem-single-retrieval-continuation-v1'
REVIEW_SCHEMA='amem-single-retrieval-core-independent-review-v1'
CORE=('tools/baselines_amem_retrieval_v1.py','tools/baseline_controller_guard_v3.py',
      'tools/audit_amem_retrieval_admission_v1.py','tools/run_five_baselines_amem_retrieval_v1.py',
      'tools/seal_amem_retrieval_continuation_v1.py','tools/archive_amem_continuation_v4.py',
      'tools/baseline_controller_guard_v2.py','tools/check_experiment_health.py')
validate_proof=admission.validate_proof


def marker_value(proof):
    return {'schema':'amem-single-retrieval-replacement-marker-v1','query_id':QID,
            'request_hash':KEY,'signature':proof['replacement']['signature'],
            'context':CONTEXT,'stage':'amem_query_keywords','logical_limit':1,
            'old_raw':proof['original_failure']['raw'],
            'replacement_certificate':proof['replacement_certificate']}


class ReplacementAPI(BaselineAPI):
    def __init__(self,run_dir,*,proof,context,cache_roots=(),**kwargs):
        if (Path(run_dir).resolve()!=WORK_OWNER/'work/api_method' or context!=CONTEXT or cache_roots
                or proof.get('query_id')!=QID or proof['replacement'].get('request_hash')!=KEY
                or proof['replacement'].get('retry_limit')!=1 or kwargs.get('retries',2)!=2):
            raise BaselineFatal('Only the qualified one-query owner/context and original transport policy are permitted')
        self.proof=proof;self.used=False
        self.marker=WORK_OWNER/'work/one_use_retrieval_replacement.json'
        if self.marker.exists():raise BaselineFatal('The logical retrieval replacement quota is already consumed')
        super().__init__(run_dir,context=deepcopy(context),cache_roots=(),**kwargs)

    def scoped(self,**context):
        if {**self.context,**context}!=CONTEXT:raise BaselineFatal('Replacement context cannot change')
        return self

    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        signature={'provider':provider,'model':model,'operation':operation,'body':body}
        if (self.used or not cache or signature!=self.proof['replacement']['signature'] or digest(signature)!=KEY
                or self.context!=CONTEXT or stage!='amem_query_keywords'
                or Path(self.run_dir).resolve()!=WORK_OWNER/'work/api_method'):
            raise BaselineFatal('No additional, changed, or unqualified method request is allowed')
        if any((self.run_dir/name).exists() for name in ('api_calls.jsonl','logical_events.jsonl')) or any(
                p.is_file() for name in ('api_cache','request_intents','cache_lineage','parser_lineage') for p in (self.run_dir/name).glob('*')):
            raise BaselineFatal('Fresh method API namespace required; no cache or unresolved attempt can be reused')
        save_new(self.marker,{**marker_value(self.proof),'created_utc':now()})
        self.used=True
        # The durable marker precedes the unchanged original physical retry loop.
        return super().request(provider,model,body,operation=operation,stage=stage,cache=True)

    def generate(self,prompt,*,model=MODEL,system=None,json_output=False,max_output=16384,stage='unspecified'):
        if not json_output:raise BaselineFatal('The exact original JSON generation request is required')
        body=generation_body(prompt,system=system,json_output=json_output,max_output=max_output)
        self.request('gemini',model,body,stage=stage)
        raw=self.run_dir/'api_cache'/(KEY+'.json')
        value,journal=amem.decode_generation_json(read(raw),raw_artifact_sha256=sha(raw))
        save_new(self.run_dir/'parser_lineage'/(KEY+'.json'),journal)
        return value

    def embed(self,*args,**kwargs):
        raise BaselineFatal('Provider embeddings are forbidden for this existing MiniLM bank')


def audit_sidecar(manifest,proof,final_root,work_owner,*,view=None):
    """Pure full method audit, also runnable solely from archived logical paths."""
    from tools import seal_amem_retrieval_continuation_v1 as union
    view=view or union.LiveView();P=union.path_type(view)
    if (Path(final_root)!=FINAL or Path(work_owner)!=WORK_OWNER or manifest.get('query_id')!=QID
            or manifest.get('work_owner')!=str(WORK_OWNER) or proof.get('query_id')!=QID):
        raise ValueError('Exact sidecar identity required')
    def bound(record):
        if view.sha(record['path'])!=record['sha256']:raise ValueError('Bound method evidence changed')
        return view.read(record['path'])
    if bound(manifest['proof'])!=proof or bound(proof['replacement_certificate'])!=proof['replacement']:
        raise ValueError('Method proof/certificate differs')
    oldraw=bound(proof['original_failure']['raw'])
    if oldraw.get('request')!=proof['replacement']['signature'] or digest(oldraw['request'])!=KEY:
        raise ValueError('Original qualified failed raw request differs')
    state=bound(proof['bank']['state']);oldcall=bound(proof['original_failure']['native_call'])
    if (not state.get('build_complete') or state['memory_sha256']!=proof['bank']['memory_sha256']
            or state['embeddings_sha256']!=proof['bank']['embeddings']['sha256'] or len(state['notes'])!=663):
        raise ValueError('Original terminal bank binding differs')
    api=P(WORK_OWNER/'work/api_method');native=P(WORK_OWNER/'native/c2')
    marker_path=WORK_OWNER/'work/one_use_retrieval_replacement.json';marker=view.read(marker_path)
    if {k:v for k,v in marker.items() if k!='created_utc'}!=marker_value(proof):raise ValueError('One-use marker differs')
    if {p.stem for p in (api/'api_cache').glob('*.json')}!={KEY} or {p.stem for p in (api/'request_intents').glob('*.json')}!={KEY}:
        raise ValueError('Exactly one method raw and request intent required')
    if list((api/'cache_lineage').glob('*')) or {p.stem for p in (api/'parser_lineage').glob('*.json')}!={KEY}:
        raise ValueError('No historical method cache reuse; exactly one strict parser journal required')
    rawpath=api/'api_cache'/(KEY+'.json');intentpath=api/'request_intents'/(KEY+'.json')
    raw=view.read(rawpath);intent=view.read(intentpath)
    if (intent.get('signature')!=proof['replacement']['signature'] or digest(intent['signature'])!=KEY
            or raw.get('request')!=intent['signature'] or raw.get('stage')!='amem_query_keywords' or intent.get('request_hash')!=KEY
            or intent.get('context')!=CONTEXT or intent.get('stage')!='amem_query_keywords' or intent.get('retries')!=2
            or datetime.fromisoformat(marker['created_utc'])>datetime.fromisoformat(intent['created_utc'])):
        raise ValueError('Exact request identity, context, quota order or original retry policy differs')
    events=[json.loads(line) for line in view.bytes(api/'logical_events.jsonl').splitlines() if line.strip()]
    if len(events)!=1:raise ValueError('Exactly one terminal logical method consumer required')
    event={k:v for k,v in events[0].items() if k!='recorded_utc'}
    expected={'request_hash':KEY,'stage':'amem_query_keywords','provider':'gemini','model':MODEL,
              'operation':'generateContent','source':'new_request','context':CONTEXT,
              'cache_path':str(rawpath),'cache_sha256':view.sha(rawpath),'error_type':None}
    if event!=expected:raise ValueError('Logical method consumer differs from complete fresh producer')
    calls=[json.loads(line) for line in view.bytes(api/'api_calls.jsonl').splitlines() if line.strip()]
    if (not 1<=len(calls)<=3 or [r.get('attempt') for r in calls]!=list(range(1,len(calls)+1))
            or calls[-1].get('status')!=200 or any(r.get('status') not in (429,500,502,503,504,529,'transport_error') for r in calls[:-1])):
        raise ValueError('Original bounded physical producer sequence differs')
    for call in calls:
        if (call.get('request_hash')!=KEY or call.get('context')!=CONTEXT or call.get('stage')!='amem_query_keywords'
                or call.get('cache_hit') or any(call.get(k)!=expected[k] for k in ('provider','model','operation'))):
            raise ValueError('Unexpected method physical producer')
    times=[datetime.fromisoformat(intent['created_utc']),*[datetime.fromisoformat(r['timestamp_utc']) for r in calls],
           datetime.fromisoformat(events[0]['recorded_utc'])]
    if times!=sorted(times) or calls[-1].get('usage')!=raw['response'].get('usageMetadata',{}):
        raise ValueError('Producer time order or retained response usage differs')
    if calls[-1].get('resolved_model')!=raw['response'].get('modelVersion'):
        raise ValueError('Final physical producer model differs from retained response')
    parsed,journal=amem.decode_generation_json(raw,raw_artifact_sha256=view.sha(rawpath))
    if view.read(api/'parser_lineage'/(KEY+'.json'))!=journal:raise ValueError('Strict parser replay differs')
    q=proof['missing_question'];query_key=amem._v1._sha({'question':q['question'],'memory_sha256':state['memory_sha256']})
    callname='query_'+query_key+'_amem_query_keywords.json';callpath=native/'calls'/callname
    if {p.name for p in (native/'calls').glob('*.json')}!={callname} or {p.name for p in (native/'retrieval').glob('*.json')}!={query_key+'.json'}:
        raise ValueError('Only the exact single native query artifacts are allowed')
    if (native/'state.json').exists() or list((native/'cache').glob('*')):raise ValueError('No new memory bank/cache permitted')
    call=view.read(callpath);req=oldcall['request']
    controller=amem._Controller(None,Path('.'));controller.memory_count=len(state['notes'])
    value,normalization=controller._derive(parsed,req['response_format']['json_schema']['schema'],'amem_query_keywords')
    lineage=amem._v2._transport_lineage(SimpleNamespace(events=[event]))
    if (call.get('schema')!=amem.SCHEMA or call.get('base_adapter_sha256')!=amem._v2.BASE_ADAPTER_SHA256
            or call.get('request')!=req or call.get('request_sha256')!=amem._v1._sha(req) or call.get('status')!='success'
            or call.get('raw_response')!=parsed or call.get('raw_response_payload_sha256')!=amem._v1._sha(parsed)
            or call.get('parsed_response')!=value or call.get('normalized_parsed_response')!=value
            or call.get('normalization')!=normalization or call.get('transport_lineage')!=lineage):
        raise ValueError('Original native query consumer/schema/lineage differs')
    result=view.read(native/'retrieval'/(query_key+'.json'));seeds=result['seed_indices']
    if len(seeds)!=10 or len(set(seeds))!=10 or any(type(i) is not int or not 0<=i<len(state['notes']) for i in seeds):
        raise ValueError('Original ten native seed indices required')
    # The scientific formatter is replayed exactly. Numeric seed search used the
    # bound existing vectors at live retrieval; disposable vectors are not needed
    # for retained text/evidence reconstruction after cleanup.
    classes=amem._runtime._classes(None,None);system=classes['AgenticMemorySystem']()
    for note in state['notes']:
        obj=classes['MemoryNote'].__new__(classes['MemoryNote']);obj.__dict__.update(deepcopy(note));system.memories[obj.id]=obj
    system.retriever.search=lambda query,k:seeds
    keywords=value['keywords'];context=system.find_related_memories_raw(keywords,k=10)
    notes=list(system.memories.values());evidence=[]
    for seed_rank,index in enumerate(seeds):
        for note_index,via in [(index,'seed')]+[(linked,'link') for linked in notes[index].links[:11]]:
            if type(note_index) is not int or not 0<=note_index<len(notes):raise ValueError('Original bank link invalid')
            evidence.append({'rank':len(evidence),'seed_rank':seed_rank,'seed_note_index':index,'note_index':note_index,
                             'via':via,'source':state['provenance'][note_index],'note':amem._v1._note_record(notes[note_index])})
    expected_result={'schema':amem._runtime.SCHEMA,'question':q['question'],'keywords':keywords,'seed_k':10,
        'seed_indices':seeds,'context':context,'evidence':evidence,'memory_sha256':state['memory_sha256'],
        'upstream_revision':amem.UPSTREAM_REVISION,'query_call':'calls/'+callname}
    if result!=expected_result:raise ValueError('Entire original native context/evidence differs')
    capture_path=WORK_OWNER/'results/c2/retrieval'/(QID+'.json');capture=view.read(capture_path)
    if capture.get('status')!='complete' or capture.get('value')!=result or capture.get('events')!=[event]:
        raise ValueError('Whole retrieval capture and logical consumer differ')
    if view.read(capture_path.with_suffix('.intent.json')).get('context')!=CONTEXT:raise ValueError('Retrieval intent scope differs')
    return {'status':'passed','query_id':QID,'request_hash':KEY,'logical_replacements':1,'physical_attempts':len(calls),
            'marker_sha256':view.sha(marker_path),'intent_sha256':view.sha(intentpath),'raw_sha256':view.sha(rawpath),
            'parser_sha256':view.sha(api/'parser_lineage'/(KEY+'.json')),'native_call_sha256':view.sha(callpath),
            'capture_sha256':view.sha(capture_path),'result_payload_sha256':amem._v1._sha(result),
            'bank':proof['bank'],'replacement_certificate':proof['replacement_certificate'],
            'seed_search':'unchanged native retrieval over bound source vectors; retained formatter/evidence replayed without re-encoding'}


def code_hashes(parent):
    return {**parent['code_sha256'],**{p:sha(ROOT/p) for p in CORE}}


def check_review(path,proof_path,parent):
    path=Path(path).resolve()
    if path.is_relative_to(SOURCE) or not path.is_relative_to(ROOT):raise ValueError('Independent review must be outside all new/original run evidence')
    review=read(path)
    if (review.get('schema')!=REVIEW_SCHEMA or review.get('status')!='passed'
            or review.get('scope')!='single_query_core_implementation_and_actual_admission'
            or review.get('proof')!=admission.binding(proof_path) or review.get('code_sha256')!=code_hashes(parent)
            or review.get('upstream_sha256')!=parent['upstream_sha256']
            or review.get('archive_cleanup_downstream_authorized') is not False):
        raise ValueError('Full final-hash independent core/admission review required')
    if not review.get('evidence_sha256'):raise ValueError('Independent review needs retained evidence')
    for p,h in review['evidence_sha256'].items():admission.bound({'path':p,'sha256':h})
    return admission.binding(path)


def expected_manifest(proof_path,proof,review_path):
    parent=read(admission.bound(proof['parent_manifest']))
    return {'schema':SCHEMA,'method':'amem','query_id':QID,'self_root':str(FINAL),'final_root':str(FINAL),
            'work_owner':str(WORK_OWNER),'parent_manifest':proof['parent_manifest'],
            'proof':admission.binding(proof_path),'independent_review':check_review(review_path,proof_path,parent),
            'questions':parent['questions'],'code_sha256':code_hashes(parent),'upstream_sha256':parent['upstream_sha256'],
            'replacement_certificate':proof['replacement_certificate'],'original_routes':proof['original_routes'],
            'scientific_configuration':{k:parent[k] for k in ('generation_model','thinkingBudget','temperature','internal_max_output',
                'reader_judge_max_output','evaluation_protocol','judge_prompt_sha256','metric','specification')},
            'lifecycle_authorization':'archive/cleanup/downstream require a separate reviewed extension'}


def prepare(review,proof_path=admission.DEFAULT_PROOF):
    guard.assert_stopped()
    validate_namespaces()
    if FINAL.exists() or WORK_OWNER.exists():raise FileExistsError('Both new namespaces must be absent')
    proof=validate_proof(proof_path);value=expected_manifest(proof_path,proof,review)
    save_new(FINAL/'run_manifest.json',value)
    base.status(FINAL,'prepared',query_id=QID,source_status='needs_review')
    return {'status':'prepared','manifest':admission.binding(FINAL/'run_manifest.json'),'model_calls':0}


def manifest(*,return_proof=False):
    guard.assert_stopped();validate_namespaces();m=read(FINAL/'run_manifest.json')
    admission.bound(m['proof']);proof=validate_proof(m['proof']['path'])
    if m!=expected_manifest(m['proof']['path'],proof,m['independent_review']['path']):raise ValueError('Immutable continuation manifest differs')
    for p,h in m['upstream_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen upstream source changed')
    return (m,proof) if return_proof else m


def validate_namespaces():
    from tools.archive_amem_continuation_v4 import safe_path
    for path in (SOURCE,FINAL,WORK_OWNER):safe_path(path)


def copy_new(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as handle:handle.write(source.read_bytes())
    if sha(source)!=sha(target):raise ValueError('Permanent evidence copy differs')


def run():
    m,proof=manifest(return_proof=True)
    if base.BaselineAPI is not BaselineAPI:raise ValueError('Original shared QA provider binding changed in this interpreter')
    if WORK_OWNER.exists() or (FINAL/'execution_intent.json').exists():raise FileExistsError('No implicit continuation reentry is permitted')
    save_new(FINAL/'execution_intent.json',{'created_utc':now(),'manifest':admission.binding(FINAL/'run_manifest.json'),
                                         'query_id':QID,'logical_replacement_limit':1})
    WORK_OWNER.mkdir(parents=True,exist_ok=False)
    base.status(FINAL,'retrieval',query_id=QID)
    try:
        api=ReplacementAPI(WORK_OWNER/'work/api_method',proof=proof,context=CONTEXT)
        q=proof['missing_question'];capture_path=WORK_OWNER/'results/c2/retrieval'/(QID+'.json')
        result=base.capture_stage(capture_path,lambda:adapter.retrieve(q['question'],proof['bank']['state']['path'],WORK_OWNER/'native/c2',api,
            state_sha256=proof['bank']['state']['sha256'],embeddings_sha256=proof['bank']['embeddings']['sha256']),api)
        for p in (capture_path,capture_path.with_suffix('.intent.json')):copy_new(p,FINAL/'results/technical_retrieval'/p.name)
        if admission.source_inventory()!=proof['source_inventory']:raise ValueError('Original source changed during retrieval')
        if result['status']!='complete':raise BaselineFatal('The only permitted retrieval replacement failed; no redispatch allowed')
        audit_sidecar(m,proof,FINAL,WORK_OWNER)
        base.status(FINAL,'first_qa',query_id=QID)
        base.qa_one(WORK_OWNER,'amem',q,result['value'],phase='firstpass')
        from tools import seal_amem_retrieval_continuation_v1 as union
        union.copy_qa('firstpass')
        if admission.source_inventory()!=proof['source_inventory']:raise ValueError('Original source changed during first QA')
        summary=union.finish_run(m,proof,FINAL,WORK_OWNER)
        if admission.source_inventory()!=proof['source_inventory']:raise ValueError('Original source changed during final seal')
        stage='complete_awaiting_result_review' if summary['audit_status']=='no_invalid_artifacts' else 'needs_review'
        base.status(FINAL,stage,query_id=QID,summary_sha256=sha(FINAL/'results/summary.json'))
        return {'status':stage,'final_root':str(FINAL),'summary':summary}
    except BaseException as exc:
        base.status(FINAL,'needs_review',query_id=QID,error=base.error_record(exc))
        raise


def snapshot():
    """Explicit read-only nested-owner view; original v4 status is never overwritten."""
    from tools.check_experiment_health import shared_read
    def get(p):return json.loads(shared_read(p)) if p.exists() else None
    result={'schema':'amem-single-retrieval-snapshot-v1','parent_status':get(SOURCE/'status.json'),
            'sidecar_status':get(FINAL/'status.json'),'replacement_marker':get(WORK_OWNER/'work/one_use_retrieval_replacement.json')}
    result['ledgers']={str(p):{'physical_lines':len(shared_read(p).splitlines())} for p in (WORK_OWNER/'work').glob('api_*/api_calls.jsonl')}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','prepare','check','run','snapshot'])
    parser.add_argument('--review',type=Path);parser.add_argument('--proof',type=Path,default=admission.DEFAULT_PROOF)
    parser.add_argument('--qualification',type=Path,default=admission.DEFAULT_QUALIFICATION)
    parser.add_argument('--process-probe',action='store_true');args=parser.parse_args()
    if args.process_probe:
        if args.command!='run':raise ValueError('Process probe requires actual run-style launcher arguments')
        guard.assert_stopped();result={'status':'passed','scope':'process_guard_only','model_calls':0}
    elif args.command=='audit':result=admission.audit(args.proof,args.qualification)
    elif args.command=='prepare':result=prepare(args.review,args.proof)
    elif args.command=='check':result={'status':'passed','manifest':manifest()}
    elif args.command=='snapshot':result=snapshot()
    else:result=run()
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
