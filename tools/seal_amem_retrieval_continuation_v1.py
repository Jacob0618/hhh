"""Full-cohort QA union for one separately admitted retrieval continuation.

The original 1539 successful QA leaves are immutable references. The failed
original firstpass slot remains missing; only the effective view uses the new
retrieval and its first Reader/Judge. Scientific QA functions are reused intact.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from tools import run_five_baselines_gemini25_v1 as base
from tools import archive_amem_continuation_v4 as frozen_archive
from tools.baselines_common_v1 import read, sha, save_new, now

ROOT = base.ROOT
PARENT = base.RUN / 'amem_native_v4'
WORK_OWNER = PARENT / 'work/retrieval_continuation_v1'
FINAL = PARENT / 'retrieval_continuation_v1'
QUERY_ID = 'c2_q42'
SCHEMA = 'amem-single-retrieval-full-cohort-union-v1'


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


class LiveView(frozen_archive.LiveResolver):
    def __init__(self):
        pass
    read = staticmethod(read)
    sha = staticmethod(sha)
    bytes = staticmethod(lambda path: Path(path).read_bytes())


def path_type(view):
    if isinstance(view, LiveView):
        return Path
    if not hasattr(view, '_union_path_type'):
        view._union_path_type = frozen_archive.archived_path_class(view)
    return view._union_path_type


def bound(view, record):
    if view.sha(record['path']) != record['sha256']:
        raise ValueError('Bound union evidence changed: ' + record['path'])
    return view.read(record['path'])


def contract(manifest, proof, final_root=None, work_owner=None, *, view=None):
    view = view or LiveView()
    final_root=FINAL if final_root is None else final_root
    work_owner=WORK_OWNER if work_owner is None else work_owner
    if Path(final_root) != FINAL or Path(work_owner) != WORK_OWNER:
        raise ValueError('Exact independent final and work namespaces required')
    if manifest.get('final_root') != str(FINAL) or manifest.get('work_owner') != str(WORK_OWNER):
        raise ValueError('Manifest continuation owners differ')
    if manifest.get('query_id') != QUERY_ID or proof.get('query_id') != QUERY_ID:
        raise ValueError('Only the originally missing query is admitted')
    if bound(view, manifest['proof']) != proof:
        raise ValueError('Proof bytes/object differ')
    parent = bound(view, manifest['parent_manifest'])
    if Path(manifest['parent_manifest']['path']) != PARENT / 'run_manifest.json':
        raise ValueError('Original parent manifest differs')
    if parent['questions'] != manifest['questions'] or len(manifest['questions']) != 1540:
        raise ValueError('Original complete fixed cohort must remain unchanged')
    questions = manifest['questions']
    if len({q['query_id'] for q in questions}) != 1540:
        raise ValueError('Duplicate fixed query identity')
    routes = {str(ci): str(base.RUN / ('amem_format_v2' if ci in (0,3,6,8) else 'amem_native_v4' if ci==2 else 'amem_native_v3')) for ci in range(10)}
    if proof.get('original_routes') != routes or parent['conversation_sources'] != routes:
        raise ValueError('Immutable original conversation routes differ')
    for p,h in manifest['code_sha256'].items():
        if view.sha(ROOT / p) != h:
            raise ValueError('Union frozen code differs: ' + p)
    return parent, questions, routes


def original_source(q, routes):
    owner = Path(routes[str(q['conversation'])])
    leaf = owner / 'results/firstpass' / q['query_id']
    ret = owner / 'results' / ('c' + str(q['conversation'])) / 'retrieval' / (q['query_id'] + '.json')
    return owner, leaf, ret


def replay(view, q, owner, leaf, ret, *, original_leaf=None):
    if hasattr(view, '_union_qa_replayer'):
        P, function = view._union_qa_replayer
        return function(q, P(leaf), P(owner), P(owner), P(original_leaf or leaf), P(ret))
    P = path_type(view)
    class RawReplay(base.RawReplay):
        generate = frozen_archive.clone(base.RawReplay.generate, read=view.read, sha=view.sha, Path=P)
    known = frozen_archive.clone(base.known_recoverable, RawReplay=RawReplay, read=view.read, sha=view.sha, Path=P)
    base_view = SimpleNamespace(**{**vars(base), 'RawReplay':RawReplay, 'known_recoverable':known})
    function = frozen_archive.clone(frozen_archive.continuation.controller.replay_leaf,
                                   read=view.read, sha=view.sha, Path=P, base=base_view)
    view._union_qa_replayer = (P, function)
    origin = original_leaf or leaf
    return function(q, P(leaf), P(owner), P(owner), P(origin), P(ret))


def row(q, reader, judge, *, phase, owner, leaf, retrieval):
    official, _ = base.verified_official_metric()
    answer = reader['value']['prediction']['answer'] if reader['status']=='complete' else None
    label = base.evaluator.parse_judge_label(judge['value']['raw']) if judge['status']=='complete' else None
    return {**q, 'answer':answer, 'official_f1':official(answer,q['reference'],q['category']) if answer is not None else None,
            'custom_metrics':base.evaluator.lexical_metrics(answer,q['reference']) if answer is not None else None,
            'judge_correct':label['correct'] if label else None, 'reader_status':reader['status'], 'judge_status':judge['status'],
            'retrieval_status':'complete', 'phase':phase, 'source_owner':str(owner), 'source_leaf':str(leaf), 'retrieval_source':str(retrieval)}


def aggregate(rows):
    f1=[r['official_f1'] for r in rows if r['official_f1'] is not None]
    labels=[r['judge_correct'] for r in rows if r['judge_correct'] is not None]
    custom=[r['custom_metrics']['token_f1'] for r in rows if r['custom_metrics'] is not None]
    n=len(rows)
    return {'slots':n,'reader_valid':len(f1),'judge_valid':len(labels),'missing_reader':n-len(f1),'missing_judge':n-len(labels),
            'official_f1_valid_percent':100*math.fsum(f1)/len(f1) if f1 else None,
            'custom_f1_valid_percent':100*math.fsum(custom)/len(custom) if custom else None,
            'judge_valid_percent':100*sum(labels)/len(labels) if labels else None,
            'official_f1_full_cohort_percent':100*math.fsum(f1)/n if len(f1)==n else None,
            'judge_full_cohort_percent':100*sum(labels)/n if len(labels)==n else None,'missing_imputed':False}


def original_firstpass(manifest, proof, *, view=None):
    view=view or LiveView(); _,questions,routes=contract(manifest,proof,view=view)
    P=path_type(view); sources=[]; rows=[]
    successes={s['query_id']:s for s in proof['original_successes']}
    if set(successes)!={q['query_id'] for q in questions if q['query_id']!=QUERY_ID}:
        raise ValueError('Proof must bind exactly the original 1539 successful identities')
    for q in questions:
        owner,leaf,ret=original_source(q,routes); record=view.read(ret)
        if q['query_id']==QUERY_ID:
            if record.get('status')!='failed' or P(leaf).exists():
                raise ValueError('Original failed retrieval and absent QA must be retained')
            missing=proof['original_failure']
            if missing['retrieval']!={'path':str(ret),'sha256':view.sha(ret)}:
                raise ValueError('Original missing retrieval differs from proof')
            rows.append({**q,'answer':None,'official_f1':None,'custom_metrics':None,'judge_correct':None,
                         'reader_status':'not_run','judge_status':'not_run','retrieval_status':'failed','phase':'firstpass',
                         'source_owner':str(owner),'source_leaf':None,'retrieval_source':str(ret)})
            sources.append({'query_id':q['query_id'],'conversation':q['conversation'],'owner':str(owner),'leaf':None,
                            'retrieval_path':str(ret),'bindings':{str(ret):view.sha(ret)},'status':'retrieval_failed'})
            continue
        s=successes[q['query_id']]
        if (s['owner']!=str(owner) or s['leaf']!=str(leaf) or s['retrieval_path']!=str(ret)
                or set(s['bindings'])!={str(p) for p in (ret,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')}):
            raise ValueError('Original successful source routing/binding set differs')
        for p,h in s['bindings'].items():
            if view.sha(p)!=h:raise ValueError('An original successful result changed')
        reader,judge=replay(view,q,owner,leaf,ret)
        if reader['status']!=judge['status'] or reader['status']!='complete':
            raise ValueError('Original success is no longer complete')
        outcome=view.read(leaf/'outcome.json')
        expected={'query_id':q['query_id'],'conversation':q['conversation'],'method':'amem','phase':'firstpass',
                  'reader':'complete','judge':'complete','reader_path':str(leaf/'reader.json'),'judge_path':str(leaf/'judge.json')}
        if outcome!=expected:raise ValueError('Original successful outcome differs')
        sources.append(deepcopy(s)); rows.append(row(q,reader,judge,phase='firstpass',owner=owner,leaf=leaf,retrieval=ret))
    return rows,sources


def technical_paths(phase='firstpass'):
    return WORK_OWNER/'results'/phase/QUERY_ID, WORK_OWNER/'results/c2/retrieval'/(QUERY_ID+'.json')


def verify_technical_copy(view, source, permanent):
    if view.bytes(source)!=view.bytes(permanent):
        raise ValueError('Permanent technical evidence differs from original work artifact')


def technical_source(view, q, phase):
    leaf,ret=technical_paths(phase)
    original_leaf,_=technical_paths()
    reader,judge=replay(view,q,WORK_OWNER,leaf,ret,original_leaf=original_leaf)
    value=view.read(ret)
    if value['status']!='complete' or value['value'].get('question')!=q['question']:
        raise ValueError('Technical retrieval must be complete for the exact original question')
    expected={'query_id':q['query_id'],'conversation':q['conversation'],'method':'amem','phase':phase,
              'reader':reader['status'],'judge':judge['status'],'reader_path':str(leaf/'reader.json'),'judge_path':str(leaf/'judge.json')}
    if view.read(leaf/'outcome.json')!=expected:raise ValueError('Technical QA outcome differs')
    return reader,judge,{'query_id':QUERY_ID,'owner':str(WORK_OWNER),'leaf':str(leaf),'retrieval_path':str(ret),
                        'phase':phase,'bindings':{str(p):view.sha(p) for p in (ret,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')}}


def copy_qa(phase):
    leaf,_=technical_paths(phase)
    target=FINAL/'results'/('technical_'+phase)/QUERY_ID
    for p in sorted(leaf.glob('*.json')):
        target.mkdir(parents=True,exist_ok=True)
        out=target/p.name
        with out.open('xb') as f:f.write(p.read_bytes())
        if sha(out)!=sha(p):raise ValueError('Technical permanent copy differs')


def verify_permanent_phase(view, phase):
    P=path_type(view);leaf,ret=technical_paths(phase)
    verify_technical_copy(view,ret,FINAL/'results/technical_retrieval'/(QUERY_ID+'.json'))
    target=FINAL/'results'/('technical_'+phase)/QUERY_ID
    sources={p.name:p for p in P(leaf).glob('*.json')}
    if set(sources)!={p.name for p in P(target).glob('*.json')}:
        raise ValueError('Permanent technical QA file set differs')
    for name,p in sources.items():verify_technical_copy(view,p,target/name)


def method_audit(manifest, proof, view):
    from tools.run_five_baselines_amem_retrieval_v1 import audit_sidecar
    value=audit_sidecar(manifest,proof,FINAL,WORK_OWNER,view=view)
    if value.get('status')!='passed':raise ValueError('Native query continuation audit did not pass')
    return value


def eligible(view, q, reader, judge):
    _,ret=technical_paths(); retrieval=view.read(ret)['value']
    if reader['status']=='complete' and judge['status']=='complete':return None
    kind='reader' if reader['status']=='failed' else 'judge'
    if (kind=='reader' and judge['status']!='not_run') or (kind=='judge' and (reader['status']!='complete' or judge['status']!='failed')):
        raise ValueError('Unclassified technical first QA status')
    P=path_type(view)
    class RawReplay(base.RawReplay):
        generate=frozen_archive.clone(base.RawReplay.generate,read=view.read,sha=view.sha,Path=P)
    known=frozen_archive.clone(base.known_recoverable,RawReplay=RawReplay,read=view.read,sha=view.sha,Path=P)
    if not known(reader if kind=='reader' else judge,P(WORK_OWNER),'amem',q,retrieval,kind=kind,reader=reader if kind=='judge' else None):
        raise ValueError('Unknown QA failure has no automatic recovery allowance')
    return kind


def usage(view):
    P=path_type(view); values=[]; ledgers={}
    # Parent v4/work recursively contains sidecar work, but the bounded old glob
    # api_* only sees direct API owners; count the new nested owner separately.
    roots={**frozen_archive.ROOTS,'retrieval_continuation':WORK_OWNER}
    for name,folder in roots.items():
        rows=[]
        for p in P(folder/'work').glob('api_*/api_calls.jsonl'):
            rows.extend(json.loads(line) for line in view.bytes(p).splitlines() if line.strip())
            ledgers[str(p)]=view.sha(p)
        values.extend(rows)
    physical=[r for r in values if not r.get('cache_hit')]
    fields=('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')
    return {'physical_attempts':len(physical),'exact_cache_hits':len(values)-len(physical),
            'status_counts':dict(Counter(str(r.get('status')) for r in physical)), 'ledger_sha256':ledgers,
            'tokens':{k:{'reported_sum':sum(r['usage'][k] for r in physical if isinstance(r.get('usage'),dict) and k in r['usage'])
                        if any(isinstance(r.get('usage'),dict) and k in r['usage'] for r in physical) else None,
                        'reported_attempts':sum(isinstance(r.get('usage'),dict) and k in r['usage'] for r in physical)} for k in fields},
            'cost_usd':None,'note':'Each physical owner ledger counted once; original failed attempts retained. Missing token telemetry is unknown.'}


def audit_qa_requests(view, *, recovered):
    P=path_type(view)
    result=frozen_archive.clone(base.audit_requests,read=view.read,sha=view.sha,Path=P)(P(WORK_OWNER))
    if result['errors']:raise ValueError('Sidecar transport audit failed')
    expected_dirs={'api_method','api_firstpass'} | ({'api_recovery'} if recovered else set())
    if {p.name for p in P(WORK_OWNER/'work').glob('api_*') if p.is_dir()} != expected_dirs:
        raise ValueError('Unexpected or missing sidecar API owner')
    report=[]
    clean_event=lambda e:{k:v for k,v in e.items() if k!='recorded_utc'}
    digest=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    for phase in ('firstpass','recovery') if recovered else ('firstpass',):
        api=P(WORK_OWNER/'work'/('api_'+phase)); leaf,_=technical_paths(phase)
        expected_events=[]
        for kind in ('reader','judge'):
            stage=view.read(leaf/(kind+'.json'))
            if stage.get('inherited'):
                if phase!='recovery' or kind!='reader':raise ValueError('Undeclared inherited sidecar stage')
                continue
            if stage['status']=='not_run':
                if stage.get('events'):raise ValueError('Not-run stage has a request consumer')
                continue
            if len(stage.get('events',[]))!=1:raise ValueError('One logical request per executed QA stage required')
            event=stage['events'][0]
            if (event.get('stage')!={'reader':base.evaluator.READER_STAGE,'judge':base.evaluator.JUDGE_STAGE}[kind]
                    or event.get('context')!={'method':'amem','conversation':2,'query_id':QUERY_ID,'phase':phase}):
                raise ValueError('Undeclared sidecar QA context or stage')
            expected_events.append(event)
        if not expected_events:raise ValueError('Executed QA phase has no generation')
        recorded=[json.loads(line) for line in view.bytes(api/'logical_events.jsonl').splitlines() if line.strip()]
        if Counter(digest(clean_event(e)) for e in recorded)!=Counter(digest(e) for e in expected_events):
            raise ValueError('QA logical consumers differ from exact retained stage events')
        keys={e['request_hash'] for e in expected_events}
        if (len(keys)!=len(expected_events) or {p.stem for p in (api/'request_intents').glob('*.json')}!=keys
                or {p.stem for p in (api/'api_cache').glob('*.json')}!=keys):
            raise ValueError('Extra/missing QA raw, intent, or duplicated logical request')
        ledger=[json.loads(line) for line in view.bytes(api/'api_calls.jsonl').splitlines() if line.strip()]
        if any(r.get('cache_hit') or r.get('request_hash') not in keys for r in ledger):
            raise ValueError('Fresh one-query phase cannot inherit cache or issue unbound requests')
        for event in expected_events:
            key=event['request_hash']; intent=view.read(api/'request_intents'/(key+'.json'))
            raw=view.read(api/'api_cache'/(key+'.json'))
            calls=[r for r in ledger if r['request_hash']==key]
            if (intent.get('request_hash')!=key or digest(intent.get('signature'))!=key
                    or raw.get('request')!=intent.get('signature') or intent.get('retries')!=2
                    or intent.get('context')!=event['context'] or intent.get('stage')!=event['stage']
                    or not 1<=len(calls)<=3 or [r.get('attempt') for r in calls]!=list(range(1,len(calls)+1))
                    or calls[-1].get('status')!=200
                    or any(r.get('status') not in {429,500,502,503,504,529,'transport_error'} for r in calls[:-1])):
                raise ValueError('QA physical producer sequence/signature/intent differs')
            if (event.get('source')!='new_request' or event.get('error_type') is not None
                    or Path(event.get('cache_path',''))!=Path(api/'api_cache'/(key+'.json'))
                    or any(event.get(k)!=intent['signature'][k] for k in ('provider','model','operation'))):
                raise ValueError('QA logical event raw path/provider/source differs')
            for call in calls:
                if (call.get('context')!=intent['context'] or call.get('stage')!=intent['stage']
                        or any(call.get(k)!=intent['signature'][k] for k in ('provider','model','operation'))):
                    raise ValueError('QA physical producer ownership differs')
            if event.get('cache_sha256')!=view.sha(api/'api_cache'/(key+'.json')):
                raise ValueError('QA original raw bytes changed')
            report.append({'phase':phase,'request_hash':key,'stage':event['stage'],'context':event['context'],
                           'physical_attempts':len(calls),'raw_sha256':view.sha(api/'api_cache'/(key+'.json')),
                           'intent_sha256':view.sha(api/'request_intents'/(key+'.json'))})
    return {'status':'passed','requests':report,'legacy_raw_inventory':result}


def reconstruct(manifest, proof, *, view=None, created_utc=None):
    """Pure full result reconstruction over live or content-addressed evidence."""
    view=view or LiveView(); _,questions,routes=contract(manifest,proof,view=view)
    first,sources=original_firstpass(manifest,proof,view=view)
    if view.read(FINAL/'firstpass_sources.json')!=sources:raise ValueError('Full original firstpass table changed')
    q=next(q for q in questions if q['query_id']==QUERY_ID)
    r,j,technical=technical_source(view,q,'firstpass')
    if view.read(FINAL/'technical_firstpass_sources.json')!=[technical]:raise ValueError('Technical firstpass source differs')
    plan=view.read(FINAL/'qa_recovery_plan.json'); outcome=view.read(FINAL/'qa_recovery_outcome.json')
    kind=eligible(view,q,r,j)
    expected_jobs=[] if kind is None else [{'query_id':QUERY_ID,'stage':kind,'source':technical}]
    if (plan.get('round')!=1 or plan.get('full_cohort')!=1540 or plan.get('jobs')!=expected_jobs
            or plan.get('firstpass_sources_sha256')!=view.sha(FINAL/'firstpass_sources.json')
            or plan.get('technical_firstpass_sources_sha256')!=view.sha(FINAL/'technical_firstpass_sources.json')
            or outcome.get('plan_sha256')!=view.sha(FINAL/'qa_recovery_plan.json')):
        raise ValueError('Uniform once-only full-cohort QA plan/outcome differs')
    phase='firstpass'
    P=path_type(view)
    if kind:
        phase='recovery'; r,j,effective_source=technical_source(view,q,phase)
        if outcome.get('results')!=[view.read(technical_paths(phase)[0]/'outcome.json')]:
            raise ValueError('The single allowed QA recovery outcome differs')
    else:
        if outcome.get('results')!=[]:raise ValueError('Unexpected QA recovery result')
        effective_source=technical
    if {p.name for p in P(WORK_OWNER/'results/recovery').glob('*') if p.is_dir()} != ({QUERY_ID} if kind else set()):
        raise ValueError('Unexpected/missing sidecar QA recovery directory')
    # Permanent copies are byte-preserving evidence, never a second dispatch.
    _,ret=technical_paths()
    for current_phase in ('firstpass','recovery') if kind else ('firstpass',):
        verify_permanent_phase(view,current_phase)
    effective=[{**item,'phase':'effective'} for item in first]
    replacement=row(q,r,j,phase='effective',owner=WORK_OWNER,leaf=technical_paths(phase)[0],retrieval=ret)
    effective=[replacement if item['query_id']==QUERY_ID else item for item in effective]
    audit=audit_qa_requests(view,recovered=bool(kind))
    # The entry's independent pure audit closes replacement certificate,
    # method producer/consumer, parser, native formatter, and original-bank
    # provenance. It is also replayable from retained logical archive paths.
    native_audit=method_audit(manifest,proof,view)
    summary={'schema':SCHEMA,'method':'amem','self_root':str(FINAL),'parent_root':str(PARENT),
             'work_owner':str(WORK_OWNER),'conversation_sources':routes,'effective_query_overrides':{QUERY_ID:effective_source},
             'audit_status':'no_invalid_artifacts' if r['status']==j['status']=='complete' else 'known_technical_failure_retained',
             'created_utc':created_utc or now(),'firstpass':aggregate(first),'effective':aggregate(effective),
             'proof':manifest['proof'],'parent_manifest':manifest['parent_manifest'],'usage':usage(view),
             'audit_content_sha256':{name:hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                                     for name,value in (('raw_request_audit.json',audit),('method_request_audit.json',native_audit))},
             'scientific_configuration':'original fixed Gemini 2.5 Reader/Judge and native A-Mem retrieval; one exact admitted query replacement',
             'firstpass_policy':'original q42 retrieval failure remains not_run/null; technical first QA is used only in effective view'}
    return {'firstpass_per_query.json':first,'per_query.json':effective,'raw_request_audit.json':audit,
            'method_request_audit.json':native_audit,'summary.json':summary}


def finish_run(manifest, proof, final_root, work_owner):
    """Only production caller may reach the original bounded QA recovery here."""
    contract(manifest,proof,final_root,work_owner)
    first,sources=original_firstpass(manifest,proof)
    q=next(q for q in manifest['questions'] if q['query_id']==QUERY_ID)
    r,j,technical=technical_source(LiveView(),q,'firstpass')
    verify_permanent_phase(LiveView(),'firstpass')
    audit_qa_requests(LiveView(),recovered=False)
    method_audit(manifest,proof,LiveView())
    # No recovery plan exists until all original 1540 statuses and the admitted
    # replacement's first QA are terminal and exactly replayed above.
    save_new(FINAL/'firstpass_sources.json',sources)
    save_new(FINAL/'technical_firstpass_sources.json',[technical])
    kind=eligible(LiveView(),q,r,j)
    jobs=[] if kind is None else [{'query_id':QUERY_ID,'stage':kind,'source':technical}]
    save_new(FINAL/'qa_recovery_plan.json',{'round':1,'full_cohort':1540,'jobs':jobs,
             'firstpass_sources_sha256':sha(FINAL/'firstpass_sources.json'),
             'technical_firstpass_sources_sha256':sha(FINAL/'technical_firstpass_sources.json')})
    recovered=[]
    if kind:
        _,ret=technical_paths()
        recovered.append(base.qa_one(WORK_OWNER,'amem',q,read(ret)['value'],phase='recovery',
                                     inherited_reader=r if r['status']=='complete' else None))
        copy_qa('recovery')
    save_new(FINAL/'qa_recovery_outcome.json',{'plan_sha256':sha(FINAL/'qa_recovery_plan.json'),'results':recovered})
    result=reconstruct(manifest,proof)
    for name,value in result.items():save_new(FINAL/'results'/name,value)
    return result['summary.json']
