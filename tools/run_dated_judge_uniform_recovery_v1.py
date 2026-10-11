"""One Judge-only recovery for the dated arms under the fixed SDK20 final rule.

All 3080 original Readers, selections and strict/compat responses stay unchanged.
The 218 strict failures are selected without looking at correctness; c2_q74 has
already used its one recovery and is inherited. Only the other 217 dispatch.
prepare/check/summarize are offline. No second recovery entry point exists.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_sdk20_judge_retest_v2 as legacy
from tools import recover_mem0_plain_judge_labels_v1 as overlay
from tools import summarize_mem0_regroup_plain_qa_v1 as strict
from tools import summarize_sdk20_judge_retest_v1 as judge_summary
from tools import freeze_sdk20_full1540_v1 as lock
from tools.summarize_sdk20_modules import fingerprint
from tools.summarize_mem0_sparse1540 import measure_usage
from tools.summarize_pilot400 import verified_official_metric
from tools.run_sdk20_modules import save_new, digest, now

SCHEMA = 'dated-judge-uniform-one-recovery-v1'
SOURCE_SCHEMA = SCHEMA + '-sources'
RUN = ROOT / 'runs/dated_judge_uniform_recovery_v1'
OLD = ROOT / 'runs/mem0_global_regroup_plain_qa_v2'
SEAL = ROOT / 'research/plain_qa_v2_firstpass_terminal_seal_2026-09-29.json'
STRICT = ROOT / 'research/reports/mem0_global_regroup_plain_qa_v2_strict/summary.json'
COMPAT = ROOT / 'runs/mem0_plain_judge_label_compat_v1/summary.json'
Q74 = ROOT / 'research/reports/sdk20_plain_judge_retest_2026-09-29/global_q74_supplement.json'
PINS = {SEAL: '29b2ad314ff65e3fb8d2076eba5453eb80cd008bb6ee1c8bf241d19106e04a8e',
        STRICT: '8dad0bd4a406d2c0689fec3c07e668fe51c7ca17fa18098372a5ff91ed669d06',
        COMPAT: '384392c324e23e0c987a2d756758593c4dee5e5d40f6b53459809287fa08c2fc',
        Q74: '807b620312d6a1cb20c95746d4e7d1b82af7084bb652c4fca33825b55c91e0ab'}
ARMS = ('dated_unmerged385', 'dated_merged')
EXISTING_KEY = 'dated_unmerged385/c2_q74'
POLICY = {**legacy.POLICY,
          'arms': list(ARMS), 'planned_slots': 3080,
          'strict_failure_population': 218, 'existing_once_recovery': 1,
          'new_judge_jobs': 217, 'reader': 'inherit_exact_original_no_Reader_generation',
          'phases': 'one_existing_firstpass_then_one_uniform_recovery_no_followup',
          'blocked_reader': 'reject', 'supplement_followup': 'forbidden',
          'requires_new_extension_firstpasses_terminal': True}
CODE = tuple(dict.fromkeys(('tools/run_dated_judge_uniform_recovery_v1.py',
       'tools/freeze_sdk20_full1540_v1.py', *legacy.CODE, *overlay.CODE,
       'tools/summarize_sdk20_judge_retest_v1.py')))
pointer, bound, read, verify, sha = legacy.pointer, legacy.bound, legacy.read, legacy.verify, legacy.sha


def key(row):
    return row['method'] + '/' + row['query_id']


def compat_record(sidecar):
    interpretation = sidecar['source_replay']['interpretation']
    return {'status': interpretation['status'], 'interpretation': interpretation}


def failure_population(jobs, judges):
    """Strict format status only; valid YES and NO have identical no-retry status."""
    if len(jobs) != 3080 or len({key(j) for j in jobs}) != 3080:
        raise ValueError('Exactly3080 original slots required')
    counts = Counter()
    failed = []
    for job in jobs:
        row = judges[key(job)]
        if row['status'] == 'failed':
            if (row.get('error') or {}).get('type') != 'InvalidJudgeLabel':
                raise ValueError('A new/unknown failure needs its own review, not implicit redispatch')
            failed.append(job); counts[job['method']] += 1
        elif row['status'] != 'complete':
            raise ValueError('Source Judge is not a known terminal outcome')
    if counts != {'dated_unmerged385': 110, 'dated_merged': 108}:
        raise ValueError('Frozen218 technical-failure population changed')
    if EXISTING_KEY not in {key(j) for j in failed}:
        raise ValueError('Previously recovered c2_q74 must remain in the218 denominator')
    return failed


def _ledger(path):
    rows = [json.loads(s) for s in legacy.shared_read(path).decode('utf-8-sig').splitlines() if s.strip()]
    indexed = defaultdict(list)
    for row in rows:
        indexed[row['request_hash'], row['stage'], fingerprint(row['context'])].append(row)
    return rows, indexed


def q74_evidence(job):
    """Replay the single previously consumed recovery, not another model attempt."""
    supplement = verify(Q74)['supplement']
    old = supplement['job']
    for field in ('conversation', 'method', 'query_id', 'question', 'reference', 'category', 'answer', 'answer_sha256'):
        if old[field] != job[field]:
            raise ValueError('Existing c2_q74 recovery belongs to another original answer')
    refs = supplement['pointers']
    s, c = verify(overlay.check_pointer(refs['strict'])), verify(overlay.check_pointer(refs['compat']))
    original_run = ROOT / 'runs/sdk20_plain_judge_retest_v2_recovery_v1'
    m = read(original_run / 'run_manifest.json')
    if m['protocol_sha256'] != digest({k:v for k,v in m.items() if k != 'protocol_sha256'}):
        raise ValueError('Original q74 manifest seal differs')
    matches = [j for j in m['jobs'] if j['job_id'] == old['job_id']]
    if matches != [old]:
        raise ValueError('Exactly one original q74 recovery job required')
    expected = legacy.identity(old, m)
    if any(any(row.get(k) != v for k,v in expected.items()) for row in (s,c)):
        raise ValueError('Original q74 recovery identity changed')
    if c['strict'] != refs['strict']:
        raise ValueError('q74 compatibility is not bound to its strict response')
    api_dir = original_run / 'conv_2/api'
    rows, indexed = _ledger(api_dir / 'api_calls.jsonl')
    context = {'architecture':legacy.SCHEMA, 'conversation':2, 'phase':'judge_retest',
               'phase_id':m['phase_id'], 'method':old['method'], 'query_id':old['query_id'],
               'job_id':old['job_id'], 'component':'judge'}
    from tools.replay_plain_judge_compat_v1 import replay_source
    replay = replay_source(api_dir, s, context, indexed, question=old['question'],
                           reference=old['reference'], reader_text=old['answer'])
    if replay['interpretation'] != c['interpretation'] or c['status'] != replay['interpretation']['status']:
        raise ValueError('q74 compatible label does not replay from its original raw')
    if legacy.resolve_pair(s,c) != supplement['result']:
        raise ValueError('q74 retained resolution changed')
    logical = [r for r in rows if r.get('context') == context]
    if not logical or any(r.get('model') != legacy.JUDGE_MODEL for r in logical):
        raise ValueError('q74 original same-context Judge evidence missing')
    return {'strict':s, 'compat':c, 'pointers':refs, 'manifest':pointer(original_run/'run_manifest.json'),
            'source_replay':replay, 'usage':measure_usage(logical), 'logical_recovery_already_consumed':True}


def build_source():
    """Existing terminal source audit plus exact old response replay; zero models."""
    lock.verify_lock()
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError('Pinned original summary/seal changed: '+str(path))
    admitted = overlay.source_admission(SEAL, STRICT)
    report, compatible = admitted['strict_summary'], verify(COMPAT)
    if compatible['sources'] != admitted['binding'] or compatible['source_hashes_unchanged'] is not True:
        raise ValueError('Compatibility overlay source differs')
    questions = {q['query_id']:q for q in lock.full_cohort()}
    sidecars = {}
    for ref in compatible['sidecars']:
        value = verify(overlay.check_pointer(ref))
        if key(value) in sidecars: raise ValueError('Duplicate compatibility source')
        sidecars[key(value)] = (value, ref)
    if len(sidecars) != 3080: raise ValueError('Incomplete compatibility overlay')
    snap = strict.Snapshot()
    _, indexed, _, missing = strict.ledger_index(snap, OLD, strict.driver.sources.cohort()[0])
    if missing: raise ValueError('Original ledger missing')
    jobs, judges, parents = [], {}, {}
    for row in report['slots']:
        k = key(row); ci, arm, qid = row['conversation'], row['method'], row['query_id']
        source = admitted['sealed_slots'][ci,arm,qid]
        reader = verify(overlay.check_pointer(source['artifacts']['reader']))
        judge = verify(overlay.check_pointer(source['artifacts']['judge']))
        q = questions[qid]
        if reader['status'] != 'complete' or reader['reference_date'] != q['reference_date']:
            raise ValueError('Original Reader/date is not complete and canonical')
        derived, changed = overlay.derive_slot(source,row,reader,judge,api_dir=OLD/f'conv_{ci}/api',
                         ledger=indexed,question=q['question'],reference=q['reference'])
        stored, ref = sidecars[k]
        if any(stored.get(name) != value for name,value in derived.items()):
            raise ValueError('Original compatibility sidecar does not reproduce')
        if changed != stored['compat_slot']:raise ValueError('Original compatibility projection changed')
        answer = reader['result']['prediction']['answer']
        j = {'job_id':k, 'conversation':ci, 'method':arm, 'query_id':qid, 'status':'ready',
             'question':q['question'], 'reference':q['reference'], 'category':q['category'],
             'reference_date':q['reference_date'], 'answer':answer,
             'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),
             'F1':deepcopy(reader['result']['metrics']),
             'source':{'reader':source['artifacts']['reader'], 'judge':source['artifacts']['judge'],
                       'selection':source['artifacts']['selection'], 'compatibility':ref},
             'attempt_kind':'one_uniform_postbatch_strict_format_recovery'}
        if any(j['F1'][name] != row['scores'][name] for name in ('official_locomo_f1','custom_token_f1')):
            raise ValueError('Original Reader/F1 projection differs')
        jobs.append(j); judges[k] = judge
        parents[k] = {'strict':judge, 'compat':compat_record(stored)}
    failed = failure_population(jobs,judges)
    existing = q74_evidence(next(j for j in failed if key(j)==EXISTING_KEY))
    if snap.unchanged():raise ValueError('Original source changed during admission')
    _, provenance = verified_official_metric()
    return {'schema':SOURCE_SCHEMA, 'master_lock':pointer(lock.LOCK),
            'original_sources':admitted['binding'], 'compatibility_summary':pointer(COMPAT),
            'q74_supplement':pointer(Q74), 'existing_recovery':existing,
            'failure_job_ids':[key(j) for j in failed], 'all_jobs':jobs,
            'jobs':[j for j in failed if key(j)!=EXISTING_KEY], 'parents':parents,
            'strict_slots':deepcopy(report['slots']), 'compatible_slots':deepcopy(compatible['slots']),
            'official_metric_provenance':provenance, 'new_Reader_calls':0,
            'planned_denominator_per_arm':1540, 'selection_criterion':'original_strict_failure_only'}


def prepare(run_dir=RUN):
    run = Path(run_dir).resolve()
    if run != RUN.resolve() or run.exists():raise FileExistsError('One fixed fresh recovery namespace required')
    with overlay.offline():source = build_source()
    run.mkdir(parents=True,exist_ok=False)
    save_new(run/'source_manifest.json',source)
    m = {'schema':SCHEMA, 'created_utc':now(), 'policy':POLICY, 'phase_kind':'uniform_dated_recovery',
         'phase_id':SCHEMA, 'master_lock':pointer(lock.LOCK), 'source_manifest':pointer(run/'source_manifest.json'),
         'jobs':source['jobs'], 'existing_recovery_job_id':EXISTING_KEY,
         'code_sha256':{p:sha(ROOT/p) for p in CODE}, 'cache_roots':[],
         'official_metric_provenance':source['official_metric_provenance']}
    save_new(run/'run_manifest.json',{**m,'protocol_sha256':digest(m)})
    return {'run_dir':str(run),'new_Judge_jobs':217,'inherited_recovery':1,'original_failures':218,'model_calls':0}


def manifest(run_dir, *, recheck_sources=True):
    run=Path(run_dir).resolve();m=verify(run/'run_manifest.json')
    unsigned={k:v for k,v in m.items() if k not in ('protocol_sha256','content_sha256')}
    if (run!=RUN.resolve() or m.get('schema')!=SCHEMA or m.get('policy')!=POLICY
            or m.get('phase_kind')!='uniform_dated_recovery' or m.get('phase_id')!=SCHEMA
            or m['protocol_sha256']!=digest(unsigned)):
        raise ValueError('Frozen recovery manifest differs')
    lock.verify_lock();bound(m['master_lock']);source=verify(overlay.check_pointer(m['source_manifest']))
    if m['master_lock']!=pointer(lock.LOCK) or Path(m['source_manifest']['path']).resolve()!=run/'source_manifest.json':
        raise ValueError('The new run must bind its own source and the canonical master lock')
    if m['code_sha256']!={p:sha(ROOT/p) for p in CODE} or m.get('cache_roots')!=[]:
        raise ValueError('Frozen Judge implementation/cache policy differs')
    if m['jobs']!=source['jobs'] or len(m['jobs'])!=217 or any(key(j)==EXISTING_KEY for j in m['jobs']):
        raise ValueError('Uniform217 population changed or existing recovery repeated')
    legacy.check_jobs(m['jobs'])
    if recheck_sources:
        actual=build_source()
        if actual!={k:v for k,v in source.items() if k!='content_sha256'}:
            raise ValueError('Sealed original sources or recovery population changed')
    return m


def terminal(run_dir):
    m=manifest(run_dir);run=Path(run_dir);out=verify(run/'run_outcome.json')
    if (out.get('terminal')is not True or out.get('protocol_sha256')!=m['protocol_sha256']
            or out.get('total')!=217 or len(out.get('results',[]))!=217):
        raise ValueError('Recovery is not a complete terminal217 population')
    jobs={j['job_id']:j for j in m['jobs']}
    if {r['job_id'] for r in out['results']}!=set(jobs):raise ValueError('Terminal population differs')
    counts={s:Counter() for s in ('strict','compat')}
    for row in out['results']:
        job=jobs[row['job_id']];expected=identity(job,m)
        for stage,path in legacy.result_paths(run,job).items():
            ref=row[stage]
            if ref is None:
                if path.exists():raise ValueError('Existing outcome cannot be marked missing')
                counts[stage]['unknown']+=1
                continue
            if ref!=pointer(path):raise ValueError('Terminal result pointer differs')
            value=verify(path)
            if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Terminal identity differs')
            counts[stage][value['status']]+=1
    if out.get('counts')!={s:dict(v) for s,v in counts.items()}:
        raise ValueError('Terminal stage counts differ from retained artifacts')
    return m,out


def _bind_legacy():
    # Execute the same Judge-only code objects with private namespace bindings.
    # No legacy module global or frozen function object is changed.
    namespace={**legacy.__dict__, 'SCHEMA':SCHEMA, 'manifest':manifest, 'terminal':terminal}
    for name in ('identity','response_evidence','one','run'):
        original=getattr(legacy,name)
        function=FunctionType(original.__code__,namespace,name,original.__defaults__,original.__closure__)
        function.__kwdefaults__=dict(original.__kwdefaults__ or {})
        namespace[name]=function
    return namespace


_LEGACY=_bind_legacy()
identity, one = _LEGACY['identity'], _LEGACY['one']


def require_extensions_terminal(paths):
    if len(paths)!=2:raise ValueError('Both1140 extension firstpasses must finish before this recovery')
    from tools import run_regroup_refinement_extension1140_v1 as extension
    found=set()
    for folder in paths:
        folder=Path(folder)
        if not (folder/'source_jobs.json').is_file() or not (folder/'run_outcome.json').is_file():
            raise ValueError('Extension sealed jobs and terminal outcome are required; never seal an active run')
        header=read(folder/'run_manifest.json')
        if header.get('schema')!=extension.SCHEMA:
            raise ValueError('Only the frozen extension1140 driver is admitted')
        m,out=extension.terminal(folder)
        arm=m.get('arm')
        if arm not in ('source_local_context','jev_refined') or arm in found:
            raise ValueError('Exactly the two distinct extension arms required')
        if (m.get('phase_kind')!='firstpass' or out.get('terminal')is not True
                or out.get('protocol_sha256')!=m.get('protocol_sha256') or out.get('total')!=1140):
            raise ValueError('Extension firstpass is not terminal1140')
        found.add(arm)


def run(run_dir, *, extension_runs, api_factory=None):
    require_extensions_terminal(extension_runs)
    return _LEGACY['run'](run_dir,api_factory=api_factory)


class _SummaryRunner:
    SCHEMA=SCHEMA
    terminal=staticmethod(terminal)
    identity=staticmethod(identity)
    def __getattr__(self,name):return getattr(legacy,name)


def audit_phase(run_dir):
    original=judge_summary.phase
    replay=FunctionType(original.__code__,{**original.__globals__,'runner':_SummaryRunner()},
                        original.__name__,original.__defaults__,original.__closure__)
    phase=replay(run_dir)
    # Strengthen the reused phase reader: no request outside the217 jobs, stage,
    # original prompt, or frozen physical retry budget may hide in the ledger.
    jobs={j['job_id']:j for j in phase['manifest']['jobs']}
    for row in phase['ledger']:
        ctx=row.get('context',{});j=jobs.get(ctx.get('job_id'))
        if j is None:raise ValueError('Unexpected job in recovery ledger')
        expected={'architecture':SCHEMA,'conversation':j['conversation'],'phase':'judge_retest',
                  'phase_id':SCHEMA,'method':j['method'],'query_id':j['query_id'],'job_id':j['job_id'],'component':'judge'}
        if (ctx!=expected or row.get('provider')!='gemini' or row.get('operation')!='generateContent'
                or row.get('resolved_model') not in (None,legacy.JUDGE_MODEL)):
            raise ValueError('Recovery request context/model/operation differs')
        if not row.get('cache_hit') and (type(row.get('attempt'))is not int or not 1<=row['attempt']<=3):
            raise ValueError('Physical transport bound changed')
    return phase


def summarize(run_dir):
    """Pure final3080-slot view; reuse frozen raw replay, metrics and resolution."""
    with overlay.offline():
        phase=audit_phase(run_dir);source=verify(overlay.check_pointer(phase['manifest']['source_manifest']))
        rows=deepcopy(source['strict_slots']);resolved={}
        for row in rows:
            k=key(row);a=source['parents'][k]
            b=source['existing_recovery'] if k==EXISTING_KEY else phase['records'].get(k)
            result=legacy.resolve_pair(a['strict'],a['compat'],b['strict'] if b else None,b['compat'] if b else None)
            value=result['correct'];resolved[k]=result
            row['scores']['judge_correct']=float(value) if value is not None else None
            row['judge_status']=row['qa_status']='complete' if value is not None else 'failed'
        slots={'strict_firstpass':source['strict_slots'],'compatible_firstpass':source['compatible_slots'],
               'final_with_uniform_technical_recovery':rows}
        views={name:strict.aggregate(values,denominator=1540,bootstrap_samples=2000) for name,values in slots.items()}
        return {'schema':SCHEMA+'-summary','created_utc':now(),'planned_per_arm':1540,'planned_total':3080,
                'run_dir':str(Path(run_dir).resolve()),
                'source_manifest':phase['manifest']['source_manifest'],'run_manifest':pointer(Path(run_dir)/'run_manifest.json'),
                'views':views,'slots_by_view':slots,'resolved':resolved,'audit_status':'no_invalid_artifacts',
                'original_failure_count':218,'existing_recovery_count':1,'new_recovery_count':217,
                'current_usage':phase['usage'],'existing_q74_usage_separate':source['existing_recovery']['usage'],
                'reader_and_f1_unchanged':True,'new_Reader_calls':0,'model_calls_in_summary':0,
                'phase_sources':{'outcome':pointer(Path(run_dir)/'run_outcome.json'),'ledgers':phase['ledger_refs']},
                'final_priority':POLICY['final_priority'],'remaining_failures':[key(r) for r in rows if r['judge_status']!='complete']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for name in ('prepare','check','run','summarize'):
        p=sub.add_parser(name);p.add_argument('--run-dir',type=Path,default=RUN)
        if name=='run':p.add_argument('--extension-runs',nargs=2,type=Path,required=True)
        if name=='summarize':p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='prepare':value=prepare(a.run_dir)
    elif a.action=='check':m=manifest(a.run_dir);value={'new_Judge_jobs':len(m['jobs']),'model_calls':0}
    elif a.action=='run':
        out=run(a.run_dir,extension_runs=a.extension_runs);value={k:out[k] for k in ('terminal','total','counts')}
    else:
        if a.output.exists():raise FileExistsError('Fresh summary path required')
        save_new(a.output,summarize(a.run_dir));value={'output':str(a.output),'model_calls':0}
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
