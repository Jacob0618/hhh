"""Offline full-cohort audit for fresh original-bank SDK20 plain QA.

Reuse the existing raw transport replay, official metric, paired bootstrap and
usage estimator. Never instantiate a provider or native memory client.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_sdk20_plain_qa_v1 as runner
from tools import sdk20_plain_qa_sources_v1 as sources
from tools import summarize_mem0_regroup_plain_qa_v1 as existing
from tools.replay_plain_judge_compat_v1 import replay_source
from tools.summarize_mem0_sparse1540 import Snapshot,measure_usage
from tools.summarize_sdk20_modules import fingerprint,InvalidArtifact
from tools.summarize_pilot400 import verified_official_metric
from tools.run_sdk20_modules import digest,save_new,now
from jev_mem import evaluation_plain_text_v1 as evaluator

ARMS=runner.ARMS
METRICS=('official_locomo_f1','custom_token_f1','judge_correct')


def aggregate(rows):
    f=existing.aggregate
    scoped=FunctionType(f.__code__,{**f.__globals__,'ARMS':ARMS},f.__name__,f.__defaults__,f.__closure__)
    scoped.__kwdefaults__=dict(f.__kwdefaults__ or {})
    return scoped(rows,denominator=1540,bootstrap_samples=2000)


def ledger(snapshot,run,m):
    rows=[];refs=[];indexed=defaultdict(list)
    jobs={(j['conversation'],j['method'],j['query_id']):j for j in m['jobs']}
    for ci in range(10):
        path=run/f'conv_{ci}/api/api_calls.jsonl'
        if not path.exists():continue
        data=snapshot.raw(path);refs.append({'path':str(path),'sha256':snapshot.sha(path)})
        for line in data.decode('utf-8-sig').splitlines():
            if not line.strip():continue
            row=json.loads(line);ctx=row.get('context',{});key=(ci,ctx.get('method'),ctx.get('query_id'))
            if key not in jobs:raise InvalidArtifact('Ledger job is outside phase population')
            stage=ctx.get('component')
            expected={'reader':(evaluator.READER_STAGE,evaluator.READER_MODEL),
                      'judge':(evaluator.JUDGE_STAGE,evaluator.JUDGE_MODEL)}.get(stage)
            if (ctx!=runner.stage_context(jobs[key],m,stage) or expected!=(row.get('stage'),row.get('model'))
                    or row.get('provider')!='gemini' or row.get('operation')!='generateContent'
                    or row.get('resolved_model') not in (None,row['model'])):
                raise InvalidArtifact('Unexpected QA model, stage, operation, or context')
            if row.get('cache_hit') is True:
                if ('status'in row or 'attempt'in row or row.get('usage_semantics')!='reused_response_no_new_model_usage'):
                    raise InvalidArtifact('Cache event incorrectly declares physical usage')
            elif type(row.get('attempt')) is not int or not 1<=row['attempt']<=3:
                raise InvalidArtifact('Physical attempt exceeds frozen provider bound')
            rows.append(row);indexed[row['request_hash'],row['stage'],fingerprint(ctx)].append(row)
    return rows,indexed,refs


def audit_stage(snapshot,path,expected,api_dir,context,indexed,callback,*,no_dispatch=False):
    if path.exists():
        row=existing.bound(snapshot,path,expected)
        if (row.get('error')or{}).get('type')=='APIInitializationFailure':
            if (row.get('status')!='not_run' or row.get('result') is not None or row.get('api_events')!=[]
                    or row.get('raw_invalid_label') is not None or path.with_suffix('.intent.json').exists()
                    or any(r.get('context')==context for rs in indexed.values() for r in rs)
                    or not row['error'].get('message','').startswith(('No dispatch; initialization raised ','No Judge dispatch; initialization raised '))):
                raise InvalidArtifact('Initialization failure has contradictory dispatch evidence')
            return row
    return existing.audit_stage(snapshot,path,expected,api_dir,context,indexed,callback,no_dispatch=no_dispatch)


def audit_phase(run_dir,source,*,parent=None):
    run=Path(run_dir).resolve();m,out=runner.terminal(run,_source=source)
    snapshot=Snapshot();logs,indexed,logrefs=ledger(snapshot,run,m)
    official,provenance=verified_official_metric()
    if provenance!=m['official_metric_provenance']:raise InvalidArtifact('Official metric binding differs')
    records={};audit_errors=[]
    for j in m['jobs']:
        paths=runner.stage_files(run,j);ident=runner.identity(j,m);api_dir=run/f"conv_{j['conversation']}/api"
        record={'job':j,'paths':{k:str(v) for k,v in paths.items()},'reader':None,'judge':None,'compat':None,'qa':None,'errors':[]}
        reader_success=False
        try:
            if j.get('reader_source'):
                if parent is None or j['job_id'] not in parent['records']:raise InvalidArtifact('Inherited Reader has no parent slot')
                prior=parent['records'][j['job_id']];pr=prior['reader'];rp=Path(prior['paths']['reader'])
                if (pr is None or pr['status']!='complete' or j['reader_source']!=runner.pointer(rp)):
                    raise InvalidArtifact('Reader inheritance is not the original valid answer')
                reader=existing.bound(snapshot,paths['reader'],{**ident,'stage':'reader'})
                context=runner.stage_context(j,m,'reader')
                if (reader.get('status')!='complete' or reader.get('result')!=pr['result'] or reader.get('inherited_from')!=j['reader_source']
                        or reader.get('api_events')!=[] or reader.get('model_calls')!=0 or reader.get('error') is not None
                        or reader.get('raw_invalid_label') is not None or paths['reader'].with_suffix('.intent.json').exists()
                        or any(row['context']==context for row in logs)):
                    raise InvalidArtifact('Successful Reader was changed or regenerated')
            else:
                reader=audit_stage(snapshot,paths['reader'],{**ident,'stage':'reader'},api_dir,
                    runner.stage_context(j,m,'reader'),indexed,lambda api:runner.reader_call(api,j,official))
            record['reader']=reader;reader_success=reader['status']=='complete'
        except Exception as exc:
            record['errors'].append({'stage':'reader','type':type(exc).__name__,'message':str(exc)[:1000]})
        try:
            expected={**ident,'stage':'judge'}
            if paths['reader'].exists():expected['reader_sha256']=snapshot.sha(paths['reader'])
            judge=audit_stage(snapshot,paths['judge'],expected,api_dir,runner.stage_context(j,m,'judge'),indexed,
                lambda api:evaluator.judge_answer(api,j['question'],j['reference'],record['reader']['result']['prediction']['answer']),
                no_dispatch=not reader_success)
            record['judge']=judge
        except Exception as exc:
            record['errors'].append({'stage':'judge','type':type(exc).__name__,'message':str(exc)[:1000]})
        try:
            if paths['compat'].exists():
                compat=existing.bound(snapshot,paths['compat'],{**ident,'stage':'compat'})
                if (not paths['judge'].exists() or compat.get('strict')!=runner.pointer(paths['judge'])
                        or compat.get('model_calls')!=0 or compat.get('raw_or_strict_modified') is not False):
                    raise InvalidArtifact('Compatibility provenance differs')
                judge=record['judge']
                if compat['status'] in ('strict_valid','compat_valid','rejected'):
                    if not reader_success or judge is None:raise InvalidArtifact('Compatibility lacks a verified Reader/Judge')
                    replay=replay_source(api_dir,judge,runner.stage_context(j,m,'judge'),indexed,
                        question=j['question'],reference=j['reference'],reader_text=record['reader']['result']['prediction']['answer'])
                    if (compat.get('interpretation')!=replay['interpretation'] or compat['status']!=replay['interpretation']['status']
                            or compat.get('source_evidence')!=replay['source_evidence'] or compat.get('error') is not None):
                        raise InvalidArtifact('Compatibility does not reproduce original raw interpretation')
                elif compat['status']=='unavailable':
                    if compat.get('interpretation') is not None or compat.get('source_evidence') is not None:
                        raise InvalidArtifact('Unavailable compatibility must not supply a label')
                    if reader_success and judge and (judge['status']=='complete' or (judge.get('error')or{}).get('type')=='InvalidJudgeLabel') and not compat.get('error'):
                        raise InvalidArtifact('Eligible Judge lost compatibility without a recorded error')
                else:raise InvalidArtifact('Unknown compatibility state')
                record['compat']=compat
        except Exception as exc:
            record['errors'].append({'stage':'compat','type':type(exc).__name__,'message':str(exc)[:1000]})
        try:
            if paths['qa'].exists():
                qa=existing.bound(snapshot,paths['qa'],{**ident,'stage':'qa'})
                reader,judge,compat=(record[k] for k in ('reader','judge','compat'))
                if any(v is None for v in (reader,judge,compat)):raise InvalidArtifact('Combined QA references unverified stages')
                for stage in ('reader','judge','compat'):
                    if qa.get(stage)!=runner.pointer(paths[stage]):raise InvalidArtifact('Combined stage pointer changed')
                resolved=runner.resolve_pair(judge,compat)
                if j.get('reader_source'):
                    pr=parent['records'][j['job_id']];resolved=runner.resolve_pair(pr['judge'],pr['compat'],judge,compat)
                elif m['phase_kind']=='technical_recovery':resolved=runner.resolve_pair(None,None,judge,compat)
                if (qa.get('reader_status')!=reader['status'] or qa.get('judge_status')!=judge['status']
                        or qa.get('reader_metrics')!=(reader['result']['metrics'] if reader_success else None)
                        or qa.get('judge_correct')!=(judge['result']['correct'] if judge['status']=='complete' else None)
                        or qa.get('resolved_judge')!=resolved
                        or qa.get('status')!=('complete' if reader['status']==judge['status']=='complete' else 'failed')):
                    raise InvalidArtifact('Combined QA changes independent stage metrics or resolution')
                record['qa']=qa
        except Exception as exc:
            record['errors'].append({'stage':'qa','type':type(exc).__name__,'message':str(exc)[:1000]})
        records[j['job_id']]=record
        audit_errors.extend({'job_id':j['job_id'],**x} for x in record['errors'])
    changed=snapshot.unchanged()
    if changed:raise InvalidArtifact('Phase source changed during read-only replay: '+str(changed[:3]))
    return {'run':str(run),'manifest':m,'outcome':out,'records':records,'ledger':logs,'ledger_refs':logrefs,
            'errors':audit_errors,'usage':measure_usage(logs),'inputs':{str(k):v for k,v in snapshot.files.items()}}


def record_rows(jobs,first,recovery=None,*,compatible=False,final=False):
    result=[]
    for j in jobs:
        a=first['records'][j['job_id']];b=recovery and recovery['records'].get(j['job_id']) if final else None
        reader=(b or a)['reader']
        if a['reader'] and a['reader']['status']=='complete':reader=a['reader']
        reader_ok=reader is not None and reader['status']=='complete'
        judge=a['judge'];label=judge['result']['correct'] if judge and judge['status']=='complete' else None
        if final:
            resolved=runner.resolve_pair(a['judge'],a['compat'],b['judge'] if b else None,b['compat'] if b else None)
            label=resolved['correct']
        elif compatible:label=runner.resolve_pair(a['judge'],a['compat'])['correct']
        if not reader_ok and label is not None:raise InvalidArtifact('Judge label exists without valid Reader')
        scores={k:reader['result']['metrics'][k] if reader_ok else None for k in METRICS[:2]}
        scores['judge_correct']=float(label) if label is not None else None
        reader_state=reader['status'] if reader else 'invalid' if any(e['stage']=='reader'for e in a['errors']) else 'missing'
        result.append({'query_id':j['query_id'],'conversation':j['conversation'],'category':j['category'],'method':j['method'],
            'selection_status':'complete','reader_status':reader_state,'judge_status':'complete' if label is not None else 'failed' if judge else 'missing',
            'qa_status':'complete' if label is not None else 'failed','scores':scores,'errors':a['errors']+(b['errors'] if b else []),
            'reader_answer_characters':len(reader['result']['prediction']['answer']) if reader_ok else None,
            'reader_answer_whitespace_words':len(reader['result']['prediction']['answer'].split()) if reader_ok else None})
    return result


def summarize(firstpass,recovery=None):
    initial=runner.read(Path(firstpass)/'run_manifest.json');source=sources.validate(initial['source_manifest']['path'])
    first=audit_phase(firstpass,source);second=audit_phase(recovery,source,parent=first) if recovery else None
    if first['manifest']['phase_kind']!='firstpass':raise InvalidArtifact('Main input is not firstpass')
    if second:
        p=second['manifest']['parent']
        if p['manifest']!=runner.pointer(Path(firstpass)/'run_manifest.json') or p['outcome']!=runner.pointer(Path(firstpass)/'run_outcome.json'):
            raise InvalidArtifact('Recovery belongs to another firstpass')
    jobs=source['jobs'];views={
        'strict_firstpass':record_rows(jobs,first),
        'compatible_firstpass':record_rows(jobs,first,compatible=True),
        'final_with_uniform_technical_recovery':record_rows(jobs,first,second,compatible=True,final=True)}
    phases=[first]+([second] if second else []);rows=[r for phase in phases for r in phase['ledger']]
    errors=[{'phase':p['run'],**e} for p in phases for e in p['errors']]
    return {'schema':'sdk20-fresh-plain-qa-summary-v1','created_utc':now(),'planned_per_arm':1540,'planned_total':3080,
        'source_manifest':initial['source_manifest'],'views':{k:aggregate(v) for k,v in views.items()},'slots_by_view':views,
        'audit_status':'no_invalid_artifacts' if not errors else 'invalid_artifacts_retained_as_missing','audit_errors':errors,
        'phase_sources':[{'manifest':runner.pointer(Path(p['run'])/'run_manifest.json'),'outcome':runner.pointer(Path(p['run'])/'run_outcome.json'),
                          'ledgers':p['ledger_refs']} for p in phases],
        'phase_counts':{p['run']:p['outcome']['counts'] for p in phases},'phase_usage':{p['run']:p['usage'] for p in phases},
        'input_sha256':{k:v for p in phases for k,v in p['inputs'].items()},
        'summary_code_sha256':{p:runner.sha(ROOT/p) for p in ('tools/summarize_sdk20_plain_qa_v1.py','tools/summarize_mem0_regroup_plain_qa_v1.py','tools/summarize_mem0_sparse1540.py')},
        'answer_length':{view:{a:{'valid':len(vals:=[r for r in records if r['method']==a and r['reader_answer_characters'] is not None]),
                'mean_characters':sum(r['reader_answer_characters'] for r in vals)/len(vals) if vals else None,
                'mean_whitespace_words':sum(r['reader_answer_whitespace_words'] for r in vals)/len(vals) if vals else None}
                for a in ARMS}for view,records in views.items()},
        'total_current_usage':measure_usage(rows),
        'usage_by_method':{a:measure_usage([r for r in rows if r['context']['method']==a]) for a in ARMS},
        'usage_by_stage':{a:measure_usage([r for r in rows if r['context']['component']==a]) for a in ('reader','judge')},
        'precomputation_cost_lineage':source.get('precomputation_cost_lineage'),'model_calls_in_summary':0,'native_clients':0,
        'limitations':['All1540 slots per arm remain in planned denominators; Reader/F1 and Judge have independent valid denominators.',
          'Known cohort, not blind validation. Same20facts count is not an equal token budget.',
          'Compatibility is a separate interpretation of original Judge text; technical retries may change generation and are reported separately.',
          'Old writing, typing and retrieval computation is reused and is not free cold-start computation. Current usage counts only current phase physical attempts.',
          'Reused paired_merged_minus_unmerged field means fact_types_sdk20 minus baseline_sdk20 in this report.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--firstpass',type=Path,required=True);p.add_argument('--recovery',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('New report output required')
    with sources.historical.offline():value=summarize(args.firstpass,args.recovery)
    save_new(args.output,value);print(json.dumps({'output':str(args.output),'audit_status':value['audit_status'],'model_calls':0}))


if __name__=='__main__':main()
