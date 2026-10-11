"""Offline fixed1540 SDK20-module summary using existing evaluation/metric helpers.

No provider clients are initialized. Historical A remains exactly1525/15; new
module failures have null results. Output must be a new report directory.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.data import load_data,queries,session_envelopes,SOURCE_HASH
from jev_mem.evaluation import answer_and_judge,lexical_metrics
from jev_mem.providers import ModelAPI
from tools import sdk20_module_sources as sources
from tools.summarize_pilot400 import verified_official_metric,paired_comparison,METRICS
from tools.summarize_mem0_sparse1540 import measure_usage
from tools.summarize_experiments import PRICES_USD_PER_MILLION
from tools import run_sdk20_modules as driver
from jev_mem.sdk20_cached_api import check_legacy_request,digest as request_digest
from jev_mem.selection import tag_fact
from jev_mem.pilot400 import query_type_scores
from jev_mem.sdk20_type_adapter import typed_select

ARMS=('baseline_sdk20','fact_types_sdk20','preseg_sdk20')
SCHEMA='sdk20-modules-v1'
READ_PROTOCOL='brief_answer_local_evidence_ids_v2'

class InvalidArtifact(ValueError):pass
class DeclaredTransportFailure(RuntimeError):
    def __init__(self,error_type):
        super().__init__('Verified recorded provider failure')
        self.provider_error_type=error_type

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def pointer(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def verified(path,protocol):
    row=driver.verify(path)
    if row.get('schema')!=SCHEMA or row.get('protocol_sha256')!=protocol:
        raise InvalidArtifact('Artifact schema/protocol differs: '+str(path))
    return row

class ReplayAPI:
    """Use unchanged legacy payload construction/parsing with file-only requests."""
    generate=ModelAPI.generate
    score_many=ModelAPI.score_many

    def __init__(self,api_dir,events,context,ledger):
        self.api_dir=Path(api_dir).resolve();self.events=events;self.context=context;self.ledger=ledger;self.seen=[]
        if not isinstance(events,list):raise InvalidArtifact('Missing API event list')

    @staticmethod
    def _logs_identity(rows,signature,key,stage):
        for r in rows:
            if any(r.get(k)!=signature[k] for k in ('provider','model','operation')) or r.get('request_hash')!=key or r.get('stage')!=stage:
                raise InvalidArtifact('Ledger provider/model/operation/request identity differs')
            if r.get('cache_hit')is True:
                if 'attempt'in r or 'status'in r or r.get('usage_semantics')!='reused_response_no_new_model_usage':
                    raise InvalidArtifact('Invalid cache consumer shape')
            elif r.get('cache_hit')not in(None,False):raise InvalidArtifact('Invalid physical/cache discriminator')

    @staticmethod
    def _failed_status(value):
        return value=='transport_error' or (type(value)is int and 400<=value<600)

    @staticmethod
    def _retry_status(value):
        return value=='transport_error' or (type(value)is int and value in (429,500,502,503,504,529))

    def _scope_rows(self,key,stage):
        # Each conversation owns an independent physical cache and intent set.
        return [r for (h,st,c),rs in self.ledger.items() if h==key and st==stage for r in rs
                if r.get('context',{}).get('conversation')==self.context['conversation']]

    def _physical(self,key,stage,signature,*,success):
        physical=[r for r in self._scope_rows(key,stage) if r.get('cache_hit')is not True]
        self._logs_identity(physical,signature,key,stage)
        ip=self.api_dir/'request_intents'/f'{key}.json'
        if not ip.exists():raise InvalidArtifact('Physical request lacks dispatch intent')
        intent=read(ip)
        if (intent.get('signature')!=signature or intent.get('stage')!=stage or intent.get('request_hash')!=key
            or intent.get('retries')!=2 or not physical or len(physical)>3
            or any(r.get('context')!=intent.get('context') for r in physical)
            or any(type(r.get('attempt'))is not int for r in physical)
            or [r.get('attempt') for r in physical]!=list(range(1,len(physical)+1))):
            raise InvalidArtifact('Physical request intent/context/bounded attempt sequence differs')
        if success:
            if (not(type(physical[-1].get('status'))is int and 200<=physical[-1]['status']<300)
                or any(not self._retry_status(r.get('status')) for r in physical[:-1])):
                raise InvalidArtifact('Successful physical producer is not the unique final2xx')
        elif (not self._failed_status(physical[-1].get('status'))
              or any(not self._retry_status(r.get('status')) for r in physical[:-1])):
            raise InvalidArtifact('Unknown/nonfailed status cannot justify declared failure')
        return physical

    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',**kwargs):
        check_legacy_request(model,body,stage)
        signature={'provider':provider,'model':model,'operation':operation,'body':body};key=request_digest(signature)
        matches=[e for e in self.events if e.get('request_hash')==key and e.get('stage')==stage]
        if len(matches)!=1 or (key,stage) in self.seen:raise InvalidArtifact('Missing/duplicate reconstructed request event')
        e=matches[0]
        if (e.get('provider')!=provider or e.get('model')!=model or e.get('operation')!=operation
            or e.get('context')!=self.context):raise InvalidArtifact('Event request/context differs')
        if e.get('source') not in ('existing_run_cache','inherited_exact_request','new_request'):
            raise InvalidArtifact('Unknown request source policy')
        self.seen.append((key,stage))
        target=self.api_dir/'api_cache'/f'{key}.json'
        if Path(e.get('cache_path','')).resolve()!=target:raise InvalidArtifact('Event cache path is outside fixed API cache')
        logs=self.ledger.get((key,stage,fingerprint(self.context)),[])
        self._logs_identity(logs,signature,key,stage)
        if target.exists():
            if e.get('error_type') is not None or e.get('cache_sha256')!=sha(target):raise InvalidArtifact('Cached response event disagrees')
            raw=read(target)
            if raw.get('request')!=signature or 'response' not in raw:raise InvalidArtifact('Raw request signature differs')
            response=raw['response'];expected_usage=response.get('usageMetadata',response.get('usage',{}))
            expected_model=response.get('modelVersion',response.get('model'))
            if expected_model not in (None,model):raise InvalidArtifact('Unapproved resolved model differs from frozen request model')
            good=[r for r in logs if r.get('cache_hit')is True or (type(r.get('status'))is int and 200<=r['status']<300)]
            if not good:raise InvalidArtifact('Response lacks same-context successful consumer')
            if any(r.get('usage')!=expected_usage or r.get('resolved_model')!=expected_model for r in good):
                raise InvalidArtifact('Consumer usage/model does not match signed raw response')
            lineage=self.api_dir/'cache_lineage'/f'{key}.json'
            if lineage.exists():
                line=read(lineage);sp=Path(line['source_path'])
                if (self.api_dir/'request_intents'/f'{key}.json').exists() or e.get('source')=='new_request':
                    raise InvalidArtifact('Inherited cache conflicts with local dispatch provenance')
                if (line['request_hash']!=key or Path(line['target_path']).resolve()!=target
                    or line['source_sha256']!=sha(sp) or line['target_sha256']!=sha(target)
                    or sha(sp)!=sha(target)):raise InvalidArtifact('Inherited raw cache lineage differs')
                candidates=[p/f'{key}.json' for p in driver.cache_roots(self.context['conversation']) if (p/f'{key}.json').exists()]
                if not candidates or sp.resolve()!=candidates[0].resolve():
                    raise InvalidArtifact('Inherited cache source violates frozen source precedence')
                if any(r.get('cache_hit')is not True for r in self._scope_rows(key,stage)):
                    raise InvalidArtifact('Inherited cache unexpectedly has new physical producers')
            else:
                if e.get('source')=='inherited_exact_request':raise InvalidArtifact('Inherited event lacks lineage')
                producer=self._physical(key,stage,signature,success=True)[-1]
                if producer.get('usage')!=expected_usage or producer.get('resolved_model')!=expected_model:
                    raise InvalidArtifact('Producer usage/model does not match raw response')
            return deepcopy(raw['response'])
        if e.get('cache_sha256')is not None or e.get('error_type')is None:raise InvalidArtifact('Missing raw response not declared failed')
        intent=self.api_dir/'request_intents'/f'{key}.json'
        if not intent.exists():raise InvalidArtifact('Transport failure lacks dispatch intent')
        inp=read(intent)
        if inp.get('signature')!=signature or inp.get('context')!=self.context or inp.get('stage')!=stage or inp.get('retries')!=2:
            raise InvalidArtifact('Transport failure intent differs')
        self._physical(key,stage,signature,success=False)
        raise DeclaredTransportFailure(e['error_type'])

    def assert_consumed(self):
        if len(self.seen)!=len(self.events):raise InvalidArtifact('Unused API events cannot be attributed to reconstructed requests')

def finite_score(v):return type(v)in(int,float) and math.isfinite(v) and 0<=v<=1

def validate_row(row,q,arm,protocol):
    if (row.get('schema')!=SCHEMA or row.get('protocol_sha256')!=protocol or row.get('method')!=arm
        or row.get('query_id')!=q['id'] or row.get('conversation')!=q['conversation']
        or any(row.get(k)!=q[k] for k in ('question','reference','category'))):
        raise InvalidArtifact('QA identity, protocol, question or reference differs')
    status=row.get('status');ids=row.get('selected_ids')
    if status not in ('complete','failed'):raise InvalidArtifact('Nonterminal or unrecognized QA status')
    if (not isinstance(ids,list) or any(not isinstance(v,str) or not v for v in ids)
        or len(set(ids))!=len(ids) or len(ids)>20):raise InvalidArtifact('Selected fact IDs malformed')
    if status=='failed':
        if row.get('result') is not None or not row.get('error'):
            raise InvalidArtifact('Failure must retain error with null result')
        return None
    if len(ids)!=20 or row.get('error') is not None:raise InvalidArtifact('Complete QA needs20 facts and no error')
    r=row.get('result')
    if not isinstance(r,dict):raise InvalidArtifact('Missing successful result')
    if (r.get('reader_protocol')!=READ_PROTOCOL or r.get('reader_model')!='gemini-2.5-flash'
        or r.get('judge_model')!='gemini-3.5-flash-lite' or r.get('selected_count')!=20):
        raise InvalidArtifact('Legacy reader/model/budget differs')
    pred=r.get('prediction');judge=r.get('judgment')
    if not isinstance(pred,dict) or not isinstance(pred.get('answer'),str):raise InvalidArtifact('Reader answer is not string')
    if (not isinstance(judge,dict) or type(judge.get('correct'))is not bool
        or not finite_score(judge.get('completeness')) or not isinstance(judge.get('reason'),str)):
        raise InvalidArtifact('Judge contract differs')
    # Preserve original evaluator contract: duplicate citations are allowed there.
    used,local=pred.get('used_fact_ids'),pred.get('used_local_fact_ids')
    if (not isinstance(used,list) or not isinstance(local,list) or len(used)!=len(local)
        or any(not isinstance(x,str) for x in used+local) or not set(used)<=set(ids)):
        raise InvalidArtifact('Citation IDs are outside selected evidence')
    mapping={f'f{i+1}':fid for i,fid in enumerate(ids)}
    if any(mapping.get(k)!=v for k,v in zip(local,used)):raise InvalidArtifact('Short/actual citation mapping differs')
    lexical=lexical_metrics(pred['answer'],q['reference'])
    if any(not finite_score(r.get(k)) or abs(r[k]-v)>1e-12 for k,v in lexical.items()):
        raise InvalidArtifact('Stored lexical metrics differ from existing evaluator')
    return r

def score_result(q,result,official):
    return {'query_id':q['id'],'conversation':q['conversation'],'category':q['category'],
       'official_locomo_f1':official(result['prediction']['answer'],q['reference'],q['category']),
       'custom_token_f1':result['token_f1'],'judge_correct':float(result['judgment']['correct']),
       'judge_completeness':result['judgment']['completeness']}

def aggregate(questions,slots,bootstrap_samples=2000):
    """Reuse existing ten-conversation paired bootstrap; preserve null outcomes."""
    n=len(questions);arms={};complete={}
    for arm in ARMS:
        values=[slots[(arm,q['id'])] for q in questions]
        counts={s:sum(v['status']==s for v in values) for s in ('complete','failed','missing','invalid')}
        records=[v['scores'] for v in values if v['status']=='complete']
        if sum(counts.values())!=n:raise InvalidArtifact('Denominator does not close')
        complete[arm]={r['query_id']:r for r in records}
        means={m:sum(r[m] for r in records)/len(records) if records else None for m in METRICS}
        arms[arm]={'cohort_denominator':n,'status_counts':counts,'completed_only_metrics':means,
            'full_cohort_observed_metrics':means if len(records)==n else {m:None for m in METRICS},
            'missing_zero_sensitivity':{'is_imputation_not_observed_scoring':True,'denominator':n,
               'imputed_count':n-len(records),'means':{m:sum(r[m] for r in records)/n for m in METRICS}},
            'judge_correct_count':sum(r['judge_correct'] for r in records),
            'scored_questions':records,'by_conversation':{},'by_category':{}}
        for field,vals in (('conversation',range(10)),('category',range(1,5))):
            for v in vals:
                sub=[x for x in values if x[field]==v];ss=[x['scores'] for x in sub if x['status']=='complete']
                arms[arm]['by_'+field][str(v)]={'denominator':len(sub),'status_counts':dict(Counter(x['status'] for x in sub)),
                    'completed_only_metrics':{m:sum(r[m] for r in ss)/len(ss) if ss else None for m in METRICS}}
    comparisons={arm+'_minus_baseline_sdk20':paired_comparison(complete[ARMS[0]],complete[arm],samples=bootstrap_samples,seed=0,denominator=n) for arm in ARMS[1:]}
    for c in comparisons.values():
        c['interpretation']='Exploratory common-success paired comparison across ten previously exposed conversations; missingness can bias the subset. No blind-test or causal significance claim.'
    return arms,comparisons

def read_ledgers(run):
    rows=[];pointers=[];missing=[]
    for ci in range(10):
        p=run/f'conv_{ci}/api/api_calls.jsonl'
        if not p.exists():missing.append(str(p));continue
        pointers.append(pointer(p))
        for lineno,line in enumerate(p.read_text(encoding='utf8').splitlines(),1):
            if not line.strip():continue
            r=json.loads(line);context=r.get('context',{})
            if context.get('conversation')!=ci:raise InvalidArtifact('Ledger conversation scope differs')
            if r.get('stage') in ('qa_reader','qa_judge'):
                expected_model='gemini-2.5-flash' if r['stage']=='qa_reader' else 'gemini-3.5-flash-lite'
                if (context.get('phase')!='qa' or context.get('method') not in ARMS[1:]
                    or r.get('provider')!='gemini' or r.get('operation')!='generateContent' or r.get('model')!=expected_model):
                    raise InvalidArtifact('QA ledger stage/model/context violates module-only dispatch')
            if r.get('resolved_model') not in (None,r.get('model')):
                raise InvalidArtifact('Unapproved ledger resolved model differs from requested model')
            if r.get('cache_hit') is True:
                if 'status' in r or 'attempt' in r or r.get('usage_semantics')!='reused_response_no_new_model_usage':
                    raise InvalidArtifact('Cache consumer cannot declare HTTP status/attempt')
            elif type(r.get('attempt'))is not int or r['attempt']<1:
                raise InvalidArtifact('Physical attempt lacks a positive attempt number')
            rows.append(r)
    return rows,pointers,missing

def usage_report(run,rows,pointers,missing):
    groups=defaultdict(list)
    for r in rows:groups[(r.get('stage','unspecified'),r.get('model','unknown'))].append(r)
    totals=measure_usage(rows)
    totals.update(source_ledgers=pointers,missing_ledgers=missing,scope_complete=not missing)
    if missing:totals['total_cost_usd']=None
    lineage=[]
    for p in sorted(run.glob('conv_*/api/cache_lineage/*.json')):
        value=read(p);lineage.append({**pointer(p),'document':value})
    model_resolution=Counter((r.get('stage'),r.get('model'),r.get('resolved_model')) for r in rows)
    return {'new_current_run':totals,'by_stage_model':[{'stage':k[0],'model':k[1],**measure_usage(v)} for k,v in sorted(groups.items())],
        'requested_to_resolved_models':[{'stage':k[0],'requested':k[1],'resolved':k[2],'events':v} for k,v in model_resolution.items()],
        'cache_lineage':lineage,'inherited_costs_recomputed':False,'full_cold_start_total_cost_usd':None,
        'prices_usd_per_million':PRICES_USD_PER_MILLION,
        'interpretation':'Configured-price estimates, not an invoice. New physical attempts counted once including failures; reused responses have no new token charge. Inherited source work is not free cold-start work, and its original costs are not included again. Missing usage, transport uncertainty and unpriced models remain unknown.'}

def summarize(run_dir,*,bootstrap_samples=2000):
    run=Path(run_dir).resolve();m=driver.manifest(run);protocol=m['protocol_sha256']
    if (m.get('questions')!=1540 or m.get('dataset_sha256')!=SOURCE_HASH or m.get('response_schema')is not False
        or m.get('reader_protocol')!=READ_PROTOCOL or m.get('native_state')!='original_unrepaired'):
        raise InvalidArtifact('Not the frozen1540 legacy-SDK20 protocol')
    questions=[{**q,'conversation':ci} for ci,item in enumerate(load_data()) for q in queries(item,ci)]
    if len(questions)!=1540 or m['query_ids']!={str(ci):[q['id'] for q in questions if q['conversation']==ci] for ci in range(10)}:
        raise InvalidArtifact('Manifest cohort changed')
    official,provenance=verified_official_metric();logs,log_paths,missing=read_ledgers(run)
    if missing:raise InvalidArtifact('Final summary requires all ten API ledgers')
    ledger=defaultdict(list)
    for r in logs:ledger[(r.get('request_hash'),r.get('stage'),fingerprint(r.get('context',{})))].append(r)
    slots={};artifacts={str(run/'run_manifest.json'):sha(run/'run_manifest.json')};expected_paths=set();banks={};alltags={};tagcounts={};dates={}
    for ci,item in enumerate(load_data()):
        folder=run/f'conv_{ci}';sessions={s['session_id']:s for s in session_envelopes(item)}
        dates[ci]=list(sessions.values())[-1]['observation_date']
        for kind in ('session','preseg'):
            bank,bhash=driver.get_bank(run,ci,kind,m);banks[ci,kind]=(bank,bhash)
            if bank['native_state']!='original_unrepaired':raise InvalidArtifact('Bank indexed state differs')
            artifacts[str(folder/kind/'bank.json')]=bhash
        tags={};bank,bhash=banks[ci,'session']
        for f in bank['facts']:
            p=folder/'tags'/f'{f["id"]}.json';t=verified(p,protocol)
            context='\n'.join(x['content'] for x in sessions[f['input_scope']['session_id']]['messages'])
            if (t['status']!='complete' or t['fact_id']!=f['id'] or t['conversation']!=ci or t['bank_sha256']!=bhash
                or t['source_context_sha256']!=fingerprint(context) or t['fact_text_sha256']!=fingerprint(f['text'])):
                raise InvalidArtifact('Full native fact typing gate failed')
            api=ReplayAPI(folder/'api',t['api_events'],{'conversation':ci,'phase':'write_typing','method':ARMS[1],'fact_id':f['id']},ledger)
            value=tag_fact({**f,'source_context':context},api);api.assert_consumed()
            if value!=t['result']:raise InvalidArtifact('Fact type raw replay differs')
            tags[f['id']]=value;artifacts[str(p)]=sha(p)
        if len(list((folder/'tags').glob('*.json')))!=len(tags):raise InvalidArtifact('Unexpected fact tag files')
        alltags[ci]=tags;tagcounts[str(ci)]=len(tags)
    if sum(tagcounts.values())!=2376:raise InvalidArtifact('Not all2376 Mem0 facts typed')
    from mem0.memory.main import score_and_rank
    for q in questions:
        ci=q['conversation'];folder=run/f'conv_{ci}';date=dates[ci]
        oldrow=sources.baseline_row(ci,q)
        captured={}
        for kind in ('session','preseg'):
            p=folder/'retrieval'/kind/f'{q["id"]}.json';ret=verified(p,protocol);bank,bhash=banks[ci,kind]
            if (ret['status']!='complete' or ret['kind']!=kind or ret['bank_sha256']!=bhash
                or ret['query_id']!=q['id'] or ret['question']!=q['question']):raise InvalidArtifact('Native retrieval binding differs')
            cap=ret['capture']
            if cap['capture_sha256']!=fingerprint({k:v for k,v in cap.items() if k!='capture_sha256'}):raise InvalidArtifact('Capture checksum differs')
            inp=deepcopy(cap['inputs'])
            if inp['top_k']!=20 or inp['threshold']!=.1 or len(inp['semantic_results'])>80:raise InvalidArtifact('Not nativeSDK20 pool/budget')
            base=score_and_rank(**inp);inp['top_k']=len(inp['semantic_results']);eligible=score_and_rank(**inp)
            if base!=cap['base20'] or eligible!=cap['eligible'] or [f['id'] for f in base]!=ret['selected_ids']:
                raise InvalidArtifact('Captured native rank differs on existing scorer replay')
            if kind=='session' and ret['selected_ids']!=oldrow['selected_ids']:raise InvalidArtifact('Original SDK20 does not replay')
            captured[kind]=ret;artifacts[str(p)]=sha(p)
        for arm in ARMS:
            kind='preseg' if arm==ARMS[2] else 'session';bank,bhash=banks[ci,kind];byid={f['id']:f for f in bank['facts']}
            p=folder/'qa'/arm/f'{q["id"]}.json';sp=folder/'selection'/arm/f'{q["id"]}.json'
            expected_paths.add(p.relative_to(run).as_posix())
            if not p.exists():raise InvalidArtifact('Final QA slot missing: '+str(p))
            row=verified(p,protocol);result=validate_row(row,q,arm,protocol);sel=verified(sp,protocol)
            if (sel['method']!=arm or sel['query_id']!=q['id'] or sel['bank_sha256']!=bhash
                or sel['selected_ids']!=row['selected_ids'] or any(i not in byid for i in sel['selected_ids'])):
                raise InvalidArtifact('Selection/bank/QA differs')
            if arm==ARMS[0]:
                if (any(row.get(k)!=oldrow[k] for k in ('status','result','selected_ids','error','source','retrieval_source','reference_date'))
                    or row.get('api_events')!=[] or row.get('inherited_result')is not True
                    or sel['status']!='complete' or sel.get('source')!=oldrow['retrieval_source']):
                    raise InvalidArtifact('Historical SDK20 result changed or redispatched')
            else:
                rp=folder/'retrieval'/kind/f'{q["id"]}.json'
                if row['selection_sha256']!=sha(sp) or row['bank_sha256']!=bhash or sel['retrieval_sha256']!=sha(rp) or row['reference_date']!=date:
                    raise InvalidArtifact('QA/selection/retrieval file binding differs')
                if sel['status']=='complete':
                    if arm==ARMS[1]:
                        qp=folder/'query_types'/f'{q["id"]}.json';qt=verified(qp,protocol)
                        if qt['query_id']!=q['id'] or qt['question']!=q['question'] or qt['status']!='complete':raise InvalidArtifact('Query type identity differs')
                        api=ReplayAPI(folder/'api',qt['api_events'],{'conversation':ci,'phase':'query_types','method':arm,'query_id':q['id']},ledger)
                        needs,meta=query_type_scores(q['question'],api);api.assert_consumed()
                        if qt['scores']!=needs or qt['metadata']!=meta:raise InvalidArtifact('Query type raw replay differs')
                        selected=typed_select(captured[kind]['capture'],alltags[ci],needs)
                        if selected['selected_ids']!=sel['selected_ids'] or selected['trace']!=sel['trace']:raise InvalidArtifact('Typed selection replay differs')
                        artifacts[str(qp)]=sha(qp)
                    elif sel['selected_ids']!=captured[kind]['selected_ids']:raise InvalidArtifact('Preseg native20 selection differs')
                    api=ReplayAPI(folder/'api',row['api_events'],{'conversation':ci,'phase':'qa','query_id':q['id'],'method':arm},ledger)
                    try:
                        replay=answer_and_judge(api,q['question'],q['reference'],[byid[i] for i in sel['selected_ids']],reference_date=date)
                    except InvalidArtifact:raise
                    except Exception as exc:
                        if row['status']!='failed':raise InvalidArtifact('Successful QA cannot replay') from exc
                        if isinstance(exc,DeclaredTransportFailure) and row['error'].get('type')!=exc.provider_error_type:
                            raise InvalidArtifact('Declared provider error type differs') from exc
                        if not isinstance(exc,DeclaredTransportFailure) and row['error']!={'type':type(exc).__name__,'message':str(exc)}:
                            raise InvalidArtifact('Declared parsing failure differs from raw replay') from exc
                        replay=None
                    api.assert_consumed()
                    if replay!=result:raise InvalidArtifact('Saved QA result differs from exact legacy raw-response replay')
                else:
                    # Selection failures require a separate recorded query-stage audit;
                    # do not silently admit errors without raw request provenance.
                    raise InvalidArtifact('Terminal selection failure requires reviewed provenance before final summary')
            scores=score_result(q,result,official) if result is not None else None
            slots[arm,q['id']]={'query_id':q['id'],'conversation':ci,'category':q['category'],'method':arm,
                'status':row['status'],'scores':scores,'error':row.get('error'),'artifact':pointer(p)}
            artifacts[str(p)]=sha(p);artifacts[str(sp)]=sha(sp)
    extras=[p.relative_to(run).as_posix() for p in run.glob('conv_*/qa/*/*.json') if p.relative_to(run).as_posix() not in expected_paths]
    if extras:raise InvalidArtifact('Unexpected QA artifact paths')
    arms,paired=aggregate(questions,slots,bootstrap_samples)
    if arms[ARMS[0]]['status_counts']!={'complete':1525,'failed':15,'missing':0,'invalid':0} or arms[ARMS[0]]['judge_correct_count']!=1001:
        raise InvalidArtifact('Historical baseline no longer1525/15/1001')
    if any(sha(p)!=v for p,v in artifacts.items()) or any(sha(p['path'])!=p['sha256'] for p in log_paths):
        raise InvalidArtifact('Files changed during final offline summary')
    return {'schema':SCHEMA+'-summary-v1','created_utc':datetime.now(timezone.utc).isoformat(),'run_dir':str(run),
        'protocol_sha256':protocol,'cohort_per_arm':1540,'expected_slots':4620,'all_terminal_slots_valid':True,
        'official_metric_provenance':provenance,'native_tag_counts':tagcounts,'arms':arms,'paired_differences':paired,
        'outcomes':list(slots.values()),'usage':usage_report(run,logs,log_paths,missing),'integrity_errors':[],
        'input_hashes':artifacts,'model_calls':0,
        'limitations':['Legacy baseline remains1525 successes and15 failed/null, not1540 scored answers.',
                       'All conversations were previously exposed; common-success comparisons can have nonrandom missingness.',
                       'Native source scope is not manually verified fact support. Sparse C only splits; no proven cross-session topic merges.',
                       'A downstream no-event guard failure or terminal selection failure requires separate provenance review; this summary refuses unsupported failure shapes rather than imputing them.',
                       'Physical-cost estimates exclude inherited cold work and retain unknown usage/unpriced-model costs.']}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError('Output directory must be new')
    if args.bootstrap_samples<1:raise ValueError('Positive bootstrap samples required')
    report=summarize(args.run_dir,bootstrap_samples=args.bootstrap_samples)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources._dump_new(args.output_dir/'summary.json',report)
    lines=['# SDK20 原协议：两个模块结果','', '完成题均值、固定1540分母的缺失计零敏感性值和完整观察均值分别报告。失败是null结果，未伪造生成答案。','',
           '| 方法 | 完成/1540 | 失败 | 缺失 | 无效 | 官方 F1(%) | Judge(%) |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,a in report['arms'].items():
        c=a['status_counts'];m=a['completed_only_metrics']
        fmt=lambda v:'—' if v is None else f'{v*100:.3f}'
        lines.append(f'| {arm} | {c["complete"]}/1540 | {c["failed"]} | {c["missing"]} | {c["invalid"]} | {fmt(m["official_locomo_f1"])} | {fmt(m["judge_correct"])} |')
    lines+=['','同题配对仅使用双方都完成的题，失败选择偏差不能靠bootstrap消除；完整来源、费用未知项及逐题核验见summary.json。']
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'output_dir':str(args.output_dir),'integrity_errors':len(report['integrity_errors'])}))

if __name__=='__main__':main()
