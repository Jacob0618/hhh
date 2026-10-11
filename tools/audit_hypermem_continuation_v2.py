"""Read-only producer/native/QA evidence checks for the new HyperMem run."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from tools import audit_hypermem_terminal_v2 as admission
from tools import audit_simplemem_terminal_v2 as common
from tools import baselines_hypermem_native_v2 as adapter
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body, response_text, check_request, READER_STAGE, JUDGE_STAGE

require, read, sha, binding, jsonl = common.require, common.read, common.file_sha, common.binding, common.jsonl
STAGES = {'hypermem_boundary','hypermem_episode','hypermem_topic_create','hypermem_topic_update',
          'hypermem_episode_roles','hypermem_topic_match','hypermem_fact_extract','hypermem_fact_roles'}


def api_audit(root, proof):
    root = Path(root)
    calls = jsonl(root/'api_calls.jsonl'); events = jsonl(root/'logical_events.jsonl')
    physical = defaultdict(list); consumers = defaultdict(list)
    for c in calls:
        if not c.get('cache_hit'): physical[c['request_hash']].append(c)
    for e in events:
        require(e.get('error_type') is None and e.get('source') in ('new_request','existing_run_cache','inherited_exact_request'), 'Unknown logical result')
        consumers[e['request_hash']].append(e)
        context=e.get('context',{});ci=context.get('conversation');phase=context.get('phase')
        require(type(ci) is int and ci in range(10) and context.get('method')=='hypermem','Logical method/conversation differs')
        if root.name=='api_method' and phase=='build':
            require(context=={'method':'hypermem','conversation':ci,'phase':'build'},'Build context differs')
        else:
            expected_phase='retrieval' if root.name=='api_method' else root.name[4:]
            require(context=={'method':'hypermem','conversation':ci,'phase':expected_phase,'query_id':context.get('query_id')}
                    and any(q['query_id']==context.get('query_id') and q['conversation']==ci for q in proof['questions']), 'QA/retrieval owner/phase/question differs')
    intents = {p.stem:p for p in (root/'request_intents').glob('*.json')}
    rawpaths = {p.stem:p for p in (root/'api_cache').glob('*.json')}
    lineages = {p.stem:p for p in (root/'cache_lineage').glob('*.json')}
    require(not(set(intents)&set(lineages)) and set(intents)==set(physical)
            and set(rawpaths)==set(consumers)==set(intents)|set(lineages), 'Raw/intent/producer/lineage sets differ')
    require(not lineages or root.name=='api_method', 'QA cannot import parent responses')
    records = {}
    for key,p in rawpaths.items():
        raw=read(p); sig=raw['request']; require(digest(sig)==key,'Raw signature digest differs')
        es=consumers[key]; operation=sig['operation']; model=sig['model']; stage=raw['stage']
        require(sig['provider']=='gemini','Wrong provider')
        if operation=='generateContent':
            require(model=='gemini-2.5-flash','Wrong generation model')
            if root.name=='api_method':
                cfg=sig['body']['generationConfig']
                require(stage in STAGES and cfg=={'maxOutputTokens':16384,'responseMimeType':'application/json','thinkingConfig':{'thinkingBudget':0}}, 'Internal generation body/stage differs')
            else:
                require(stage in (READER_STAGE,JUDGE_STAGE),'Unknown QA stage')
                check_request(sig['provider'],model,sig['body'],operation,stage)
        else:
            require(root.name=='api_method' and model=='gemini-embedding-001' and operation in ('batchEmbedContents','countTokens'), 'Unknown embedding endpoint')
        response=raw['response']; rawsha=sha(p)
        for e in es:
            require(e['cache_path']==str(p) and e['cache_sha256']==rawsha and e['stage']==stage
                    and all(e[k]==sig[k] for k in ('provider','model','operation')), 'Logical raw/stage/endpoint binding differs')
        if key in lineages:
            require(key in proof['registry'],'Unadmitted imported key')
            origin=proof['registry'][key]; source=origin['raw']
            require(sha(source['path'])==source['sha256']==rawsha and sig==origin['signature'], 'Original imported raw bytes changed')
            require(read(lineages[key])=={'source_path':source['path'],'source_sha256':source['sha256'],
                    'target_path':str(p.resolve()),'target_sha256':source['sha256'],'request_hash':key}, 'Imported lineage differs')
            require(es[0]['source']=='inherited_exact_request' and all(e['context']==origin['context'] for e in es)
                    and all(e['source']=='existing_run_cache' for e in es[1:]),'Imported prefix consumer order/context differs')
        else:
            require(key not in proof['registry'], 'Original physical request was regenerated')
            intent=read(intents[key]); cc=physical[key]
            require(intent['signature']==sig and intent['request_hash']==key and type(intent['retries']) is int and intent['retries']==2, 'Intent signature/retry authority differs')
            expected={'context':intent['context'],'stage':intent['stage'],'request_hash':key,**{k:sig[k] for k in ('provider','model','operation')}}
            require([c['attempt'] for c in cc]==list(range(1,len(cc)+1)) and len(cc)<=3 and cc[-1]['status']==200
                    and all(c['status'] in common.RETRYABLE for c in cc[:-1]), 'Physical attempts unresolved or unbounded')
            require(all(common.identity(c)==expected for c in cc), 'Physical retry identity differs')
            times=[datetime.fromisoformat(c['timestamp_utc']) for c in cc]
            require(times==sorted(times) and datetime.fromisoformat(intent['created_utc'])<=times[0], 'Intent/attempt order differs')
            require(cc[-1].get('usage')==response.get('usageMetadata',response.get('usage',{}))
                    and cc[-1].get('resolved_model')==response.get('modelVersion',response.get('model')), 'Raw/physical usage or model differs')
            producers=[e for e in es if e['source']=='new_request']
            require(len(producers)==1 and common.identity(producers[0])==expected
                    and datetime.fromisoformat(producers[0]['recorded_utc'])>=times[-1]
                    and all(e['source']=='existing_run_cache' for e in es if e is not producers[0]), 'Logical producer order/identity differs')
        records[key]={'signature':sig,'raw_value':raw,'raw':binding(p),'events':es}
    cached=[c for c in calls if c.get('cache_hit')]
    require(Counter(digest(common.identity(e)) for e in events if e['source']!='new_request')==
            Counter(digest(common.identity(c)) for c in cached), 'Cache consumer multiplicity differs')
    for c in cached:
        r=records[c['request_hash']]['raw_value']['response']
        require(c.get('usage_semantics')=='reused_response_no_new_model_usage' and c.get('usage')==r.get('usageMetadata',r.get('usage',{}))
                and c.get('resolved_model')==r.get('modelVersion',r.get('model')), 'Cache usage metadata differs')
    pacing=jsonl(root/'embedding_pacing_events.jsonl')
    require(Counter((e['request_hash'],e['physical_attempt']) for e in pacing if e['event']=='physical_embedding_admitted')==
            Counter((key,c['attempt']) for key,cc in physical.items() for c in cc if c['operation']=='batchEmbedContents'), 'Embedding admission mismatch')
    return {'root':str(root),'physical':len([c for c in calls if not c.get('cache_hit')]),'cache':len(cached),
            'logical':len(events),'imported_keys':sorted(lineages),'raws':{key:r['raw'] for key,r in records.items()},
            'ledger':binding(root/'api_calls.jsonl'),'logical_events':binding(root/'logical_events.jsonl')},records,events,calls


def native_audit(folder, proof, records, events):
    runtime=adapter._runtime; generation=Counter(); embedded=Counter(); native_bindings=[]; compatibility=[]
    modules=runtime._load()
    require({p.name for p in (folder/'work/prefix').iterdir()}=={f'c{ci}{suffix}' for ci in range(10) for suffix in ('.intent.json','.complete.json','.jsonl')},'Extra/missing prefix evidence files')
    data=admission.chain.base.load_data()
    for ci in range(10):
        native=folder/f'work/native/c{ci}'; context={'method':'hypermem','conversation':ci,'phase':'build'}
        case=proof['rows'][ci]; prefix=folder/'work/prefix'
        require(read(prefix/f'c{ci}.complete.json')=={'conversation':ci,'request_hash':case['request_hash'],'original_raw':case['raw']},'Incomplete prefix')
        require(read(prefix/f'c{ci}.intent.json')=={'conversation':ci,'request_hash':case['request_hash'],'signature':case['signature'],
                'context':context,'stage':case['stage'],'original_raw':case['raw']},'Prefix intent differs')
        journal=jsonl(prefix/f'c{ci}.jsonl'); ee=[e for e in events if e['context']['conversation']==ci]
        require(len(journal)==len(ee) and journal and journal[0]['first_prefix'] is True and not any(j['first_prefix'] for j in journal[1:]), 'Prefix journal count/order differs')
        expected_journal=[{'event':'request_admitted','first_prefix':i==0,'request_hash':e['request_hash'],
                'context':e['context'],'stage':e['stage'],'operation':e['operation'],
                'source':'original_exact_raw' if e['request_hash'] in proof['registry'] else 'new_namespace_request'} for i,e in enumerate(ee)]
        require(journal[0]==expected_journal[0] and Counter(map(digest,journal))==Counter(map(digest,expected_journal)), 'Request admission journal differs')
        conv=admission.chain.base.conversation_only(data[ci]); identity=read(native/'identity.json')
        require(identity=={'schema':adapter.SCHEMA,'input_sha256':runtime._sha(conv),'upstream_revision':adapter.UPSTREAM_REVISION,'config':adapter.CONFIG},'Source/config identity differs')
        require(sha(native/'source_messages.json')==case['source_messages']['sha256'],'Rebuilt original source messages differ')
        state=read(native/'state.json'); require(state['work_dir']==str(native),'State path differs'); runtime._verify_state(state)
        runtime._verify_artifacts(native,state['document_embedding_sha256'])
        for name,filename in {'bm25':'hypergraph_bm25_index_conv_0.pkl','vectors':'hypergraph_embedding_index_conv_0.pkl','enhanced':'hypergraph_embedding_index_enhanced_conv_0.pkl'}.items():
            require(sha(native/'cache'/filename)==state['cache_sha256'][name],'Native index changed')
        for p in sorted((native/'calls').glob('*.json')):
            runtime._verify_seal(p); value=read(p); r=value['request']; stage=r['stage']
            require(value['status']=='success' and value['request_sha256']==runtime._sha(r) and stage in STAGES
                    and r=={'prompt':r['prompt'],'response_format':{'type':'json_object'},'stage':stage,'json_output':True,'max_output':16384,'system':None},'Invalid native call')
            sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':generation_body(r['prompt'],system=None,json_output=True,max_output=16384)}
            key=digest(sig); rr=records[key]; parsed=json.loads(response_text(rr['raw_value']['response']))
            require(rr['signature']==sig and value['parsed_response']==value['raw_response']==parsed,'Native value differs from exact raw')
            if stage=='hypermem_boundary':
                adapter._Generation._validate(parsed,stage,{})
                if 'should_wait' not in parsed: compatibility.append({'call':binding(p),'request_hash':key,'native_default_should_wait':True,'payload_mutated':False})
            generation[(digest(context),stage,key)]+=1
            if p.name=='episode_000001_hypermem_boundary.json':
                old=read(case['native_call']['path'])
                require(r==old['request'] and parsed==old['raw_response'] and key==case['request_hash'],'First successful native call differs from original failure')
            native_bindings.append(binding(p))
        for p in sorted((native/'embedding_evidence').glob('*.json')):
            runtime._verify_seal(p); value=read(p); ident=value['identity']
            require(value['status']=='success','Failed native embedding')
            vectors=[]
            for stage,sig in common.embedding_signatures(ident):
                key=digest(sig); rr=records[key]; require(rr['signature']==sig,'Native embedding request differs')
                if sig['operation']=='countTokens': require(type(rr['raw_value']['response']['totalTokens']) is int and rr['raw_value']['response']['totalTokens']<=2048,'Embedding length invalid')
                else:
                    rows=rr['raw_value']['response']['embeddings']; require(len(rows)==len(sig['body']['requests']),'Embedding row count differs')
                    for row in rows:
                        v=row['values']; norm=math.sqrt(sum(x*x for x in v)); require(len(v)==768 and math.isfinite(norm) and norm>0,'Invalid vector'); vectors.append([x/norm for x in v])
                embedded[(ci,stage,key)]+=1
            require(value['vectors']==vectors,'Original normalized native embedding values differ')
            native_bindings.append(binding(p))
        build=read(folder/f'results/c{ci}/build.json'); require(build['status']=='complete','Incomplete build')
        expected_memory=[]
        for i,relative in enumerate(state['export_paths']):
            source=native/relative;target=folder/f'results/c{ci}/memory'/f'{i}_{source.name}'
            require(sha(target)==sha(source),'Permanent export differs from actual native source')
            expected_memory.append(binding(target))
        require(build['memory']==expected_memory,'Memory export set/order/source binding differs')
        for q in [q for q in proof['questions'] if q['conversation']==ci]:
            result=read(folder/f'results/c{ci}/retrieval/{q["query_id"]}.json'); require(result['status']=='complete','Retrieval incomplete')
            value=result['value']; key=runtime._sha({'question':q['question'],'graph_sha256':state['graph_sha256'],'config':adapter.CONFIG})
            p=native/'retrieval'/(key+'.json'); runtime._verify_seal(p); require(read(p)==value,'Persisted retrieval differs from native artifact')
            require(value['question']==q['question'] and value['graph_sha256']==state['graph_sha256'] and value['config']==adapter.CONFIG,'Retrieval identity differs')
            grouped={layer:[] for layer in ('topics','episodes','facts')}
            for evidence in value['evidence']:
                layer=evidence['layer'];require(layer in grouped and evidence['rank']==len(grouped[layer]) and evidence['included_in_context']==(layer!='topics'),'Retrieval evidence layout differs')
                grouped[layer].append((evidence['node'],evidence['score']))
            context=modules['main/stage4_hypergraph_retrieval.py'].format_hierarchical_results(grouped,state['speaker_a'],state['speaker_b'],'011')
            require(value['context']==context,'Complete original native context formatter differs')
            expected_events=[common.normalized_event(e) for e in events if e['context'].get('query_id')==q['query_id']]
            require(result['events']==expected_events,'Retrieval capture does not match actual method consumers')
            native_bindings.append(binding(p))
    expected_generation=Counter((digest(e['context']),e['stage'],e['request_hash']) for e in events if e['operation']=='generateContent')
    require(generation==expected_generation,'Generation native/logical multiplicity differs')
    expected_embedding=Counter((e['context']['conversation'],e['stage'],e['request_hash']) for e in events if e['operation']!='generateContent')
    require(embedded==expected_embedding,'Embedding native/logical multiplicity differs')
    return {'status':'passed','native_bindings':native_bindings,'compatibility':compatibility,'prefix_conversations':10,
            'generation_consumers':sum(generation.values()),'embedding_consumers':sum(embedded.values()),
            'scope':'Actual source/index/vector byte verification plus raw-to-native/QA artifacts; no regenerated model output.'}


def audit_all(folder,proof):
    folder=Path(folder)
    require(admission.inventory(admission.OLD)==proof['original_inventory'],'Original parent inventory changed')
    roots=sorted((folder/'work').glob('api_*'))
    require({p.name for p in roots}<={'api_method','api_firstpass','api_recovery'} and {'api_method','api_firstpass'}<={p.name for p in roots},'API phase roots differ')
    api=[]; all_calls=[]; all_events=[]; method=None
    for root in roots:
        report,records,events,calls=api_audit(root,proof);api.append(report);all_calls.extend(calls)
        if root.name=='api_method':method=(records,events)
        else:all_events.extend(events)
    native=native_audit(folder,proof,*method)
    qs={q['query_id']:q for q in proof['questions']};plan=read(folder/'qa_recovery_plan.json')['jobs'];outcome=read(folder/'qa_recovery_outcome.json')['results']
    require(len({r['query_id'] for r in plan})==len(plan) and len({r['query_id'] for r in outcome})==len(outcome) and {r['query_id'] for r in plan}=={r['query_id'] for r in outcome}
            =={p.name for p in (folder/'results/recovery').glob('*') if p.is_dir()},'Recovery plan/outcome set differs')
    require({p.name for p in (folder/'results/firstpass').glob('*') if p.is_dir()}==set(qs),'Firstpass cohort differs')
    eligible={}
    for qid,q in qs.items():
        leaf=folder/'results/firstpass'/qid;reader=read(leaf/'reader.json');judge=read(leaf/'judge.json')
        require(not reader.get('inherited'),'Original firstpass cannot have an inherited Reader')
        if reader['status']=='complete' and judge['status']=='complete':continue
        kind='reader' if reader['status']=='failed' else 'judge';failed=reader if kind=='reader' else judge
        ret=read(folder/f'results/c{q["conversation"]}/retrieval/{qid}.json')['value']
        require(admission.chain.base.known_recoverable(failed,folder,'hypermem',q,ret,kind=kind,reader=reader),'Unqualified QA recovery')
        eligible[qid]=read(leaf/'outcome.json')
    require({r['query_id']:r for r in plan}==eligible,'Recovery plan is not exactly the eligible failed firstpass set')
    for row in outcome:
        qid=row['query_id'];leaf=folder/'results/recovery'/qid
        require(row==read(leaf/'outcome.json'),'Recovery outcome differs from durable leaf')
        original=read(folder/'results/firstpass'/qid/'reader.json');current=read(leaf/'reader.json')
        if original['status']=='complete':
            require(current.get('inherited') is True and {k:v for k,v in current.items() if k!='inherited'}==original,'Successful original Reader regenerated or changed')
        else:require(not current.get('inherited'),'Failed Reader falsely inherited')
    expected=Counter()
    for phase in ('firstpass','recovery'):
        qids=set(qs) if phase=='firstpass' else {r['query_id'] for r in plan}
        for qid in qids:
            leaf=folder/'results'/phase/qid
            for name in ('reader','judge'):
                stage=read(leaf/(name+'.json'))
                if stage.get('inherited'):continue
                for event in stage.get('events',[]):expected[digest(common.normalized_event(event))]+=1
    require(expected==Counter(digest(common.normalized_event(e)) for e in all_events),'Extra/missing QA logical consumers')
    return {'status':'passed','errors':[],'api':api,'native':native,'parent_producers':10,
            'parent_usage':proof['original_physical_usage'],'new_usage':admission.usage_presence(all_calls),
            'combined_physical_usage':admission.usage_presence([r['physical'] for r in proof['rows']]+all_calls),
            'parent_inventory_sha256':digest(proof['original_inventory'])}
