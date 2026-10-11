"""A-Mem c2-only continuation: exact v3 raw prefix, immutable inherited QA.

This separate entry privately reuses the frozen v3 controller. No source module,
old run, original prompt, transport policy, or paid retry allowance is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_amem_native_v3_admission_v2 as previous
from tools import baselines_amem_v1 as original
from tools.baselines_common_v1 import read, sha, save_new, now, BaselineFatal
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body, MODEL

FROZEN_PATH = ROOT/'tools/run_five_baselines_amem_native_v3_admission_v2.py'
FROZEN_SHA = '73cfff288099e72c5e20e87bbe504f1c835ec144e81140f18719b868613d9ac1'
SOURCE = previous.NEW
NEW = previous.base.RUN/'amem_native_v4'
HISTORY = {'v1': previous.ANCESTOR, 'v2': previous.SOURCE, 'v3': SOURCE}
DEFAULT_PROOF = previous.base.RESEARCH/'amem_native_v4/terminal_admission_v4.json'
DEFAULT_QUALIFICATION = previous.base.RESEARCH/'heartbeat_20261008T1945/independent_c2_token_qualification_v1.json'
PREFLIGHT = previous.base.RESEARCH/'heartbeat_20261008T1945/amem_v4_actual_minilm_prefix_preflight_v1.json'
SCHEMA = 'amem-native-v4-terminal-admission-v1'
EXTRA = ('tools/run_five_baselines_amem_native_v4.py', 'tools/baselines_amem_v4.py')
FAILED_KEY = '4a67f80cbb747d131e51949d111adb2451f00b12bed600417bfac1feadb20965'
FAILED_RAW_SHA = '28e9ab6f4901e6f556707dd31f65b9c1997f9190afccb3c9ecbd332b07c58a87'
KIND = 'single_quoted_json_value_strings'


def _private_controller():
    data = FROZEN_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != FROZEN_SHA:
        raise ValueError('Frozen v3 controller differs')
    source = data.decode('utf-8')
    # Exhaustively declared import/schema-only substitutions. Scientific and QA
    # function bodies remain those of the hash-locked controller.
    replacements = {
        'from tools import baselines_amem_v3 as amem': ('from tools import baselines_amem_v4 as amem', 1),
        'from tools import baselines_amem_v3\n': ('from tools import baselines_amem_v4 as baselines_amem_v3\n', 3),
        'from tools.baselines_amem_v3 import decode_generation_json': ('from tools.baselines_amem_v4 import decode_generation_json', 1),
        "schema='five-baselines-amem-native-v3'": ("schema='five-baselines-amem-native-v4'", 1),
        "technical_continuation={'proof'": ("self_root=str(NEW), conversation_sources=proof['conversation_sources'], technical_continuation={'proof'", 1),
        "'schema':'five-baseline-result-native-continuation-v3'": ("'schema':'five-baseline-result-native-continuation-v4'", 1),
        "'source_manifests':{'v1':_PROOF['ancestor_manifest_sha256'],'v2':_PROOF['source_manifest_sha256']}":
            ("'source_manifests':_PROOF['history_manifests'], 'self_root':str(NEW), 'conversation_sources':_PROOF['conversation_sources']", 1),
    }
    for before, (after, count) in replacements.items():
        if source.count(before) != count:
            raise ValueError('Frozen controller substitution count differs: '+before)
        source = source.replace(before, after)
    module = types.ModuleType('_jev_amem_v4_private_controller')
    module.__file__ = str(FROZEN_PATH)
    exec(compile(source, str(FROZEN_PATH), 'exec'), module.__dict__)
    module.SOURCE, module.NEW = SOURCE, NEW
    module.DEFAULT_PROOF, module.EXTRA = DEFAULT_PROOF, EXTRA
    module.HISTORICAL_CACHE = ('Only the terminal-proof-bound full 908-call c2 v3 writer prefix is inherited; '
        'all nine successful conversation memories and 1388 QA outcomes are read-only referenced at their original v2/v3 owners. '
        'The full v1/v2/v3 ancestry is immutable. No paid replacement authority is carried forward; c5/c7 quotas are consumed.')
    return module


controller = _private_controller()
inventory = previous.inventory
check_binding = previous.check_binding


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def _active_controllers(processes, current_pid=None):
    """Permit only this interpreter and its verified venv launcher parent."""
    current_pid=os.getpid() if current_pid is None else current_pid
    by_id={int(p['ProcessId']):p for p in processes}
    pattern=re.compile(r'run_five_baselines[\w]*(?:\.py)?(?:["\s]|$)')
    run_token=re.compile(r'(?:^|["\s])run(?:["\s]|$)')
    def active(p):
        return (re.fullmatch(r'python(?:w|[0-9.]*)?\.exe',p.get('Name',''),re.I) is not None
                and pattern.search(p.get('CommandLine') or '') and run_token.search(p.get('CommandLine') or ''))
    allowed={current_pid}
    current=by_id.get(current_pid)
    if current is None:raise ValueError('Current interpreter missing from process snapshot')
    parent=by_id.get(int(current.get('ParentProcessId',-1)))
    if parent and active(current) and active(parent):
        def tail(p):
            match=re.fullmatch(r'\s*(?:"[^"]+"|\S+)(.*)',p.get('CommandLine') or '')
            return match.group(1) if match else None
        normalize=lambda p:os.path.normcase(os.path.abspath(p))
        if (tail(current) is not None and tail(current)==tail(parent)
                and re.search(r'run_five_baselines_amem_native_v4(?:\.py)?(?:["\s]|$)',tail(current))
                and normalize(parent.get('ExecutablePath') or '')==normalize(ROOT/'runtime/baseline_env_v1/Scripts/python.exe')
                and normalize(current.get('ExecutablePath') or '')==normalize(sys._base_executable)
                and datetime.fromisoformat(parent['CreatedUtc'])<=datetime.fromisoformat(current['CreatedUtc'])):
            allowed.add(int(parent['ProcessId']))
    return [p for p in processes if active(p) and int(p['ProcessId']) not in allowed]


def assert_source_stopped():
    if os.name != 'nt':
        raise ValueError('Production process admission requires the Windows host')
    command=("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine,ExecutablePath,"
        "@{Name='CreatedUtc';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}}) | ConvertTo-Json -Compress")
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True)
    if result.returncode!=0 or not result.stdout.strip():
        raise ValueError('Process enumeration failed')
    values=json.loads(result.stdout);values=values if isinstance(values,list) else [values]
    if _active_controllers(values):
        raise ValueError('A baseline controller is active or process enumeration failed')


def _qualification(path):
    report = read(path)
    if (report.get('schema') != 'amem-single-quoted-json-value-qualification-v1'
            or report.get('status') != 'passed' or report.get('scope') != 'individual_original_raw_token_representation_only'
            or report.get('kind') != KIND or report.get('retry_limit') != 0
            or report.get('conversation') != 2 or report.get('request_hash') != FAILED_KEY
            or report.get('context') != {'method':'amem','conversation':2,'phase':'build'}
            or report.get('stage') != 'amem_evolution' or report.get('raw_sha256') != FAILED_RAW_SHA):
        raise ValueError('Exact zero-retry individual token qualification required')
    check_binding(report['raw_path'], report['raw_sha256'], SOURCE/'work/api_method/api_cache')
    check_binding(report['native_call_path'], report['native_call_sha256'], SOURCE/'work/native/c2/calls')
    support = {'path':report['review_path'],'sha256':report['review_sha256']}
    check_binding(support['path'], support['sha256'], ROOT)
    evidence = read(support['path'])
    # This is an earlier scoped evidence report, explicitly not a terminal launch
    # authorization. Its own contract is checked separately from this proof.
    if (evidence.get('schema') != 'amem-native-v3-independent-c2-terminal-failure-review-v1'
            or evidence.get('status') != 'review_complete_known_failure_no_recovery_started'
            or evidence.get('unknown_or_inflight_requests') != []
            or evidence.get('source_inventory_unchanged') is not True
            or evidence.get('new_api_requests') != 0 or evidence.get('credential_reads') != 0
            or evidence.get('run_files_modified') != 0 or evidence.get('launch_admitted') is not False):
        raise ValueError('Qualified scoped supporting evidence differs')
    if (evidence['source_manifest_sha256']!=sha(SOURCE/'run_manifest.json')
            or evidence['status_sha256']!=sha(SOURCE/'status.json')
            or evidence['outcomes_sha256']!=sha(SOURCE/'rebuild_firstpass_outcome.json')
            or Path(evidence['source_run']).resolve()!=SOURCE.resolve()):
        raise ValueError('Scoped source manifest/terminal evidence changed')
    failed=evidence['c2_failure']
    if any(failed[k] != report[k] for k in ('conversation','stage','request_hash','raw_path','raw_sha256','native_call_path','native_call_sha256')):
        raise ValueError('Qualification/supporting failure binding differs')
    from tools import baselines_amem_v4 as adapter
    parsed,journal=adapter.decode_generation_json(read(report['raw_path']),raw_artifact_sha256=report['raw_sha256'])
    if (journal['original_text_utf8_sha256']!=report['original_text_sha256']
            or journal['derived_text_utf8_sha256']!=report['derived_text_sha256']
            or original._sha(parsed)!=report['derived_parsed_sha256'] or len(journal['changes'])!=14
            or journal.get('inverse_verified') is not True or journal.get('literal_values_unchanged') is not True):
        raise ValueError('Parser derivation differs from independent token qualification')
    return report


def _call_entry(path):
    call = read(path); request = call['request']
    signature={'provider':'gemini','model':MODEL,'operation':'generateContent',
        'body':generation_body(request['prompt'],system=request['system'],json_output=True,max_output=16384)}
    key=digest(signature); raw_path=SOURCE/'work/api_method/api_cache'/(key+'.json')
    if (request['json_output'] is not True or request['max_output'] != 16384
            or request['stage'] not in ('amem_analysis','amem_evolution')
            or original._sha(request) != call['request_sha256'] or read(raw_path)['request'] != signature):
        raise ValueError('Original c2 native request/signature differs')
    return {'mode':'inherit','request_hash':key,'signature':signature,
        'context':{'method':'amem','conversation':2,'phase':'build'},'stage':request['stage'],
        'raw_path':str(raw_path.resolve()),'raw_sha256':sha(raw_path),
        'native_call_path':str(path.resolve()),'native_call_sha256':sha(path)}


def _audit_source_native(old_proof):
    """Complete source native/consumer audit, with exactly one admitted failure."""
    from tools import baselines_amem_v3 as old_adapter, baselines_amem_v4 as adapter
    events=[{k:v for k,v in json.loads(line).items() if k!='recorded_utc'}
        for line in (SOURCE/'work/api_method/logical_events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    event_counts=Counter(original._sha(e) for e in events)
    by_hash={original._sha(e):e for e in events}; consumed=Counter(); raw_keys=set(); parser_keys=set(); calls=[]
    questions=previous.base.questions()
    for ci in (1,2,4,5,7,9):
        native=SOURCE/'work/native'/f'c{ci}'; out=SOURCE/'results'/f'c{ci}'
        for path in sorted((native/'calls').glob('*.json')):
            call=read(path); req=call['request']; stage=req['stage']
            sig={'provider':'gemini','model':MODEL,'operation':'generateContent',
                'body':generation_body(req['prompt'],system=req['system'],json_output=True,max_output=16384)}
            key=digest(sig); rp=SOURCE/'work/api_method/api_cache'/(key+'.json'); raw=read(rp); raw_keys.add(key)
            if raw['request']!=sig or call['request_sha256']!=original._sha(req):raise ValueError('Source native raw/request differs')
            if call['status']!='success':
                if ci!=2 or path.name!='note_000454_amem_evolution.json' or key!=FAILED_KEY or sha(rp)!=FAILED_RAW_SHA:
                    raise ValueError('Unqualified failed source call')
                parsed,journal=adapter.decode_generation_json(raw,raw_artifact_sha256=sha(rp))
                if (SOURCE/'work/api_method/parser_lineage'/(key+'.json')).exists():raise ValueError('Failed raw has unexpected accepted parser journal')
                candidates=[e for e in events if e['request_hash']==key]
                if len(candidates)!=1:raise ValueError('Failed source call consumer count differs')
                event=candidates[0]; lineage=old_adapter._v2._transport_lineage(types.SimpleNamespace(events=[event]))
            else:
                parsed,journal=old_adapter.decode_generation_json(raw,raw_artifact_sha256=sha(rp)); parser_keys.add(key)
                if (read(SOURCE/'work/api_method/parser_lineage'/(key+'.json'))!=journal
                        or parsed!=call['raw_response'] or original._sha(parsed)!=call['raw_response_payload_sha256']):
                    raise ValueError('Source native parsed raw/journal differs')
                lineage=call['transport_lineage']; event=by_hash.get(lineage.get('event_sha256'))
                if event is None or old_adapter._v2._transport_lineage(types.SimpleNamespace(events=[event]))!=lineage:
                    raise ValueError('Source native logical consumer differs')
            context={'method':'amem','conversation':ci,'phase':'retrieval' if stage=='amem_query_keywords' else 'build'}
            if stage=='amem_query_keywords':
                qid=event['context'].get('query_id')
                if qid not in {q['query_id'] for q in questions if q['conversation']==ci}:raise ValueError('Foreign source query')
                context['query_id']=qid
            if (event['context']!=context or event['stage']!=stage or lineage['request_hash']!=key
                    or lineage['cache_sha256']!=sha(rp) or Path(lineage['cache_path']).resolve()!=rp.resolve()):
                raise ValueError('Source consumer context/stage/raw differs')
            consumed[original._sha(event)]+=1
            c=adapter._Controller(None,native)
            if stage=='amem_evolution':
                c.memory_count=int(path.name.split('_')[1])-1
                text=req['prompt'].split('The nearest neighbors memories:',1)[-1].split('Based on this information',1)[0]
                c.neighbor_indices=[int(x) for x in re.findall(r'memory index:(\d+)',text)]
            value,normalization=c._derive(parsed,req['response_format']['json_schema']['schema'],stage)
            if call['status']=='success' and (value!=call['parsed_response'] or normalization!=call['normalization']):
                raise ValueError('Source native normalization differs')
            calls.append({'path':str(path),'sha256':sha(path),'request_hash':key,'raw_sha256':sha(rp),'status':call['status']})
        if ci==2:
            state=read(native/'checkpoint.json')
            if state['completed_notes']!=453 or (out/'build.json').exists() or (out/'retrieval').exists():raise ValueError('Partial c2 unexpectedly complete')
            continue
        state=read(native/'state.json'); build=read(out/'build.json')
        if (not state.get('build_complete') or build['status']!='complete'
                or state['memory_sha256']!=original._sha({k:state[k] for k in ('notes','provenance','corpus','evo_cnt')})):
            raise ValueError('Source complete memory integrity differs')
        if len(build['memory'])!=len(state['export_paths']):raise ValueError('Source exports incomplete')
        for record,relative in zip(build['memory'],state['export_paths']):
            p=check_binding(record['path'],record['sha256'],out/'memory')
            if sha(native/relative)!=sha(p):raise ValueError('Source permanent/native export differs')
        updates=[json.loads(line) for line in (native/'memory_updates.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        notes={}
        for row in updates:
            for note in row['changed_notes_after_evolution']:notes[note['id']]=note
        if ([r['ordinal'] for r in updates]!=list(range(1,state['completed_notes']+1)) or list(notes.values())!=state['notes']
                or [r['source'] for r in updates]!=state['provenance'] or updates[-1]['evo_cnt']!=state['evo_cnt']):
            raise ValueError('Source evolution journal/final state differs')
        for q in (q for q in questions if q['conversation']==ci):
            ret=read(out/'retrieval'/(q['query_id']+'.json')); key=original._sha({'question':q['question'],'memory_sha256':state['memory_sha256']})
            if ret['status']!='complete' or ret['value']!=read(native/'retrieval'/(key+'.json')):raise ValueError('Source retrieval export differs')
    if (consumed!=event_counts or {p.stem for p in (SOURCE/'work/api_method/api_cache').glob('*.json')}!=raw_keys
            or {p.stem for p in (SOURCE/'work/api_method/parser_lineage').glob('*.json')}!=parser_keys):
        raise ValueError('Source method raw/parser/native/consumer inventories differ')
    return {'status':'passed','calls':calls}


def _collect(qualification_path):
    assert_source_stopped()
    before={name:inventory(folder) for name,folder in HISTORY.items()}
    m,old_proof=previous.manifest(return_proof=True)
    binding=m['technical_continuation']['proof']
    if m['questions']!=previous.base.questions():raise ValueError('Original fixed questions differ')
    if read(SOURCE/'status.json')['stage']!='needs_review':raise ValueError('Source is not terminal needs_review')
    outcomes=read(SOURCE/'rebuild_firstpass_outcome.json')['outcomes']; by_conv={r['conversation']:r for r in outcomes}
    if len(outcomes)!=6 or set(by_conv)!={1,2,4,5,7,9} or {c for c,r in by_conv.items() if r['status']!='terminal'}!={2}:
        raise ValueError('Expected exactly c2 terminal build failure')
    for p in ('qa_recovery_plan.json','qa_recovery_outcome.json','results/recovery','results/summary.json','results/firstpass_sources.json'):
        if (SOURCE/p).exists():raise ValueError('Earlier full-cohort recovery/summary already exists')
    qualification=_qualification(qualification_path)
    support=read(qualification['review_path'])
    if original._sha(before['v3'])!=support['source_inventory_sha256'] or len(before['v3'])!=support['source_inventory_files']:
        raise ValueError('Independent terminal review source inventory differs')
    raw_audit=types.FunctionType(previous.audit_new_requests.__code__,
        {**previous.audit_new_requests.__globals__,'_PROOF':old_proof},'source_raw_audit')()
    if raw_audit['status']!='passed' or raw_audit['errors']:raise ValueError('Source raw producer/consumer audit failed')
    native_audit=_audit_source_native(old_proof)
    prefix=[_call_entry(p) for p in sorted((SOURCE/'work/native/c2/calls').glob('*.json'))]
    expected_names=[f'note_{n:06d}_amem_{stage}.json' for n in range(1,455) for stage in ('analysis','evolution')]
    if [Path(e['native_call_path']).name for e in prefix]!=expected_names or len(prefix)!=908:
        raise ValueError('Full ordered 908-call source prefix required')
    if any(read(e['native_call_path'])['status']!=('failed' if i==907 else 'success') for i,e in enumerate(prefix)):
        raise ValueError('Exactly the final source call must be failed')
    last=prefix[-1]
    if any(last[k]!=qualification[k] for k in ('request_hash','raw_path','raw_sha256','native_call_path','native_call_sha256','context','stage')):
        raise ValueError('Prefix tail and individual qualification differ')
    last['failure_certificate']={**pointer(qualification_path),'kind':KIND,'retry_limit':0,'status':'passed'}
    preflight=read(PREFLIGHT)
    if (preflight.get('status')!='passed' or preflight.get('conversation')!=2 or preflight.get('replayed_calls')!=908
            or preflight.get('previous_success_calls_unchanged')!=907 or preflight.get('new_model_calls')!=0
            or preflight.get('network_or_vault_attempts')!=[] or preflight.get('credential_reads')!=0
            or preflight.get('copied_partial_state') is not False or preflight.get('temporary_work_deleted') is not True
            or preflight.get('original_neighbors_recomputed') is not True or preflight.get('all_full_prompts_and_signatures_exact') is not True
            or preflight.get('source_native_inventory_unchanged') is not True or preflight['boundary454']['completed_notes']!=454
            or preflight['stop']['reason']!='first_unseen_request' or preflight['stop']['call_ordinal']!=909):
        raise ValueError('Real MiniLM full prefix author preflight is incomplete')
    for p,h in preflight['code_sha256'].items():check_binding(p,h,ROOT)
    for binding_ in (preflight['audit_program'],preflight['source_checkpoint'],preflight['source_dataset']):
        check_binding(binding_['path'],binding_['sha256'],ROOT)
    if (len(preflight['ordered_original_requests'])!=908
            or [r['ordinal'] for r in preflight['notes1_to453_exact_journals']]!=list(range(1,454))):
        raise ValueError('Actual native prefix evidence coverage differs')
    for ordinal,(entry,row) in enumerate(zip(prefix,preflight['ordered_original_requests']),1):
        if (row['ordinal']!=ordinal or row['request_hash']!=entry['request_hash'] or row['stage']!=entry['stage']
                or any(row['raw'][k]!=entry['raw_'+k] for k in ('path','sha256'))
                or any(row['native_call'][k]!=entry['native_call_'+k] for k in ('path','sha256'))):
            raise ValueError('Actual MiniLM preflight source request binding differs')
    routes={}; decisions={}; qa=[]
    for ci in range(10):
        owner=NEW if ci==2 else previous.SOURCE if old_proof['conversations'][str(ci)]['mode']=='inherit' else SOURCE
        routes[str(ci)]=str(owner)
        decisions[str(ci)]={'mode':'rebuild' if ci==2 else 'inherit','owner':str(owner),'prefix':prefix if ci==2 else []}
        if ci==2:continue
        build=read(owner/'results'/f'c{ci}'/'build.json')
        if build['status']!='complete':raise ValueError('Inherited native build incomplete')
        for q in (q for q in m['questions'] if q['conversation']==ci):
            leaf=owner/'results/firstpass'/q['query_id'];ret=owner/'results'/f'c{ci}'/'retrieval'/(q['query_id']+'.json')
            row=read(leaf/'outcome.json');reader,judge=previous.replay_leaf(q,leaf,owner,owner,leaf,ret)
            if (row['query_id']!=q['query_id'] or row['conversation']!=ci or row['method']!='amem' or row['phase']!='firstpass'
                    or row['reader']!=reader['status'] or row['judge']!=judge['status']
                    or Path(row['reader_path']).resolve()!=(leaf/'reader.json').resolve()
                    or Path(row['judge_path']).resolve()!=(leaf/'judge.json').resolve()):raise ValueError('Inherited QA outcome identity differs')
            qa.append({'query_id':q['query_id'],'conversation':ci,'owner':str(owner),'leaf':str(leaf),'retrieval_path':str(ret),
                'bindings':{str(p):sha(p) for p in (leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json',ret)}})
    if len(qa)!=1388 or len({r['query_id'] for r in qa})!=1388:raise ValueError('Exactly 1388 inherited QA slots required')
    after={name:inventory(folder) for name,folder in HISTORY.items()}
    if before!=after:raise ValueError('Historical evidence changed during terminal admission')
    return {'schema':SCHEMA,'status':'passed','source_run':str(SOURCE),'self_root':str(NEW),
        'source_manifest_sha256':sha(SOURCE/'run_manifest.json'),'ancestor_manifest_sha256':sha(HISTORY['v1']/'run_manifest.json'),
        'history_manifests':{k:sha(v/'run_manifest.json') for k,v in HISTORY.items()},
        'history_roots':{k:str(v) for k,v in HISTORY.items()},'history_inventories':before,
        'prior_terminal_proof':binding,'failure_qualification':pointer(qualification_path),
        'actual_minilm_prefix_preflight':pointer(PREFLIGHT),
        'source_terminal':{'process_stopped':True,'status_sha256':sha(SOURCE/'status.json'),'outcomes_sha256':sha(SOURCE/'rebuild_firstpass_outcome.json')},
        'conversations':decisions,'conversation_sources':routes,'source_raw_audit':raw_audit,'source_native_audit':native_audit,
        'inherited_qa':qa,'paid_replacement_permissions':[],
        'consumed_ancestor_replacements':{p.name:sha(p) for p in sorted((SOURCE/'work/api_method/replacements').glob('*.json'))},
        'model_calls_during_admission':0}


def audit(qualification_path=DEFAULT_QUALIFICATION, destination=DEFAULT_PROOF):
    if Path(destination).exists():raise FileExistsError('Terminal proof is immutable')
    proof=_collect(qualification_path)
    save_new(destination,proof)
    return {'status':'passed','proof':pointer(destination),'model_calls':0}


def validate_proof(path):
    proof=read(path)
    if proof.get('schema')!=SCHEMA or proof.get('status')!='passed':raise ValueError('Passed v4 terminal admission required')
    q=proof['failure_qualification'];check_binding(q['path'],q['sha256'],ROOT)
    actual=_collect(q['path'])
    if actual!=proof:raise ValueError('Terminal v4 proof differs from complete live evidence replay')
    return proof


def source_for(q):
    owner=Path(controller._PROOF['conversation_sources'][str(q['conversation'])])
    return owner,owner/'results/firstpass'/q['query_id'],owner/'results'/f"c{q['conversation']}"/'retrieval'/(q['query_id']+'.json')


class NativeAPI(controller.NativeAPI):
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        if self.is_build:
            prefix=controller._PROOF['conversations'][str(self.context['conversation'])]['prefix']
            if self.position<len(prefix) and prefix[self.position]['mode']!='inherit':
                raise BaselineFatal('No paid writer replacement is permitted in v4')
            key=digest({'provider':provider,'model':model,'operation':operation,'body':body})
            if self.position>=len(prefix) and any((p/'work/api_method/api_cache'/(key+'.json')).exists() for p in HISTORY.values()):
                raise BaselineFatal('Historical writer request outside complete admitted prefix cannot be redrawn')
        return super().request(provider,model,body,operation=operation,stage=stage,cache=cache)


def usage_union():
    views={};physical=[]
    for name,folder in {**HISTORY,'v4':NEW}.items():
        rows=[]
        for path in (folder/'work').glob('api_*/api_calls.jsonl'):
            rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        paid=[r for r in rows if not r.get('cache_hit')];physical.extend(paid)
        views[name]={'physical_attempts':len(paid),'cache_hits':len(rows)-len(paid)}
    return {'sources':views,'physical_attempts':len(physical),'status_counts':dict(Counter(str(r.get('status')) for r in physical)),
        'usage_unknown_or_missing':sum(bool(r.get('usage_unknown')) or not r.get('usage') for r in physical),
        'known_partial_tokens':{k:sum(r.get('usage',{}).get(k,0) for r in physical) for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')},
        'cost_usd':None,'cost_note':'Each of four source ledgers counted once; lineage copies not billed. Unknown usage is not zero.'}


controller.validate_proof=validate_proof
controller.source_for=source_for
controller.NativeAPI=NativeAPI
controller.usage_union=usage_union
_private_manifest=controller.manifest


def prepare(review,proof_path=DEFAULT_PROOF):
    return controller.prepare(review,proof_path)


def manifest(method='amem',*,return_proof=False):
    if method!='amem':raise ValueError('This continuation entry only handles A-Mem')
    m,proof=_private_manifest(return_proof=True)
    if m.get('self_root')!=str(NEW) or m.get('conversation_sources')!=proof['conversation_sources']:
        raise ValueError('Manifest final namespace/conversation ownership differs')
    return (m,proof) if return_proof else m


controller.manifest=manifest


def run():
    return controller.run()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['audit','prepare','check','run'])
    parser.add_argument('--method',choices=['amem'],default='amem')
    parser.add_argument('--review',type=Path)
    parser.add_argument('--proof',type=Path,default=DEFAULT_PROOF)
    parser.add_argument('--qualification',type=Path,default=DEFAULT_QUALIFICATION)
    args=parser.parse_args()
    if args.command=='audit':result=audit(args.qualification,args.proof)
    elif args.command=='prepare':result=prepare(args.review,args.proof)
    elif args.command=='check':result={'status':'passed','manifest':manifest()}
    else:
        try:result=run()
        except FileExistsError:raise
        except BaseException as exc:
            if NEW.exists():previous.base.status(NEW,'needs_review',error=previous.base.error_record(exc))
            raise
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
