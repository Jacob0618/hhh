"""Versioned A-Mem continuation: immutable sources, exact raw prefixes, union QA.

Importing this module performs no I/O to credentials or providers. Production
preparation requires an independently reviewed terminal-source admission proof.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_gemini25_v1 as base
from tools import run_five_baselines_format_v2 as v2
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, read, sha, save_new, now, append
from jev_mem.sdk20_cached_api import digest, _publish_bytes
from jev_mem.gemini25_ablation_provider_v1 import generation_body, MODEL

SOURCE = base.RUN / 'amem_format_v2'
ANCESTOR = base.RUN / 'amem'
NEW = base.RUN / 'amem_native_v3'
DEFAULT_PROOF = base.RESEARCH / 'amem_native_v3/terminal_admission.json'
SCHEMA = 'amem-native-continuation-admission-v3'
EXTRA = ('tools/run_five_baselines_amem_native_v3.py', 'tools/baselines_amem_v3.py',
         'tools/audit_amem_terminal_admission_v3.py')
KINDS = {'literal_tab_in_json_string', 'native_open_string_action', 'known_complete_invalid_json', 'known_local_checkpoint_io'}
HISTORICAL_CACHE = ('Only terminal-proof-admitted full v2 writer prefixes, including their forty original v1 ancestor requests; '
                    'successful v2 conversations and QA are read-only referenced; all other historical request reuse is forbidden.')
_paths, _adapter, _manifest = base.paths, base.adapter, base.manifest
_PROOF = None


def inventory(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def bound_path(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise ValueError('Missing or foreign proof-bound artifact: ' + str(path))
    return path


def check_binding(path, expected, folder):
    p = bound_path(path, folder)
    if sha(p) != expected:
        raise ValueError('Artifact SHA differs: ' + str(p))
    return p


def assert_source_stopped():
    if os.name != 'nt':
        raise ValueError('Production terminal-process validation requires this Windows host')
    command = ("$p = Get-CimInstance Win32_Process | Where-Object { "
               "$_.Name -match '^python(w)?\\.exe$' -and "
               "$_.CommandLine -match 'run_five_baselines_format_v2\\.py.*\\brun\\b.*--method[ =]+amem\\b' }; "
               "if ($p) { exit 23 }")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True)
    if result.returncode != 0:
        raise ValueError('Source controller is active or process validation failed')


def _certificate(entry, certificate=None):
    cert = certificate if certificate is not None else entry.get('failure_certificate')
    if not isinstance(cert, dict) or cert.get('status') != 'passed' or cert.get('kind') not in KINDS:
        raise ValueError('Missing reviewed failure certificate')
    check_binding(cert['review_path'], cert['review_sha256'], ROOT)
    if cert.get('original_raw_sha256') != entry['raw_sha256']:
        raise ValueError('Failure certificate binds a different original response')
    replacement = entry.get('mode') == 'replace_once'
    if cert.get('retry_limit') != (1 if replacement else 0):
        raise ValueError('Failure certificate retry allowance differs')
    if replacement != (cert['kind'] == 'known_complete_invalid_json'):
        raise ValueError('Representation/native-contract correction cannot request a replacement')
    return cert


def validate_proof(path):
    proof = read(path)
    if proof.get('schema') != SCHEMA or proof.get('status') != 'passed':
        raise ValueError('Passed terminal admission proof required')
    if Path(proof['source_run']).resolve() != SOURCE.resolve() or Path(proof['ancestor_run']).resolve() != ANCESTOR.resolve():
        raise ValueError('Source/ancestor namespace differs')
    assert_source_stopped()
    terminal = proof['source_terminal']
    if terminal.get('process_stopped') is not True:
        raise ValueError('Source is not terminal')
    check_binding(SOURCE/'status.json', terminal['status_sha256'], SOURCE)
    check_binding(SOURCE/'firstpass_outcome.json', terminal['outcomes_sha256'], SOURCE)
    if read(SOURCE/'status.json').get('stage') != 'needs_review':
        raise ValueError('Unexpected source method state')
    if (SOURCE/'qa_recovery_plan.json').exists() or (SOURCE/'results/recovery').exists():
        raise ValueError('An earlier QA recovery round already exists')
    for folder, name in ((SOURCE, 'source'), (ANCESTOR, 'ancestor')):
        check_binding(folder/'run_manifest.json', proof[name+'_manifest_sha256'], folder)
        if inventory(folder) != proof[name+'_inventory']:
            raise ValueError('Frozen '+name+' inventory differs')
        manifest = read(folder/'run_manifest.json')
        for p, expected in {**manifest['code_sha256'], **manifest['upstream_sha256']}.items():
            check_binding(ROOT/p, expected, ROOT)
    original = proof['original_prefix_proof']
    check_binding(original['path'], original['sha256'], ROOT)
    if Path(original['path']).resolve() != v2.PROOF.resolve():
        raise ValueError('Original v1 prefix proof differs')
    qualification_binding=proof['failure_qualification']
    qualification_path=check_binding(qualification_binding['path'],qualification_binding['sha256'],ROOT)
    qualification=read(qualification_path)
    if qualification.get('status')!='passed':raise ValueError('Independent failure qualification is not passed')
    for record in qualification['certificates'].values():
        if record.get('status')!='passed':raise ValueError('Unpassed nested failure qualification')
        nested=check_binding(record['review_path'],record['review_sha256'],ROOT)
        if read(nested).get('status')!='passed':raise ValueError('Nested independent failure review is not passed')
    source_audit = v2.audit_requests(SOURCE)
    if source_audit['status'] != 'passed' or source_audit['errors']:
        raise ValueError('Source producer/consumer/ancestor-lineage audit failed')
    m = read(SOURCE/'run_manifest.json')
    if m['questions'] != base.questions():
        raise ValueError('Fixed 1540 question identities changed')
    outcomes = read(SOURCE/'firstpass_outcome.json')['outcomes']
    if len(outcomes) != 10 or {x['conversation'] for x in outcomes} != set(range(10)):
        raise ValueError('All ten source conversation outcomes required')
    by_conv = {x['conversation']: x for x in outcomes}
    if set(proof['conversations']) != {str(i) for i in range(10)}:
        raise ValueError('All ten conversation modes required')
    for ci in range(10):
        decision, outcome = proof['conversations'][str(ci)], by_conv[ci]
        prefix = decision['prefix']
        out = SOURCE/'results'/f'c{ci}'
        if decision['mode'] == 'inherit':
            if prefix or outcome['status'] != 'terminal' or read(out/'build.json')['status'] != 'complete':
                raise ValueError('Inherited conversation is not a complete source result')
            expected = {q['query_id'] for q in m['questions'] if q['conversation'] == ci}
            rows = outcome['results']
            if len(rows) != len(expected) or {r['query_id'] for r in rows} != expected:
                raise ValueError('Inherited firstpass identities are incomplete')
            for row in rows:
                if row.get('status') == 'retrieval_failed' or row.get('conversation') != ci:
                    raise ValueError('Incomplete inherited retrieval')
                ret = read(out/'retrieval'/(row['query_id']+'.json'))
                if ret['status'] != 'complete' or not isinstance(ret['value']['context'], str):
                    raise ValueError('Invalid inherited context')
                for kind in ('reader', 'judge'):
                    p = bound_path(row[kind+'_path'], SOURCE/'results/firstpass'/row['query_id'])
                    if read(p)['status'] != row[kind]:
                        raise ValueError('Inherited stage/outcome differs')
        elif decision['mode'] == 'rebuild':
            if outcome['status'] != 'needs_review' or (out/'build.json').exists() or not prefix:
                raise ValueError('Only admitted failed builds can be rebuilt')
            for ordinal, entry in enumerate(prefix):
                if entry.get('mode') not in ('inherit', 'replace_once'):
                    raise ValueError('Unknown prefix mode')
                if entry.get('context') != {'method':'amem','conversation':ci,'phase':'build'}:
                    raise ValueError('Prefix context differs')
                if entry.get('stage') not in ('amem_analysis', 'amem_evolution'):
                    raise ValueError('Non-writer call in prefix')
                p = check_binding(entry['raw_path'], entry['raw_sha256'], SOURCE/'work/api_method/api_cache')
                raw = read(p)
                if raw['request'] != entry['signature'] or digest(raw['request']) != entry['request_hash'] or p.stem != entry['request_hash']:
                    raise ValueError('Prefix full request signature differs')
                native = check_binding(entry['native_call_path'], entry['native_call_sha256'], SOURCE/'work/native'/f'c{ci}'/'calls')
                call = read(native)
                req = call['request']
                signature = {'provider':'gemini','model':MODEL,'operation':'generateContent',
                             'body':generation_body(req['prompt'],system=req['system'],json_output=True,max_output=16384)}
                if signature != entry['signature'] or req['stage'] != entry['stage']:
                    raise ValueError('Native call/full raw signature differs')
                if ordinal != len(prefix)-1 and (call['status'] != 'success' or entry['mode'] != 'inherit' or 'failure_certificate' in entry):
                    raise ValueError('Only terminal failed call may be corrected/replaced')
                if ordinal == len(prefix)-1:
                    cert = decision.get('failure_certificate',entry.get('failure_certificate'))
                    is_io = isinstance(cert,dict) and cert.get('kind')=='known_local_checkpoint_io'
                    if call['status'] != ('success' if is_io else 'failed'):
                        raise ValueError('Terminal prefix call is not the failed call')
                    _certificate(entry,cert)
                    qualified=qualification['certificates'][str(ci)]
                    if (Path(cert['review_path']).resolve()!=qualification_path.resolve() or cert['review_sha256']!=qualification_binding['sha256']
                            or any(cert[k]!=qualified[k] for k in ('kind','retry_limit','original_raw_sha256'))):
                        raise ValueError('Prefix correction differs from the exact independent qualification')
            names = {Path(e['native_call_path']).name for e in prefix}
            actual = {p.name for p in (SOURCE/'work/native'/f'c{ci}'/'calls').glob('*.json')}
            if names != actual or len(names) != len(prefix):
                raise ValueError('Prefix must cover all original native calls exactly once')
            if [Path(e['native_call_path']).name for e in prefix] != sorted(names, key=lambda n:(n.split('_amem_')[0], 0 if n.endswith('_analysis.json') else 1)):
                raise ValueError('Prefix is not in original note/stage order')
        else:
            raise ValueError('Unknown conversation mode')
    if set(qualification['certificates'])!={ci for ci,rec in proof['conversations'].items() if rec['mode']=='rebuild'}:
        raise ValueError('Qualified failure cohort differs')
    return proof


class NativeAPI(BaselineAPI):
    def __init__(self, run_dir, *, cache_roots=(), **kwargs):
        if cache_roots:
            raise ValueError('Unrestricted historical roots forbidden')
        self.position = 0
        context=kwargs.get('context') or {}
        self.is_build = context.get('method') == 'amem' and context.get('phase') == 'build'
        if self.is_build:
            if os.path.normcase(os.path.abspath(run_dir)) != os.path.normcase(os.path.abspath(NEW/'work/api_method')):
                raise BaselineFatal('Build outside admitted v3 API namespace')
            if _PROOF is None or _PROOF['conversations'].get(str(context.get('conversation')),{}).get('mode') != 'rebuild':
                raise BaselineFatal('Inherited/unadmitted conversation cannot invoke build API')
        super().__init__(run_dir, cache_roots=(), **kwargs)

    def scoped(self, **context):
        return type(self)(self.run_dir, keys=self.keys,retries=self.retries,timeout=self.timeout,context={**self.context,**context})

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        signature = {'provider':provider,'model':model,'operation':operation,'body':body}
        key = digest(signature)
        entry = None
        if self.is_build:
            prefix = _PROOF['conversations'][str(self.context['conversation'])]['prefix']
            if self.position < len(prefix):
                entry = prefix[self.position]
                if signature != entry['signature'] or key != entry['request_hash'] or stage != entry['stage'] or self.context != entry['context']:
                    raise BaselineFatal('Exact original prefix reconstruction differs; no dispatch')
                source = check_binding(entry['raw_path'],entry['raw_sha256'],SOURCE/'work/api_method/api_cache')
                if entry['mode'] == 'inherit':
                    target = self.run_dir/'api_cache'/source.name
                    lineage = {'source_path':str(source),'source_sha256':entry['raw_sha256'],
                               'target_path':str(target.resolve()),'target_sha256':entry['raw_sha256'],'request_hash':key}
                    lp = self.run_dir/'cache_lineage'/source.name
                    if (self.run_dir/'request_intents'/source.name).exists():
                        raise BaselineFatal('Inherited prefix already has a local producer')
                    if lp.exists():
                        if read(lp) != lineage:
                            raise BaselineFatal('Prefix lineage changed')
                    else:
                        _publish_bytes(lp,(json.dumps(lineage,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
                    if target.exists():
                        if sha(target) != entry['raw_sha256']:
                            raise BaselineFatal('Imported prefix raw changed')
                    else:
                        _publish_bytes(target,source.read_bytes())
                else:
                    _certificate(entry)
                    # The replacement consumption record precedes common request
                    # intent/transport. Any reentry is an explicit hard stop.
                    save_new(self.run_dir/'replacements'/(key+'.json'),{
                        'schema':'amem-exact-known-format-replacement-v3','request_hash':key,
                        'signature':signature,'context':self.context,'stage':stage,'prefix_ordinal':self.position,
                        'original_raw_path':str(source),'original_raw_sha256':entry['raw_sha256'],
                        'failure_certificate':entry['failure_certificate'],'logical_replacement_limit':1,'created_utc':now()})
            else:
                # All source requests, including non-prefix/other conversations,
                # remain forbidden as new requests in the fresh build namespace.
                if any((folder/'work/api_method/api_cache'/(key+'.json')).exists() for folder in (SOURCE,ANCESTOR)):
                    raise BaselineFatal('Known source request outside admitted prefix cannot be redrawn')
        result = super().request(provider,model,body,operation=operation,stage=stage,cache=cache)
        if entry is not None:
            append(NEW/'work/prefix_consumption.jsonl',{'conversation':self.context['conversation'],
                'ordinal':self.position,'request_hash':key,'mode':entry['mode'],
                'source_raw_sha256':entry['raw_sha256'],'event':deepcopy(self.events[-1])})
            self.position += 1
        return result

    def generate(self, prompt, *, model=MODEL, system=None, json_output=False,max_output=16384,stage='unspecified'):
        if not (stage.startswith('amem_') and json_output):
            return super().generate(prompt,model=model,system=system,json_output=json_output,max_output=max_output,stage=stage)
        from tools.baselines_amem_v3 import decode_generation_json
        body = generation_body(prompt,system=system,json_output=True,max_output=max_output)
        response = self.request('gemini',model,body,stage=stage)
        event = self.events[-1]
        raw = read(event['cache_path'])
        if raw.get('response') != response or sha(event['cache_path']) != event['cache_sha256']:
            raise BaselineFatal('Parser raw/event binding differs')
        try:
            parsed,journal = decode_generation_json(raw,raw_artifact_sha256=event['cache_sha256'])
        except Exception as exc:
            raise BaselineFatal('Retained method JSON failed strict v3 parsing: '+type(exc).__name__) from exc
        path = self.run_dir/'parser_lineage'/(event['request_hash']+'.json')
        if path.exists():
            if read(path) != journal:
                raise BaselineFatal('Derived parser lineage changed')
        else:
            save_new(path,journal)
        return parsed


def prepare(review, proof_path=DEFAULT_PROOF):
    if NEW.exists():
        raise FileExistsError('Fresh amem_native_v3 namespace required')
    proof = validate_proof(proof_path)
    checked = read(review)
    old = read(SOURCE/'run_manifest.json')
    code = {**old['code_sha256'], **{p:sha(ROOT/p) for p in EXTRA}}
    if checked.get('status') != 'passed' or checked.get('proof_sha256') != sha(proof_path) or checked.get('code_sha256') != code:
        raise ValueError('Independent v3 review must bind complete code and terminal proof')
    if checked.get('upstream_sha256') != old['upstream_sha256']:
        raise ValueError('Independent review must bind pinned upstream')
    from tools import baselines_amem_v3
    manifest = deepcopy(old)
    manifest.update(schema='five-baselines-amem-native-v3',created_utc=now(),code_sha256=code,
        historical_cache=HISTORICAL_CACHE,
        specification=baselines_amem_v3.describe(),review={'path':str(Path(review).resolve()),'sha256':sha(review)},
        technical_continuation={'proof':{'path':str(Path(proof_path).resolve()),'sha256':sha(proof_path)},
            'source_run':str(SOURCE),'source_manifest_sha256':proof['source_manifest_sha256'],
            'ancestor_manifest_sha256':proof['ancestor_manifest_sha256'],
            'inherit_conversations':[int(k) for k,v in proof['conversations'].items() if v['mode']=='inherit'],
            'rebuild_conversations':[int(k) for k,v in proof['conversations'].items() if v['mode']=='rebuild']})
    save_new(NEW/'run_manifest.json',manifest)
    return {'manifest_sha256':sha(NEW/'run_manifest.json'),'model_calls':0}


def manifest(method='amem', *, return_proof=False):
    if method != 'amem':
        return _manifest(method)
    m = read(NEW/'run_manifest.json')
    if m['questions'] != base.questions():
        raise ValueError('Fixed questions differ')
    for p,h in {**m['code_sha256'],**m['upstream_sha256']}.items():
        check_binding(ROOT/p,h,ROOT)
    for binding in (m['review'],m['technical_continuation']['proof']):
        check_binding(binding['path'],binding['sha256'],ROOT)
    proof = validate_proof(m['technical_continuation']['proof']['path'])
    continuation=m['technical_continuation']
    if (continuation['source_manifest_sha256'] != proof['source_manifest_sha256']
            or continuation['ancestor_manifest_sha256'] != proof['ancestor_manifest_sha256']
            or Path(continuation['source_run']).resolve()!=SOURCE.resolve()):
        raise ValueError('Manifest/source admission mismatch')
    for mode in ('inherit','rebuild'):
        if sorted(m['technical_continuation'][mode+'_conversations']) != sorted(int(k) for k,v in proof['conversations'].items() if v['mode']==mode):
            raise ValueError('Manifest conversation routing differs from admission')
    from tools import baselines_amem_v3
    checked=read(m['review']['path']);old=read(SOURCE/'run_manifest.json')
    expected_code={**old['code_sha256'],**{p:sha(ROOT/p) for p in EXTRA}}
    if (m['code_sha256']!=expected_code or m['upstream_sha256']!=old['upstream_sha256'] or m.get('historical_cache')!=HISTORICAL_CACHE
            or m['specification']!=baselines_amem_v3.describe()
            or checked.get('status')!='passed' or checked.get('proof_sha256')!=continuation['proof']['sha256']
            or checked.get('code_sha256')!=expected_code or checked.get('upstream_sha256')!=m['upstream_sha256']):
        raise ValueError('Manifest/review/full frozen configuration differs')
    return (m,proof) if return_proof else m


def install():
    base.paths = lambda method: NEW if method=='amem' else _paths(method)
    from tools import baselines_amem_v3
    base.adapter = lambda method: baselines_amem_v3 if method=='amem' else _adapter(method)
    base.BaselineAPI = NativeAPI


def source_for(q):
    mode = _PROOF['conversations'][str(q['conversation'])]['mode']
    folder = SOURCE if mode=='inherit' else NEW
    return folder,folder/'results/firstpass'/q['query_id'],folder/'results'/('c'+str(q['conversation']))/'retrieval'/(q['query_id']+'.json')


def replay_leaf(q, leaf, leaf_owner, original_owner, original_leaf, retrieval_path):
    reader,judge = read(leaf/'reader.json'),read(leaf/'judge.json')
    ret = read(retrieval_path)
    if ret['status'] != 'complete':
        raise ValueError('QA lacks complete original retrieval')
    reader_owner = leaf_owner
    if reader.get('inherited'):
        original = read(original_leaf/'reader.json')
        if {k:v for k,v in reader.items() if k!='inherited'} != original or original['status']!='complete':
            raise ValueError('Inherited successful Reader differs')
        reader_owner = original_owner
    for stage,owner,phase in ((reader,reader_owner,'firstpass' if reader.get('inherited') else leaf.parent.name),
                              (judge,leaf_owner,leaf.parent.name)):
        for event in stage.get('events',[]):
            if event.get('context')!={'method':'amem','conversation':q['conversation'],'query_id':q['query_id'],'phase':phase}:
                raise ValueError('QA original conversation/phase ownership differs')
    for kind,stage in (('reader',reader),('judge',judge)):
        stage_leaf=original_leaf if kind=='reader' and reader.get('inherited') else leaf
        intent=stage_leaf/(kind+'.intent.json')
        if stage['status'] in ('complete','failed'):
            expected_context={'method':'amem','conversation':q['conversation'],'query_id':q['query_id'],'phase':stage_leaf.parent.name}
            if read(intent).get('context')!=expected_context:raise ValueError('QA stage intent ownership differs')
        elif intent.exists():raise ValueError('Not-run stage unexpectedly has an intent')
    if reader['status'] == 'complete':
        replay = base.RawReplay(reader['events'],reader_owner,'amem',q['query_id'])
        expected = base.evaluator.read_answer(replay,q['question'],q['reference'],
            [{'id':'native_context','text':ret['value']['context'],'observation_date':None}],reference_date=q['reference_date'])
        official,_ = base.verified_official_metric()
        expected['official_f1'] = official(expected['prediction']['answer'],q['reference'],q['category'])
        if expected != reader['value']:
            raise ValueError('Reader/F1 differs from exact original raw replay')
        if judge['status']=='complete':
            replay = base.RawReplay(judge['events'],leaf_owner,'amem',q['query_id'])
            expected_j = base.evaluator.judge_answer(replay,q['question'],q['reference'],expected['prediction']['answer'])
            if expected_j != judge['value']:
                raise ValueError('Judge differs from exact original raw replay')
        elif judge['status']!='failed' or not base.known_recoverable(judge,leaf_owner,'amem',q,ret['value'],kind='judge',reader=reader):
            raise ValueError('Unresolved Judge stage')
    elif reader['status']!='failed' or judge['status']!='not_run' or not base.known_recoverable(reader,leaf_owner,'amem',q,ret['value'],kind='reader'):
        raise ValueError('Unresolved Reader stage')
    return reader,judge


def firstpass_sources(m):
    if len(m['questions'])!=1540 or len({q['query_id'] for q in m['questions']})!=1540:
        raise ValueError('All 1540 firstpass slots must be terminal before recovery')
    rows=[]
    for q in m['questions']:
        folder,leaf,ret=source_for(q)
        row=read(leaf/'outcome.json')
        if (row['query_id']!=q['query_id'] or row['conversation']!=q['conversation'] or row['method']!='amem'
                or row['phase']!='firstpass' or Path(row['reader_path']).resolve()!=(leaf/'reader.json').resolve()
                or Path(row['judge_path']).resolve()!=(leaf/'judge.json').resolve()):
            raise ValueError('Firstpass source identity differs')
        reader,judge=replay_leaf(q,leaf,folder,folder,leaf,ret)
        if row['reader']!=reader['status'] or row['judge']!=judge['status']:raise ValueError('Firstpass outcome stage status differs')
        rows.append({'query_id':q['query_id'],'conversation':q['conversation'],'owner':str(folder),
                     'leaf':str(leaf),'retrieval_path':str(ret),'bindings':{str(p):sha(p) for p in (ret,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')},'outcome':row})
    if len(rows)!=1540 or len({r['query_id'] for r in rows})!=1540:
        raise ValueError('All 1540 firstpass slots must be terminal before recovery')
    return rows


def run():
    global _PROOF
    if (NEW/'execution_intent.json').exists():
        raise FileExistsError('Immutable execution intent forbids controller reentry')
    if (NEW/'work').exists() or (NEW/'results').exists():
        raise FileExistsError('Fresh empty continuation work/results required before execution intent')
    m,_PROOF=manifest(return_proof=True)
    install()
    save_new(NEW/'execution_intent.json',{'created_utc':now(),'pid':os.getpid(),'manifest_sha256':sha(NEW/'run_manifest.json')})
    base.status(NEW,'firstpass_continuation')
    data=base.load_data();outcomes=[]
    rebuild=m['technical_continuation']['rebuild_conversations']
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures=[pool.submit(base.conv_worker,NEW,'amem',ci,[q for q in m['questions'] if q['conversation']==ci],data[ci]) for ci in rebuild]
        for f in as_completed(futures):
            outcomes.append(f.result());base.status(NEW,'firstpass_continuation',rebuild_outcomes=outcomes)
    save_new(NEW/'rebuild_firstpass_outcome.json',{'completed_utc':now(),'outcomes':outcomes})
    if any(x['status']!='terminal' or any(r.get('status')=='retrieval_failed' for r in x['results']) for x in outcomes):
        base.status(NEW,'needs_review',reason='Continuation build/retrieval failure; no additional automatic method recovery')
        return {'status':'needs_review'}
    # Revalidate immutable source before admitting a full-cohort recovery plan.
    validate_proof(m['technical_continuation']['proof']['path'])
    sources=firstpass_sources(m)
    save_new(NEW/'results/firstpass_sources.json',sources)
    recover=[]
    for q,source in zip(m['questions'],sources):
        folder,leaf,ret=Path(source['owner']),Path(source['leaf']),Path(source['retrieval_path'])
        reader,judge=read(leaf/'reader.json'),read(leaf/'judge.json')
        if reader['status']=='complete' and judge['status']=='complete':continue
        failed=reader if reader['status']=='failed' else judge
        if not base.known_recoverable(failed,folder,'amem',q,read(ret)['value'],kind='reader' if reader['status']=='failed' else 'judge',reader=reader):
            raise ValueError('QA failure needs separate local/raw review')
        recover.append({'query_id':q['query_id'],'conversation':q['conversation'],
                        'stage':'reader' if reader['status']=='failed' else 'judge','source':source})
    save_new(NEW/'qa_recovery_plan.json',{'created_utc':now(),'round':1,'full_cohort':1540,
             'firstpass_sources_sha256':sha(NEW/'results/firstpass_sources.json'),'jobs':recover})
    def recover_conv(ci):
        def one(job):
            q=next(q for q in m['questions'] if q['query_id']==job['query_id'])
            source=job['source'];reader=read(Path(source['leaf'])/'reader.json')
            return base.qa_one(NEW,'amem',q,read(source['retrieval_path'])['value'],phase='recovery',
                               inherited_reader=reader if reader['status']=='complete' else None)
        with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(one,[r for r in recover if r['conversation']==ci]))
    recovered=[]
    with ThreadPoolExecutor(max_workers=10) as pool:
        for values in pool.map(recover_conv,range(10)):recovered.extend(values)
    save_new(NEW/'qa_recovery_outcome.json',{'completed_utc':now(),'plan_sha256':sha(NEW/'qa_recovery_plan.json'),
             'firstpass_sources_sha256':sha(NEW/'results/firstpass_sources.json'),'results':recovered})
    summary=seal(m)
    base.status(NEW,'completed_awaiting_result_review',summary_sha256=sha(NEW/'results/summary.json'))
    return summary


def audit_new_requests():
    records={e['request_hash']:e for c in _PROOF['conversations'].values() for e in c['prefix'] if e['mode']=='inherit'}
    inventory_out=[];inherited=set()
    for root in sorted((NEW/'work').glob('api_*')):
        ledger=[json.loads(x) for x in (root/'api_calls.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (root/'api_calls.jsonl').exists() else []
        events=[json.loads(x) for x in (root/'logical_events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (root/'logical_events.jsonl').exists() else []
        intents={p.stem:p for p in (root/'request_intents').glob('*.json')};lineages={p.stem:p for p in (root/'cache_lineage').glob('*.json')};raws={p.stem:p for p in (root/'api_cache').glob('*.json')}
        if set(intents)&set(lineages) or set(raws)!=set(intents)|set(lineages) or {r['request_hash'] for r in ledger}!=set(raws) or {e['request_hash'] for e in events}!=set(raws):
            raise ValueError('New API producer/consumer/raw inventory mismatch')
        calls_by_key=defaultdict(list);events_by_key=defaultdict(list)
        for row in ledger:calls_by_key[row['request_hash']].append(row)
        for event in events:events_by_key[event['request_hash']].append(event)
        for key,p in raws.items():
            raw=read(p);calls=calls_by_key[key];consumers=events_by_key[key]
            if digest(raw['request'])!=key or any(e.get('error_type') or e.get('cache_sha256')!=sha(p) or Path(e['cache_path']).resolve()!=p.resolve() for e in consumers):
                raise ValueError('New raw/event signature binding differs')
            if any(any(r.get(k)!=raw['request'][k] for k in ('provider','model','operation')) for r in calls+consumers):
                raise ValueError('Physical/logical API identity differs')
            if key in intents:
                intent=read(intents[key]);physical=[r for r in calls if not r.get('cache_hit')]
                if (intent.get('signature')!=raw['request'] or intent.get('request_hash')!=key or not physical
                        or type(physical[-1].get('status')) is not int or not 200<=physical[-1]['status']<300
                        or any(r.get('context')!=intent.get('context') or r.get('stage')!=intent.get('stage') for r in physical)
                        or any(e.get('stage')!=intent.get('stage') for e in consumers)):
                    raise ValueError('New physical producer missing')
                producers=[e for e in consumers if e.get('source')=='new_request']
                if (len(producers)!=1 or producers[0]['context']!=intent['context']
                        or [r.get('attempt') for r in physical]!=list(range(1,len(physical)+1))
                        or type(intent.get('retries')) is not int or not 0<len(physical)<=intent['retries']+1
                        or any(r.get('status') not in (429,500,502,503,504,529,'transport_error') for r in physical[:-1])
                        or any(e.get('source') not in ('new_request','existing_run_cache') for e in consumers)):
                    raise ValueError('New request attempt/consumer history differs')
                logical_calls=[physical[0]]+[r for r in calls if r.get('cache_hit')]
            else:
                e=records.get(key);lineage=read(lineages[key])
                expected={'source_path':str(Path(e['raw_path']).resolve()),'source_sha256':e['raw_sha256'],'target_path':str(p.resolve()),'target_sha256':e['raw_sha256'],'request_hash':key} if e else None
                if root.name!='api_method' or e is None or lineage!=expected or sha(p)!=e['raw_sha256'] or any(not r.get('cache_hit') for r in calls) or any(c['context']!=e['context'] or c['stage']!=e['stage'] for c in consumers):
                    raise ValueError('New inherited lineage mismatch')
                inherited.add(key)
                logical_calls=calls
            identity=lambda r:json.dumps({'context':r['context'],'stage':r['stage']},sort_keys=True)
            if Counter(map(identity,logical_calls))!=Counter(map(identity,consumers)):
                raise ValueError('Producer/cache-hit and logical consumer attribution differs')
            inventory_out.append({'api':root.name,'request_hash':key,'raw_sha256':sha(p),'origin':'inherited' if key in lineages else 'local'})
    if inherited!=set(records):raise ValueError('Not all admitted successful prefix responses were inherited')
    consumed=[json.loads(x) for x in (NEW/'work/prefix_consumption.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    replacements={}
    for ci,c in _PROOF['conversations'].items():
        rows=sorted([r for r in consumed if r['conversation']==int(ci)],key=lambda r:r['ordinal'])
        expected=[(i,e['request_hash'],e['mode'],e['raw_sha256']) for i,e in enumerate(c['prefix'])]
        if [(r['ordinal'],r['request_hash'],r['mode'],r['source_raw_sha256']) for r in rows]!=expected:raise ValueError('Prefix consumption order/count differs')
        for ordinal,entry in enumerate(c['prefix']):
            if entry['mode']=='replace_once':
                replacement=read(NEW/'work/api_method/replacements'/(entry['request_hash']+'.json'))
                expected={'schema':'amem-exact-known-format-replacement-v3','request_hash':entry['request_hash'],
                    'signature':entry['signature'],'context':entry['context'],'stage':entry['stage'],'prefix_ordinal':ordinal,
                    'original_raw_path':str(Path(entry['raw_path']).resolve()),'original_raw_sha256':entry['raw_sha256'],
                    'failure_certificate':entry['failure_certificate'],'logical_replacement_limit':1}
                if {k:v for k,v in replacement.items() if k!='created_utc'}!=expected:raise ValueError('Replacement provenance differs')
                intent=read(NEW/'work/api_method/request_intents'/(entry['request_hash']+'.json'))
                if (intent['signature']!=entry['signature'] or intent['context']!=entry['context'] or intent['stage']!=entry['stage']
                        or not replacement['created_utc']<=intent['created_utc']):raise ValueError('Replacement was not consumed before the exact local request')
                replacements[entry['request_hash']]=replacement
    if {p.stem for p in (NEW/'work/api_method/replacements').glob('*.json')}!=set(replacements):raise ValueError('Unexpected/missing replacement consumption record')
    return {'status':'passed','errors':[],'inventory':inventory_out}


def usage_union():
    views={};physical=[]
    for name,folder in (('v1',ANCESTOR),('v2',SOURCE),('v3',NEW)):
        rows=[]
        for p in (folder/'work').glob('api_*/api_calls.jsonl'):
            rows.extend(json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip())
        paid=[r for r in rows if not r.get('cache_hit')];physical.extend(paid)
        views[name]={'physical_attempts':len(paid),'cache_hits':len(rows)-len(paid)}
    return {'sources':views,'physical_attempts':len(physical),'status_counts':dict(Counter(str(r.get('status')) for r in physical)),
            'usage_unknown_or_missing':sum(bool(r.get('usage_unknown')) or not r.get('usage') for r in physical),
            'known_partial_tokens':{k:sum(r.get('usage',{}).get(k,0) for r in physical) for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')},
            'cost_usd':None,'cost_note':'Each source ledger counted once; lineage copies not billed. Unknown usage is not zero.'}


def audit_native_outputs(m):
    """Verify every new writer/query response and complete native text export."""
    from tools import baselines_amem_v3 as amem
    from tools import baselines_amem_v1 as original
    checked=[];parser_keys=set();consumed_events=Counter()
    event_path=NEW/'work/api_method/logical_events.jsonl'
    events=[{k:v for k,v in json.loads(line).items() if k!='recorded_utc'} for line in event_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    event_counts=Counter(original._sha(event) for event in events)
    events_by_sha={original._sha(event):event for event in events}
    for ci in m['technical_continuation']['rebuild_conversations']:
        native=NEW/'work/native'/f'c{ci}';out=NEW/'results'/f'c{ci}'
        state=read(native/'state.json');build=read(out/'build.json')
        if (not state.get('build_complete') or state['memory_sha256']!=original._sha({k:state[k] for k in ('notes','provenance','corpus','evo_cnt')})):
            raise ValueError('Final native memory integrity differs')
        exports=state['export_paths']
        if len(build['memory'])!=len(exports):raise ValueError('Permanent native exports incomplete')
        for record,relative in zip(build['memory'],exports):
            p=bound_path(record['path'],out/'memory')
            if sha(p)!=record['sha256'] or sha(native/relative)!=record['sha256']:raise ValueError('Native/permanent export differs')
        updates=[json.loads(x) for x in (native/'memory_updates.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        if [r['ordinal'] for r in updates]!=list(range(1,state['completed_notes']+1)):raise ValueError('Native update journal incomplete')
        notes={}
        for update in updates:
            for note in update['changed_notes_after_evolution']:notes[note['id']]=note
        if list(notes.values())!=state['notes'] or [u['source'] for u in updates]!=state['provenance'] or updates[-1]['evo_cnt']!=state['evo_cnt']:
            raise ValueError('Final native memory differs from complete evolution journal')
        for p in sorted((native/'calls').glob('*.json')):
            call=read(p);req=call['request'];stage=req['stage']
            signature={'provider':'gemini','model':MODEL,'operation':'generateContent',
                       'body':generation_body(req['prompt'],system=req['system'],json_output=True,max_output=16384)}
            key=digest(signature);raw_path=NEW/'work/api_method/api_cache'/(key+'.json');raw=read(raw_path)
            parser_keys.add(key)
            if call['status']!='success' or raw['request']!=signature or call['request_sha256']!=original._sha(req):
                raise ValueError('Native call/raw signature mismatch')
            parsed,journal=amem.decode_generation_json(raw,raw_artifact_sha256=sha(raw_path))
            if read(NEW/'work/api_method/parser_lineage'/(key+'.json'))!=journal or parsed!=call['raw_response'] or original._sha(parsed)!=call['raw_response_payload_sha256']:
                raise ValueError('Native parser lineage/raw value differs')
            lineage=call['transport_lineage']
            if (lineage.get('available') is not True or lineage.get('request_hash')!=key or lineage.get('cache_sha256')!=sha(raw_path)
                    or Path(lineage['cache_path']).resolve()!=raw_path.resolve()):raise ValueError('Native call transport lineage differs')
            event=events_by_sha.get(lineage.get('event_sha256'))
            if event is None or amem._v2._transport_lineage(SimpleNamespace(events=[event]))!=lineage:
                raise ValueError('Native transport lineage differs from original logical event')
            expected_context={'method':'amem','conversation':ci,'phase':'retrieval' if stage=='amem_query_keywords' else 'build'}
            if stage=='amem_query_keywords':
                query_id=event['context'].get('query_id')
                if query_id not in {q['query_id'] for q in m['questions'] if q['conversation']==ci}:raise ValueError('Native query consumer identity differs')
                expected_context['query_id']=query_id
            if event['stage']!=stage or event['context']!=expected_context:raise ValueError('Native consumer context/stage differs')
            consumed_events[lineage['event_sha256']]+=1
            controller=amem._Controller(None,native)
            if stage=='amem_evolution':
                ordinal=int(p.name.split('_')[1]);controller.memory_count=ordinal-1
                context=req['prompt'].split('The nearest neighbors memories:',1)[-1].split('Based on this information',1)[0]
                controller.neighbor_indices=[int(x) for x in re.findall(r'memory index:(\d+)',context)]
            value,normalization=controller._derive(parsed,req['response_format']['json_schema']['schema'],stage)
            if value!=call['parsed_response'] or normalization!=call['normalization']:raise ValueError('Native derived response differs')
            checked.append({'conversation':ci,'call':str(p.relative_to(NEW)),'call_sha256':sha(p),'request_hash':key,'raw_sha256':sha(raw_path)})
        for q in (q for q in m['questions'] if q['conversation']==ci):
            ret=read(out/'retrieval'/(q['query_id']+'.json'))
            key=original._sha({'question':q['question'],'memory_sha256':state['memory_sha256']})
            if ret['status']!='complete' or ret['value']!=read(native/'retrieval'/(key+'.json')):raise ValueError('Native/permanent retrieval differs')
    if {p.stem for p in (NEW/'work/api_method/parser_lineage').glob('*.json')}!=parser_keys:
        raise ValueError('Unexpected/missing method parser lineage')
    if {p.stem for p in (NEW/'work/api_method/api_cache').glob('*.json')}!=parser_keys:
        raise ValueError('Unaccounted method raw response without native call')
    if consumed_events!=event_counts:raise ValueError('Logical method consumers and native calls differ')
    return {'status':'passed','generation_calls':checked,'source_native_audit':'terminal admission proof and immutable source inventory'}


def seal(m):
    validate_proof(m['technical_continuation']['proof']['path'])
    raw_audit=audit_new_requests();native_audit=audit_native_outputs(m);sources=read(NEW/'results/firstpass_sources.json')
    if [r['query_id'] for r in sources]!=[q['query_id'] for q in m['questions']]:raise ValueError('Result source table differs')
    plan=read(NEW/'qa_recovery_plan.json');outcome=read(NEW/'qa_recovery_outcome.json')
    jobs=plan['jobs'];job_ids=[r['query_id'] for r in jobs]
    eligible=[]
    for q,source in zip(m['questions'],sources):
        owner,leaf,ret=source_for(q)
        if (Path(source['owner']).resolve()!=owner.resolve() or Path(source['leaf']).resolve()!=leaf.resolve()
                or Path(source['retrieval_path']).resolve()!=ret.resolve()
                or set(source['bindings'])!={str(p) for p in (ret,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')}):
            raise ValueError('Query source routing differs from admitted conversation owner')
        if read(leaf/'reader.json')['status']!='complete' or read(leaf/'judge.json')['status']!='complete':eligible.append(q['query_id'])
    recovered=outcome['results'];recovered_ids=[r['query_id'] for r in recovered]
    source_sha=sha(NEW/'results/firstpass_sources.json')
    if (plan.get('round')!=1 or plan.get('full_cohort')!=1540 or plan.get('firstpass_sources_sha256')!=source_sha
            or outcome.get('firstpass_sources_sha256')!=source_sha or outcome.get('plan_sha256')!=sha(NEW/'qa_recovery_plan.json')
            or len(job_ids)!=len(set(job_ids))
            or set(job_ids)!=set(eligible) or len(recovered_ids)!=len(set(recovered_ids)) or set(recovered_ids)!=set(job_ids)
            or {p.name for p in (NEW/'results/recovery').glob('*') if p.is_dir()}!=set(job_ids)):
        raise ValueError('Recovery must be exactly one complete eligible full-cohort round')
    by_id={s['query_id']:s for s in sources}
    if any(j['source']!=by_id[j['query_id']] or j['stage']!=('reader' if read(Path(j['source']['leaf'])/'reader.json')['status']=='failed' else 'judge') for j in jobs):raise ValueError('Recovery source/stage binding differs')
    for row in recovered:
        leaf=NEW/'results/recovery'/row['query_id']
        if (row!=read(leaf/'outcome.json') or row['phase']!='recovery' or row['method']!='amem'
                or Path(row['reader_path']).resolve()!=(leaf/'reader.json').resolve()
                or Path(row['judge_path']).resolve()!=(leaf/'judge.json').resolve()
                or row['conversation']!=by_id[row['query_id']]['conversation']
                or row['reader']!=read(leaf/'reader.json')['status'] or row['judge']!=read(leaf/'judge.json')['status']):raise ValueError('Recovery outcome binding differs')
    rows=[];firstpass=[];official,_=base.verified_official_metric()
    for q,source in zip(m['questions'],sources):
        for p,h in source['bindings'].items():check_binding(p,h,Path(source['owner']))
        original_owner,original_leaf,ret=Path(source['owner']),Path(source['leaf']),Path(source['retrieval_path'])
        for phase,out in (('firstpass',firstpass),('effective',rows)):
            recovery=NEW/'results/recovery'/q['query_id']
            leaf,owner=(recovery,NEW) if phase=='effective' and recovery.exists() else (original_leaf,original_owner)
            reader,judge=replay_leaf(q,leaf,owner,original_owner,original_leaf,ret)
            ans=reader['value']['prediction']['answer'] if reader['status']=='complete' else None
            label=base.evaluator.parse_judge_label(judge['value']['raw']) if judge['status']=='complete' else None
            out.append({**q,'answer':ans,'official_f1':official(ans,q['reference'],q['category']) if ans is not None else None,
                'custom_metrics':base.evaluator.lexical_metrics(ans,q['reference']) if ans is not None else None,
                'judge_correct':label['correct'] if label else None,'reader_status':reader['status'],'judge_status':judge['status'],
                'phase':phase,'source_owner':str(owner),'source_leaf':str(leaf),'retrieval_source':str(ret)})
    def aggregate(values):
        f1=[r['official_f1'] for r in values if r['official_f1'] is not None];labels=[r['judge_correct'] for r in values if r['judge_correct'] is not None]
        custom=[r['custom_metrics'].get('token_f1',r['custom_metrics'].get('f1')) for r in values if r['custom_metrics'] is not None]
        return {'slots':1540,'reader_valid':len(f1),'judge_valid':len(labels),'custom_f1_valid_percent':100*sum(custom)/len(custom) if custom else None,
            'official_f1_valid_percent':100*sum(f1)/len(f1) if f1 else None,'official_f1_fixed_missing_zero_percent':100*sum(f1)/1540,
            'judge_valid_percent':100*sum(labels)/len(labels) if labels else None,'judge_fixed_missing_zero_percent':100*sum(labels)/1540,'missing_reader':1540-len(f1),'missing_judge':1540-len(labels)}
    summary={'schema':'five-baseline-result-native-continuation-v3','method':'amem','audit_status':'no_invalid_artifacts',
        'firstpass':aggregate(firstpass),'effective':aggregate(rows),'usage':usage_union(),'created_utc':now(),
        'continuation_proof':m['technical_continuation']['proof'],'source_manifests':{'v1':_PROOF['ancestor_manifest_sha256'],'v2':_PROOF['source_manifest_sha256']}}
    for name,value in (('raw_request_audit.json',raw_audit),('native_output_audit.json',native_audit),('per_query.json',rows),('firstpass_per_query.json',firstpass),('summary.json',summary)):
        save_new(NEW/'results'/name,value)
    return summary


def require_reviewed_predecessor():
    """Fail closed until a separately reviewed cross-root cleanup is complete."""
    completion=read(NEW/'results/predecessor_completion.json')
    if completion.get('schema')!='amem-native-v3-predecessor-completion-v1' or completion.get('status')!='passed':
        raise ValueError('Reviewed aggregate predecessor completion required')
    if completion.get('summary_sha256')!=sha(NEW/'results/summary.json') or completion.get('cleanup_receipt_sha256')!=sha(NEW/'results/cleanup_receipt.json'):
        raise ValueError('Aggregate summary/cleanup receipt changed')
    review=completion['result_review'];check_binding(review['path'],review['sha256'],ROOT)
    value=read(review['path'])
    if value.get('status')!='passed' or value.get('summary_sha256')!=completion['summary_sha256']:
        raise ValueError('Independent aggregate result review differs')
    if read(NEW/'results/summary.json').get('audit_status')!='no_invalid_artifacts':
        raise ValueError('Aggregate result has invalid artifacts')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','run']);p.add_argument('--method',choices=base.METHODS,default='amem');p.add_argument('--review',type=Path);p.add_argument('--proof',type=Path,default=DEFAULT_PROOF)
    args=p.parse_args()
    if args.method!='amem':
        # Prior-method gate resolves the accepted aggregate; no old source cache
        # or prefix privilege is installed for later baseline methods.
        base.paths=lambda method:NEW if method=='amem' else _paths(method)
        if args.command=='prepare':
            checked=read(args.review)
            dispatcher='tools/run_five_baselines_amem_native_v3.py'
            if checked.get('code_sha256',{}).get(dispatcher)!=sha(ROOT/dispatcher):
                raise ValueError('Later-method review supplement must bind the actual v3 dispatcher')
            result=base.prepare(args.method,args.review)
        elif args.command=='check':result=base.manifest(args.method)
        else:
            require_reviewed_predecessor()
            result=base.run(args.method)
    elif args.command=='prepare':result=prepare(args.review,args.proof)
    elif args.command=='check':result={'status':'passed','manifest':manifest()}
    else:
        try:result=run()
        except FileExistsError:raise
        except BaseException as exc:
            if NEW.exists():base.status(NEW,'needs_review',error=base.error_record(exc))
            raise
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
