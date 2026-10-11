"""R07 exact native numeric technical continuation, in a fresh namespace only.

Twelve banks are referenced; four completed failed cheap banks are audited and
metadata-exported read-only; four never-started JEV banks receive their first ADD.
No scientific protocol, accepted statement, old outcome or native SDK is changed.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import materialize_write_omission_r07 as old
from tools import r07_numeric_native_support as support

TECH_SPEC = ROOT/'research/jev_autoresearch_2026-10-04/R07_numeric_compatibility_protocol_v1.json'
TECH_SHA = '5f7c9e1e1f42b094a24bfc29e849a272e8c327bfae2c89dc35ad8569e41cb2ce'
SCHEMA = 'write-omission-r07-native-numeric-continuation-v1'
SOURCE_SCHEMA = old.SOURCE_SCHEMA
CODE = tuple(dict.fromkeys((*old.CODE, 'tools/r07_numeric_native_support.py',
    'tools/r07_native_numeric_replay.py', 'tools/materialize_write_omission_numeric_r07.py',
    'tools/run_write_omission_numeric_qa_r07.py', 'tools/summarize_write_omission_numeric_r07.py')))
require, pointer, bound, sealed_read, save_new = old.require, old.pointer, old.bound, old.sealed_read, old.save_new


def spec():
    require(old.sha(TECH_SPEC) == TECH_SHA, 'Three-voted numeric protocol changed')
    value = old.read(TECH_SPEC)
    for field in ('scientific_protocol','original_selector_manifest','original_admission',
                  'original_native_manifest','original_terminal_outcome'):
        bound(value[field])
    for ref in value['evidence']: bound(ref)
    return value


def original_folder(s, ci, arm):
    return old.bank_folder(s['original_run'], ci, arm)


def native_folder(run, ci, arm):
    return old.bank_folder(run, ci, arm)


def retrieval_folder(run, ci, arm):
    require(type(ci) is int and ci in range(10) and arm in old.NEW_ARMS, 'Invalid retrieval identity')
    return Path(run).resolve()/f'retrieval/conv_{ci}/{arm}'


def api_dir(run, ci):
    return Path(run).resolve()/f'materialization/conv_{ci}/embedding_api'


def addition_rows(folder, candidates):
    paths = sorted((Path(folder)/'additions').glob('*.json'))
    require([p.name for p in paths] == [f'{i:04d}.json' for i in range(len(candidates))],
            'Complete ordered ADD records required; no orphan or partial candidate')
    rows = [{k:v for k,v in sealed_read(p).items() if k != 'content_sha256'} for p in paths]
    require([r['candidate'] for r in rows] == candidates, 'Accepted candidate sequence changed')
    dispatch = sorted((Path(folder)/'add_dispatch').glob('*.json'))
    expected = [f'{i:04d}.json' for i, r in enumerate(rows) if r['status'] == 'added']
    require([p.name for p in dispatch] == expected, 'ADD dispatch inventory differs')
    for p in dispatch:
        r = rows[int(p.stem)]; d = sealed_read(p)
        require(d['candidate'] == r['candidate'] and d['memory_id'] == r['memory_id']
            and d['native_result'] == r['native_result']
            and d['status'] == 'add_persisted_before_entity_link', 'Persisted UUID/ADD raw result differs')
    return rows


def _dead(outcome):
    from tools.recover_gemini38_session_io_v1 import pid_alive
    require(not pid_alive(outcome['pid']), 'Native controller/owner is still alive')


def numeric_runtime_pins(reports):
    pins = {}
    for report in reports:
        for path,value in report['native_code_sha256'].items():
            require(path not in pins or pins[path] == value, 'Numeric runtime changed during preparation')
            pins[path] = value
    require(bool(pins), 'Numeric runtime code pins missing')
    return pins


def prepare(run_dir):
    run = Path(run_dir).resolve(); s = spec()
    require(str(run) == s['namespace'] and not run.exists(), 'Fresh exact numeric namespace required')
    orig_m = old.manifest(s['original_run'])
    admit = old.admission(s['original_run'])
    terminal = bound(s['original_terminal_outcome'], sealed=True)
    _dead(bound(terminal['intent'], sealed=True))
    require(len(terminal['banks']) == 16 and terminal['status'] == 'failed', 'Original terminal scope changed')
    inputs = {ref['path']: ref for ref in [pointer(TECH_SPEC), *s['evidence'],
        *[s[f] for f in ('scientific_protocol','original_selector_manifest','original_admission',
                        'original_native_manifest','original_terminal_outcome')]]}
    old_banks = {}; old_audits = {}; original_inventory = {}
    for ci in range(10):
        by_arm = {a: [] for a in old.NEW_ARMS}
        for arm in old.NEW_ARMS:
            key = f'{ci}/{arm}'; folder = original_folder(s, ci, arm)
            if key in s['fixed_scope']['never_started_first_attempt_banks']:
                require(not folder.exists(), 'Previously unstarted bank unexpectedly exists')
                continue
            oref = pointer(folder/'materialization_outcome.json'); o = bound(oref, sealed=True); _dead(o)
            failed = key in s['fixed_scope']['reviewed_export_banks']
            require(o['status'] == ('failed' if failed else 'complete') and not o['warnings'],
                    'Original bank strict outcome scope differs')
            if failed:
                require(o['bank'] is None and o['error']['phase'] == 'native_delta_audit'
                    and o['error']['message'] == 'Original entity content/vector/link removed or changed',
                    'Reviewed export must be the original numeric-only gate failure')
            rows = addition_rows(folder, old.accepted(admit, ci, arm)); by_arm[arm] = rows
            native_path = folder/'native'; after = old.snapshot(native_path)
            before = bound(orig_m['source_banks'][str(ci)]['snapshot'], sealed=True)
            # Full producer closure is checked before any cache root can be used.
            dense = old.audit_added_embeddings(api_dir(s['original_run'],ci), ci, arm, rows, after)
            numeric = support.numeric_delta(orig_m['source_banks'][str(ci)]['native_dir'], before,
                                          after, rows, orig_m['source_banks'][str(ci)]['user_id'])
            inv = old.inventory(native_path); original_inventory[key] = {'native_dir':str(native_path), 'inventory': inv}
            refs = [oref, o['materialization_intent'], *[pointer(p) for p in sorted((folder/'additions').glob('*.json'))],
                    *[pointer(p) for p in sorted((folder/'add_dispatch').glob('*.json'))]]
            if not failed:
                b = bound(o['bank'], sealed=True)
                require(b['additions'] == rows and b['native_inventory'] == inv
                    and b['dense_embedding_proof'] == dense, 'Original successful bank changed')
                refs.append(o['bank'])
            for ref in refs: inputs[ref['path']] = ref
            old_banks[key] = {'outcome':oref, 'native_dir':str(native_path), 'native_inventory':inv,
                'additions_refs':[pointer(p) for p in sorted((folder/'additions').glob('*.json'))],
                'original_bank':o['bank'], 'original_write_status':o['status'],
                'initial_numeric_audit':numeric, 'initial_dense_embedding_proof':dense}
        old_audits[str(ci)] = old.audit_embedding_inventory(api_dir(s['original_run'],ci),ci,by_arm)
        directory = api_dir(s['original_run'],ci)
        for p in sorted(directory.rglob('*')):
            if p.is_file(): inputs[str(p.resolve())] = pointer(p)
    selector = bound(s['original_selector_manifest'])
    new_selector = {k:v for k,v in copy.deepcopy(selector).items() if k != 'content_sha256'}
    new_selector.update(original_selector_manifest=s['original_selector_manifest'],
        numeric_compatibility_protocol=pointer(TECH_SPEC), numeric_technical_continuation=True,
        code_sha256={p:old.sha(ROOT/p) for p in CODE})
    save_new(run/'run_manifest.json', new_selector)
    m = {k:v for k,v in copy.deepcopy(orig_m).items() if k != 'content_sha256'}
    m.update(schema=SCHEMA, created_utc=old.now(), admission=s['original_admission'],
        original_selector_manifest=s['original_selector_manifest'], original_native_manifest=s['original_native_manifest'],
        numeric_compatibility_protocol=pointer(TECH_SPEC), input_refs=list(inputs.values()),
        original_effective_banks=old_banks, original_embedding_audits=old_audits,
        original_native_inventory=original_inventory, code_sha256={p:old.sha(ROOT/p) for p in CODE},
        numeric_runtime_code_sha256=numeric_runtime_pins(
            [v['initial_numeric_audit']['numeric_trajectory'] for v in old_banks.values()]),
        execution={'materialization_concurrency':4,'capture_concurrency':8,
            'first_attempt_banks':s['fixed_scope']['never_started_first_attempt_banks'],
            'reviewed_export_banks':s['fixed_scope']['reviewed_export_banks'],
            're_add_completed_banks':False,'automatic_rebuild':False})
    return save_new(run/'materialization/run_manifest.json',m)


def manifest(run_dir, validate_admission=True):
    run = Path(run_dir).resolve(); s = spec(); m = sealed_read(run/'materialization/run_manifest.json')
    require(str(run) == s['namespace'] and m['schema'] == SCHEMA
        and m['numeric_compatibility_protocol'] == pointer(TECH_SPEC)
        and m['admission'] == s['original_admission'], 'Numeric manifest identity differs')
    original = bound(s['original_native_manifest'],sealed=True)
    for field in ('protocol','source_banks','resources','embedding_cache_pack','retrieval_manifest','native_sdk_sha256'):
        require(m[field] == original[field], 'Inherited original native/source/resource field changed')
    for path, value in m['code_sha256'].items(): require(old.sha(ROOT/path) == value, 'Numeric implementation changed')
    for path,value in m['numeric_runtime_code_sha256'].items():
        require(old.sha(path) == value, 'Pinned native numeric runtime changed')
    require(m['native_sdk_sha256'] == old.native_sdk_identity(), 'Native SDK changed')
    for ref in m['input_refs']: bound(ref)
    for info in [*m['source_banks'].values(), *m['original_native_inventory'].values()]:
        require(old.inventory(info['native_dir']) == info['inventory'], 'Immutable original native bytes changed')
    for info in m['source_banks'].values():
        bound(info['bank'],sealed=True); bound(info['snapshot'],sealed=True)
    if validate_admission:
        require(old.admission(s['original_run']) == bound(m['admission'],sealed=True), 'Scientific admission changed')
    selector = sealed_read(run/'run_manifest.json')
    require(selector['original_selector_manifest'] == s['original_selector_manifest']
        and selector['numeric_compatibility_protocol'] == m['numeric_compatibility_protocol']
        and selector['code_sha256'] == m['code_sha256'], 'Numeric selector/code lineage differs')
    return m


def _native_context(run, ci, m):
    source_api = api_dir(spec()['original_run'],ci)
    class ImportEmbeddingOnlyAPI(old.EmbeddingOnlyAPI):
        def __init__(self, run_dir, **kwargs):
            roots = kwargs.pop('cache_roots', (source_api/'api_cache',))
            require(tuple(Path(p).resolve() for p in roots) == ((source_api/'api_cache').resolve(),),
                    'Only corresponding conversation cache can be imported')
            super().__init__(run_dir, cache_roots=roots, **kwargs)
    clone = support.isolated({'EmbeddingOnlyAPI':ImportEmbeddingOnlyAPI,
        'manifest':manifest,'admission':lambda r:old.admission(spec()['original_run']),
        'validate_delta':lambda before,after,rows,user:support.numeric_delta(
            m['source_banks'][str(ci)]['native_dir'],before,after,rows,user),
        'audit_embedding':lambda directory,text,context,ledger:support.audit_import(
            directory,source_api,text,context,ledger)})
    return clone


def materialize_first_attempt(run, ci, m):
    require(ci in (4,5,6,7), 'Only never-started four conversations may dispatch')
    clone = _native_context(run,ci,m)
    return clone.materialize_bank(run,ci,'jev',m,old.admission(spec()['original_run']))


def materialize(run_dir):
    run = Path(run_dir).resolve(); m = manifest(run)
    intent = save_new(run/'materialization/controller_intent.json', {'schema':SCHEMA,'pid':os.getpid(),
        'created_utc':old.now(),'first_attempt_only':True,'conversations':[4,5,6,7], 'max_processes':4})
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(materialize_first_attempt,run,ci,m):ci for ci in (4,5,6,7)}
        for f in as_completed(futures):
            ci = futures[f]
            try: rows.append(f.result())
            except Exception as ex:
                rows.append({'conversation':ci,'arm':'jev','status':'failed',
                    'error':old.safe_error(ex,'first_attempt_process'),'namespace_retained':True})
    rows.sort(key=lambda r:r['conversation'])
    return save_new(run/'materialization/controller_outcome.json', {'schema':SCHEMA,
        'status':'complete' if len(rows)==4 and all(r['status']=='complete' for r in rows) else 'failed',
        'finished_utc':old.now(),'banks':rows,'intent':intent,'referenced_original_complete':12,
        'reviewed_completed_failed_cheap':4,'new_first_attempt_jev':4})


def _export_bank(run, ci, arm, m, rows, numeric, dense, bm25, native_path, original_outcome):
    info = m['source_banks'][str(ci)]; original = bound(info['bank'],sealed=True)
    snap = old.snapshot(native_path); facts = copy.deepcopy(original['facts'])
    for row in rows:
        if row['status'] != 'added': continue
        c,mid = row['candidate'],row['memory_id']
        facts.append({'id':mid,'text':c['statement'],'event':'ADD','user_id':info['user_id'],
            'observation_date':c['observation_date'],'embedding':snap['facts'][mid]['vector'][''],
            'input_scope':{'session_id':c['source_session_id'],
                'source_turn_ids':list(dict.fromkeys(e['turn_id'] for e in c['evidence']))},
            'verified_support':[],'support_status':'unverified_proposed_statement',
            'supplement_candidate_id':c['candidate_id'],'evidence':c['evidence'],'admission':m['admission']})
    return {'schema':old.SCHEMA,'complete':True,'conversation':ci,'arm':arm,
        'protocol_sha256':old.digest(m['protocol']),'user_id':info['user_id'],'native_dir':str(Path(native_path).resolve()),
        'facts':facts,'original_bank':info['bank'],'admission':m['admission'],'additions':rows,
        'gate':numeric,'bm25':bm25,'dense_embedding_proof':dense,'native_inventory':old.inventory(native_path),
        'numeric_compatibility_protocol':m['numeric_compatibility_protocol'],
        'original_outcome':original_outcome,'original_write_status':'failed',
        'reviewed_export_status':'numeric_trajectory_compatible','model_calls_for_review_export':0}


def effective_audit(run_dir, publish=False):
    run = Path(run_dir).resolve(); m = manifest(run); s = spec(); admit = bound(m['admission'],sealed=True)
    terminal = sealed_read(run/'materialization/controller_outcome.json'); _dead(bound(terminal['intent'],sealed=True))
    require(terminal['status'] == 'complete' and len(terminal['banks']) == 4
        and {(r['conversation'],r['arm']) for r in terminal['banks']} == {(ci,'jev') for ci in (4,5,6,7)},
        'Four first-attempt banks are not all terminal/complete')
    refs = {}; numeric_audits = {}; origins = {}; first_audits = {}; original_audits = {}
    for ci in range(10):
        original_rows = {a:[] for a in old.NEW_ARMS}
        for arm in old.NEW_ARMS:
            key = f'{ci}/{arm}'; first = key in s['fixed_scope']['never_started_first_attempt_banks']
            reviewed = key in s['fixed_scope']['reviewed_export_banks']
            folder = native_folder(run,ci,arm) if first else original_folder(s,ci,arm)
            oref = pointer(folder/'materialization_outcome.json'); outcome = bound(oref,sealed=True); _dead(outcome)
            require(outcome['status'] == ('failed' if reviewed else 'complete') and not outcome['warnings'],
                    'Effective bank outcome identity differs')
            if first:
                row = next(r for r in terminal['banks'] if r['conversation'] == ci)
                require({k:v for k,v in outcome.items() if k != 'content_sha256'} == row,
                        'New controller/per-bank outcome differs')
            rows = addition_rows(folder, old.accepted(admit,ci,arm)); native_path = folder/'native'
            intent = bound(outcome['materialization_intent'],sealed=True)
            require(intent['conversation'] == ci and intent['arm'] == arm
                and intent['admission'] == m['admission'] and intent['candidates_sha256'] == old.digest(old.accepted(admit,ci,arm))
                and intent['pid'] == outcome['pid'], 'Native owner intent/candidate binding differs')
            if not first: original_rows[arm] = rows
            after = old.snapshot(native_path); info = m['source_banks'][str(ci)]
            before = bound(info['snapshot'],sealed=True)
            directory = api_dir(run,ci) if first else api_dir(s['original_run'],ci)
            clone = _native_context(run,ci,m) if first else old
            dense = clone.audit_added_embeddings(directory,ci,arm,rows,after)
            numeric = support.numeric_delta(info['native_dir'],before,after,rows,info['user_id'])
            require(numeric['numeric_trajectory']['native_code_sha256'] == m['numeric_runtime_code_sha256'],
                    'Effective numeric audit used a different native runtime')
            bm25 = old.compare_persisted_fact_vectors([native_path],m['resources']['snapshot'],m['resources']['files_sha256'])
            require(bm25['facts_compared'] == bm25['exact_matches'] == len(after['facts']), 'Native BM25 differs')
            if reviewed:
                expected = _export_bank(run,ci,arm,m,rows,numeric,dense,bm25,native_path,oref)
                export_path = run/f'reviewed_exports/conv_{ci}/{arm}/bank.json'
                ref = save_new(export_path,expected) if publish else pointer(export_path)
                exported = bound(ref,sealed=True)
                require({k:v for k,v in exported.items() if k!='content_sha256'} == expected,
                        'Read-only reviewed bank export differs')
            else:
                ref = outcome['bank']; b = bound(ref,sealed=True)
                require(b['conversation'] == ci and b['arm'] == arm and b['complete']
                    and b['admission'] == m['admission'] and b['additions'] == rows
                    and b['native_inventory'] == old.inventory(native_path)
                    and b['dense_embedding_proof'] == dense, 'Effective bank payload/provenance changed')
                if first: require(b['gate'] == numeric, 'First-attempt exact numeric gate changed')
                expected_facts = _export_bank(run,ci,arm,m,rows,numeric,dense,bm25,native_path,oref)['facts']
                require(b['facts'] == expected_facts, 'Effective bank fact export differs from actual native ADD')
            refs[key] = ref; numeric_audits[key] = numeric
            origins[key] = {'kind':'first_attempt' if first else 'reviewed_export' if reviewed else 'original_success',
                'outcome':oref,'original_write_status':'not_run' if first else outcome['status'],
                'native_dir':str(native_path.resolve()),'native_inventory':old.inventory(native_path),
                'additions_refs':[pointer(p) for p in sorted((folder/'additions').glob('*.json'))]}
        original_audits[str(ci)] = old.audit_embedding_inventory(api_dir(s['original_run'],ci),ci,original_rows)
        require(original_audits[str(ci)] == m['original_embedding_audits'][str(ci)],
                'Original whole embedding producer/consumer ledger changed')
        if ci in (4,5,6,7):
            new_rows = bound(refs[f'{ci}/jev'],sealed=True)['additions']
            first_audits[str(ci)] = support.audit_inventory(api_dir(run,ci),api_dir(s['original_run'],ci),ci,
                                                         {'cheap':[],'jev':new_rows})
    result = {'schema':SCHEMA,'status':'passed','admission':m['admission'],'banks':refs,'new_banks':20,
        'source_original_facts':1637,'semantic_support_certified':False,'model_calls_for_audit':0,
        'numeric_compatibility_protocol':m['numeric_compatibility_protocol'],'numeric_audits':numeric_audits,
        'bank_origins':origins,'embedding_audits':{'original':original_audits,
                                               'first_attempt':first_audits},
        'original_strict_status_counts':{'complete':12,'failed':4,'not_run':4},
        'effective_status_counts':{'complete':20},'first_attempt_controller':pointer(run/'materialization/controller_outcome.json')}
    if publish: return save_new(run/'materialization/index_admission.json',result)
    return result


def _retrieval_context(run,m):
    return support.isolated({'manifest':manifest,'bank_folder':retrieval_folder,
        'admission':lambda r:bound(m['admission'],sealed=True)})


def retrieve(run_dir):
    run = Path(run_dir).resolve(); m = manifest(run)
    gate = sealed_read(run/'materialization/index_admission.json')
    require(effective_audit(run) == {k:v for k,v in gate.items() if k != 'content_sha256'},
            'Effective all-bank gate does not replay')
    for key,ref in gate['banks'].items():
        ci,arm = key.split('/'); b = bound(ref,sealed=True)
        old.copy_closed(b['native_dir'],retrieval_folder(run,int(ci),arm)/'native',b['native_inventory'])
    # Reuse the frozen controller, replacing only its submitted function by a
    # module-level picklable worker. The child then builds its isolated context.
    controller = support.isolated({'manifest':manifest,'bank_folder':retrieval_folder,
        'retrieve_conversation':retrieve_conversation})
    return controller.retrieve(run)


def retrieve_conversation(run,ci,m,gate,jobs,entries):
    return _retrieval_context(run,m).retrieve_conversation(run,ci,m,gate,jobs,entries)


def _seal_value(run):
    m = manifest(run); clone = _retrieval_context(run,m); gate = sealed_read(run/'materialization/index_admission.json')
    require(effective_audit(run) == {k:v for k,v in gate.items() if k!='content_sha256'}, 'Effective gate changed')
    original = bound(m['protocol']['frozen_source']); jobs = old.old_jobs(original)
    clone.validate_retrieval_terminal(run,m,gate,jobs)
    for key,ref in gate['banks'].items():
        ci,arm = key.split('/'); b = bound(ref,sealed=True)
        require(old.inventory(retrieval_folder(run,int(ci),arm)/'native') == b['native_inventory'],
                'QA retrieval native copy changed')
    p = m['protocol']
    return {'schema':SOURCE_SCHEMA,'status':'ready','arms':list(old.ARMS),'jobs':clone.construct_jobs(run,m,gate,jobs),
        'questions_per_arm':1540,'frozen_source':p['frozen_source'],'frozen_qa_run_manifest':p['frozen_qa_run_manifest'],
        'frozen_baseline_reuse_proof':p['frozen_baseline_reuse_proof'],'reuse_proof_qa_manifest':p['reuse_proof_qa_manifest'],
        'write_omission_protocol':p['research_protocol'],'selector_manifest':pointer(run/'run_manifest.json'),
        'numeric_manifest':pointer(run/'run_manifest.json'),'numeric_compatibility_protocol':pointer(TECH_SPEC),
        'effective_bank_gate':pointer(run/'materialization/index_admission.json'),
        'admission_manifest':m['admission'],'materialization_manifest':pointer(run/'materialization/run_manifest.json'),
        'index_admission':pointer(run/'materialization/index_admission.json'),'retrieval_outcome':pointer(run/'retrieval_outcome.json'),
        'official_metric_provenance':original['official_metric_provenance'],'source_dataset_sha256':original['source_dataset_sha256'],
        'protocol_sha256':old.digest(p),'code_sha256':m['code_sha256'],
        'selection_counts':{a:{'jobs':1540} for a in old.ARMS}}


def seal(run_dir):
    run = Path(run_dir).resolve(); value = _seal_value(run)
    return save_new(run/'selected_inputs.json',{**value,'created_utc':old.now()})


def validate_source(path, fresh=False):
    path = Path(path).resolve(); before = old.sha(path); source = sealed_read(path)
    expected = _seal_value(path.parent)
    require({k:v for k,v in source.items() if k not in ('content_sha256','created_utc')} == expected,
            'Numeric selected source, native gate or full job replay differs')
    require(old.sha(path) == before,'Selected source changed during audit')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','materialize','audit','retrieve','seal','check'))
    parser.add_argument('--run-dir',required=True,type=Path); args=parser.parse_args()
    actions = {'prepare':prepare,'materialize':materialize,'audit':lambda r:effective_audit(r,publish=True),
        'retrieve':retrieve,'seal':seal,'check':lambda r:{'valid_jobs':len(validate_source(Path(r)/'selected_inputs.json')['jobs']),
                                                     'model_calls':0}}
    print(json.dumps(actions[args.command](args.run_dir),ensure_ascii=False))


if __name__ == '__main__': main()
