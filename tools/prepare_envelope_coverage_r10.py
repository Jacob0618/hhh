"""Offline uniform R09 raw import, R10 score compilation and admission."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import envelope_coverage_r10 as core
from tools import prepare_qualifier_repair_r09 as old
from tools import score_qualifier_repair_r09 as old_dispatch
Reader, save_new, file_sha, now = old.Reader, old.save_new, old.file_sha, old.now
_canonical_ref, _inside, _body = old._canonical_ref, old._inside, old._body
RESEARCH_PROTOCOL = ROOT/'research/jev_autoresearch_2026-10-04/R10_envelope_coverage_protocol_v1.json'
SCHEMA = 'envelope-coverage-r10-selector-manifest-v1'
COMPILED_SCHEMA = 'envelope-coverage-r10-score-manifest-v1'
ADMISSION_SCHEMA = 'envelope-coverage-r10-admission-manifest-v1'
INPUT_SCHEMA = 'envelope-coverage-r10-source-input-v1'
REQUEST_SCHEMA = 'envelope-coverage-r10-request-v1'
SCORE_SCHEMA = 'envelope-coverage-r10-score-result-v1'
AUDIT_SCHEMA = 'envelope-coverage-r10-terminal-audit-v1'
SOURCE_SCHEMA = 'envelope-coverage-r10-selected-inputs-v1'
CODE = tuple(dict.fromkeys((*old.CODE,
    'jev_mem/envelope_coverage_r10.py', 'tools/prepare_envelope_coverage_r10.py',
    'tools/score_envelope_coverage_r10.py', 'tools/materialize_envelope_coverage_r10.py',
    'tools/r09_native_update_audit.py', 'tools/r09_snapshot_native_support.py',
    'tools/retrieve_qualifier_repair_compatible_r09.py',
    'tools/run_envelope_coverage_qa_r10.py', 'tools/summarize_envelope_coverage_r10.py',
    'tools/snapshot_envelope_coverage_r10.py')))


def _codes():
    return {p:file_sha(ROOT/p) for p in CODE}


def _unchanged(reader):
    for ref in reader.refs.values():
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('Immutable input changed before phase completion')


def _imports(reader):
    """Replay both original terminal producer closures once per byte-bound Reader."""
    if hasattr(reader, 'r10_imports'):
        return reader.r10_imports
    spec = reader.read(RESEARCH_PROTOCOL, sealed=False)
    sref = reader.raw_pointer(RESEARCH_PROTOCOL)
    if sref['sha256'] != core.PROTOCOL_SHA256:
        raise ValueError('Three-voted R10 scientific protocol changed')
    for pointer in spec['input_refs'].values():
        reader.raw_pointer(pointer['path'], ref=_canonical_ref(pointer))
    oldrun = Path(spec['input_refs']['R09_proposer_manifest']['path']).resolve().parent
    manifest, normative = old._initial(reader, oldrun)
    if reader.pointer(oldrun/'run_manifest.json') != _canonical_ref(spec['input_refs']['R09_proposer_manifest']):
        raise ValueError('Original proposer manifest differs')
    compiled = old._score_manifest(reader, oldrun, manifest, normative)
    if reader.pointer(oldrun/'score_manifest.json') != _canonical_ref(spec['input_refs']['R09_score_manifest']):
        raise ValueError('Original score registry differs')
    audits = {}
    for phase, refs in [('proposer', manifest['request_refs']), ('jev', compiled['request_refs'])]:
        auditpath = oldrun/'reviews'/f'{phase}_terminal_audit_v1.json'
        saved = reader.read(auditpath)
        fresh = old_dispatch.audit_replay(oldrun, phase, refs)
        without_time = lambda x:{k:v for k,v in _body(x).items() if k != 'created_utc'}
        if without_time(saved) != without_time(fresh) or fresh['status'] != 'passed':
            raise ValueError('Historical complete producer closure differs')
        for ref in fresh['input_refs']:
            reader.bound(ref, sealed=False)
        audits[phase] = reader.pointer(auditpath)
    # The old shared ledger is terminal and immutable, not a current append-only ledger.
    ledger = reader.raw_pointer(oldrun/'api_calls.jsonl')
    items = []; registry = {}
    byid = {i['query_id']:i for i in compiled['items']}
    for original_item in manifest['items']:
        qid = original_item['query_id']; source = reader.bound(original_item['source_input'])
        request = reader.bound(original_item['request'])
        result = old._result(reader, oldrun, original_item['request'], 'proposer', normative)
        response = None
        if result['raw_response'] is not None:
            response = reader.bound(result['raw_response'], sealed=False)['response']
        if response is None:
            replay = {'status':result['status'], 'reason':result['reason'], 'parsed':None,
                      'decoding':{'branch':'unavailable','visible_text':None,
                                  'decoded_slice':None,'decoding_count':0,
                                  'removed_delimiter_ranges':[]}}
        else:
            replay = core.replay_proposal(response, source['state'])
        if result['status'] == 'complete':
            if replay['status'] != 'complete' or replay['parsed'] != result['parsed']:
                raise ValueError('Old valid session changed parsed identity/noop/order')
        candidates = (core.stable_candidates(replay['parsed'], qid, source['state'])
                      if replay['parsed'] is not None else [])
        if result['status'] == 'complete' and candidates != byid[qid]['candidates']:
            raise ValueError('Old valid canonical candidate identity differs')
        item = {'query_id':qid,'conversation':source['conversation'],
                'session_id':source['session_id'], 'old_source_input':original_item['source_input'],
                'original_proposer_request':original_item['request'],
                'original_proposer_result':result, 'replay':replay,
                'candidates':candidates}
        items.append(item)
    if len(items) != 272 or sum(i['original_proposer_result']['status']=='complete' for i in items) != 54:
        raise ValueError('Uniform272/original54 control differs')
    for item in compiled['items']:
        if item['request'] is None:
            continue
        req = reader.bound(item['request'])
        result = old._result(reader, oldrun, item['request'], 'jev', normative)
        if result['status'] != 'complete':
            raise ValueError('Pinned old38 complete score control differs')
        registry.setdefault(req['request_hash'], []).append((item, req, result))
    if sum(len(v) for v in registry.values()) != 38:
        raise ValueError('Expected38 original whole score requests')
    value = {'spec':spec,'spec_ref':sref,'normative':normative,
             'manifest':manifest,'compiled':compiled,'items':items,
             'registry':registry,'audits':audits,'ledger':ledger,'old_run':oldrun}
    reader.r10_imports = value
    return value


def audit_old_score_cache(reader, signature, state, candidates, qid):
    """All matching complete batches, including rejected candidates; no score stitching."""
    imported = _imports(reader); matches = []
    for item, req, result in imported['registry'].get(core.request_hash(signature), []):
        if req['signature'] != signature:
            raise ValueError('Historical score hash collision')
        if req['state'] != state or req['candidates'] != candidates:
            raise ValueError('Matching HTTP has different target/proposal provenance')
        matches.append((item, req, result))
    if not matches:
        return None
    item, req, result = sorted(matches, key=lambda x:x[0]['query_id'])[0]
    if core.build_request(state,candidates,imported['normative']) != signature:
        raise ValueError('Historical whole predicate identity differs')
    proof = {'source':'r09_whole_request','source_query_id':item['query_id'],
             'consumer_query_id':qid,'request_hash':core.request_hash(signature),
             'request':item['request'],'source_input':item['source_input'],
             'response':result['response'],'raw':result['raw_response'],
             'state_sha256':core.digest(state),'candidates_sha256':core.digest(candidates),
             'scores':deepcopy(result['parsed']), 'proposer_terminal_audit':imported['audits']['proposer'],
             'score_terminal_audit':imported['audits']['jev'],'original_ledger':imported['ledger']}
    paths = {r['path'] for phase in ('proposer','jev')
             for r in reader.bound(imported['audits'][phase])['input_refs']}
    paths.update(r['path'] for r in (proof['request'],proof['source_input'],proof['response'],proof['raw'],
                                    proof['original_ledger'],*imported['audits'].values()))
    proof['input_refs'] = [reader.raw_pointer(p) for p in sorted(paths)]
    return proof


def _protocol(reader, imported):
    p = deepcopy(imported['manifest']['protocol'])
    p.update(schema=SCHEMA,policy=deepcopy(core.POLICY),research_protocol=imported['spec_ref'],
             envelope_coverage_protocol=imported['spec_ref'],
             qualifier_repair_protocol=_canonical_ref(imported['spec']['input_refs']['R09_scientific_protocol']),
             code_sha256=_codes(), original_proposer_terminal_audit=imported['audits']['proposer'],
             original_score_terminal_audit=imported['audits']['jev'])
    return p


def _request_value(unit, normative, psha, candidates, reader):
    req = old._request_value(unit,normative,psha,candidates=candidates)
    req['schema'] = REQUEST_SCHEMA
    req['score_cache'] = audit_old_score_cache(reader,req['signature'],unit['state'],candidates,unit['query_id'])
    return req


def prepare(run_dir):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT/'runs').resolve() or not run.name.startswith('envelope_coverage_r10') or run.exists():
        raise FileExistsError('Fresh direct R10 selector namespace required')
    reader=Reader();im=_imports(reader);protocol=_protocol(reader,im);psha=core.digest(protocol)
    _unchanged(reader);run.mkdir(parents=True,exist_ok=False);items=[]
    for item in im['items']:
        unit=_body(reader.bound(item['old_source_input']))
        unit.update(schema=INPUT_SCHEMA,protocol_sha256=psha)
        iref=save_new(run/'source_inputs'/f'{item["query_id"]}.json',unit)
        items.append({**deepcopy(item),'source_input':iref})
    refs=deepcopy(list(reader.refs.values()));_unchanged(reader)
    value={'schema':SCHEMA,'phase':'raw_import','status':'prepared','created_utc':now(),
           'protocol':protocol,'protocol_sha256':psha,'items':items,
           'inherited_input_refs':refs,'summary':dict(Counter(i['replay']['status'] for i in items)),
           'original_valid_identity_checks':54,'new_proposer_calls':0,'model_calls_during_prepare':0}
    return {'manifest':save_new(run/'run_manifest.json',value),'sessions':272,'model_calls':0}


def _initial(reader,run):
    m=reader.read(run/'run_manifest.json');im=_imports(reader);p=_protocol(reader,im)
    if (m.get('schema')!=SCHEMA or m.get('phase')!='raw_import' or m.get('status')!='prepared'
        or m.get('protocol')!=p or m.get('protocol_sha256')!=core.digest(p)
        or len(m.get('items',[]))!=272 or m.get('original_valid_identity_checks')!=54
        or m.get('new_proposer_calls')!=0):
        raise ValueError('R10 uniform import/protocol/code differs')
    expected=[]
    for olditem,item in zip(im['items'],m['items']):
        iref=item['source_input']
        if _body(item)!={**olditem,'source_input':iref}:
            raise ValueError('Uniform imported proposal sequence changed')
        unit=_body(reader.bound(olditem['old_source_input']))
        unit.update(schema=INPUT_SCHEMA,protocol_sha256=m['protocol_sha256'])
        if (_inside(iref,run)!=run/'source_inputs'/f'{item["query_id"]}.json'
            or _body(reader.bound(iref))!=unit):
            raise ValueError('Original five-field source/current birth differs')
        expected.append(iref)
    old._no_orphans(run/'source_inputs',expected)
    declared=m.get('inherited_input_refs',[])
    if len({r['path'] for r in declared}) != len(declared):
        raise ValueError('Complete unique imported input refs required')
    for ref in declared:reader.raw_pointer(ref['path'],ref=ref)
    required={r['path'] for r in reader.refs.values() if not Path(r['path']).is_relative_to(run)}
    if not required <= {r['path'] for r in declared}:
        raise ValueError('Import omits complete authenticated lineage')
    return m,im['normative']


def compile_score(run_dir):
    run=Path(run_dir).resolve()
    if (run/'score_manifest.json').exists() or (run/'requests').exists():
        raise FileExistsError('Compiled requests immutable')
    reader=Reader();m,normative=_initial(reader,run);prepared=[]
    for item in m['items']:
        req=None
        if item['candidates']:
            unit=reader.bound(item['source_input'])
            req=_request_value(unit,normative,m['protocol_sha256'],item['candidates'],reader)
            if item['original_proposer_result']['status']=='complete' and req['score_cache'] is None:
                raise ValueError('Old38 whole score control must inherit')
        prepared.append((item,req))
    _unchanged(reader);items=[];refs=[];counts=Counter()
    for item,req in prepared:
        rref=None
        if req is not None:
            rref=save_new(run/'requests'/f'{item["query_id"]}.json',req);refs.append(rref)
            counts['logical_nouls']+=req['logical_noul_count']
            counts['inherited_whole_batches']+=req['score_cache'] is not None
            counts['oversize']+=not req['dispatch_eligible']
        counts['candidates']+=len(item['candidates']);items.append({**deepcopy(item),'request':rref})
    value={'schema':COMPILED_SCHEMA,'phase':'jev','status':'compiled','created_utc':now(),
           'protocol':deepcopy(m['protocol']),'protocol_sha256':m['protocol_sha256'],
           'initial_manifest':reader.pointer(run/'run_manifest.json'),'items':items,'request_refs':refs,
           'summary':dict(counts),'model_calls_during_compile':0}
    return {'manifest':save_new(run/'score_manifest.json',value),'requests':len(refs),'summary':dict(counts),'model_calls':0}


def _score_manifest(reader,run,m,normative):
    score=reader.read(run/'score_manifest.json');refs=[]
    if (score.get('schema')!=COMPILED_SCHEMA or score.get('phase')!='jev' or score.get('status')!='compiled'
        or score.get('protocol')!=m['protocol'] or score.get('protocol_sha256')!=m['protocol_sha256']
        or score.get('initial_manifest')!=reader.pointer(run/'run_manifest.json') or len(score['items'])!=272):
        raise ValueError('R10 score manifest binding differs')
    for original,item in zip(m['items'],score['items']):
        ref=item['request']
        if item!={**original,'request':ref} or bool(item['candidates'])!=(ref is not None):
            raise ValueError('Exactly one complete request per nonempty session')
        if ref:
            unit=reader.bound(item['source_input'])
            expected=_request_value(unit,normative,m['protocol_sha256'],item['candidates'],reader)
            if _inside(ref,run)!=run/'requests'/f'{item["query_id"]}.json' or _body(reader.bound(ref))!=expected:
                raise ValueError('Exact R09 state/whole body/provenance/cache differs')
            if item['original_proposer_result']['status']=='complete' and expected['score_cache'] is None:
                raise ValueError('Old38 cache control differs')
            refs.append(ref)
    if score['request_refs']!=refs:raise ValueError('Score cohort sequence differs')
    old._no_orphans(run/'requests',refs)
    return score


def validate_prepared(run_dir,stage='score'):
    reader=Reader();run=Path(run_dir).resolve();m,n=_initial(reader,run)
    if stage=='raw_import':value,refs=m,[]
    elif stage=='score':value=_score_manifest(reader,run,m,n);refs=value['request_refs']
    else:raise ValueError('No R10 proposer dispatch; raw_import or score only')
    _unchanged(reader);return value,refs


def _admission_value(reader,run,m,n,score):
    from tools.score_envelope_coverage_r10 import audit_replay
    auditpath=run/'reviews/jev_terminal_audit_v1.json';saved=reader.read(auditpath)
    actual=audit_replay(run,'jev',score['request_refs'],reader=reader)
    if ({k:v for k,v in _body(saved).items() if k!='created_utc'}!=
        {k:v for k,v in actual.items() if k!='created_utc'} or actual['status']!='passed'):
        raise ValueError('Complete passed R10 JEV terminal audit required')
    accepted={a:{str(ci):[] for ci in range(10)} for a in ('cheap','jev')};rows=[]
    for item in score['items']:
        candidates=item['candidates'];result=None
        if item['request']:
            req=reader.bound(item['request']);row=reader.read(run/'responses'/f'{item["query_id"]}.json')
            result={'status':row['status'],'reason':row.get('reason'),'parsed':row['scores'],
                    'response':reader.pointer(run/'responses'/f'{item["query_id"]}.json'),
                    'raw_response':row['raw'],'score_source':row['score_source'],'score_cache':row['score_cache']}
        decision=core.admit_candidates(candidates,result['parsed'] if result else None,
                                      status=result['status'] if result else 'complete')
        ids={'cheap':[c['candidate_id'] for c in candidates],'jev':decision['accepted_ids']}
        for arm in accepted:
            accepted[arm][str(item['conversation'])].extend(deepcopy(c) for c in candidates if c['candidate_id'] in ids[arm])
        rows.append({**deepcopy(item),'score_request':item['request'],'score_result':result,'accepted':ids,'decision':decision})
    return {'schema':ADMISSION_SCHEMA,'status':'admitted','protocol':deepcopy(m['protocol']),
            'protocol_sha256':m['protocol_sha256'],'initial_manifest':reader.pointer(run/'run_manifest.json'),
            'score_manifest':reader.pointer(run/'score_manifest.json'),'jev_terminal_audit':reader.pointer(auditpath),
            'items':rows,'accepted_by_conversation':accepted,
            'summary':{a:sum(len(v) for v in accepted[a].values()) for a in accepted},'model_calls_during_admit':0}


def admit(run_dir):
    run=Path(run_dir).resolve();reader=Reader();m,n=_initial(reader,run);score=_score_manifest(reader,run,m,n)
    value=_admission_value(reader,run,m,n,score);_unchanged(reader)
    return {'manifest':save_new(run/'admission_manifest.json',value),'summary':value['summary'],'model_calls':0}


def validate_admission(path_or_run):
    path=Path(path_or_run).resolve();run=path if path.is_dir() else path.parent
    reader=Reader();m,n=_initial(reader,run);score=_score_manifest(reader,run,m,n);value=reader.read(run/'admission_manifest.json')
    if _body(value)!=_admission_value(reader,run,m,n,score):raise ValueError('Admission changed')
    _unchanged(reader);return value


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','check','compile-score','check-score','admit','check-admission']);p.add_argument('--run-dir',required=True)
    a=p.parse_args()
    if a.mode=='prepare':v=prepare(a.run_dir)
    elif a.mode=='compile-score':v=compile_score(a.run_dir)
    elif a.mode=='admit':v=admit(a.run_dir)
    elif a.mode=='check-admission':v={'status':'passed','sessions':len(validate_admission(a.run_dir)['items']),'model_calls':0}
    else:
        _,refs=validate_prepared(a.run_dir,'score' if a.mode=='check-score' else 'raw_import');v={'status':'passed','requests':len(refs),'model_calls':0}
    print(json.dumps(v,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
