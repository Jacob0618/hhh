"""Immutable R08 offline import, score compilation, sealing and exact replay."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import facet_discovery_r08 as core
from tools import prepare_evidence_selection_r03 as prior
from tools import prepare_evidence_coverage_r06 as r06
Reader, save_new, now, file_sha = prior.Reader, prior.save_new, prior.now, prior.file_sha
SCHEMA = 'facet-discovery-r08-selector-manifest-v1'
COMPILED_SCHEMA = 'facet-discovery-r08-score-manifest-v1'
SOURCE_SCHEMA = 'facet-discovery-r08-selected-inputs-v1'
REQUEST_SCHEMA = 'facet-discovery-r08-request-v1'
SELECTION_SCHEMA = 'facet-discovery-r08-selection-v1'
RESEARCH_PROTOCOL = ROOT/'research/jev_autoresearch_2026-10-04/R08_facet_discovery_protocol_final_v1.json'
DEFAULT_SOURCE = prior.DEFAULT_SOURCE
CODE = tuple(dict.fromkeys((*r06.CODE, 'jev_mem/facet_discovery_r08.py',
    'tools/prepare_facet_discovery_r08.py', 'tools/score_facet_discovery_r08.py',
    'tools/run_facet_discovery_qa_r08.py', 'tools/summarize_facet_discovery_r08.py')))
_VALIDATION_CACHE = {}


def _body(value):
    return {k: v for k, v in value.items() if k != 'content_sha256'}


def _codes():
    return {p: file_sha(ROOT/p) for p in CODE}


def _research_protocol(reader):
    spec = reader.read(RESEARCH_PROTOCOL, sealed=False)
    ref = reader.raw_pointer(RESEARCH_PROTOCOL)
    if ref['sha256'] != core.PROTOCOL_SHA256:
        raise ValueError('Exact three-voted R08 scientific bytes changed')
    for pointer in spec['input_refs'].values():
        reader.raw_pointer(ROOT/pointer['path'], ref=prior._canonical_ref(
            {'path': str(ROOT/pointer['path']), 'sha256': pointer['sha256']}))
    return spec, ref


def _inputs(reader, source_path, spec):
    ref = spec['input_refs']['R01_pool_manifest']
    source, sref, jobs, banks, pools = prior._inputs(reader, source_path,
        {'basis': {'r01_pool_manifest': {'path': str(ROOT/ref['path']), 'sha256': ref['sha256']}}})
    oldrun = (ROOT/spec['input_refs']['R06_producer_manifest']['path']).parent
    oldm, oldspec, oldjobs, oldpools = r06._manifest(reader, oldrun)
    facets, auditref = r06._facet_results(reader, oldrun, oldm, oldspec)
    compiled = reader.bound({'path': str(ROOT/spec['input_refs']['R06_compiled_facet_manifest']['path']),
                             'sha256': spec['input_refs']['R06_compiled_facet_manifest']['sha256']})
    if compiled['facet_terminal_audit'] != auditref:
        raise ValueError('Imported R06 facet audit lineage differs')
    if set(oldjobs) != {j['query_id'] for j in jobs}:
        raise ValueError('Imported facet cohort differs')
    imported = {}
    ciitems = {i['query_id']: i for i in compiled['items']}
    for job in jobs:
        qid = job['query_id']; pool = pools[qid]
        core.build_pool(pool['pool_ids'], job['selected_ids'], pool['native_other_ids'],
                        pool['original_native80_ids'], [])
        if oldjobs[qid] != job or oldpools[qid] != pool:
            raise ValueError('Imported query/date/bank/pool lineage differs')
        result = facets[qid]
        if ciitems[qid]['facet_result'] != result or ciitems[qid]['facets'] != (
                result['parsed']['facets'] if result['status'] == 'complete' else None):
            raise ValueError('Compiled facets differ from audited raw parse')
        imported[qid] = {'query_id': qid, 'conversation': job['conversation'],
            'status': result['status'], 'facets': None if result['parsed'] is None else result['parsed']['facets'],
            'result': result, 'terminal_audit': auditref}
    # The R01 canonical fact projection has no vectors. Bind/read the original full banks separately.
    vectors = {}
    for ci, bank in banks.items():
        raw = reader.bound(bank['ref'])
        vectors[ci] = [{'id': f['id'], 'embedding': deepcopy(f['embedding'])} for f in raw['facts']]
        if [f['id'] for f in vectors[ci]] != [f['id'] for f in bank['facts']]:
            raise ValueError('Original bank ordinal differs')
        for f in vectors[ci]: core.normalize_vector(f['embedding'])
    return source, sref, jobs, banks, pools, imported, vectors


def _protocol(reader, source, sref, banks, spec, spec_ref):
    reuse = reader.read(prior.REUSE_QA_MANIFEST, sealed=False)
    prior._validate_qa_manifest(reuse)
    if reuse['baseline_reuse_proof'] != reader.raw_pointer(prior.FROZEN_REUSE_PROOF):
        raise ValueError('Original N0 reuse proof producer mismatch')
    return {'schema': SCHEMA, 'research_protocol': spec_ref, 'scientific_basis': spec['input_refs'],
        'frozen_source': sref, 'frozen_qa_run_manifest': reader.raw_pointer(prior.FROZEN_QA_MANIFEST),
        'frozen_baseline_reuse_proof': reader.raw_pointer(prior.FROZEN_REUSE_PROOF),
        'reuse_proof_qa_manifest': reader.raw_pointer(prior.REUSE_QA_MANIFEST),
        'source_dataset_sha256': source['source_dataset_sha256'],
        'frozen_protocol_sha256': source['protocol_sha256'], 'code_sha256': _codes(),
        'questions_per_arm': 1540, 'arms': list(core.ARMS), 'fact_count': 1637,
        'bank_refs': {str(ci): banks[ci]['ref'] for ci in sorted(banks)},
        'official_metric_provenance': source['official_metric_provenance']}


def _request(signature, request_id, ci, protocol_sha, **extra):
    size = len(core.serialized_signature(signature))
    return {'schema': REQUEST_SCHEMA, 'request_id': request_id, 'query_id': request_id,
        'conversation': ci, 'protocol_sha256': protocol_sha, 'signature': signature,
        'request_hash': core.request_hash(signature), 'signature_utf8_bytes': size,
        'signature_utf8_byte_cap': core.MAX_BYTES, 'dispatch_eligible': size <= core.MAX_BYTES,
        'max_physical_attempts_per_phase': 3, **extra}


def audit_old_score_cache(reader, spec, signature, ordered_ids, qid):
    """Whole R03 request donor with exact external ID and physical producer closure.

    No provider object, credentials, dispatch or cache import is involved. A broken
    claimed hit is an error; only a genuine signature miss or audited terminal
    known failure returns None.
    """
    if signature is None:
        return None
    pins = set()

    def bound(ref, sealed=True):
        canonical = prior._canonical_ref(ref)
        value = reader.bound(canonical, sealed=sealed); pins.add(canonical['path'])
        return value

    def local(path, sealed=True):
        value = reader.read(path, sealed=sealed); pins.add(str(Path(path).resolve()))
        return value

    mref = spec['input_refs']['R03_manifest']
    manifest = bound({'path': str(ROOT/mref['path']), 'sha256': mref['sha256']})
    protocol = manifest['protocol']; items = {i['query_id']: i for i in manifest['items']}
    if not hasattr(reader, 'r08_r03_codes'):
        reader.r08_r03_codes = prior._codes()
    if (manifest.get('schema') != prior.SCHEMA or manifest.get('status') != 'prepared'
            or manifest['protocol_sha256'] != core.digest(protocol) or len(items) != 1540
            or protocol['code_sha256'] != reader.r08_r03_codes):
        raise ValueError('R03 donor manifest/protocol/code changed')
    if qid not in items:
        raise ValueError('Current query absent from frozen R03 cohort')
    consumer_qid = qid
    item = items[qid]; req = bound(item['request']); proposal = bound(item['proposal'])
    if (req['signature'] != signature or
            [f['id'] for f in proposal['frozen_pool']['candidate_records']] != list(ordered_ids)):
        # Current-qid is a fast path, not the definition of a legitimate donor.
        # Registry construction is read-only and memoized per byte-bound Reader.
        if not hasattr(reader, 'r08_r03_registry'):
            registry = {}
            for candidate in manifest['items']:
                oldreq = reader.bound(candidate['request'])
                if oldreq['request_hash'] != core.request_hash(oldreq['signature']):
                    raise ValueError('R03 registry request hash differs')
                registry.setdefault(oldreq['request_hash'], []).append(candidate)
            reader.r08_r03_registry = registry
        matches = []
        for candidate in reader.r08_r03_registry.get(core.request_hash(signature), []):
            oldreq = reader.bound(candidate['request'])
            if oldreq['signature'] == signature:
                oldprop = reader.bound(candidate['proposal'])
                matches.append((candidate, oldreq, oldprop))
        if not matches:
            return None
        matches = [x for x in matches if [f['id'] for f in x[2]['frozen_pool']['candidate_records']] == list(ordered_ids)]
        if not matches:
            raise ValueError('Matching old HTTP body has different ordered-ID provenance')
        matches.sort(key=lambda x: (x[0]['conversation'], int(x[0]['query_id'].split('_q')[1])))
        item, req, proposal = matches[0]; qid = item['query_id']
        # Pin the selected donor independently of registry traversal/memo state.
        req = bound(item['request']); proposal = bound(item['proposal'])
    producer_records = proposal['frozen_pool']['candidate_records']
    core.same_score_provenance(ordered_ids, [f['id'] for f in producer_records])
    r03ref = spec['input_refs']['R03_protocol']
    r03spec = bound({'path': str(ROOT/r03ref['path']), 'sha256': r03ref['sha256']}, sealed=False)
    source_ref = prior._canonical_ref({'path': str(ROOT/spec['input_refs']['N0_inputs']['path']),
                                      'sha256': spec['input_refs']['N0_inputs']['sha256']})
    if (protocol['research_protocol'] != reader.raw_pointer(ROOT/r03ref['path'])
            or protocol['frozen_source'] != source_ref or proposal['frozen_source'] != source_ref
            or proposal['schema'] != prior.PROPOSAL_SCHEMA or proposal['query_id'] != qid
            or proposal['request'] != item['request'] or proposal['protocol_sha256'] != manifest['protocol_sha256']
            or proposal['candidate_records_sha256'] != core.digest(producer_records)
            or proposal['source_job_sha256'] != core.digest(proposal['original_job'])
            or req['protocol_sha256'] != manifest['protocol_sha256'] or req['query_id'] != qid
            or req['request_hash'] != core.request_hash(signature)
            or core.build_request(proposal['original_job']['question'], proposal['original_job']['reference_date'],
                                   producer_records, r03spec) != signature):
        raise ValueError('R03 whole-body/ID/query/source provenance differs')
    run = Path(mref['path']).parent
    run = (ROOT/run).resolve(); key = req['request_hash']; stage = 'evidence_selection_score_r03'
    eref = spec['input_refs']['R03_execution_intent']
    execution = bound({'path': str(ROOT/eref['path']), 'sha256': eref['sha256']}, sealed=False)
    if (execution.get('manifest_sha256') != mref['sha256'] or execution.get('phase') != 'jev'
            or execution.get('request_files') != [i['request'] for i in manifest['items']]
            or execution.get('dispatch_code_sha256') != protocol['code_sha256']['tools/score_evidence_selection_r03.py']
            or execution.get('transport_retries') != 2):
        raise ValueError('R03 controller dispatch cohort/code binding differs')
    oref = spec['input_refs']['R03_score_outcome']
    outcome = bound({'path': str(ROOT/oref['path']), 'sha256': oref['sha256']})
    if (outcome.get('status') != 'terminal' or outcome.get('local_errors') or
            outcome.get('expected_requests') != 1540 or outcome.get('terminal_requests') != 1540):
        raise ValueError('R03 full terminal phase required for donor reuse')
    row = local(run/'responses'/f'{qid}.json')
    if (row.get('schema') != prior.SCORE_SCHEMA or row.get('query_id') != qid
            or row.get('protocol_sha256') != manifest['protocol_sha256']
            or row.get('request_hash') != key or row.get('actual_request_hash') != key
            or row.get('request_path') != item['request']['path']
            or row.get('request_sha256') != item['request']['sha256']):
        raise ValueError('R03 donor response/request lineage differs')
    if row.get('status') == 'unknown':
        raise ValueError('Unknown old matching producer blocks redispatch')
    ip = run/'request_intents'/f'{key}.json'; rp = run/'api_cache'/f'{key}.json'
    if (run/'cache_lineage'/f'{key}.json').exists():
        raise ValueError('Undeclared external R03 producer cache lineage')
    if not ip.exists():
        raise ValueError('Matching R03 producer has no intent')
    intent = local(ip, sealed=False)
    producer = intent.get('context')
    if (intent.get('signature') != signature or intent.get('request_hash') != key
            or intent.get('stage') != stage or intent.get('retries') != 2
            or not isinstance(producer, dict) or producer.get('query_id') not in items
            or producer != {'conversation': items[producer['query_id']]['conversation'],
                            'query_id': producer['query_id'], 'method': 'jev_selection', 'phase': 'jev'}):
        raise ValueError('R03 producer intent is not an authorized prepared consumer')
    preq = bound(items[producer['query_id']]['request'])
    if preq['signature'] != signature:
        raise ValueError('Producer exact signature differs')
    ledgerref = spec['input_refs']['R03_score_ledger']; lp = ROOT/ledgerref['path']
    canonical = prior._canonical_ref({'path': str(lp), 'sha256': ledgerref['sha256']})
    reader.raw_pointer(lp, ref=canonical); pins.add(canonical['path'])
    if not hasattr(reader, 'r08_legacy_ledger'):
        raw = lp.read_bytes()
        if not raw.endswith(b'\n') or file_sha(lp) != ledgerref['sha256']:
            raise ValueError('Old producer ledger incomplete or changed')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        index = {}
        for event in rows: index.setdefault(event.get('request_hash'), []).append(event)
        reader.r08_legacy_ledger = index
    events = reader.r08_legacy_ledger.get(key, [])
    physical = []
    cache_contexts = set()
    for event in events:
        context = event.get('context', {}); consumer = context.get('query_id')
        if (event.get('stage') != stage or consumer not in items
                or context != {'conversation': items[consumer]['conversation'], 'query_id': consumer,
                               'method': 'jev_selection', 'phase': 'jev'}
                or (event.get('provider'), event.get('model'), event.get('operation')) !=
                   (signature['provider'], signature['model'], signature['operation'])):
            raise ValueError('Old matching producer event has unauthorized context/backend/stage')
        if bound(items[consumer]['request'])['signature'] != signature:
            raise ValueError('Old matching consumer request differs')
        if event.get('cache_hit'):
            if context == producer or consumer in cache_contexts:
                raise ValueError('Duplicate old cache consumer')
            cache_contexts.add(consumer)
        else:
            if context != producer: raise ValueError('Multiple physical old producers')
            physical.append(event)
    retryable = {429, 500, 502, 503, 504, 529, 'transport_error'}
    if (not physical or len(physical) > 3 or [e.get('attempt') for e in physical] != list(range(1, len(physical)+1))
            or any(e.get('status') not in retryable for e in physical[:-1])):
        raise ValueError('Old producer attempts incomplete/duplicated/unbounded')
    if not any(e.get('context', {}).get('query_id') == qid for e in events):
        raise ValueError('No actual old consumer event')
    if rp.exists():
        if row.get('raw') is None or Path(row['raw']['path']).resolve() != rp:
            raise ValueError('Old raw pointer missing or outside producer namespace')
        raw = bound(row['raw'], sealed=False)
        if raw.get('request') != signature or physical[-1].get('status') != 200:
            raise ValueError('Old raw response lacks completed actual producer')
        if row['status'] == 'complete':
            scores = core.parse_scores(raw['response'], signature)
            if scores != row.get('scores'): raise ValueError('Old full score vector replay differs')
            return {'kind': 'r03_whole_request', 'query_id': consumer_qid, 'source_query_id': qid, 'request_hash': key,
                'ordered_fact_ids': list(ordered_ids), 'scores': scores, 'raw': row['raw'],
                'producer_context': producer, 'response': reader.pointer(run/'responses'/f'{qid}.json'),
                'request': item['request'], 'intent': reader.raw_pointer(ip),
                'ledger': canonical, 'input_refs': [deepcopy(reader.refs[p]) for p in sorted(pins)]}
        if row['status'] != 'failed' or row.get('scores') is not None:
            raise ValueError('Unavailable old scores cannot be fabricated')
        try: core.parse_scores(raw['response'], signature)
        except (ValueError, RuntimeError, KeyError, TypeError): pass
        else: raise ValueError('Valid old scores mislabeled failed')
        return None
    if row.get('raw') is not None or row.get('scores') is not None or row['status'] != 'failed':
        raise ValueError('Old missing raw is not a proven known failure')
    tail = physical[-1].get('status')
    if not ((tail in retryable and len(physical) == 3) or (type(tail) is int and tail >= 400 and tail not in retryable)):
        raise ValueError('Old matching missing raw has unknown producer state')
    return None


old_score_cache = audit_old_score_cache


def prepare(run_dir, source_path=DEFAULT_SOURCE):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT/'runs').resolve() or not run.name.startswith('facet_discovery_r08') or run.exists():
        raise FileExistsError('Fresh direct runs/facet_discovery_r08* required')
    reader = Reader(); spec, sref = _research_protocol(reader)
    source, src, jobs, banks, pools, facets, vectors = _inputs(reader, source_path, spec)
    protocol = _protocol(reader, source, src, banks, spec, sref); psha = core.digest(protocol)
    batches = core.pack_batches(list(facets.values()))
    run.mkdir(parents=True, exist_ok=False); refs = []
    for batch in batches:
        signature = core.embedding_signature(batch['texts'])
        refs.append(save_new(run/'embedding_requests'/f'{batch["request_id"]}.json',
            _request(signature, batch['request_id'], batch['conversation'], psha, batch=batch)))
    mref = save_new(run/'run_manifest.json', {'schema': SCHEMA, 'phase': 'embedding', 'status': 'prepared',
        'created_utc': now(), 'protocol': protocol, 'protocol_sha256': psha,
        'items': [facets[j['query_id']] for j in jobs], 'request_refs': refs,
        'summary': {'queries': 1540, 'embedding_batches': len(refs),
                    'embedding_items': sum(len(b['texts']) for b in batches)}, 'model_calls_during_prepare': 0})
    return {'manifest': mref, 'requests': len(refs), 'model_calls': 0}


def _manifest(reader, run):
    m = reader.read(run/'run_manifest.json'); spec, sref = _research_protocol(reader)
    source, src, jobs, banks, pools, facets, vectors = _inputs(reader, m['protocol']['frozen_source']['path'], spec)
    expected = _protocol(reader, source, src, banks, spec, sref)
    if (m.get('schema') != SCHEMA or m.get('phase') != 'embedding' or m.get('status') != 'prepared'
            or m['protocol'] != expected or m['protocol_sha256'] != core.digest(expected)
            or m['items'] != [facets[j['query_id']] for j in jobs]):
        raise ValueError('Initial immutable R08 lineage changed')
    expectedrefs = []
    for b in core.pack_batches(list(facets.values())):
        path = run/'embedding_requests'/f'{b["request_id"]}.json'
        expectedrec = _request(core.embedding_signature(b['texts']), b['request_id'], b['conversation'],
                               m['protocol_sha256'], batch=b)
        if _body(reader.read(path)) != expectedrec: raise ValueError('Embedding batch replay differs')
        expectedrefs.append(reader.pointer(path))
    if m['request_refs'] != expectedrefs: raise ValueError('Embedding manifest refs differ')
    r06._no_orphans(run/'embedding_requests', expectedrefs)
    return m, spec, jobs, banks, pools, facets, vectors


def _terminal(reader, run, phase, manifest, refs):
    from tools.score_facet_discovery_r08 import phase_paths, audit_replay
    paths = phase_paths(run, phase)
    auditpath = run/'reviews'/f'{phase}_terminal_audit_v1.json'
    saved = reader.read(auditpath)
    replay = audit_replay(run, phase, refs, reader=reader)
    if (saved.get('status') != 'passed' or saved.get('unknown') or
            any(f['technical_recovery_eligible'] for f in saved['known_terminal_failures'])
            or any(saved.get(k) != replay[k] for k in replay)):
        raise ValueError('Exact passed phase audit required with no pending recovery')
    results = {reader.bound(ref)['request_id']: reader.read(paths['responses']/
               f'{reader.bound(ref)["request_id"]}.json') for ref in refs}
    return results, reader.pointer(auditpath)


def _compiled_values(reader, run, m, spec, jobs, banks, pools, facets, vectors):
    results, auditref = _terminal(reader, run, 'embedding', m, m['request_refs'])
    mapped = {}
    for ref in m['request_refs']:
        req = reader.bound(ref); result = results[req['request_id']]
        for slot, consumers in enumerate(req['batch']['consumers']):
            for consumer in consumers:
                mapped[(consumer['query_id'], consumer['facet_index'])] = (
                    result['status'], None if result['vectors'] is None else result['vectors'][slot],
                    reader.pointer(run/'embedding_responses'/f'{req["request_id"]}.json'))
    r03spec = reader.read(ROOT/spec['input_refs']['R03_protocol']['path'], sealed=False)
    items = []
    for old in jobs:
        qid = old['query_id']; imported = facets[qid]; original = pools[qid]; status = imported['status']
        qvectors, embedding_refs = [], []
        if status == 'complete':
            for index in range(1, len(imported['facets'])):
                state, vector, ref = mapped[(qid, index)]
                if state == 'unknown': raise ValueError('Unknown embedding blocks compilation')
                embedding_refs.append(ref)
                if state != 'complete': status = 'failed'
                qvectors.append(vector)
        discoveries = []
        if status == 'complete' and qvectors:
            discoveries = core.round_robin(core.rank_facets(vectors[old['conversation']], qvectors),
                original['original_native80_ids'], original['pool_ids'][:60])
        newpool = core.build_pool(original['pool_ids'], old['selected_ids'], original['native_other_ids'],
                                  original['original_native80_ids'], discoveries)
        lookup = {f['id']: f for f in banks[old['conversation']]['facts']}
        records = [deepcopy(lookup[fid]) for fid in newpool['pool_ids']]
        signature = core.build_request(old['question'], old['reference_date'], records, r03spec) if status == 'complete' else None
        cache = old_score_cache(reader, spec, signature, newpool['pool_ids'], qid) if signature else None
        items.append({'query_id': qid, 'conversation': old['conversation'], 'original_job': old,
            'source_job_sha256': core.digest(old), 'frozen_pool': original, 'imported_facet': imported,
            'embedding_results': embedding_refs, 'discovery_status': status, 'discoveries': discoveries,
            'pool': newpool, 'candidate_records': records, 'candidate_records_sha256': core.digest(records),
            'signature': signature, 'score_cache': cache,
            'cheap_decision': core.choose_cheap(old['question'], old['reference_date'], records) if status == 'complete'
                else core.choose_jev(records, None, score_status='failed')})
    return items, auditref


def compile_score(run_dir):
    run = Path(run_dir).resolve()
    if (run/'score_manifest.json').exists() or (run/'requests').exists():
        raise FileExistsError('Fresh compiled namespace required')
    reader = Reader(); m, spec, jobs, banks, pools, facets, vectors = _manifest(reader, run)
    items, aref = _compiled_values(reader, run, m, spec, jobs, banks, pools, facets, vectors)
    refs = []
    for item in items:
        item['request'] = None
        if item['signature'] is not None:
            ref = save_new(run/'requests'/f'{item["query_id"]}.json', _request(item['signature'],
                item['query_id'], item['conversation'], m['protocol_sha256'], logical_noul_count=80,
                ordered_fact_ids=item['pool']['pool_ids'], score_cache=item['score_cache']))
            item['request'] = ref; refs.append(ref)
    mref = save_new(run/'score_manifest.json', {'schema': COMPILED_SCHEMA, 'phase': 'jev', 'status': 'compiled',
        'created_utc': now(), 'protocol': m['protocol'], 'protocol_sha256': m['protocol_sha256'],
        'initial_manifest': reader.pointer(run/'run_manifest.json'), 'embedding_terminal_audit': aref,
        'items': items, 'request_refs': refs, 'model_calls_during_compile': 0})
    return {'manifest': mref, 'requests': len(refs), 'model_calls': 0}


def _compiled(reader, run, state):
    m, spec, jobs, banks, pools, facets, vectors = state
    cm = reader.read(run/'score_manifest.json')
    items, aref = _compiled_values(reader, run, m, spec, jobs, banks, pools, facets, vectors)
    refs = []
    for item in items:
        item['request'] = None
        if item['signature'] is not None:
            path = run/'requests'/f'{item["query_id"]}.json'
            if _body(reader.read(path)) != _request(item['signature'], item['query_id'], item['conversation'],
                    m['protocol_sha256'], logical_noul_count=80, ordered_fact_ids=item['pool']['pool_ids'],
                    score_cache=item['score_cache']): raise ValueError('Score request replay differs')
            item['request'] = reader.pointer(path); refs.append(item['request'])
    if (cm.get('schema') != COMPILED_SCHEMA or cm.get('status') != 'compiled' or cm.get('phase') != 'jev'
            or cm['protocol'] != m['protocol'] or cm['protocol_sha256'] != m['protocol_sha256']
            or cm['initial_manifest'] != reader.pointer(run/'run_manifest.json')
            or cm['embedding_terminal_audit'] != aref or cm['items'] != items or cm['request_refs'] != refs):
        raise ValueError('Compiled discovery/score manifest replay differs')
    r06._no_orphans(run/'requests', refs)
    return cm


def validate_prepared(run_dir, stage='embedding'):
    run = Path(run_dir).resolve(); reader = Reader(); state = _manifest(reader, run)
    if stage not in ('embedding', 'score'): raise ValueError('Unknown prepared stage')
    m = state[0] if stage == 'embedding' else _compiled(reader, run, state)
    return m, deepcopy(m['request_refs'])


def _expected_records(reader, run, state):
    cm = _compiled(reader, run, state)
    responses, jref = _terminal(reader, run, 'jev', cm, cm['request_refs'])
    records = []
    for item in cm['items']:
        old = item['original_job']; candidates = item['candidate_records']; qid = old['query_id']
        row = responses.get(qid); scorestatus = 'failed' if row is None else row['status']
        decisions = {'baseline': core.choose_baseline(candidates), 'cheap': item['cheap_decision'],
            'jev': core.choose_jev(candidates, None if row is None else row['scores'], score_status=scorestatus)}
        for method in core.ARMS:
            decision = decisions[method]; selected = core.selected_facts(candidates, decision['pool_selected_indices'])
            ids = decision['selected_ids']; added = set(ids) - set(old['selected_ids'])
            record = {'schema': SELECTION_SCHEMA, 'protocol_sha256': cm['protocol_sha256'],
                'query_id': qid, 'conversation': old['conversation'], 'method': method,
                'bank': old['bank'], 'original_selection': old['selection'],
                'proposal': reader.pointer(run/'score_manifest.json'), 'request': item['request'],
                'response': None if row is None else reader.pointer(run/'responses'/f'{qid}.json'),
                'raw_response': None if row is None else row['raw'], 'frozen_source': cm['protocol']['frozen_source'],
                'imported_facet': item['imported_facet'], 'embedding_results': item['embedding_results'],
                'discoveries': item['discoveries'], 'pool': item['pool'], 'discovery_status': item['discovery_status'],
                'jev_terminal_audit': jref, 'decision': decision, 'selected_ids': ids, 'selected_facts': selected,
                'selected_sha256': core.digest(selected), 'source_job_sha256': core.digest(old),
                'baseline_selected_sha256': core.digest(old['selected_facts']),
                'unchanged_ordered_reader_facts': selected == old['selected_facts'],
                'same_membership_as_baseline': set(ids) == set(old['selected_ids']),
                'added_ids': sorted(added), 'removed_ids': sorted(set(old['selected_ids']) - set(ids)),
                'added_outside_original_native80_ids': sorted(added - set(item['frozen_pool']['original_native80_ids']))}
            records.append((old, method, record))
    return cm, records, jref


def _count(records):
    counts = Counter()
    for old, method, r in records:
        counts[f'{method}_changed_queries'] += not r['unchanged_ordered_reader_facts']
        counts[f'{method}_changed_membership_queries'] += not r['same_membership_as_baseline']
        counts[f'{method}_added_fact_occurrences'] += len(r['added_ids'])
        counts[f'{method}_added_outside_original_native80_occurrences'] += len(r['added_outside_original_native80_ids'])
        counts[f'{method}_score_status_'+r['decision']['score_status']] += 1
    return dict(counts)


def _source(reader, run, m, cm, records, jref, jobs):
    return {'schema': SOURCE_SCHEMA, 'status': 'sealed', 'arms': list(core.ARMS), 'questions_per_arm': 1540,
        'jobs': jobs, 'parent_run': str(run), 'selector_manifest': reader.pointer(run/'run_manifest.json'),
        'score_manifest': reader.pointer(run/'score_manifest.json'), 'embedding_terminal_audit': cm['embedding_terminal_audit'],
        'jev_terminal_audit': jref, 'protocol_sha256': m['protocol_sha256'],
        'discovery_protocol': m['protocol']['research_protocol'],
        **{k: m['protocol'][k] for k in ('frozen_source', 'frozen_qa_run_manifest', 'frozen_baseline_reuse_proof',
             'reuse_proof_qa_manifest', 'source_dataset_sha256', 'official_metric_provenance', 'code_sha256')},
        'selection_counts': _count(records), 'model_calls_during_seal': 0}


def seal(run_dir):
    run = Path(run_dir).resolve()
    if (run/'selected_inputs.json').exists() or (run/'selections').exists():
        raise FileExistsError('Fresh selected source namespace required')
    reader = Reader(); state = _manifest(reader, run); cm, records, jref = _expected_records(reader, run, state)
    jobs = []
    for old, method, record in records:
        ref = save_new(run/'selections'/method/f'{old["query_id"]}.json', record)
        jobs.append(prior._job(old, method, record, ref, cm['protocol_sha256']))
    value = _source(reader, run, state[0], cm, records, jref, jobs)
    value['created_utc'] = now()
    return {'source': save_new(run/'selected_inputs.json', value), 'jobs': len(jobs), 'model_calls': 0}


def validate_source(path, *, fresh=False):
    path = Path(path).resolve(); key = str(path); cached = _VALIDATION_CACHE.get(key)
    if cached and not fresh and all(Path(p).is_file() and (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == mark
            for p, mark in cached['stats'].items()) and all(not Path(p).exists() for p in cached['absent']):
        return deepcopy(cached['value'])
    reader = Reader(); source = reader.read(path); run = Path(source['parent_run']).resolve()
    if path != run/'selected_inputs.json': raise ValueError('Exact selected source namespace required')
    state = _manifest(reader, run); cm, records, jref = _expected_records(reader, run, state); jobs = []
    for old, method, r in records:
        p = run/'selections'/method/f'{old["query_id"]}.json'
        if _body(reader.read(p)) != r: raise ValueError('Selected20 exact replay differs')
        jobs.append(prior._job(old, method, r, reader.pointer(p), cm['protocol_sha256']))
    expected = _source(reader, run, state[0], cm, records, jref, jobs)
    if {k: v for k, v in _body(source).items() if k != 'created_utc'} != expected:
        raise ValueError('Sealed4620 source/code/query/lineage differs')
    stats = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in reader.refs if Path(p).is_file()}
    for name in CODE:
        codepath = str((ROOT/name).resolve())
        stats[codepath] = (Path(codepath).stat().st_size, Path(codepath).stat().st_mtime_ns)
    absent = [p for p in getattr(reader, 'absent_paths', set()) if not Path(p).exists()]
    _VALIDATION_CACHE[key] = {'stats': stats, 'absent': absent, 'value': deepcopy(source)}
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('prepare', 'compile-score', 'seal', 'validate-source'), required=True)
    p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = p.parse_args()
    if args.action == 'prepare': result = prepare(args.run_dir, args.source)
    elif args.action == 'compile-score': result = compile_score(args.run_dir)
    elif args.action == 'seal': result = seal(args.run_dir)
    else: result = {'valid': bool(validate_source(args.run_dir/'selected_inputs.json', fresh=True)), 'model_calls': 0}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
