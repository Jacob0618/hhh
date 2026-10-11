"""Offline uniform20 review and optional metadata export; never repairs native data.

All20 original banks must be terminal before review/export. The one currently
supported failure exception is pinned to independently reviewed source evidence.
No Memory/Qdrant client is constructed. Production execution is opt-in CLI only.
"""
from __future__ import annotations
import argparse
import copy
import os
import stat
from pathlib import Path
import sys
from types import FunctionType
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import audit_regroup_refinement_banks_v1 as strict
from tools import mem0_regroup_firstpass_review_v2 as old
from jev_mem.local_bm25_context_v2 import forbid_network

ROOT = old.ROOT
ARMS = ('source_local_context', 'jev_refined')
SCHEMA = 'regroup-refinement-uniform20-review-v2'
EXPORT_SCHEMA = 'regroup-refinement-reviewed-export-v2'
STATUS = 'passed_complete_facts_with_disclosed_native_defects'
ERROR = 'Native entity linking omitted extracted entity-bearing facts'
SUPPORTED_FAILURES = {
    (3, 'source_local_context', '607a8ed53abe1a31f5a2a1dbc8b462f76ca983db36d51553f100a199e07f33f5'):
        'eda4bc6c807d40ef2a4ce09a3074b1ce51e14d006b9626db43b0a729d130c87c',
}
digest, verify, pointer = old.digest, old.verify, old.pointer
DEPENDENCIES = ('tools/audit_regroup_refinement_banks_v1.py',
                'tools/mem0_regroup_firstpass_review_v2.py',
                'jev_mem/local_bm25_context_v2.py', 'tools/check_experiment_health.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def _checked(ref):
    require(pointer(ref['path']) == ref, 'Source pointer drift')
    return old.read(ref['path'])


def _population(rows):
    require(len(rows) == 20 and {(r['conversation'], r['arm']) for r in rows}
            == {(c, arm) for c in range(10) for arm in ARMS}, 'Exactly current20 banks required')


def terminal(runs):
    """Read all statuses first. No SQLite or partial review when any is running."""
    require(set(runs) == set(ARMS), 'Exactly two declared writer runs required')
    rows = []
    for arm in ARMS:
        run = Path(runs[arm]).resolve()
        m = old.read(run / 'run_manifest.json')
        require(m['protocol_sha256'] == digest({k: v for k, v in m.items() if k != 'protocol_sha256'}), 'Manifest seal drift')
        require(m['arms'] == [arm] and m['refinement']['arm'] == arm, 'Writer arm mismatch')
        for ci in range(10):
            folder = run / f'conv_{ci}' / arm
            s = old.read(folder / 'write_status.json')
            require(s.get('status') in ('complete', 'failed') and s.get('finished_utc'), 'All20 must be terminal')
            o = verify(folder / 'write_outcome.json')
            require(o.get('status') == s['status'] and o['write_status'] == pointer(folder / 'write_status.json'), 'Terminal outcome drift')
            require(all(v.get('protocol_sha256') == m['protocol_sha256']
                        and (v.get('conversation'), v.get('arm')) == (ci, arm) for v in (s, o)), 'Terminal identity drift')
            require(s.get('completed_groups') == s.get('expected_groups') and s.get('expected_groups', 0) > 0,
                    'Partial facts are never eligible')
            if s['status'] == 'failed':
                require(s.get('error', {}).get('message') == ERROR and o.get('error') == s.get('error'), 'Unknown failure class')
            rows.append({'conversation': ci, 'arm': arm, 'run': run, 'folder': folder,
                         'manifest': m, 'state': s, 'outcome_value': o})
    _population(rows)
    return rows


def _failure_binding(path, target):
    ref = pointer(path)
    s, m = target['state'], target['manifest']
    key = (target['conversation'], target['arm'], m['protocol_sha256'])
    require(SUPPORTED_FAILURES.get(key) == ref['sha256'], 'Failure needs its own independently pinned proof')
    p = verify(path)
    require(p.get('schema') == 'context-c3-entity-failure-audit-v1'
            and p.get('status') == 'real_native_same_canonical_stale_payload_omission_confirmed', 'Unsupported diagnostic schema')
    require((p['conversation'], p['arm'], p['protocol_sha256']) == key
            and p['manifest'] == pointer(target['run'] / 'run_manifest.json')
            and p['write_status'] == pointer(target['folder'] / 'write_status.json')
            and p['write_outcome'] == pointer(target['folder'] / 'write_outcome.json'), 'Proof bound to another source')
    require(p['counts']['groups'] == s['expected_groups'] and p['counts']['facts'] == s['facts']
            and p.get('qa_gate_passed') is False and p.get('recovery_or_export_performed') is False, 'Proof scope differs')
    require(p['checks']['all36_original_group_source_date_effective_prompt_raw_add_chains'] is True
            and p['checks']['facts_dense768_sparse_and_bm25_exact'] is True
            and p['checks']['existing_entity_dense_and_links_valid'] is True, 'Complete source/index proof required')
    for name, h in p['source_file_hashes'].items():
        require(old.sha(name) == h, 'Failure proof source drift')
    require(_folder_inventory(target['folder']) == p['source_folder_inventory'],
            'Failure proof folder inventory drift')
    require(not (target['folder'] / 'bank.json').exists(), 'Original failed namespace unexpectedly gained a bank')
    return p, ref


def _folder_inventory(folder):
    """Closed bank's exact ordinary-file set; never follow links or junctions."""
    folder = Path(folder).absolute()
    def reject_link(path):
        # Python 3.11 has no Path.is_junction; Windows lstat exposes reparse points.
        attributes = getattr(path.lstat(), 'st_file_attributes', 0)
        require(not path.is_symlink() and not attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                'Source folder symlink or junction prohibited')
    reject_link(folder)
    require(folder.is_dir(), 'Source folder missing')
    result = {}
    for base, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            reject_link(p)
            if name in files:
                require(p.is_file(), 'Non-ordinary source file prohibited')
                if p.name != '.lock':
                    result[str(p)] = old.sha(p)
    return result


def _isolated_overwrite_proof(folder, arm, orphan):
    """Reuse the exact pure algorithm with private path bindings, no monkey-patch."""
    original = old.overwrite_proof
    bindings = dict(original.__globals__)
    bindings.update(folder=lambda ci, chosen_arm: Path(folder), ORPHANS={arm: orphan})
    return FunctionType(original.__code__, bindings, original.__name__,
                        original.__defaults__, original.__closure__)


def _failure_facts(target, proof):
    """Recheck exact native facts and overwrite proof on the current source."""
    n = target['folder'] / 'native'
    js = sorted([(old.read(p), p) for p in (n / 'group_audit').rglob('*.json')], key=lambda x: x[0]['sequence'])
    require(len(js) == target['state']['expected_groups'] and all(j['status'] == 'success' for j, p in js), 'Incomplete journals')
    records = [f for j, p in js for f in j['fact_records']]
    facts = old.points(n / 'qdrant/collection/facts/storage.sqlite')
    entities = old.points(n / 'qdrant/collection/facts_entities/storage.sqlite')
    old.validate_records(records, facts, old.rows(n / 'history.sqlite', 'history'))
    extractor, nlp = old.load_extractor()
    extracted = {f['id']: extractor._extract_entities_from_doc(doc)
                 for f, doc in zip(records, nlp.pipe([x['text'] for x in records], batch_size=32))}
    linked = {mid for e in entities.values() for mid in e.payload['linked_memory_ids']}
    orphans = sorted({mid for mid, es in extracted.items() if es} - linked)
    require(orphans == proof['orphan_fact_ids'] and len(orphans) == 1, 'Orphan identity changed')
    bound_proof = _isolated_overwrite_proof(target['folder'], target['arm'], orphans[0])
    current = bound_proof(target['arm'], records, entities, js, extracted)
    require(current == proof['cause_evidence'], 'Current raw/vector/overwrite proof differs')
    return [dict(f, embedding=facts[f['id']].vector['']) for f in records]


def audit(runs, failure_audits, output):
    """Create a new uniform review. This performs no export and no API calls."""
    output = Path(output).resolve()
    require(not output.exists() and not output.with_suffix('.evidence').exists(), 'New review namespace required')
    old.HASHES.clear()
    with forbid_network() as guards:
        targets = terminal(runs)
        failed = {(t['conversation'], t['arm']) for t in targets if t['state']['status'] == 'failed'}
        require(set(failure_audits) == failed, 'Exactly one explicit proof per failed bank required')
        # Preflight all failures before generating evidence for successful banks.
        proofs = {key: _failure_binding(path, next(t for t in targets if (t['conversation'], t['arm']) == key))
                  for key, path in failure_audits.items()}
        evidence = output.with_suffix('.evidence')
        evidence.mkdir(parents=True)
        reports, all_hashes = {}, dict(old.HASHES)
        for arm in ARMS:
            convs = [t['conversation'] for t in targets if t['arm'] == arm and t['state']['status'] == 'complete']
            if not convs:
                continue
            result = strict.audit(Path(runs[arm]), convs)
            require(result['strict_passed'] is True, 'New anomaly in a strict-complete bank')
            ref = old.save(evidence / f'{arm}_strict.json', {k: v for k, v in result.items() if k != 'content_sha256'})
            reports[arm] = (result, ref)
            all_hashes.update(result['source_file_hashes'])
        rows = []
        for t in targets:
            ci, arm, f, s = t['conversation'], t['arm'], t['folder'], t['state']
            native = f / 'native'; m = t['manifest']
            common = {'conversation': ci, 'arm': arm, 'original_write_status': s['status'],
                'write_status': pointer(f / 'write_status.json'), 'outcome': pointer(f / 'write_outcome.json'),
                'source_manifest': pointer(t['run'] / 'run_manifest.json'), 'group_inputs': pointer(f / 'group_inputs.json'),
                'native_dir': str(native), 'native_inventory': old.native_inventory(native),
                'groups': s['completed_groups'], 'expected_groups': s['expected_groups'], 'facts': s['facts']}
            if s['status'] == 'complete':
                result, ref = reports[arm]
                inspected = next(x for x in result['banks'] if x['conversation'] == ci)
                bank = verify(f / 'bank.json')
                common.update(source_kind='strict_complete', bank=pointer(f / 'bank.json'), proof=None,
                    orphan_fact_ids=[], facts_projection_sha256=digest(bank['facts']),
                    audits={'source_date': [ref], 'index': [ref]}, native_index_limitations=[
                        'Native per-fact support is unverified group scope; partial entity edges are not exhaustively reconstructed.',
                        f"Unused entity sparse missing={len(inspected['unused_entity_bm25_missing_ids'])}; empty={len(inspected['unused_entity_bm25_empty_ids'])}."])
            else:
                proof, ref = proofs[ci, arm]
                projection = _failure_facts(t, proof)
                require(len(projection) == s['facts'], 'Failed source count drift')
                all_hashes.update(proof['source_file_hashes'])
                common.update(source_kind='reviewed_export', bank=None, proof=ref,
                    orphan_fact_ids=proof['orphan_fact_ids'], facts_projection_sha256=digest(projection),
                    audits={'source_date': [ref], 'index': [ref]}, native_index_limitations=[
                        'Original strict writer failed despite complete original facts; only separately reviewed metadata admission is eligible.',
                        'One proved same-canonical stale-payload overwrite remains in the native unrepaired index; no entity boost for its orphan fact.',
                        'Partial entity edges in other facts are not exhaustively reconstructed.'])
            rows.append(common)
        all_hashes.update(old.HASHES)
        _population(rows)
        require(not any(g.call_count for g in guards), 'Network attempted')
        for p, h in all_hashes.items():
            require(old.sha(p) == h, 'Source drift during uniform review')
        value = {'schema': SCHEMA, 'status': STATUS, 'created_utc': old.now(), 'banks': rows,
            'runs': {arm: str(Path(runs[arm]).resolve()) for arm in ARMS},
            'checks': {'all20_terminal': True, 'all20_complete_fact_groups': True,
                       'uniform_source_date_raw_history_required_vectors_reviewed': True,
                       'original_native_unchanged': True, 'no_unknown_failure_admitted': True},
            'native_index_policy': 'original_unrepaired', 'source_file_hashes': all_hashes,
            'implementation_sha256': {name: old.sha(ROOT/name) for name in DEPENDENCIES},
            'review_tool': pointer(Path(__file__)), 'model_calls': 0, 'network_attempts': 0,
            'source_mutations': 0, 'qa_gold_used': False, 'qa_gate_passed': False,
            'limitations': ['Post-failure complete-facts admission is distinct from original strict success.',
                           'This index/source review is not semantic correctness or an online-causal guarantee.']}
        return old.save(output, value)


def _validate_row(row, target):
    ci, arm, folder = target['conversation'], target['arm'], target['folder']
    s, o = target['state'], target['outcome_value']
    require((row['conversation'], row['arm']) == (ci, arm) and row['original_write_status'] == s['status'], 'Review row identity differs')
    for k, p in [('write_status', folder/'write_status.json'), ('outcome', folder/'write_outcome.json'),
                 ('source_manifest', target['run']/'run_manifest.json'), ('group_inputs', folder/'group_inputs.json')]:
        require(row[k] == pointer(p), 'Review original lineage differs')
    require(row['groups'] == row['expected_groups'] == s['completed_groups'] == s['expected_groups']
            and row['facts'] == s['facts'], 'Review count differs')
    require(Path(row['native_dir']).resolve() == (folder/'native').resolve()
            and row['native_inventory'] == old.native_inventory(folder/'native'), 'Native inventory drift')
    require(isinstance(row.get('audits'), dict) and set(row['audits']) == {'source_date', 'index'},
            'Both source_date and index audits required')
    for refs in row['audits'].values():
        require(isinstance(refs, list) and bool(refs), 'Audit evidence missing')
        for ref in refs: _checked(ref)
    require(row['native_index_limitations'], 'Native limitation disclosure required')
    if s['status'] == 'complete':
        require(row['source_kind'] == 'strict_complete' and row['proof'] is None and row['orphan_fact_ids'] == [], 'Strict source kind differs')
        require(row['bank'] == pointer(folder/'bank.json') == s['bank'] == o['bank'], 'Original bank pointer differs')
        bank = verify(folder/'bank.json')
        require(bank['complete'] is True and digest(bank['facts']) == row['facts_projection_sha256'], 'Strict fact projection differs')
    else:
        require(row['source_kind'] == 'reviewed_export' and row['bank'] is None, 'Failure must remain original failed')
        proof, ref = _failure_binding(row['proof']['path'], target)
        require(ref == row['proof'] and row['orphan_fact_ids'] == proof['orphan_fact_ids'], 'Failure evidence differs')


def check_review(path):
    """Static consumer: recheck full20 terminals, source hashes and provenance."""
    with forbid_network():
        review = verify(path)
        require(review.get('schema') == SCHEMA and review.get('status') == STATUS
                and review.get('native_index_policy') == 'original_unrepaired', 'Wrong uniform review')
        require(review['review_tool'] == pointer(Path(__file__)), 'Review implementation drift')
        require(all(review['checks'].get(k) is True for k in ('all20_terminal', 'all20_complete_fact_groups',
                    'uniform_source_date_raw_history_required_vectors_reviewed', 'original_native_unchanged', 'no_unknown_failure_admitted')), 'Incomplete uniform review')
        _population(review['banks'])
        require(set(review.get('implementation_sha256', {})) == set(DEPENDENCIES), 'Review dependency bindings missing')
        for name, expected in review['implementation_sha256'].items():
            require(old.sha(ROOT/name) == expected, 'Review dependency drift')
        targets = terminal(review['runs'])
        for path, h in review['source_file_hashes'].items():
            require(old.sha(path) == h, 'Reviewed source drift')
        for row in review['banks']:
            _validate_row(row, next(t for t in targets if (t['conversation'],t['arm']) == (row['conversation'],row['arm'])))
        return review


def export(review_path, destination):
    """Explicit production action, unused by imports/tests; metadata files only."""
    dest = Path(destination).resolve()
    require(dest.parent == (ROOT/'runs').resolve() and not dest.exists(), 'Fresh direct runs namespace required')
    with forbid_network():
        review = check_review(review_path); ref = pointer(review_path)
        targets = [r for r in review['banks'] if r['source_kind'] == 'reviewed_export']
        require(bool(targets), 'No reviewed failures to export')
        dest.mkdir()
        old.save(dest/'export_intent.json', {'schema': EXPORT_SCHEMA, 'source_review': ref, 'native_index_policy': 'original_unrepaired'})
        exported = []
        for row in targets:
            ci, arm = row['conversation'], row['arm']; n = Path(row['native_dir'])
            js = sorted([(old.read(p),p) for p in (n/'group_audit').rglob('*.json')],key=lambda x:x[0]['sequence'])
            points = old.points(n/'qdrant/collection/facts/storage.sqlite')
            facts = [dict(f,embedding=points[f['id']].vector['']) for j,p in js for f in j['fact_records']]
            require(digest(facts) == row['facts_projection_sha256'], 'Export facts/vector projection changed')
            m = _checked(row['source_manifest'])
            value = {'schema': 'regroup-refinement-reviewed-bank-v2', 'protocol_sha256': m['protocol_sha256'],
                'conversation': ci, 'arm': arm, 'complete': True,
                'complete_scope': 'all_original_groups_and_generated_facts; original_writer_failed',
                'user_id': f'dated_regroup_c{ci}_{arm}', 'facts': facts, 'group_inputs': row['group_inputs'],
                'journal_files': [pointer(p) for j,p in js], 'native_dir': str(n),
                'source_dataset_sha256': m['dataset_sha256'], 'fact_support': 'unverified_group_input_scope',
                'explicit_empty_group_ids': [j['group_id'] for j,p in js if j.get('explicit_empty_extraction')],
                'chunk_embedding_group_ids': [j['group_id'] for j,p in js if j.get('group_context_embedding',{}).get('triggered')],
                'source_kind': 'reviewed_export', 'native_index_policy': 'original_unrepaired',
                'source_inventory': row['native_inventory'], 'source_review': ref,
                'native_index_audit': {'orphan_fact_ids': row['orphan_fact_ids'], 'limitations': row['native_index_limitations']}}
            folder = dest/f'conv_{ci}'/arm; bank = old.save(folder/'bank.json', value)
            record = {'schema': EXPORT_SCHEMA, 'conversation': ci, 'arm': arm, 'source_kind': 'reviewed_export',
                'original_write_status': 'failed', 'reviewed_export_status': 'complete', 'bank': bank,
                'original_write_status_ref': row['write_status'], 'original_outcome_ref': row['outcome'],
                'source_manifest': row['source_manifest'], 'source_review': ref, 'uniform_twenty_bank_review': ref,
                'review_row_sha256': digest(row), 'native_dir': str(n), 'native_inventory': row['native_inventory'],
                'native_index_policy': 'original_unrepaired', 'orphan_fact_ids': row['orphan_fact_ids'],
                'facts': row['facts'], 'groups': row['groups'], 'model_calls': 0, 'source_mutations': 0,
                'native_bytes_copied': 0, 'qa_gate_passed': False, 'export_tool': pointer(Path(__file__))}
            ep = old.save(folder/'reviewed_export.json', record)
            exported.append({'conversation': ci, 'arm': arm, 'bank': bank, 'reviewed_export': ep})
        check_review(review_path)
        return old.save(dest/'export_outcome.json', {'schema': EXPORT_SCHEMA, 'status': 'reviewed_metadata_exports_complete',
            'source_review': ref, 'exports': exported, 'source_mutations': 0, 'native_bytes_copied': 0, 'model_calls': 0})


def check_export_bank(binding_or_dir):
    """Read metadata and hashes only; never opens SQLite or calls a model."""
    with forbid_network():
        if isinstance(binding_or_dir, dict):
            _checked(binding_or_dir); path = Path(binding_or_dir['path'])
        else:
            path = Path(binding_or_dir)
            if path.is_dir(): path = path/'reviewed_export.json'
        record = verify(path)
        require(record.get('schema') == EXPORT_SCHEMA and record.get('source_kind') == 'reviewed_export'
                and record.get('original_write_status') == 'failed' and record.get('reviewed_export_status') == 'complete'
                and record.get('native_index_policy') == 'original_unrepaired', 'Wrong reviewed export')
        require(record['source_review'] == record['uniform_twenty_bank_review'], 'Uniform review binding differs')
        _checked(record['source_review'])
        review = check_review(record['source_review']['path'])
        row = next(r for r in review['banks'] if (r['conversation'],r['arm']) == (record['conversation'],record['arm']))
        require(row['source_kind'] == 'reviewed_export' and digest(row) == record['review_row_sha256'], 'Export row differs')
        require(record['original_write_status_ref'] == row['write_status'] and record['original_outcome_ref'] == row['outcome']
                and record['source_manifest'] == row['source_manifest'] and record['orphan_fact_ids'] == row['orphan_fact_ids'], 'Original failure lineage differs')
        require(record['bank'] == pointer(path.parent/'bank.json') and record['export_tool'] == pointer(Path(__file__)), 'Export file/tool drift')
        bank = verify(path.parent/'bank.json'); m = _checked(row['source_manifest'])
        require(bank['schema'] == 'regroup-refinement-reviewed-bank-v2' and bank['complete'] is True
                and (bank['conversation'],bank['arm']) == (row['conversation'],row['arm'])
                and bank['protocol_sha256'] == m['protocol_sha256'] and bank['source_dataset_sha256'] == m['dataset_sha256'], 'Export bank identity differs')
        require(bank['group_inputs'] == row['group_inputs'] and bank['source_review'] == record['source_review']
                and bank['source_kind'] == 'reviewed_export' and bank['native_index_policy'] == 'original_unrepaired', 'Export source scope differs')
        js = [_checked(p) for p in bank['journal_files']]
        require(len(js) == row['groups'] and [j['sequence'] for j in js] == list(range(row['groups']))
                and all(j['status'] == 'success' for j in js), 'Export incomplete journals')
        require([f for j in js for f in j['fact_records']] == [{k:v for k,v in f.items() if k != 'embedding'} for f in bank['facts']]
                and len(bank['facts']) == row['facts'] and digest(bank['facts']) == row['facts_projection_sha256'], 'Export text/vector projection differs')
        require(bank['native_index_audit']['orphan_fact_ids'] == row['orphan_fact_ids'], 'Native defect erased')
        n = Path(row['native_dir'])
        require(Path(record['native_dir']).resolve() == n.resolve() == Path(bank['native_dir']).resolve()
                and record['native_inventory'] == bank['source_inventory'] == row['native_inventory'] == old.native_inventory(n), 'Native source inventory differs')
        aggregate = verify(path.parent.parent.parent/'export_outcome.json')
        require(aggregate['status'] == 'reviewed_metadata_exports_complete' and aggregate['source_review'] == record['source_review']
                and any(x['reviewed_export'] == pointer(path) and x['bank'] == record['bank'] for x in aggregate['exports']), 'Export aggregate missing')
        return {'bank': bank, 'native': n, 'uniform_row': copy.deepcopy(row), 'export': record, 'source_kind': 'reviewed_export',
            'review': {'bank': record['bank'], 'native_inventory': row['native_inventory'], 'source_kind': 'reviewed_export',
                       'original_write_status': row['write_status'], 'original_outcome': row['outcome'],
                       'reviewed_export': pointer(path), 'source_manifest': row['source_manifest'], 'source_review': record['source_review']}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('audit','check','export','check-export'))
    p.add_argument('--context-run',type=Path); p.add_argument('--jev-run',type=Path)
    p.add_argument('--failure-audit', action='append', default=[], help='CI:ARM:PATH; explicit diagnosed failure only')
    p.add_argument('--review',type=Path); p.add_argument('--destination',type=Path)
    args = p.parse_args()
    if args.stage == 'audit':
        if not all((args.context_run,args.jev_run,args.review)): p.error('audit requires both runs and --review')
        proofs = {}
        for spec in args.failure_audit:
            ci,arm,path = spec.split(':',2); key=(int(ci),arm)
            if key in proofs: p.error('Duplicate failure proof')
            proofs[key] = Path(path)
        result = audit(dict(zip(ARMS,(args.context_run,args.jev_run))),proofs,args.review)
    elif args.stage == 'check': result = {'valid': bool(check_review(args.review))}
    elif args.stage == 'export': result = export(args.review,args.destination)
    else: result = {'valid': bool(check_export_bank(args.destination))}
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
