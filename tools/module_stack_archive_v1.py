"""Offline import of frozen retrieval inputs; never a deleted HTTP-cache replay."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from jev_mem.evidence_selection_r03 import choose_baseline, choose_cheap, choose_jev, digest
from jev_mem.source_append_r05 import source_turns

ARCHIVE_SHA256 = 'b7466b44fe60174dfc6cc5acb2f8851f238afb6905fdc4941b1911da7fd57eaa'
COHORT_SIZE = 1540
CONVERSATIONS = tuple(range(10))
ARMS = ('pool_old_bm25', 'pool_old_jev', 'pool_new_bm25', 'pool_new_jev')
IDENTITY = ('question', 'reference', 'reference_date', 'category', 'conversation')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fact(raw):
    return {k: raw.get(k) for k in ('id', 'text', 'observation_date')}


def _ids(values, count, context):
    _require(isinstance(values, list) and len(values) == count
             and all(isinstance(x, str) and x for x in values)
             and len(set(values)) == count, context + ': unique IDs required')


class _Archive:
    def __init__(self, z):
        self.z = z
        self.members = {}

    def raw(self, name):
        raw = self.z.read(name)  # Reading to EOF verifies the member CRC.
        self.members[name] = _sha(raw)
        return raw

    def doc(self, name):
        return json.loads(self.raw(name))

    def lines(self, name):
        h = hashlib.sha256()
        with self.z.open(name) as stream:
            for line, raw in enumerate(stream, 1):
                h.update(raw)
                _require(bool(raw.strip()), name + ': empty JSONL row')
                yield line, _sha(raw), json.loads(raw)
        self.members[name] = h.hexdigest()


def _check_sources(a, sources):
    _require(set(sources) == set(CONVERSATIONS), 'Exact conversation sources required')
    canonical = {}
    for ci in CONVERSATIONS:
        src = sources[ci]
        bank = a.doc(f'facts/conv_{ci}.json')
        _require(bank['conversation'] == ci and bank['complete'] is True,
                 'Archived bank identity/completeness differs')
        facts = bank['facts']
        _ids([f['id'] for f in facts], len(facts), 'Archived bank')
        archived = {f['id']: f for f in facts}
        live = src['rawfacts']
        _require(set(archived) == set(live), 'Archive/current bank ID sets differ')
        for fid, f in archived.items():
            _require(_fact(f) == _fact(live[fid]), 'Archive/current fact text/date differs')
            _require(f.get('input_scope') == live[fid].get('input_scope'),
                     'Archive/current fact scope differs')
        groups = a.doc(f'facts/conv_{ci}_group_inputs.json')
        _require(groups['conversation'] == ci and groups['arm'] == 'session',
                 'Archive group identity differs')
        projected = source_turns(groups, src.get('group_ref', {'archive_member': ci}))
        fields = ('source_turn_id', 'session_id', 'observation_date', 'session_index',
                  'turn_index', 'global_index', 'view', 'projection', 'json_pointer', 'content_sha256')
        _require([{k: t[k] for k in fields} for t in projected]
                 == [{k: t[k] for k in fields} for t in src['turns']],
                 'Archive/current source projection differs')
        byid = {t['source_turn_id']: t for t in projected}
        for f in facts:
            scope = f.get('input_scope')
            if scope is None:
                continue
            ids = scope.get('source_turn_ids', [])
            _ids(ids, len(ids), 'Fact source scope')
            _require(all(t in byid and byid[t]['session_id'] == scope.get('session_id') for t in ids),
                     'Fact scope crosses a session or references an absent turn')
        canonical[ci] = {fid: _fact(f) for fid, f in live.items()}
    return canonical


def _records(ids, ci, banks):
    _require(all(fid in banks[ci] for fid in ids), 'Foreign/absent bank fact')
    return [deepcopy(banks[ci][fid]) for fid in ids]


def _old_pool(row, baseline, banks):
    q = row['query_id']
    _require(q in baseline and row['conversation'] == baseline[q]['conversation'], 'Pool identity differs')
    pool = row['pool']; ids = pool['pool_ids']; base = baseline[q]['selected_ids']
    _ids(ids, 80, 'Original pool')
    _require(ids[:20] == base == pool['baseline_ids'], 'Original pool baseline prefix differs')
    native = pool['original_native80_ids']
    _ids(native, 80, 'Original native80')
    others = [fid for fid in native if fid not in set(base)][:40]
    _require(len(others) == 40 and pool['native_other_ids'] == others
             and ids[:60] == base + others, 'Protected original60 differs')
    _records(ids + native, row['conversation'], banks)
    return pool


def _new_pool(result, old, ci, banks):
    _require(result.get('discovery_status') == 'complete', 'Pinned archive discovery not complete')
    discoveries = result['discoveries']
    discovered = [d['id'] for d in discoveries]
    _ids(discovered, len(discovered), 'Discovery IDs')
    _require(len(discovered) <= 20 and not set(discovered) & set(old['original_native80_ids']),
             'Discovery quota/native exclusion differs')
    _records(discovered, ci, banks)
    ids = list(old['pool_ids'][:60])
    for fid in discovered + old['pool_ids']:
        if fid not in ids and len(ids) < 80:
            ids.append(fid)
    _ids(ids, 80, 'Discovered pool')
    _require(result['pool']['pool_ids'] == ids, 'R08 discovery/fill order differs')
    return ids


def load_archive(path, baseline_jobs: list, sources: dict) -> dict:
    """Validate the pinned ZIP and return only four frozen selected-input arms.

    baseline_jobs/sources must already come from the live R05 baseline audit.
    Old QA is checked for identity only, never imported as a cache producer.
    """
    path = Path(path).resolve()
    with path.open('rb') as stream:
        actual_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    _require(actual_sha == ARCHIVE_SHA256, 'Pinned review ZIP SHA differs')
    archive_ref = {'path': str(path), 'sha256': actual_sha}
    _require(len(baseline_jobs) == COHORT_SIZE, 'Wrong baseline cohort count')
    baseline = {j['query_id']: j for j in baseline_jobs}
    _require(len(baseline) == COHORT_SIZE, 'Duplicate baseline query')
    arms = {name: {} for name in ARMS}
    audit = {'schema': 'module-stack-archive-input-audit-v1', 'queries_per_arm': COHORT_SIZE,
             'conversations': len(CONVERSATIONS), 'model_calls': 0,
             'old_http_cache_replayed': False, 'old_qa_results_imported': False,
             'score_replay_semantics': 'archived numeric decision replay only; no HTTP/ledger claim'}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        _require(len(names) == len(set(names)), 'Duplicate ZIP member')
        _require(z.testzip() is None, 'ZIP CRC failure')
        a = _Archive(z)
        index = a.doc('archive_index.json')
        # Only entries explicitly preserved as original bytes have this guarantee.
        for entry in index.values():
            if entry.get('preserved_original_bytes'):
                _require(_sha(a.raw(entry['entry'])) == entry['sha256'], 'Original-byte archive member differs')
        a.raw('README.txt')
        banks = _check_sources(a, sources)
        for q, j in baseline.items():
            ci = j['conversation']; _require(ci in banks, 'Unknown baseline conversation')
            _ids(j['selected_ids'], 20, 'Baseline selected20')
            _require(j['selected_facts'] == _records(j['selected_ids'], ci, banks), 'Baseline canonical records differ')
        pools = {}
        for _, _, row in a.lines('shared/original_candidate_pools.jsonl'):
            q = row['query_id']; _require(q not in pools, 'Duplicate archived pool query')
            pools[q] = _old_pool(row, baseline, banks)
        _require(set(pools) == set(baseline), 'Archived pools cohort differs')
        for rn, prefix in (('R03', 'pool_old'), ('R08', 'pool_new')):
            a.doc(rn + '/protocol.json'); a.doc(rn + '/scientific_protocol.json')
            meta = a.doc(rn + '/source_metadata.json')
            _require(meta['questions_per_arm'] == COHORT_SIZE and meta['status'] == 'sealed', 'Unsealed archive source')
            decisions = {}; pool_by_q = {}
            member = rn + '/retrieval_or_choice.jsonl'
            for line, line_sha, row in a.lines(member):
                result = row['result']; q = result['query_id']; method = result['method']
                key = (method, q)
                _require(q in baseline and method in ('baseline', 'cheap', 'jev') and key not in decisions,
                         'Unexpected/duplicate archive selection')
                _require(row['job_id'] == method + '/' + q and result['conversation'] == baseline[q]['conversation'],
                         'Archive selection identity differs')
                ci = result['conversation']; old = pools[q]
                ids = old['pool_ids'] if rn == 'R03' else _new_pool(result, old, ci, banks)
                _require(q not in pool_by_q or pool_by_q[q] == ids, 'Arms do not share the same candidate pool')
                pool_by_q[q] = ids
                if rn == 'R03':
                    _require(result['pool_sha256'] == digest(old), 'R03 pool binding differs')
                records = _records(ids, ci, banks)
                decision = result['decision']
                if method == 'baseline':
                    expected = choose_baseline(records)
                elif method == 'cheap':
                    expected = choose_cheap(baseline[q]['question'], baseline[q]['reference_date'], records)
                else:
                    scores = result.get('raw_scores') if rn == 'R03' else result['preserved_score_response'].get('scores')
                    _require(decision['score_status'] != 'unknown', 'Archived unknown selection is not admissible')
                    expected = choose_jev(records, scores, score_status=decision['score_status'])
                _require(decision == expected, 'Archived selection does not replay from its frozen pool/scores')
                selected_ids = expected['selected_ids']; selected = _records(selected_ids, ci, banks)
                _require(result['selected_ids'] == selected_ids and result['selected_facts'] == selected
                         and result['selected_sha256'] == digest(selected)
                         and result['baseline_selected_sha256'] == digest(baseline[q]['selected_facts']),
                         'Archived selected records/digests differ')
                decisions[key] = {'selected_facts': selected, 'selected_ids': selected_ids,
                    'selected_sha256': digest(selected), 'pool_ids': deepcopy(ids),
                    'archived_input': {'zip': deepcopy(archive_ref), 'member': member, 'line': line,
                        'exported_line_sha256': line_sha, 'historical_original_ref': deepcopy(row['original_ref'])},
                    'selection_provenance': {'round': rn, 'method': method,
                        'score_status': decision['score_status'], 'reason': decision['reason'],
                        'pool_sha256': digest(ids), 'interpretation': audit['score_replay_semantics']}}
            expected_keys = {(m, q) for m in ('baseline', 'cheap', 'jev') for q in baseline}
            _require(set(decisions) == expected_keys, 'Archived selection cohort differs')
            seen = set(); qa_member = rn + '/qa_and_selected20.jsonl'
            for line, line_sha, row in a.lines(qa_member):
                q = row['query_id']; method = row['method']; key = (method, q)
                _require(key in expected_keys and key not in seen, 'Unexpected/duplicate archived QA identity')
                seen.add(key)
                _require(row['job_id'] == method + '/' + q
                         and all(row[k] == baseline[q][k] for k in IDENTITY), 'Archived QA question identity differs')
                rec = decisions[key]
                _require(row['selected_facts'] == rec['selected_facts'], 'QA/selection input mismatch')
                if method == 'baseline':
                    _require(rec['selected_facts'] == baseline[q]['selected_facts'], 'Archive/native baseline mismatch')
                else:
                    rec['archived_input']['qa_input_binding'] = {'member': qa_member, 'line': line,
                        'exported_line_sha256': line_sha}
                    arms[prefix + ('_bm25' if method == 'cheap' else '_jev')][q] = rec
            _require(seen == expected_keys, 'Archived QA cohort differs')
        members = dict(a.members)
    audit.update(validated_selected_inputs=sum(map(len, arms.values())),
                 member_count=len(members), all_input_invariants_passed=True)
    return {'archive_ref': archive_ref, 'members': members, 'arms': arms, 'audit': audit}
