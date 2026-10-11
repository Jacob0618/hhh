"""Offline exact-proof tests; synthetic raw producers, no transport clients."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r03_r05_reuse_proofs_v1 as proof


def job(method='baseline', qid='c0_q0', text='Paris'):
    facts = [{'id': f'f{i}', 'text': text + str(i), 'observation_date': '2023-01-01'} for i in range(20)]
    return {'job_id': method + '/' + qid, 'query_id': qid, 'conversation': 0, 'method': method,
            'question': 'Where?', 'reference': 'Paris', 'category': 1, 'reference_date': '2023-02-01',
            'selected_ids': [f['id'] for f in facts], 'selected_facts': facts, 'status': 'ready'}


def write(path, value, seal=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if seal:
        value = {**value, 'content_sha256': proof.digest(value)}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'path': str(path.resolve()), 'sha256': proof._file_sha(path)}


class RawFixture:
    def __init__(self, root, *, label='[[no]]', method='baseline', answer='Paris'):
        self.root = Path(root).resolve()
        self.j = job(method)
        self.m = {'protocol_sha256': 'protocol', 'phase_kind': 'firstpass', 'phase_id': 'fixture',
                  'source_manifest': {'path': 'source', 'sha256': 'sourcehash'}}
        self.paths = proof.prior.stage_files(self.root, self.j)
        self.api = self.root / 'conv_0' / 'api'
        self.rows = []
        self.records = {}
        self.signatures = {}
        self.official = lambda answer, reference, category: float(answer == reference)
        for stage, text in (('reader', answer), ('judge', label)):
            sig = proof.prior.request_signature(self.j, stage, answer)
            self.signatures[stage] = sig
            h = proof.request_hash(sig)
            ctx = proof.prior.stage_context(self.j, self.m, stage)
            modelstage = proof.prior.evaluator.READER_STAGE if stage == 'reader' else proof.prior.evaluator.JUDGE_STAGE
            response = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]}}],
                        'modelVersion': sig['model'], 'usageMetadata': {'totalTokenCount': 11}}
            rawref = write(self.api / 'api_cache' / (h + '.json'), {'request': sig, 'response': response})
            write(self.api / 'request_intents' / (h + '.json'),
                  {'request_hash': h, 'signature': sig, 'context': ctx, 'stage': modelstage, 'retries': 2})
            ledger = {k: sig[k] for k in ('provider', 'model', 'operation')}
            ledger.update(request_hash=h, context=ctx, stage=modelstage, attempt=1, status=200,
                          timestamp_utc='2026-10-06T00:00:00Z', resolved_model=sig['model'], usage=response['usageMetadata'])
            self.rows.append(ledger)
            ident = {**proof.prior.identity(self.j, self.m), 'stage': stage}
            if stage == 'judge':
                ident['reader_sha256'] = proof._file_sha(self.paths['reader'])
            cap = proof.prior.SignatureCapture(text)
            result = (proof.prior.reader_call(cap, self.j, self.official) if stage == 'reader' else
                      proof.prior.evaluator.judge_answer(cap, self.j['question'], self.j['reference'], answer))
            event = {k: sig[k] for k in ('provider', 'model', 'operation')}
            event.update(request_hash=h, context=ctx, stage=modelstage, source='new_request',
                         cache_path=rawref['path'], cache_sha256=rawref['sha256'], error_type=None)
            write(self.paths[stage].with_suffix('.intent.json'), ident)
            write(self.paths[stage], {**ident, 'status': 'complete', 'result': result,
                                     'error': None, 'api_events': [event], 'raw_invalid_label': None}, seal=True)
            self.records[stage] = json.loads(self.paths[stage].read_text(encoding='utf-8'))
        self.record = {'job': self.j, 'paths': {k: str(v) for k, v in self.paths.items()}, **self.records}

    def index(self):
        out = {}
        for row in self.rows:
            key = row['request_hash'], row['stage'], proof.fingerprint(row['context'])
            out.setdefault(key, []).append(row)
        return out

    def replay(self, stage, *, snapshot=None, indexed=None):
        context = proof.prior.stage_context(self.j, self.m, stage)
        expected = {**proof.prior.identity(self.j, self.m), 'stage': stage}
        if stage == 'reader':
            callback = lambda api: proof.prior.reader_call(api, self.j, self.official)
        else:
            expected['reader_sha256'] = proof._file_sha(self.paths['reader'])
            callback = lambda api: proof.prior.evaluator.judge_answer(api, self.j['question'], self.j['reference'],
                                        self.records['reader']['result']['prediction']['answer'])
        return proof._projection(snapshot or proof.Snapshot(), self.paths[stage], expected, self.api, context,
                                 self.index() if indexed is None else indexed, callback)

    def mini_proof(self, consumers=None):
        consumers = consumers or [self.j]
        entries = {proof._scope(self.j, s, self.signatures[s]): proof._entry(self.j, self.record, s, self.signatures[s])
                   for s in ('reader', 'judge')}
        return {'schema': proof.SCHEMA, 'model_calls_during_proof': 0, 'entries': entries,
                'consumer_jobs': {j['job_id']: proof._consumer(j) for j in consumers},
                'reader_matches': {j['job_id']: proof._scope(j, 'reader', proof.prior.request_signature(j, 'reader')) for j in consumers}}


class OriginalReplayTests(unittest.TestCase):
    def test_real_reader_and_no_judge_replay_without_api(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d)
            with patch.object(proof.prior.Gemini38API, '__init__', side_effect=AssertionError('No API construction')):
                self.assertEqual(f.replay('reader')['result'], f.records['reader']['result'])
                self.assertFalse(f.replay('judge')['result']['correct'])
                self.assertEqual(proof._inventory(proof.Snapshot(), f.api, f.rows)['unknown_intent_count'], 0)

    def test_altered_raw_response_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d)
            p = f.api / 'api_cache' / (proof.request_hash(f.signatures['reader']) + '.json')
            raw = json.loads(p.read_text(encoding='utf-8'))
            raw['response']['candidates'][0]['content']['parts'][0]['text'] = 'Different'
            write(p, raw)
            with self.assertRaises(ValueError):
                f.replay('reader')

    def test_same_request_other_context_producer_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d)
            stage = f.records['reader']
            producer = {**f.rows[0], 'context': {**f.rows[0]['context'], 'method': 'pool_old_jev'}}
            intent_path = f.api / 'request_intents' / (f.rows[0]['request_hash'] + '.json')
            intent = json.loads(intent_path.read_text(encoding='utf-8'))
            intent['context'] = producer['context']; write(intent_path, intent)
            consumer = {k: v for k, v in f.rows[0].items() if k not in ('attempt', 'status')}
            consumer.update(cache_hit=True, usage_semantics='reused_response_no_new_model_usage')
            f.rows = [producer, consumer, f.rows[1]]
            stage['api_events'][0]['source'] = 'existing_run_cache'
            stage.pop('content_sha256'); write(f.paths['reader'], stage, seal=True)
            self.assertEqual(f.replay('reader')['status'], 'complete')

    def test_context_extra_dispatch_not_hidden_by_projection(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d)
            f.rows.append({**f.rows[0], 'request_hash': 'undeclared'})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                f.replay('reader')

    def test_missing_or_orphan_intent_and_unended_producer_rejected(self):
        for mutation in ('missing', 'orphan', 'tail', 'signature'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                f = RawFixture(d)
                ip = f.api / 'request_intents' / (f.rows[0]['request_hash'] + '.json')
                if mutation == 'missing':
                    ip.unlink()
                elif mutation == 'orphan':
                    write(ip.with_name('orphan.json'), {})
                elif mutation == 'tail':
                    f.rows[0]['status'] = 'transport_error'
                else:
                    value = json.loads(ip.read_text(encoding='utf-8'))
                    value['signature']['model'] = 'changed'; write(ip, value)
                with self.assertRaises(ValueError):
                    proof._inventory(proof.Snapshot(), f.api, f.rows)

    def test_request_hash_preserves_original_spaces(self):
        signature = proof.prior.request_signature(job(), 'reader')
        self.assertNotEqual(proof.digest(signature), proof.request_hash(signature))


class LookupTests(unittest.TestCase):
    def test_reader_exact_and_durable_actual_answer_no_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d)
            p = f.mini_proof()
            self.assertEqual(proof.lookup_reader(p, f.j)['reader_result'], f.records['reader']['result'])
            ref = {'path': str(f.paths['reader']), 'sha256': proof._file_sha(f.paths['reader'])}
            self.assertFalse(proof.lookup_judge(p, f.j, ref)['judge_result']['correct'])
            with self.assertRaises((TypeError, KeyError)):
                proof.lookup_judge(p, f.j, 'Paris')
            reader = deepcopy(f.records['reader']); reader.pop('content_sha256')
            reader['result']['prediction']['answer'] = 'A new actual answer'
            ref = write(f.paths['reader'], reader, seal=True)
            self.assertIsNone(proof.lookup_judge(p, f.j, ref))

    def test_foreign_reader_changed_reference_or_order_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d); p = f.mini_proof()
            for field in ('reference', 'reference_date', 'selected_facts'):
                j = deepcopy(f.j)
                if field == 'selected_facts':
                    j[field].reverse(); j['selected_ids'].reverse()
                else:
                    j[field] = 'changed'
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Consumer job'):
                    proof.lookup_reader(p, j)
            reader = deepcopy(f.records['reader']); reader.pop('content_sha256'); reader['job_id'] = 'other/c0_q0'
            ref = write(f.paths['reader'], reader, seal=True)
            with self.assertRaisesRegex(ValueError, 'actual complete durable Reader'):
                proof.lookup_judge(p, f.j, ref)

    def test_proof_signature_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d); p = f.mini_proof()
            p['entries'][p['reader_matches'][f.j['job_id']]]['signatures']['reader']['model'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'Exact inherited request'):
                proof.lookup_reader(p, f.j)

    def test_new_arm_name_allowed_but_duplicate_missing_and_changed_identity_rejected(self):
        a = job(); originals = {a['query_id']: a}; b = job('new_unfrozen_name')
        proof._validate_consumers([a, b], originals)
        with self.assertRaisesRegex(ValueError, 'Duplicate job'):
            proof._validate_consumers([a, a], originals)
        with self.assertRaisesRegex(ValueError, 'cohort differs'):
            proof._validate_consumers([a], {**originals, 'c0_q1': job(qid='c0_q1')})
        b['reference'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'QA identity changed'):
            proof._validate_consumers([a, b], originals)

    def test_missing_archive_score_only_source_not_a_proof(self):
        with self.assertRaisesRegex(ValueError, 'exactly four'):
            proof.build_reuse_proofs({'jobs': [job()], 'archived_score': .9})

    def test_builder_precedence_is_frozen_arm_order_not_label_or_input_order(self):
        with tempfile.TemporaryDirectory() as d:
            baseline = RawFixture(Path(d) / 'base', label='[[no]]')
            other = RawFixture(Path(d) / 'other', label='[[yes]]', method='pool_old_jev')
            jobs = [other.j, baseline.j]
            source = {'jobs': [job('refined')], 'prior_qa_reuse': {}, 'official_metric_provenance': {}}
            phase = {'records': {baseline.j['job_id']: baseline.record, other.j['job_id']: other.record}, 'ledger': []}
            snapshot = proof.Snapshot()
            with patch.object(proof, '_load_inputs', return_value=(snapshot, Path(d), {}, {'jobs': jobs})), \
                 patch.object(proof, '_audit', return_value=phase), \
                 patch.object(proof, '_producer_inventory', return_value=[]), \
                 patch.object(proof.prior, 'proof_for', return_value={'inherited_physical_usage': {}}), \
                 patch.object(proof, 'CODE', ()):
                result = proof.build_reuse_proofs(source)
            reader = proof.lookup_reader(result, source['jobs'][0])
            self.assertEqual(reader['source_job']['method'], 'baseline')
            key = proof._scope(baseline.j, 'judge', baseline.signatures['judge'])
            self.assertFalse(result['entries'][key]['judge_result']['correct'])
            self.assertNotIn('selected_facts', result['entries'][key]['source_job'])
            self.assertEqual(result['historical_reader_judge_stages_replayed'], 4)

    def test_revalidation_rejects_new_orphan_not_in_old_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            source = {'jobs': [job()], 'prior_qa_reuse': {'run_manifest': {'path': str(root / 'run_manifest.json')}}}
            p = {'schema': proof.SCHEMA, 'prior_qa_reuse': source['prior_qa_reuse'],
                 'model_calls_during_proof': 0, 'not_independent_samples': True, 'proof_code_sha256': {},
                 'consumer_jobs': {source['jobs'][0]['job_id']: proof._consumer(source['jobs'][0])},
                 'input_sha256': {}, 'producer_inventory': {'run': str(root), 'conversations': list(range(10)), 'files': []}}
            write(root / 'conv_0/api/request_intents/orphan.json', {})
            with patch.object(proof, 'CODE', ()), self.assertRaisesRegex(ValueError, 'inventory changed'):
                proof.revalidate(p, source)


if __name__ == '__main__':
    unittest.main()
