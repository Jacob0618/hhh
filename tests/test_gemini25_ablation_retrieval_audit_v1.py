"""Offline retrieval provenance tests using native capture fixtures and mock HTTP."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_mem.gemini25_ablation_provider_v1 import Gemini25API
from tools import audit_gemini25_ablation_retrieval_v1 as audit
from tools import prepare_gemini25_ablation_v1 as prep
from tools import prepare_gemini38_hybrid_v1 as native
from tests.test_gemini38_hybrid_v1 import Memory, bank as make_bank


def replace(path, value, *, sealed=True):
    path = Path(path)
    if path.exists():
        path.unlink()
    if sealed:
        return prep.save_new(path, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return prep.pointer(path)


class MockHTTP:
    def post(self, url, **kwargs):
        body = kwargs['json']
        raw = ({'totalTokens': 500} if url.endswith(':countTokens') else
               {'embeddings': [{'values': [1.] + [0.] * 767} for _ in body['requests']],
                'usageMetadata': {'promptTokenCount': 7, 'totalTokenCount': 7}})
        return type('Response', (), {'ok': True, 'status_code': 200, 'headers': {}, 'json': lambda self: raw})()


class Fixture:
    def __init__(self, root, *, question='Alice likes hiking?', duplicate=False, entities=True):
        self.run = Path(root).resolve()
        self.folder = self.run / 'conv_0'
        source = self.run / 'writer_native'
        copy = self.folder / 'native'
        for folder in (source, copy):
            folder.mkdir(parents=True)
            (folder / 'history.sqlite').write_bytes(b'closed fixture database')
        memory = Memory()
        bank = make_bank(memory)
        bank['native_dir'] = str(source)
        bank['schema'] = 'gemini25-ablation-session-writer-v1'
        self.bank_ref = prep.save_new(self.run / 'writer_bank.json', bank)
        self.bank = prep.Reader().bound(self.bank_ref)
        self.item = {'conversation': 0, 'bank': self.bank_ref, 'native_source': str(source),
                     'native_copy': str(copy), 'native_inventory': audit.inventory(source)}
        self.manifest = {'protocol_sha256': 'new25'}
        self.queries = [{'id': 'c0_q0', 'question': question}]
        if duplicate:
            self.queries.append({'id': 'c0_q1', 'question': question})
        self.batches = [[question.strip()]] + ([['Alice']] if entities else [])
        prep.save_new(self.folder / 'retrieval_intent.json', {'protocol_sha256': 'new25'})
        prep.save_new(self.folder / 'post_retrieval.json', {'status': 'complete', 'source_unchanged': True,
                      'protocol_sha256': 'new25', 'native_inventory': audit.inventory(copy)})
        replace(self.folder / 'retrieval_status.json', {'status': 'complete', 'conversation': 0,
                'completed': len(self.queries), 'expected': len(self.queries), 'finished_utc': '2026-10-07T00:00:00Z'}, sealed=False)
        self.api_dir = self.folder / 'retrieval_api'
        api = Gemini25API(self.api_dir, keys={'GEMINI_API_KEY': 'offline-placeholder'}, retries=2,
                          gate_path=self.run / 'gate.sqlite', context={'conversation': 0, 'phase': 'retrieval'})
        with patch('requests.Session', side_effect=MockHTTP), patch('jev_mem.paced_embedding_api.SharedEmbeddingGate.acquire', return_value={}):
            for q in self.queries:
                scoped = api.scoped(query_id=q['id'])
                for batch in self.batches:
                    scoped.embed(batch, query=True, stage=audit.STAGE)
                captured = native.capture_current(memory, q['question'], self.bank, self.bank_ref, _scorer_module=memory.module)
                prep.save_new(self.folder / 'retrieval' / (q['id'] + '.json'), {
                    'conversation': 0, 'query_id': q['id'], 'question': q['question'], 'bank': self.bank_ref,
                    'protocol_sha256': 'new25', 'status': 'complete', 'capture': captured, 'api_events': scoped.events})
        self.memory = memory

    def audit(self):
        def validate(cap, facts, user_id):
            return native.validate_native_capture(cap, facts, user_id, _scorer_module=self.memory.module)
        with patch.object(audit, 'query_batches', return_value=self.batches), patch.object(audit, 'validate_native_capture', side_effect=validate):
            return audit.audit_retrieval(prep.Reader(), self.run, self.manifest, self.item, self.bank, self.queries)

    def row(self, index=0):
        path = self.folder / 'retrieval' / (self.queries[index]['id'] + '.json')
        return path, prep.Reader().read(path)


class RetrievalAuditTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.create_connection', 'socket.socket.connect', 'requests.Session.request', 'jev_mem.providers.load_credentials'):
            guard = patch(target, side_effect=AssertionError('No network or credentials'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_new_raw_query_entity_and_same_conversation_exact_cache(self):
        with tempfile.TemporaryDirectory() as t:
            fixture = Fixture(t, duplicate=True)
            before = (fixture.api_dir / 'api_calls.jsonl').read_bytes()
            self.assertTrue(fixture.audit())
            self.assertEqual((fixture.api_dir / 'api_calls.jsonl').read_bytes(), before)
            rows = [json.loads(line) for line in before.splitlines()]
            self.assertEqual(sum(r.get('cache_hit') is True for r in rows), 2)

    def test_count_tokens_replayed_for_long_query_without_dispatch(self):
        with tempfile.TemporaryDirectory() as t:
            fixture = Fixture(t, question='Alice ' + 'long ' * 400, entities=False)
            self.assertTrue(fixture.audit())
            _, row = fixture.row()
            self.assertEqual([r['operation'] for r in row['api_events']], ['countTokens', 'batchEmbedContents'])

    def test_wrong_query_search_capture_is_rejected_even_when_resealed(self):
        with tempfile.TemporaryDirectory() as t:
            fixture = Fixture(t)
            path, row = fixture.row()
            row['capture']['native_search_request']['query'] = 'Different question'
            replace(path, row)
            with self.assertRaisesRegex(ValueError, 'Captured native search'):
                fixture.audit()

    def test_old_embedding_lineage_and_omitted_entity_event_rejected(self):
        for kind in ('lineage', 'missing_entity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
                fixture = Fixture(t)
                path, row = fixture.row()
                if kind == 'lineage':
                    row['api_events'][0]['source'] = 'inherited_exact_request'
                else:
                    row['api_events'].pop()
                replace(path, row)
                with self.assertRaises(ValueError):
                    fixture.audit()

    def test_foreign_unlogged_intent_and_missing_producer_rejected(self):
        for kind in ('extra_intent', 'no_producer'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
                fixture = Fixture(t)
                if kind == 'extra_intent':
                    replace(fixture.api_dir / 'request_intents/foreign.json', {}, sealed=False)
                else:
                    (fixture.api_dir / 'api_calls.jsonl').write_text('')
                with self.assertRaises(ValueError):
                    fixture.audit()

    def test_resealed_nonfinite_vector_and_raw_usage_mismatch_rejected(self):
        for kind in ('bad_vector', 'wrong_usage'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
                fixture = Fixture(t)
                path, row = fixture.row()
                event = row['api_events'][0]
                rawpath = Path(event['cache_path'])
                raw = json.loads(rawpath.read_text())
                if kind == 'bad_vector':
                    raw['response']['embeddings'][0]['values'][0] = True
                else:
                    raw['response']['usageMetadata']['promptTokenCount'] = 999
                replace(rawpath, raw, sealed=False)
                event['cache_sha256'] = prep.file_sha(rawpath)
                replace(path, row)
                with self.assertRaises(ValueError):
                    fixture.audit()

    def test_native_source_copy_and_post_protocol_binding(self):
        for kind in ('source', 'copy', 'post_protocol'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
                fixture = Fixture(t)
                if kind == 'post_protocol':
                    p = fixture.folder / 'post_retrieval.json'
                    row = prep.Reader().read(p)
                    row['protocol_sha256'] = 'old38'
                    replace(p, row)
                else:
                    name = 'native_source' if kind == 'source' else 'native_copy'
                    (Path(fixture.item[name]) / 'history.sqlite').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    fixture.audit()


if __name__ == '__main__':
    unittest.main()
