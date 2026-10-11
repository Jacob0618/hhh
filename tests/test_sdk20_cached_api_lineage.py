"""Offline crash/provenance tests; no real credentials or model calls."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_mem import sdk20_cached_api as cache_api
from jev_mem.sdk20_cached_api import CachedLegacyAPI, digest

KEYS = {'GEMINI_API_KEY': 'unit-test-not-a-key', 'TYPESAFE_API_KEY': 'unit-test-not-a-key'}
MODEL = 'gemini-2.5-flash'
BODY = {'contents': [{'role': 'user', 'parts': [{'text': 'Question?'}]}],
        'generationConfig': {'temperature': 0, 'maxOutputTokens': 2048,
                             'thinkingConfig': {'thinkingBudget': 0},
                             'responseMimeType': 'application/json'}}
SIGNATURE = {'provider': 'gemini', 'model': MODEL, 'operation': 'generateContent', 'body': BODY}
KEY = digest(SIGNATURE)


def response(text='{"answer":"first"}'):
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]}}]}


def write_source(root, value=None):
    path = root / (KEY + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'request': SIGNATURE, 'response': value if value is not None else response()},
                               ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


class CacheLineageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = self.root / 'first'
        self.later = self.root / 'later'
        self.api = CachedLegacyAPI(self.root / 'run', keys=KEYS, cache_roots=[self.first, self.later])
        self.target = self.api.cache_dir / (KEY + '.json')
        self.lineage = self.api.run_dir / 'cache_lineage' / (KEY + '.json')
        self.intent = self.api.run_dir / 'request_intents' / (KEY + '.json')
        # Any accidental fall-through to HTTP is a failing test.
        http = patch('requests.Session.post', side_effect=AssertionError('must not dispatch'))
        self.http = http.start()
        self.addCleanup(http.stop)

    def request(self):
        return self.api.request('gemini', MODEL, copy.deepcopy(BODY), stage='lineage_test')

    def write_intent(self, *, signature=SIGNATURE):
        self.intent.parent.mkdir(parents=True, exist_ok=True)
        self.intent.write_text(json.dumps({'request_hash': KEY, 'signature': signature}), encoding='utf-8')

    def interrupted_import(self):
        original = cache_api._publish_bytes

        def fail_before_raw(path, raw):
            if Path(path) == self.target:
                raise OSError('simulated interruption before raw publication')
            return original(path, raw)

        with patch.object(cache_api, '_publish_bytes', side_effect=fail_before_raw):
            with self.assertRaisesRegex(OSError, 'simulated interruption'):
                self.request()
        self.assertTrue(self.lineage.exists())
        self.assertFalse(self.target.exists())
        self.assertFalse(self.intent.exists())

    def test_lineage_is_published_first_and_exact_bytes_survive_reuse(self):
        source = write_source(self.first)
        published = []
        original = cache_api._publish_bytes

        def observe(path, raw):
            path = Path(path)
            if path == self.target:
                self.assertTrue(self.lineage.exists())
            published.append(path)
            return original(path, raw)

        with patch.object(cache_api, '_publish_bytes', side_effect=observe):
            self.assertEqual(self.request(), response())
        self.assertEqual(published, [self.lineage, self.target])
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        lineage = json.loads(self.lineage.read_text())
        self.assertEqual(lineage['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(lineage['target_sha256'], lineage['source_sha256'])
        self.assertEqual(self.request(), response())
        self.assertEqual(self.http.call_count, 0)

    def test_interrupted_import_resumes_the_pinned_source_only(self):
        source = write_source(self.first)
        self.interrupted_import()
        frozen_lineage = self.lineage.read_bytes()
        write_source(self.later, response('{"answer":"later"}'))
        self.assertEqual(self.request(), response())
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        self.assertEqual(self.lineage.read_bytes(), frozen_lineage)
        self.assertEqual(self.http.call_count, 0)

    def test_interrupted_changed_source_fails_without_trying_later_answer(self):
        write_source(self.first)
        self.interrupted_import()
        write_source(self.first, response('{"answer":"changed"}'))
        write_source(self.later, response('{"answer":"later valid"}'))
        with self.assertRaisesRegex(ValueError, 'source hash differs'):
            self.request()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.http.call_count, 0)

    def test_orphan_target_is_rejected_even_with_usable_source(self):
        write_source(self.api.cache_dir)
        write_source(self.first)
        with self.assertRaisesRegex(ValueError, 'Orphan current run cache'):
            self.request()
        self.assertFalse(self.lineage.exists())
        self.assertEqual(self.http.call_count, 0)

    def test_modified_imported_response_with_same_request_is_rejected(self):
        write_source(self.first)
        self.request()
        write_source(self.api.cache_dir, response('{"answer":"changed"}'))
        with self.assertRaisesRegex(ValueError, 'imported source hash'):
            self.request()
        self.assertEqual(self.http.call_count, 0)

    def test_local_intent_allows_existing_raw_but_unresolved_cannot_import(self):
        self.write_intent()
        write_source(self.first)
        with self.assertRaisesRegex(RuntimeError, 'Prior unresolved logical request'):
            self.request()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.lineage.exists())
        write_source(self.api.cache_dir)
        self.assertEqual(self.request(), response())
        self.assertEqual(self.http.call_count, 0)

    def test_mismatched_local_intent_is_rejected(self):
        write_source(self.api.cache_dir)
        self.write_intent(signature={**SIGNATURE, 'model': 'different-model'})
        with self.assertRaisesRegex(ValueError, 'intent binding differs'):
            self.request()
        self.assertEqual(self.http.call_count, 0)

    def test_failed_first_response_is_replayed_without_quality_selection(self):
        source = write_source(self.first, response('not valid json'))
        write_source(self.later, response('{"answer":"later valid"}'))
        for _ in range(2):
            with self.assertRaises(json.JSONDecodeError):
                self.api.generate('Question?', json_output=True, max_output=2048, stage='lineage_test')
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        self.assertEqual(json.loads(self.lineage.read_text())['source_path'], str(source))
        self.assertEqual(self.http.call_count, 0)

    def test_bad_first_source_signature_does_not_fall_through(self):
        source = write_source(self.first)
        saved = json.loads(source.read_text())
        saved['request']['model'] = 'different-model'
        source.write_text(json.dumps(saved), encoding='utf-8')
        write_source(self.later)
        with self.assertRaisesRegex(ValueError, 'Source cache request/hash mismatch'):
            self.request()
        self.assertFalse(self.lineage.exists())
        self.assertFalse(self.target.exists())
        self.assertEqual(self.http.call_count, 0)


if __name__ == '__main__':
    unittest.main()
