"""Independent candidate checks; local captures and temporary artifacts only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools import baselines_hypermem_native_v4 as a

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/new_suffix_diagnosis_v1'


def finish(coro):
    try:
        coro.send(None)
    except StopIteration as end:
        return end.value
    raise AssertionError('Unexpected asynchronous transport in pure fixture')


def exact_case(ci):
    report = json.loads((CAPTURE / 'capture03/report.json').read_bytes())
    row = next(x for x in report['rows'] if x['conversation'] == ci)
    raw_path = Path(row['raw']['captured']['path'])
    blob = raw_path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == row['raw']['captured']['sha256']
    text = json.loads(blob)['response']['candidates'][0]['content']['parts'][0]['text']
    native = json.loads(Path(row['native_call']['captured']['path']).read_bytes())
    cert = {'conversation': ci, 'request_hash': row['request_hash'], 'stage': row['stage'],
            'response_text_sha256': hashlib.sha256(text.encode()).hexdigest()}
    if ci in a.INSERTIONS:
        if ci == 5:
            syntax = json.loads((CAPTURE / 'c5_syntax_candidate_v1/report.json').read_bytes())
        else:
            data = json.loads((CAPTURE / 'syntax_candidates_v1/report.json').read_bytes())
            syntax = next(x for x in data['rows'] if x['conversation'] == ci)
        cert.update(edits=syntax['edits'], derived_parsed_sha256=syntax['derived_parsed_sha256'])
    return row, native, text, cert


class ExactDerivations(unittest.TestCase):
    def test_native_generation_source_has_only_three_declared_boundary_substitutions(self):
        restored = a._derived_source
        decode = 'value = self._decode_exact(raw, stage, frame.f_locals, path, request)'
        validate = "self._validate_qualified(value, stage, frame.f_locals, path, request, cached_native='cached' in locals())"
        self.assertEqual(restored.count(decode), 1)
        self.assertEqual(restored.count(validate), 2)
        restored = restored.replace(decode, 'value = raw if isinstance(raw, dict) else json.loads(raw)')
        restored = restored.replace(validate, 'self._validate(value, stage, frame.f_locals)')
        self.assertEqual(restored, a._v3._derived_source)

    def test_four_actual_texts_have_exact_insertions_inverse_and_no_other_edits(self):
        for ci in (1, 4, 5, 8):
            with self.subTest(conversation=ci):
                row, native, text, cert = exact_case(ci)
                before = deepcopy(cert)
                value, journal = a.derive_exact_text(text, cert)
                self.assertEqual(cert, before)
                self.assertEqual(journal['original_text'], text)
                self.assertTrue(journal['inverse_utf8_equal'])
                self.assertEqual(journal['edits'], cert['edits'])
                self.assertEqual(a._runtime._sha(value), cert['derived_parsed_sha256'])
                self.assertEqual(a.strict_json(journal['derived_text']), value)
                changed = text
                for e in reversed(cert['edits']):
                    changed = changed[:e['offset']] + e['insert'] + changed[e['offset']:]
                self.assertEqual(changed, journal['derived_text'])

    def test_changed_raw_or_edit_or_derived_hash_is_rejected(self):
        _, _, text, cert = exact_case(8)
        cases = [(text + ' ', cert)]
        bad = deepcopy(cert); bad['edits'][0]['offset'] += 1; cases.append((text, bad))
        bad = deepcopy(cert); bad['derived_parsed_sha256'] = '0' * 64; cases.append((text, bad))
        for raw, candidate in cases:
            with self.assertRaises(a.BaselineFatal):
                a.derive_exact_text(raw, candidate)

    def test_actual_false_payloads_preserved_and_true_branch_stays_strict(self):
        for ci in (0, 7):
            row, _, text, _ = exact_case(ci)
            value = json.loads(text); before = deepcopy(value)
            mapping = row['diagnosis']['candidate_mapping']
            report = a.false_row_contract(value, {'simple_to_real': mapping})
            self.assertEqual(value, before)
            self.assertFalse(report['payload_changed'])
            for replacement in ('topic_999', None, 7):
                bad = deepcopy(value)
                next(x for x in bad['results'] if x['match'])['topic_id'] = replacement
                with self.assertRaises((a.BaselineFatal, KeyError, TypeError)):
                    a.false_row_contract(bad, {'simple_to_real': mapping})

    def test_actual_topics_batch_objects_are_retained_in_local_evidence(self):
        topics = [SimpleNamespace(to_dict=lambda: {'topic_id': 'actual-id', 'summary': 'same display'})]
        value = a._Generation._local_record({'topics_batch': topics, 'episode': SimpleNamespace(to_dict=lambda: {'event_id': 'actual-episode'}),
                                             'simple_to_real': {'topic_1': 'actual-id'}, 'irrelevant': 1})
        self.assertEqual(value['topics_batch'], [{'topic_id': 'actual-id', 'summary': 'same display'}])
        self.assertNotIn('irrelevant', value)

    def test_update_new_episode_actual_parameter_is_retained(self):
        value = a._Generation._local_record({'new_episode': SimpleNamespace(to_dict=lambda: {'event_id': 'actual-new-episode'}),
                                             'topic': SimpleNamespace(to_dict=lambda: {'topic_id': 'actual-topic'})})
        self.assertEqual(value['new_episode'], {'event_id': 'actual-new-episode'})


class NativeCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'research', prefix='independent_hyper_v4_')
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.row, self.old_native, self.text, self.cert = exact_case(8)
        self.receipt = {'status': 'authorized', 'fixture_only': True}
        self.deliveries = 0
        self.authorizations = 0
        self.api = SimpleNamespace(generate=self.deliver,
            compatibility_certificate=lambda **kw: deepcopy(self.cert),
            authorize_exact_compatibility=self.authorize)

    def deliver(self, *args, **kwargs):
        self.deliveries += 1
        return self.text

    def authorize(self, *args, **kwargs):
        self.authorizations += 1
        return deepcopy(self.receipt)

    def call(self):
        generation = a._Generation(self.api, self.folder, 'topic')
        async def _update_existing_topic():
            topic = SimpleNamespace(to_dict=lambda: {'topic_id': 'fixture-topic'})
            new_episode = SimpleNamespace(to_dict=lambda: {'event_id': 'fixture-episode'})
            return await generation.generate(self.old_native['request']['prompt'], response_format={'type': 'json_object'})
        return finish(_update_existing_topic())

    def test_real_generation_keeps_original_text_separate_from_parsed_object(self):
        self.call()
        native = a._runtime._read_json(self.folder / 'calls/topic_000001_hypermem_topic_update.json')
        self.assertEqual(native['raw_response'], self.text)
        self.assertIsInstance(native['parsed_response'], dict)
        self.assertEqual(self.deliveries, 1)
        journal = self.folder / 'compatibility_v4/topic_000001_hypermem_topic_update.json'
        a._runtime._verify_seal(journal)
        self.assertEqual(a._runtime._read_json(journal)['derivation']['original_text'], self.text)

    def test_cached_syntax_requires_retained_compatibility_journal(self):
        self.call()
        journal = self.folder / 'compatibility_v4/topic_000001_hypermem_topic_update.json'
        journal.unlink()  # Only this test's private temporary artifact.
        with self.assertRaises(a.BaselineFatal):
            self.call()
        self.assertEqual(self.deliveries, 1)
        self.assertFalse(journal.exists())

    def test_cached_syntax_rederives_parsed_value_from_original_text(self):
        self.call()
        path = self.folder / 'calls/topic_000001_hypermem_topic_update.json'
        native = a._runtime._read_json(path)
        native['parsed_response']['summary'] += ' changed local cache'
        a._runtime._write_json(path, native)
        a._runtime._seal_path(path).unlink()  # Local adversarial fixture only.
        a._runtime._seal(path)  # Consistent local reseal is not raw authority.
        with self.assertRaises(a.BaselineFatal):
            self.call()
        self.assertEqual(self.deliveries, 1)


class FalseNativeCache(NativeCache):
    def setUp(self):
        super().setUp()
        self.row, self.old_native, self.text, self.cert = exact_case(0)

    def deliver(self, *args, **kwargs):
        self.deliveries += 1
        return json.loads(self.text)

    def call(self):
        generation = a._Generation(self.api, self.folder, 'topic')
        async def _llm_match_topics_batch():
            simple_to_real = deepcopy(self.row['diagnosis']['candidate_mapping'])
            topics_batch = [SimpleNamespace(to_dict=lambda ident=real: {'topic_id': ident}) for real in simple_to_real.values()]
            episode = SimpleNamespace(to_dict=lambda: {'event_id': 'fixture-episode'})
            return await generation.generate(self.old_native['request']['prompt'], response_format={'type': 'json_object'})
        return finish(_llm_match_topics_batch())

    # Replace inherited syntax tests because this class targets a distinct
    # no-edit consumer branch.
    def test_real_generation_keeps_original_text_separate_from_parsed_object(self):
        self.call()
        native = a._runtime._read_json(self.folder / 'calls/topic_000001_hypermem_topic_match.json')
        self.assertEqual(native['raw_response'], json.loads(self.text))
        self.assertEqual(native['parsed_response'], native['raw_response'])
        self.assertEqual(self.deliveries, 1)

    def test_cached_syntax_requires_retained_compatibility_journal(self):
        self.call()
        journal = self.folder / 'compatibility_v4/topic_000001_hypermem_topic_match.json'
        journal.unlink()
        with self.assertRaises(a.BaselineFatal):
            self.call()
        self.assertEqual(self.deliveries, 1)
        self.assertFalse(journal.exists())

    def test_cached_syntax_rederives_parsed_value_from_original_text(self):
        self.call()
        native_path = self.folder / 'calls/topic_000001_hypermem_topic_match.json'
        native = a._runtime._read_json(native_path)
        native['parsed_response']['results'][0]['match'] = not native['parsed_response']['results'][0]['match']
        a._runtime._write_json(native_path, native)
        a._runtime._seal_path(native_path).unlink()
        a._runtime._seal(native_path)
        with self.assertRaises(a.BaselineFatal):
            self.call()
        self.assertEqual(self.deliveries, 1)


if __name__ == '__main__':
    unittest.main()
