"""Recovery writer contract tests: synthetic temp sources, no credentials/network."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_group_writer_v2 import FakeAPI as BaseFakeAPI, dump, http
from tools import pilot400_recovery_writer_v1 as recovery
from jev_mem import group_writer_v2 as w
from jev_mem.topic_grouping import render_group


class FakeAPI(BaseFakeAPI):
    def request(self, provider, model, body, *, operation='generateContent', stage, cache=True):
        return super().request(provider, model, body, operation=operation, stage=stage)


class RecoveryWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.api = FakeAPI(self.root / 'cache')
        self.api.retries = 0
        self.raw = [{'source_turn_id': 'D1:' + str(i), 'session_id': 's1', 'speaker': 'A',
                     'observation_date': '2022-01-0' + str(i + 1),
                     'text': 'Original assertion ' + str(i)} for i in range(5)]
        self.members = [self.raw[2], self.raw[4]]
        self.invalid = {'facts': [{'text': 'Invalid association',
            'source_turn_ids': ['D1:4', 'D1:0'], 'temporal_status': 'not_temporal'}]}
        self.valid = {'facts': [{'text': 'Original assertion 2',
            'source_turn_ids': ['D1:2'], 'temporal_status': 'not_temporal'}]}
        block = patch.object(recovery.ModelAPI, '__init__', side_effect=AssertionError('Real API forbidden'))
        block.start()
        self.addCleanup(block.stop)

    def pending(self):
        store = self.root / 'original_failure'
        if not (store / 'pending_group.json').exists():
            self.api.responses = [http(self.invalid), http(self.invalid)]
            with self.assertRaises(w.GroupWriteError):
                w.GroupFactWriter(self.api, store).add_group(user_id='u', group_id='g',
                    members=self.members, raw_turns=self.raw, grouping_method='jev_topics')
        path = store / 'pending_group.json'
        return recovery.read(path), path

    def recover(self, response=None):
        pending, path = self.pending()
        self.api.responses = [http(self.valid if response is None else response)]
        return recovery.recover_group(pending, path, self.root / 'recovered', self.api)

    def test_member_packets_preserve_source_dates_associations_exclude_old_output_and_gold(self):
        pending, _ = self.pending()
        pending['qa'] = 'FORBIDDEN_QA'
        pending['gold'] = 'FORBIDDEN_GOLD'
        pending['raw_response'] = {'facts': [{'text': 'FORBIDDEN_OLD_FACT'}]}
        signature, payload = recovery.packet_request(pending)
        rendered = json.dumps(signature, sort_keys=True)
        self.assertFalse(any(x in rendered for x in ['FORBIDDEN_QA', 'FORBIDDEN_GOLD', 'FORBIDDEN_OLD_FACT']))
        self.assertEqual([p['member'] for p in payload['MEMBER_PACKETS']], self.members)
        context = {t['source_turn_id']: t for t in pending['read_only_context']}
        for packet in payload['MEMBER_PACKETS']:
            allowed = pending['member_context_ids'][packet['member']['source_turn_id']]
            self.assertEqual(packet['allowed_context_ids'], allowed)
            self.assertEqual(packet['read_only_context'], [context[i] for i in allowed])

    def test_schema_has_original_types_and_visible_source_enum(self):
        pending, _ = self.pending()
        request, payload = recovery.packet_request(pending)
        self.assertEqual(json.loads(request['body']['contents'][0]['parts'][0]['text']), payload)
        schema = request['body']['generationConfig']['responseSchema']['properties']['facts']['items']
        self.assertEqual(set(schema['required']), {'text', 'source_turn_ids', 'temporal_status'})
        self.assertEqual(schema['properties']['text']['type'], 'STRING')
        self.assertEqual(schema['properties']['source_turn_ids']['items']['enum'],
                         pending['member_source_turn_ids'] + pending['context_source_turn_ids'])
        self.assertEqual(request['model'], 'gemini-2.5-flash')

    def test_unchanged_parser_rejects_context_only(self):
        pending, path = self.pending()
        self.api.responses = [http({'facts': [{'text': 'Context', 'source_turn_ids': ['D1:0'],
                                             'temporal_status': 'not_temporal'}]})]
        with self.assertRaisesRegex(w.GroupWriteError, 'Context-only'):
            recovery.recover_group(pending, path, self.root / 'recovered', self.api)
        self.assertFalse((self.root / 'recovered/recovery_record.json').exists())

    def test_unchanged_parser_rejects_context_unassociated_with_cited_member(self):
        pending, path = self.pending()
        self.api.responses = [http(self.invalid)]
        with self.assertRaisesRegex(w.GroupWriteError, 'not associated'):
            recovery.recover_group(pending, path, self.root / 'recovered', self.api)

    def test_empty_is_valid_and_does_not_embed_or_invent_facts(self):
        result = self.recover({'facts': []})
        self.assertEqual(result['fact_records'], [])
        self.assertEqual(result['raw_response'], {'facts': []})
        self.assertEqual(self.api.embed_calls, 0)
        self.assertEqual(recovery.validate_recovery_record(result), [])

    def test_valid_recovery_replays_without_new_request_and_original_bytes_unchanged(self):
        pending, path = self.pending()
        original = path.read_bytes()
        result = self.recover()
        requests = len(self.api.requests)
        again = recovery.recover_group(pending, path, self.root / 'recovered', self.api)
        self.assertEqual(result, again)
        self.assertEqual(len(self.api.requests), requests)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(result['fact_records']), 1)
        self.assertEqual(len(result['fact_records'][0]['embedding']), 768)

    def test_modified_request_schema_or_payload_is_rejected(self):
        record = self.recover()
        for mutate in ('request', 'payload', 'request_hash', 'facts'):
            bad = copy.deepcopy(record)
            if mutate == 'request': bad['request']['body']['generationConfig'].pop('responseSchema')
            elif mutate == 'payload': bad['payload']['MEMBER_PACKETS'][0]['member']['text'] = 'Changed'
            elif mutate == 'request_hash': bad['request_hash'] = '0' * 64
            else: bad['fact_records'][0]['text'] = 'Changed'
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, w.GroupWriteError)):
                recovery.validate_recovery_record(bad)

    def test_intent_without_http_response_never_reissues(self):
        pending, path = self.pending()
        self.api.raise_stage = 'pilot_recovery_member_first_writer'
        with self.assertRaises(RuntimeError):
            recovery.recover_group(pending, path, self.root / 'recovered', self.api)
        self.assertTrue((self.root / 'recovered/intent.json').exists())
        attempts = len(self.api.requests)
        self.api.raise_stage = None
        self.api.responses = [http(self.valid)]
        with self.assertRaises(w.ExtractionUnknown):
            recovery.recover_group(pending, path, self.root / 'recovered', self.api)
        self.assertEqual(len(self.api.requests), attempts)

    def test_changed_original_failure_file_rejected_on_replay(self):
        record = self.recover()
        pending, path = self.pending()
        pending['error_type'] = 'Changed'
        dump(path, pending)
        with self.assertRaises(ValueError):
            recovery.recover_group(record['original_pending'], path, self.root / 'recovered', self.api)

    def test_wrong_signature_cache_rejected_before_provider(self):
        pending, path = self.pending()
        request, _ = recovery.packet_request(pending)
        cache_path = self.api.cache_dir / (w.provider_request_hash(request) + '.json')
        bad_request = copy.deepcopy(request)
        bad_request['model'] = 'gemini-other'
        dump(cache_path, {'request': bad_request, 'response': http(self.valid),
                          'stage': 'pilot_recovery_member_first_writer'})
        calls = len(self.api.requests)
        with self.assertRaisesRegex(ValueError, 'cache'):
            recovery.recover_group(pending, path, self.root / 'recovered', self.api)
        self.assertEqual(len(self.api.requests), calls)

    def test_wrong_stage_cache_rejected_before_provider(self):
        pending, path = self.pending()
        request, _ = recovery.packet_request(pending)
        dump(self.api.cache_dir / (w.provider_request_hash(request) + '.json'),
             {'request': request, 'response': http(self.valid), 'stage': 'unrelated_stage'})
        calls = len(self.api.requests)
        with self.assertRaisesRegex(ValueError, 'cache'):
            recovery.recover_group(pending, path, self.root / 'recovered', self.api)
        self.assertEqual(len(self.api.requests), calls)

    def test_changed_embedding_transaction_or_vector_rejected(self):
        record = self.recover()
        bad = copy.deepcopy(record)
        bad['fact_records'][0]['embedding'] = [0., 1.] + [0.] * 766
        with self.assertRaisesRegex(ValueError, 'Embedding'):
            recovery.validate_recovery_record(bad)
        transaction = Path(record['embedding_transaction_path'])
        data = recovery.read(transaction)
        data['embeddings'] = [[0., 1.] + [0.] * 766]
        dump(transaction, data)
        with self.assertRaisesRegex(ValueError, 'Embedding'):
            recovery.validate_recovery_record(record)

    def test_raw_response_alias_is_bound_to_http_response(self):
        record = self.recover()
        record['raw_response'] = {'facts': []}
        with self.assertRaises(ValueError):
            recovery.validate_recovery_record(record)

    def test_validator_binds_original_checkpoint_bytes_and_embedded_pending(self):
        record = self.recover()
        pending, path = self.pending()
        pending['error_type'] = 'Changed original'
        dump(path, pending)
        with self.assertRaises(ValueError):
            recovery.validate_recovery_record(record)

    def make_overlay(self):
        original = self.root / 'original_complete_prefix'
        original.mkdir()
        records, groups = [], []
        for index, members in enumerate((self.raw[:2], self.raw[2:])):
            gid = 'g' + str(index)
            self.api.responses = [http({'facts': []})]
            record = w.GroupFactWriter(self.api, original / gid).add_group(
                user_id='u', group_id=gid, members=members, raw_turns=self.raw, grouping_method='jev_topics')
            records.append(record)
            groups.append({'group_id': gid, 'turns': members,
                           'source_turn_ids': [m['source_turn_id'] for m in members], 'text': render_group(members)})
        raw_path, grouping_path, partial_path = [original / name for name in
                                                ('raw_source.json', 'grouping.json', 'group_fact_bank.json')]
        dump(raw_path, {'user_id': 'u', 'source_sha256': w._hash(self.raw), 'raw_turns': self.raw})
        dump(grouping_path, {'groups': groups})
        dump(partial_path, {'schema': w.SCHEMA, 'user_id': 'u', 'grouping_method': 'jev_topics',
                            'source_sha256': w._hash(self.raw), 'groups': records, 'facts': []})
        value = {'schema': 'pilot400-mixed-bank-overlay-v1', 'conversation': 1,
            'original_raw_path': str(raw_path), 'original_raw_sha256': recovery.sha(raw_path),
            'original_grouping_path': str(grouping_path), 'original_grouping_sha256': recovery.sha(grouping_path),
            'original_partial_bank_path': str(partial_path), 'original_partial_bank_sha256': recovery.sha(partial_path),
            'groups': [{'group_id': row['group_id'], 'kind': 'reused_v2', 'record': row} for row in records], 'facts': []}
        path = self.root / 'overlay.json'
        dump(path, value)
        return path, value

    def test_v2_records_reused_and_exact_partition_accepted(self):
        path, value = self.make_overlay()
        self.assertEqual(recovery.validate_overlay(path), value)

    def test_missing_group_duplicate_group_and_changed_member_rejected(self):
        path, value = self.make_overlay()
        cases = []
        bad = copy.deepcopy(value); bad['groups'].pop(); cases.append(bad)
        bad = copy.deepcopy(value); bad['groups'][1] = copy.deepcopy(bad['groups'][0]); cases.append(bad)
        bad = copy.deepcopy(value); bad['groups'][1]['record']['members'][0]['text'] = 'Changed'; cases.append(bad)
        for index, bad in enumerate(cases):
            dump(path, bad)
            with self.subTest(index=index), self.assertRaises((ValueError, w.GroupWriteError)):
                recovery.validate_overlay(path)

    def test_reused_record_must_equal_exact_original_prefix(self):
        path, value = self.make_overlay()
        # Timestamp is not an extraction semantic, but changing it means this is
        # no longer the exact sealed original record claimed as reused_v2.
        value['groups'][0]['record']['finished_at'] = 'FORGED'
        dump(path, value)
        with self.assertRaises(ValueError):
            recovery.validate_overlay(path)


if __name__ == '__main__':
    unittest.main()
