"""Synthetic offline archive integrity tests; no model clients or live mutations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import module_stack_archive_v1 as mod
from jev_mem.evidence_selection_r03 import choose_baseline, choose_cheap, choose_jev, digest
from jev_mem.source_append_r05 import source_turns


def fixture():
    facts = [{'id': str(i), 'text': f'Person remembers activity number {i}.',
              'observation_date': '2023-01-01', 'input_scope': {'session_id': 's1', 'source_turn_ids': ['D1:1']}}
             for i in range(82)]
    bank = {'conversation': 0, 'complete': True, 'facts': facts}
    groups = {'conversation': 0, 'arm': 'session', 'envelopes': [{'session_id': 's1',
              'source_turn_ids': ['D1:1'], 'messages': [{'content': 'Person: I did things.'}],
              'observation_date': '2023-01-01'}]}
    canon = [mod._fact(f) for f in facts]
    base = {'query_id': 'c0_q0', 'conversation': 0, 'question': 'What activity number 3?',
            'reference': 'PRIVATE_GOLD', 'reference_date': '2023-02-01', 'category': 4,
            'selected_ids': [str(i) for i in range(20)], 'selected_facts': canon[:20]}
    sources = {0: {'rawfacts': {f['id']: deepcopy(f) for f in facts},
                  'turns': source_turns(groups, {'path': 'unused', 'sha256': 'unused'})}}
    old = {'pool_ids': [str(i) for i in range(80)], 'baseline_ids': base['selected_ids'],
           'original_native80_ids': [str(i) for i in range(80)], 'native_other_ids': [str(i) for i in range(20, 60)]}
    members = {'archive_index.json': {}, 'README.txt': 'No caches.', 'facts/conv_0.json': bank,
               'facts/conv_0_group_inputs.json': groups,
               'shared/original_candidate_pools.jsonl': [{'query_id': 'c0_q0', 'conversation': 0, 'pool': old}]}
    for rn in ('R03', 'R08'):
        ids = old['pool_ids'] if rn == 'R03' else old['pool_ids'][:60] + ['80', '81'] + old['pool_ids'][60:78]
        records = [canon[int(fid)] for fid in ids]
        scores = {f'f{i:02d}': (i + 1) / 100 for i in range(80)}
        qa = []; selections = []
        for method in ('baseline', 'cheap', 'jev'):
            dec = (choose_baseline(records) if method == 'baseline' else
                   choose_cheap(base['question'], base['reference_date'], records) if method == 'cheap' else
                   choose_jev(records, scores))
            selected = [canon[int(fid)] for fid in dec['selected_ids']]
            r = {'query_id': 'c0_q0', 'conversation': 0, 'method': method, 'decision': dec,
                 'selected_ids': dec['selected_ids'], 'selected_facts': selected,
                 'selected_sha256': digest(selected), 'baseline_selected_sha256': digest(canon[:20]),
                 'pool_sha256': digest(old), 'raw_scores': scores if method == 'jev' else None,
                 'discovery_status': 'complete', 'discoveries': [{'id': '80'}, {'id': '81'}],
                 'pool': {'pool_ids': ids}, 'preserved_score_response': {'scores': scores}}
            selections.append({'job_id': method + '/c0_q0', 'result': r,
                               'original_ref': {'path': 'DELETED_HISTORY', 'sha256': '0' * 64}})
            qa.append({**{k: base[k] for k in mod.IDENTITY}, 'query_id': 'c0_q0', 'method': method,
                       'job_id': method + '/c0_q0', 'selected_facts': selected,
                       'scores': {'judge_correct': 1}, 'reader_answer': 'PRIVATE_ANSWER', 'judge_result': 'PRIVATE_JUDGE'})
        members.update({rn + '/protocol.json': {}, rn + '/scientific_protocol.json': {},
                        rn + '/source_metadata.json': {'questions_per_arm': 1, 'status': 'sealed'},
                        rn + '/retrieval_or_choice.jsonl': selections, rn + '/qa_and_selected20.jsonl': qa})
    return members, [base], sources


class ArchiveTests(unittest.TestCase):
    def load(self, members, jobs, sources, *, wrong_sha=False):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'test.zip'
            with zipfile.ZipFile(p, 'w') as z:
                for name, value in members.items():
                    raw = (''.join(json.dumps(r) + '\n' for r in value) if name.endswith('.jsonl') else
                           value if name.endswith('.txt') else json.dumps(value))
                    z.writestr(name, raw)
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(mod, 'ARCHIVE_SHA256', '0' * 64 if wrong_sha else h), \
                 patch.object(mod, 'COHORT_SIZE', 1), patch.object(mod, 'CONVERSATIONS', (0,)):
                return mod.load_archive(p, jobs, sources)

    def test_four_arms_no_old_answers_gold_or_fake_cache(self):
        args = fixture(); before = deepcopy(args)
        got = self.load(*args)
        self.assertEqual(args, before)
        self.assertEqual(set(got['arms']), set(mod.ARMS))
        self.assertEqual(got['audit']['validated_selected_inputs'], 4)
        serialized = json.dumps(got)
        for secret in ('PRIVATE_GOLD', 'PRIVATE_ANSWER', 'PRIVATE_JUDGE', 'judge_correct'):
            self.assertNotIn(secret, serialized)
        self.assertFalse(got['audit']['old_http_cache_replayed'])
        rec = got['arms']['pool_new_jev']['c0_q0']
        self.assertEqual(rec['archived_input']['line'], 3)
        self.assertEqual(len(rec['pool_ids']), 80)

    def test_zip_pin_rejects_changed_archive(self):
        with self.assertRaisesRegex(ValueError, 'ZIP SHA'):
            self.load(*fixture(), wrong_sha=True)

    def test_question_or_date_or_gold_identity_cannot_drift(self):
        for key in ('question', 'reference_date', 'reference', 'category', 'conversation'):
            with self.subTest(key=key):
                m, b, s = fixture(); m['R08/qa_and_selected20.jsonl'][1][key] = 'changed'
                with self.assertRaisesRegex(ValueError, 'question identity'):
                    self.load(m, b, s)

    def test_selected_text_change_is_rejected(self):
        m, b, s = fixture(); m = deepcopy(m)
        m['R03/qa_and_selected20.jsonl'][1]['selected_facts'] = deepcopy(m['R03/qa_and_selected20.jsonl'][1]['selected_facts'])
        m['R03/qa_and_selected20.jsonl'][1]['selected_facts'][0]['text'] += ' appended source'
        with self.assertRaisesRegex(ValueError, 'input mismatch'):
            self.load(m, b, s)

    def test_score_rank_replay_rejects_changed_score(self):
        m, b, s = fixture()
        m['R03/retrieval_or_choice.jsonl'][2]['result']['raw_scores']['f00'] = 1.0
        with self.assertRaisesRegex(ValueError, 'does not replay'):
            self.load(m, b, s)

    def test_nonfinite_boolean_missing_scores_rejected(self):
        for bad in (float('nan'), True, None):
            m, b, s = fixture(); m['R03/retrieval_or_choice.jsonl'][2]['result']['raw_scores']['f00'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load(m, b, s)

    def test_new_pool_cannot_change_protected_prefix(self):
        m, b, s = fixture(); ids = m['R08/retrieval_or_choice.jsonl'][0]['result']['pool']['pool_ids']
        ids[0], ids[1] = ids[1], ids[0]
        with self.assertRaisesRegex(ValueError, 'discovery/fill order'):
            self.load(m, b, s)

    def test_duplicate_or_missing_selection_and_qa_rows_rejected(self):
        for member in ('R03/retrieval_or_choice.jsonl', 'R08/qa_and_selected20.jsonl'):
            for duplicate in (True, False):
                m, b, s = fixture()
                if duplicate:
                    m[member].append(deepcopy(m[member][0]))
                else:
                    m[member].pop()
                with self.subTest(member=member, duplicate=duplicate), self.assertRaises(ValueError):
                    self.load(m, b, s)

    def test_same_corrupt_scope_in_archive_and_live_still_rejected(self):
        m, b, s = fixture()
        m['facts/conv_0.json']['facts'][0]['input_scope']['session_id'] = 'wrong'
        s[0]['rawfacts']['0']['input_scope']['session_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'scope crosses'):
            self.load(m, b, s)

    def test_live_projection_drift_rejected(self):
        m, b, s = fixture(); s[0]['turns'][0]['view']['speaker_text'] = 'Changed source'
        with self.assertRaisesRegex(ValueError, 'source projection'):
            self.load(m, b, s)

    def test_original_byte_binding_only_when_declared(self):
        m, b, s = fixture()
        m['archive_index.json']['historical'] = {'entry': 'README.txt', 'preserved_original_bytes': True,
                                               'sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'Original-byte'):
            self.load(m, b, s)


if __name__ == '__main__':
    unittest.main()
