"""Pure overlay guards and synthetic original-response replay; zero network."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import recover_mem0_plain_judge_labels_v1 as o
from tools.run_sdk20_modules import verify
from tests import test_judge_terminal_label_compat_v1 as fixtures


class OverlayTests(unittest.TestCase):
    def source(self, text):
        f = fixtures.SourceReplayTests(); f.setUp(); self.addCleanup(f.doCleanups)
        judge, _ = f.source(text)
        reader = verify(f.root / 'conv_0/reader' / f.arm / 'c0_q0.json')
        statuses = {'selection': 'complete', 'reader': 'complete', 'judge': judge['status'],
                    'qa': 'complete' if judge['status'] == 'complete' else 'failed'}
        slot = {'conversation': 0, 'method': f.arm, 'query_id': 'c0_q0', 'category': 1,
                **{k + '_status': v for k, v in statuses.items()},
                'scores': {'official_locomo_f1': .75, 'custom_token_f1': 1.,
                           'judge_correct': float(judge['result']['correct']) if judge['status'] == 'complete' else None}, 'errors': []}
        source_slot = {'statuses': statuses, 'artifacts': {stage: o.pointer(f.root / 'conv_0' / stage / f.arm / 'c0_q0.json') for stage in statuses}}
        return f, source_slot, slot, reader, judge

    def derive(self, items):
        f, source_slot, slot, reader, judge = items
        return o.derive_slot(source_slot, slot, reader, judge, api_dir=f.api.root,
                             ledger=f.api.shared['ledger'], question='Where?', reference='Paris')

    def test_recovered_no_is_valid_false_and_all_source_and_f1_unchanged(self):
        items = self.source('Insufficient detail.\n[[no]]')
        before = copy.deepcopy(items[1:])
        hashes = {p: o.sha(p) for p in items[0].root.rglob('*.json')}
        sidecar, row = self.derive(items)
        self.assertEqual(sidecar['status'], 'compat_recovered')
        self.assertIs(sidecar['judge_result']['correct'], False)
        self.assertEqual(row['scores'], {'official_locomo_f1': .75, 'custom_token_f1': 1., 'judge_correct': 0.})
        self.assertEqual(items[1:], before)
        self.assertEqual(hashes, {p: o.sha(p) for p in items[0].root.rglob('*.json')})

    def test_strict_success_replayed_without_replacement(self):
        items = self.source('[[yes]]'); sidecar, row = self.derive(items)
        self.assertEqual(sidecar['status'], 'strict_inherited'); self.assertEqual(row, items[2])

    def test_conflict_retains_null_not_a_false_score(self):
        items = self.source('The prompt mentions [[no]].\n[[yes]]')
        sidecar, row = self.derive(items)
        self.assertEqual(sidecar['status'], 'retained_failure'); self.assertIsNone(sidecar['judge_result'])
        self.assertEqual(row, items[2]); self.assertIsNone(row['scores']['judge_correct'])

    def test_other_failure_is_preserved_without_attempting_compatibility(self):
        items = self.source('Explanation.\n[[yes]]'); items[-1]['error']['type'] = 'RuntimeError'
        with patch.object(o, 'replay_source', side_effect=AssertionError('Ineligible failure replayed')):
            sidecar, row = self.derive(items)
        self.assertEqual(sidecar['status'], 'retained_failure'); self.assertEqual(row, items[2])

    def test_identity_or_strict_score_drift_rejected(self):
        items = self.source('[[yes]]'); items[-1]['question'] = 'A different question'
        with self.assertRaises(ValueError): self.derive(items)
        items[-1]['question'] = 'Where?'; items[2]['scores']['judge_correct'] = 0.
        with self.assertRaises(ValueError): self.derive(items)

    def slots(self, n=30):
        return [{'query_id': f'c{i//3}_q{i%3}', 'conversation': i//3, 'category': 1, 'method': arm,
                 **{s + '_status': 'complete' for s in o.sealing.STAGES},
                 'scores': {m: 1. for m in o.strict.METRICS}, 'errors': []}
                for arm in o.strict.ARMS for i in range(n)]

    def test_separate_strict_compat_denominators_and_pairs_f1_identical(self):
        rows = self.slots(); rows[0]['judge_status'] = rows[0]['qa_status'] = 'failed'; rows[0]['scores']['judge_correct'] = None
        original = {'slots': rows, **o.strict.aggregate(rows, denominator=30, bootstrap_samples=20)}
        changed = copy.deepcopy(rows); changed[0]['judge_status'] = changed[0]['qa_status'] = 'complete'; changed[0]['scores']['judge_correct'] = 0.
        a, b = o.compare_views(original, changed, denominator=30, bootstrap_samples=20)
        arm = o.strict.ARMS[0]
        self.assertEqual((a['arms'][arm]['metrics']['judge_correct']['valid_denominator'], b['arms'][arm]['metrics']['judge_correct']['valid_denominator']), (29, 30))
        self.assertEqual(a['paired_merged_minus_unmerged']['official_locomo_f1'], b['paired_merged_minus_unmerged']['official_locomo_f1'])
        self.assertEqual(a['arms'][arm]['metrics']['official_locomo_f1'], b['arms'][arm]['metrics']['official_locomo_f1'])
        changed[0]['scores']['official_locomo_f1'] = .5
        with self.assertRaises(ValueError): o.compare_views(original, changed, denominator=30, bootstrap_samples=20)

    def test_all3080_slots_required_duplicate_or_status_drift_rejected(self):
        manifest = {'query_ids': {str(ci): [f'c{ci}_q{i}' for i in range(154)] for ci in range(10)}}
        slots = [{'conversation': ci, 'method': arm, 'query_id': qid,
                  **{s + '_status': 'complete' for s in o.sealing.STAGES}, 'scores': {'judge_correct': 1.}, 'errors': []}
                 for ci in range(10) for arm in o.strict.ARMS for qid in manifest['query_ids'][str(ci)]]
        seal = {'slots': [{'conversation': s['conversation'], 'arm': s['method'], 'query_id': s['query_id'],
                           'statuses': {k: s[k + '_status'] for k in o.sealing.STAGES}} for s in slots]}
        self.assertEqual(len(o.validate_population(manifest, seal, {'slots': slots})[0]), 3080)
        with self.assertRaises(ValueError): o.validate_population(manifest, seal, {'slots': slots[:-1]})
        with self.assertRaises(ValueError): o.validate_population(manifest, seal, {'slots': slots[:-1] + [slots[0]]})
        seal['slots'][0]['statuses']['judge'] = 'failed'
        with self.assertRaises(ValueError): o.validate_population(manifest, seal, {'slots': slots})

    def test_new_namespace_and_pointer_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'runs').mkdir(); source = root / 'runs/source'; source.mkdir()
            with patch.object(o, 'ROOT', root):
                self.assertEqual(o.validate_namespace(root / 'runs/new', source), root / 'runs/new')
                for path in (source, source / 'nested', root / 'new', root / 'runs'):
                    with self.assertRaises(ValueError): o.validate_namespace(path, source)
            p = source / 'raw.json'; p.write_text('{}'); ref = o.pointer(p); o.check_pointer(ref)
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError): o.check_pointer(ref)

    def test_offline_guard_blocks_credentials_and_network(self):
        import socket
        from jev_mem import providers
        with o.offline():
            with self.assertRaises(AssertionError): socket.create_connection(('example.com', 443))
            with self.assertRaises(AssertionError): providers.load_credentials()


if __name__ == '__main__': unittest.main()
