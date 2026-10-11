"""Independent offline original R03 + new R05/C source and dispatcher review."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_gemini25_ablation_v1 as prep
from tools import score_gemini25_ablation_v1 as score
from tools import run_gemini25_ablation_qa_v1 as qa
from tests import test_module_stack_dispatch_v1 as dispatch_fixtures
from tests.test_r05_refinement_v1 import fixture


class ScientificFixture:
    def __init__(self, n=14, count=80):
        records, scopes, turns = fixture(n)
        for i in range(20, count):
            records.append({'id': f'f{i}', 'text': f'Alice likes boots hiking item {i}.', 'observation_date': '2023-06-01'})
            scopes.append(deepcopy(scopes[0]))
        self.records, self.scopes, self.turns = records, scopes, turns
        self.m = {'protocol_sha256': 'new25'}
        self.q = {'conversation': 0, 'query_id': 'c0_q0', 'question': 'boots hiking', 'reference': 'GOLD_MUST_NEVER_APPEAR',
                  'category': 1, 'reference_date': '2023-06-01', 'records': records,
                  'bank': {'path': 'fresh25bank', 'sha256': 'b'}, 'retrieval': {'path': 'fresh25retrieval', 'sha256': 'r'}}
        self.sources = {0: {'bank': self.q['bank'], 'turns': turns,
                           'rawfacts': {f['id']: {**deepcopy(f), 'input_scope': deepcopy(s)} for f, s in zip(records, scopes)}}}
        self.us, self.results, self.requests = {}, {}, {}

    def build(self, phase):
        with patch.object(prep, 'inputs', return_value=(self.m, self.sources, [self.q])), \
             patch.object(prep, 'units', side_effect=lambda reader, run, p: self.us[p]), \
             patch.object(prep, 'phase_results', side_effect=lambda reader, run, p: self.results[p]):
            values = list(prep.phase_values(prep.Reader(), Path('offline'), phase))
        self.us[phase] = [({'path': phase + '/' + u['unit_id'], 'sha256': phase}, u) for u, _ in values]
        self.requests[phase] = [r for _, requests in values for r in requests]
        self.results[phase] = {r['request_id']: {'status': 'complete',
            'scores': {k: (.1 + i / 100 if phase == 'r03' else .9)
                       for i, k in enumerate(r['signature']['body']['questions'])}} for r in self.requests[phase]}
        return values

    def selections(self):
        with patch.object(prep, 'inputs', return_value=(self.m, self.sources, [self.q])), \
             patch.object(prep, 'units', side_effect=lambda reader, run, p: self.us[p]), \
             patch.object(prep, 'phase_results', side_effect=lambda reader, run, p: self.results[p]):
            return list(prep.selections(prep.Reader(), Path('offline')))


def phase_graph(root):
    prep.save_new(root / 'run_manifest.json', {})
    m = {'protocol_sha256': 'new25'}
    expected = {}
    for phase in prep.PHASES:
        unit = {'unit_id': 'c0_q0', 'phase': phase, 'scientific_input': 'original'}
        expected[phase] = unit
        ref = prep.save_new(root / 'phases' / phase / 'units/c0_q0.json', unit)
        prep.save_new(root / 'phases' / phase / 'manifest.json', {'schema': 'gemini25-ablation-phase-v1',
            'phase': phase, 'protocol_sha256': 'new25', 'parent_manifest': prep.pointer(root / 'run_manifest.json'),
            'units': [ref], 'requests': []})
    return m, expected


class SourceIndependentTests(unittest.TestCase):
    def setUp(self):
        for target in ('requests.Session.post', 'requests.Session.request', 'socket.create_connection',
                       'jev_mem.providers.load_credentials'):
            guard = patch(target, side_effect=AssertionError('No network or credentials'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_r03_full80_exact_predicate_and_source_rank_f_keys_both_admitted(self):
        f = ScientificFixture()
        sig = prep.r03.build_request(f.q['question'], f.q['reference_date'], f.records, prep.spec())
        self.assertEqual(score.validate_request(sig), 'r03')
        self.assertEqual(len(sig['body']['questions']), 80)
        loc = prep.source_core.build_candidates(f.q['question'], f.records[:20], f.scopes[:20], f.turns, conversation=0)
        self.assertEqual(len(loc['candidates']), 4)
        rank = prep.stack.build_rank_request(f.q['question'], f.q['reference_date'], f.records[:20], loc['candidates'])
        self.assertEqual(score.validate_request(rank), 'f')
        self.assertEqual(len(rank['body']['questions']), 4)
        for original in (sig, rank):
            bad = deepcopy(original)
            bad['body']['state']['reference'] = 'GOLD_MUST_NEVER_APPEAR'
            with self.assertRaises(ValueError):
                score.validate_request(bad)
        broken = deepcopy(sig)
        broken['body']['questions']['f00']['instructions'] += '\nAltered predicate'
        with self.assertRaises(ValueError):
            score.validate_request(broken)

    def test_actual_five_phase_wiring_matches_original_functions_and_has_no_gold(self):
        f = ScientificFixture()
        for phase in prep.PHASES:
            f.build(phase)
        self.assertEqual(len(f.us['support']), 80)
        self.assertEqual({u['plan']['fact']['id'] for _, u in f.us['support']}, {x['id'] for x in f.records})
        all_requests = [r for requests in f.requests.values() for r in requests]
        self.assertNotIn('GOLD_MUST_NEVER_APPEAR', json.dumps(all_requests))
        for request in all_requests:
            score.validate_request(request['signature'])
        r03decision = prep.r03.choose_jev(f.records, f.results['r03']['c0_q0']['scores'])
        chosen = prep.r03.selected_facts(f.records, r03decision['pool_selected_indices'])
        self.assertEqual([x['id'] for x in chosen], [f'f{i}' for i in range(60, 80)])
        support = {}
        for _, unit in f.us['support']:
            values, status = prep.merged_scores(unit['request_ids'], f.results['support'])
            support[unit['plan']['fact']['id']] = {'plan': unit['plan'], 'scores': values, 'status': status}
        expected = prep.r05.build_candidate_variants(f.q['question'], chosen,
            [f.sources[0]['rawfacts'][x['id']]['input_scope'] for x in chosen], f.turns,
            conversation=0, support_results=support, reference_date=f.q['reference_date'])['multi_plan']
        multi = f.us['multi_compare'][0][1]
        self.assertEqual(multi['records'], chosen)
        self.assertEqual(multi['plan'], expected)
        scores, status = prep.merged_scores(multi['request_ids'], f.results['multi_compare'])
        expected_locator = prep.r05.resolve_multi(expected, scores, score_status=status)['locator']
        closure = f.us['closure'][0][1]
        self.assertEqual(closure['locator'], expected_locator)
        for candidate, plan in zip(expected_locator['candidates'], closure['plans']):
            scope = f.sources[0]['rawfacts'][chosen[candidate['origin']['anchor_slot']]['id']]['input_scope']
            self.assertEqual(plan, prep.stack.build_closure_plan(f.q['question'], f.q['reference_date'], chosen,
                candidate, scope, f.turns, conversation=0))
        resolved = []
        for plan, ids in zip(closure['plans'], closure['request_ids']):
            values, status = prep.merged_scores(ids, f.results['closure'])
            resolved.append(prep.stack.resolve_closure(plan, values, score_status=status))
        rank = f.us['rank'][0][1]
        self.assertEqual(rank['locator'], prep.stack.apply_closures(expected_locator, resolved, chosen, closure_plans=closure['plans']))
        selections = f.selections()
        self.assertEqual([arm for _, arm, _ in selections], list(prep.ARMS))
        self.assertEqual(selections[0][2]['selected_facts'], f.records[:20])
        self.assertEqual(selections[1][2]['selected_facts'], chosen)
        jobs = [{**{k: q[k] for k in ('conversation', 'query_id', 'question', 'reference', 'category', 'reference_date', 'bank')},
            'job_id': arm + '/' + q['query_id'], 'method': arm, 'status': 'ready',
            'selected_ids': rec['selected_ids'], 'selected_facts': rec['selected_facts'],
            'selection': {'path': 'fresh', 'sha256': 'x'}} for q, arm, rec in selections]
        qa.check_jobs(jobs)

    def test_no_rank_is_real_none_and_zero_request_phase_never_constructs_api(self):
        f = ScientificFixture(n=1, count=20)
        for phase in prep.PHASES:
            f.build(phase)
        self.assertEqual(f.requests['rank'], [])
        selected = f.selections()[-1][2]
        self.assertEqual(selected['rank_decision']['score_status'], 'not_dispatched')
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            prep.save_new(root / 'phases/rank/manifest.json', {'requests': [], 'units': [], 'protocol_sha256': 'new25'})
            with patch.object(prep, 'validate_phase', return_value=({'protocol_sha256': 'new25'}, [])), \
                 patch.object(score, 'GateAPI', side_effect=AssertionError('Zero-request phase cannot construct API')):
                out = score.run_phase(root, 'rank')
            self.assertEqual(out['status'], 'terminal')
            self.assertEqual(out['expected_requests'], 0)
            self.assertEqual(score.phase_results(prep.Reader(), root, 'rank'), {})

    def test_resealed_support_ancestor_tamper_blocks_rank_and_selected_source(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            manifest, expected = phase_graph(root)
            unit_path = root / 'phases/support/units/c0_q0.json'
            unit_path.unlink()
            ref = prep.save_new(unit_path, {**expected['support'], 'scientific_input': 'tampered'})
            pm_path = root / 'phases/support/manifest.json'
            pm = prep.Reader().read(pm_path)
            pm_path.unlink()
            prep.save_new(pm_path, {**pm, 'units': [ref]})
            sp = root / 'selected_inputs.json'
            prep.save_new(sp, {'schema': prep.SOURCE_SCHEMA, 'status': 'sealed', 'arms': list(prep.ARMS),
                'parent_run': str(root), 'selector_manifest': prep.pointer(root / 'run_manifest.json'),
                'audited_bindings': [], 'jobs': []})
            with patch.object(prep, 'manifest', return_value=manifest), \
                 patch.object(prep, 'inputs', return_value=(manifest, {}, [])), \
                 patch.object(prep, 'phase_values', side_effect=lambda reader, run, phase: iter([(expected[phase], [])])):
                with self.assertRaisesRegex(ValueError, 'Scientific unit'):
                    prep.validate_phase(root, 'rank')
                with self.assertRaisesRegex(ValueError, 'Scientific unit'):
                    prep.validate_source(sp)

    def test_snapshot_graph_keeps_all_five_phase_bindings(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            manifest, expected = phase_graph(root)
            reader = prep.Reader()
            with patch.object(prep, 'inputs', return_value=(manifest, {}, [])), \
                 patch.object(prep, 'phase_values', side_effect=lambda reader, run, phase: iter([(expected[phase], [])])):
                prep.validate_phase(root, 'rank', reader=reader)
            for phase in prep.PHASES:
                self.assertIn(str((root / 'phases' / phase / 'manifest.json').resolve()), reader.refs)
                self.assertIn(str((root / 'phases' / phase / 'units/c0_q0.json').resolve()), reader.refs)

    def test_manifest_rechecks_new_writer_review_banks_and_fixed_protocol(self):
        from tools import audit_gemini25_ablation_banks_v1 as admission
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            writer = root / 'writer'
            writer.mkdir()
            prep.save_new(writer / 'run_manifest.json', {})
            review = prep.save_new(writer / 'review.json', {})
            resources = {'fixture': 'new25resources'}
            admitted, banks = [], []
            for ci in range(10):
                native_dir = writer / f'conv_{ci}/session/native'
                ref = prep.save_new(writer / f'conv_{ci}/bank.json', {'native_dir': str(native_dir)})
                admitted.append({'conversation': ci, 'bank': ref, 'native_inventory': {}})
                banks.append({'conversation': ci, 'bank': ref, 'native_inventory': {},
                              'native_source': str(native_dir), 'native_copy': str(root / f'conv_{ci}/native')})
            qs, dates = prep.cohort()
            protocol = prep.fixed_protocol()
            base = {'schema': prep.SCHEMA, 'arms': list(prep.ARMS), 'code_sha256': {},
                'dataset_sha256': prep.SOURCE_HASH, 'r03_spec': prep.pointer(prep.R03_SPEC),
                'cohort_sha256': prep.digest(qs), 'reference_dates': dates, 'protocol': protocol,
                'protocol_sha256': prep.digest(protocol), 'writer_manifest': prep.pointer(writer / 'run_manifest.json'),
                'bank_review': review, 'resources': resources, 'banks': banks,
                'official_metric_provenance': prep.verified_official_metric()[1]}
            path = root / 'run_manifest.json'
            prep.save_new(path, base)
            with patch.object(prep, 'CODE', ()), patch.object(admission, 'admit_review',
                return_value={'manifest': {'resources': resources}, 'banks': admitted}) as gate:
                self.assertEqual(prep.manifest(root)['banks'], banks)
                gate.assert_called_with(writer, review['path'], conversations=tuple(range(10)))
                for kind in ('old_model', 'missing_bank', 'different_bank', 'changed_writer_pointer', 'resources'):
                    value = deepcopy(base)
                    if kind == 'old_model':
                        value['protocol']['writer'] = 'gemini-3.8-flash'
                        value['protocol_sha256'] = prep.digest(value['protocol'])
                    elif kind == 'missing_bank':
                        value['banks'].pop()
                    elif kind == 'different_bank':
                        value['banks'][0]['bank'] = value['banks'][1]['bank']
                    elif kind == 'changed_writer_pointer':
                        value['writer_manifest']['sha256'] = 'old38'
                    else:
                        value['resources'] = {'fixture': 'different'}
                    path.unlink()
                    prep.save_new(path, value)
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        prep.manifest(root)

    def replay_dispatch_fixture(self, name):
        with patch.object(dispatch_fixtures, 'prepare', prep), patch.object(dispatch_fixtures, 'score', score):
            case = dispatch_fixtures.DispatchTests(name)
            getattr(case, name)()

    def test_full_raw_producer_consumer_and_same_phase_cache(self):
        self.replay_dispatch_fixture('test_real_exact_cache_producer_and_consumer_raw_replay')

    def test_known_transport_failure_requires_uniform_recovery(self):
        self.replay_dispatch_fixture('test_exhausted_transport_consumer_without_ledger_event_is_known_failure')

    def test_partial_transport_and_missing_success_raw_stay_unknown(self):
        self.replay_dispatch_fixture('test_partial_transport_and_http200_without_raw_stay_unknown')

    def test_raw_bad_scores_and_changed_signature_rejected(self):
        self.replay_dispatch_fixture('test_failed_raw_replays_parser_and_changed_signature_rejected')

    def test_consumer_context_and_request_hash_pair_required(self):
        self.replay_dispatch_fixture('test_complete_result_requires_its_consumer_event')
        self.replay_dispatch_fixture('test_ledger_hash_and_context_must_be_admitted_as_a_pair')


if __name__ == '__main__':
    unittest.main()
