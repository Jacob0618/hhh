"""Synthetic overlay/projection wiring tests; no live QA, gold or API access."""
from collections import Counter
from copy import deepcopy
import marshal
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import prepare_source_set_rounding_r11 as overlay
from tools import project_source_set_strict_r11 as projection
from tools import run_source_set_rounding_qa_r11 as qa
from tools import run_source_set_qa_r11 as prior_qa
from tools import resume_source_set_r11_reviewed_v1 as resume
from jev_mem import source_set_r11 as core
from test_source_set_rounding_r11 import memories, candidates, response


class DictReader:
    def __init__(self, values): self.values = values
    def bound(self, ref, **kwargs): return deepcopy(self.values[ref['path']])


class WiringTests(unittest.TestCase):
    def test_overlay_keeps_strict_failed_reference_and_rejects_nonmax(self):
        policy = {'p': {'instructions': 'Each pair only.', 'criteria_template': '{first} and {second}',
            'presentation_seed': core.POLICY['presentation_seed']}}
        values = {}; jobs = []
        for i, r in enumerate([response([.93, .05, .01, 0, 0, 0]),
            response([.27, .27, .14, .13, .11, .07]),
            response([.29, .30, .18, .16, .05, .02], choice='p00')]):
            qid = f'c0_q{i}'; sig = core.build_p_request(qid, 'Synthetic Q', None, memories(), candidates(), policy)
            selection = {'path': 'selection' + str(i)}; raw = {'path': 'raw' + str(i)}; prop = {'path': 'prop' + str(i)}
            jobs.append({'job_id': 'p/' + qid, 'query_id': qid, 'method': 'p',
                'protocol_sha256': 'strict_protocol', 'selection': selection, 'selected_facts': memories(),
                'unchanged_ordered_reader_facts': True})
            values[selection['path']] = {'schema': 'strict', 'raw_response': raw, 'proposal': prop,
                'decision': {'score_status': 'failed'}, 'selected_facts': memories()}
            values[raw['path']] = {'request': sig, 'response': r}
            values[prop['path']] = {'original_job': {'selected_facts': memories()},
                'locator': {'candidates': candidates()}}
        snapshot = deepcopy(jobs)
        rows, counts = overlay.derive(DictReader(values), {'jobs': jobs}, 'overlay_protocol')
        self.assertEqual(jobs, snapshot)
        self.assertEqual(counts['new_pair_inputs'], 1)
        self.assertEqual(counts['P_rounded_mass'], 2)
        self.assertEqual(counts['P_rejected'], 1)
        self.assertEqual(rows[0][1]['schema'], 'source-set-r11-rounding-selection-v1')
        self.assertEqual(rows[0][1]['strict_selection'], jobs[0]['selection'])
        self.assertEqual(rows[0][0]['strict_protocol_sha256'], 'strict_protocol')
        self.assertFalse(rows[0][1]['unchanged_ordered_reader_facts'])
        self.assertTrue(rows[1][1]['unchanged_ordered_reader_facts'])
        self.assertIsNone(rows[2][1])
        self.assertEqual(rows[2][0]['selected_facts'], memories())
        original_counts = {'p_score_failed': 3, 'p_score_complete': 0,
            'p_changed_queries': 0, 'p_attachments': 0}
        effective = overlay.effective_counts({'jobs': jobs, 'selection_counts': original_counts}, rows)
        self.assertEqual(effective, {'p_score_failed': 1, 'p_score_complete': 2,
            'p_changed_queries': 1, 'p_attachments': 2})
        self.assertEqual(original_counts['p_score_failed'], 3)

    def test_new_qa_bytecode_preserved_and_resume_has_no_scoring_entrypoint(self):
        for name in ('prepare', 'manifest', 'one', 'recovery_jobs', 'prepare_recovery',
            'terminal', 'reader_call', 'compatibility', 'resolve_pair', 'known_failure', 'run'):
            self.assertEqual(marshal.dumps(getattr(qa, name).__code__), marshal.dumps(getattr(prior_qa, name).__code__))
        self.assertIs(qa.evaluator, prior_qa.evaluator)
        self.assertEqual(qa.ARMS, ('baseline', 'g', 'p'))
        self.assertNotIn('scoring', resume.main.__code__.co_names)
        self.assertNotIn('score_source_set_r11', resume.main.__code__.co_names)

    def test_strict_projection_changed_p_uses_actual_n0_not_compatible_score(self):
        oldjobs = []; newjobs = []; rows = []
        for n in range(1540):
            q = f'c0_q{n}'
            for arm in ('baseline', 'g', 'p'):
                job = {'job_id': arm + '/' + q, 'method': arm, 'query_id': q,
                    'conversation': 0, 'question': 'Synthetic Q', 'reference': 'Synthetic reference',
                    'category': 1, 'reference_date': None, 'selected_facts': memories(),
                    'selected_ids': [str(i) for i in range(20)], 'selection': {'path': 'strict/' + arm + '/' + q}}
                # Every P except q0 has the same strict/effective synthetic suffix.
                if arm == 'p' and n != 0: job['selected_facts'][0]['text'] += ' existing suffix'
                oldjobs.append(job); effective = deepcopy(job)
                if arm == 'p' and n == 0: effective['selected_facts'][0]['text'] += ' compatibility suffix'
                newjobs.append(effective)
                rows.append({'job_id': job['job_id'], 'method': arm, 'query_id': q,
                    'reader_answer': 'Compatible answer' if arm == 'p' and n == 0 else 'Actual answer',
                    'scores': {'judge_correct': 1 if arm == 'p' and n == 0 else 0},
                    'reader_artifact': {'path': 'actual/' + job['job_id'] + '/reader'},
                    'judge_artifact': {'path': 'actual/' + job['job_id'] + '/judge'}})
        original = {'jobs': oldjobs, 'selection_counts': {'p_attachments': 3078},
            'selector_manifest': {'path': 'strict_selector', 'sha256': 'h'},
            'source_set_protocol': {'path': 'strict_protocol', 'sha256': 'h'}}
        effective = {'jobs': newjobs, 'strict_source': {'path': 'strict_source', 'sha256': 'h'}}
        actual = {'audit_status': 'no_invalid_artifacts', 'rounding_sensitivity': True,
            'source_manifest': {'path': 'effective_source', 'sha256': 'h'},
            'slots_by_view': {'final_with_uniform_technical_recovery': rows}, 'input_sha256': {}}
        objects = {'effective_source': effective, 'strict_source': original, 'strict_protocol': {}}
        output = {}
        with patch.object(projection, 'verify', return_value=deepcopy(actual)), \
             patch.object(projection, 'bound', side_effect=lambda ref: deepcopy(objects[ref['path']])), \
             patch.object(projection, 'pointer', side_effect=lambda p: {'path': str(p), 'sha256': 'h'}), \
             patch.object(projection, 'sha', return_value='h'), \
             patch.object(projection, 'save_new', side_effect=lambda p, value: output.update(value=value)), \
             patch.object(projection.strict_summary, 'augmentation_audit', return_value={'synthetic': True}), \
             patch.object(projection.strict_summary, 'aggregate', side_effect=lambda rows, **kw: {'n': len(rows)}), \
             patch.object(projection.strict_summary, 'stopping_result', return_value={}):
            result = projection.project(Path('actual_summary.json'), Path('strict_projection.json'))
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(output['value']['schema'], 'source-set-r11-strict-exact-qa-projection-v1')
        projected = {r['job_id']: r for r in output['value']['slots_by_view']['final_with_uniform_technical_recovery']}
        changed = projected['p/c0_q0']
        self.assertEqual(changed['scores']['judge_correct'], 0)
        self.assertEqual(changed['reader_answer'], 'Actual answer')
        self.assertEqual(changed['strict_exact_projection']['producer_job_id'], 'baseline/c0_q0')
        self.assertEqual(changed['reader_artifact']['path'], 'actual/baseline/c0_q0/reader')
        unchanged = projected['p/c0_q1']
        self.assertEqual(unchanged['strict_exact_projection']['producer_job_id'], 'p/c0_q1')
        self.assertEqual(len(projected), 4620)


if __name__ == '__main__':
    unittest.main()
