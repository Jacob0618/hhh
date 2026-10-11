"""Focused continuation evidence, resource-boundary and controller tests."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import gemini25_billing_admission_v1 as billing
from tools import resume_gemini25_ablation_after_billing_v1 as resume


class ContinuationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, *, independent_status='review_passed', independent_sha='adapter-sha'):
        failure_path = self.root / 'failure.json'
        independent_path = self.root / 'independent.json'
        resource_path = self.root / 'resource.json'
        failure = {'status': 'review_passed_original_firstpass_terminal_resource_failure',
                   'run_dir': str(billing.RUN), 'phase': 'r03',
                   'firstpass_dir': str(billing.FIRSTPASS), 'firstpass_file_sha256': {'a': 'b'}}
        independent = {'schema': 'gemini25-ablation-billing-independent-review-v1',
                       'status': independent_status, 'model_calls': 0,
                       'code_sha256': {billing.TECHNICAL_CODE[0]: independent_sha,
                                      billing.TECHNICAL_CODE[1]: 'wrapper-sha'}}
        # External reviews need not carry a content seal; their bytes are pinned.
        failure_path.write_text(json.dumps(failure), encoding='utf-8')
        independent_path.write_text(json.dumps(independent), encoding='utf-8')
        billing.save_new(resource_path, {'schema': 'gemini25-billing-resource-confirmation-v1',
                         'firstpass_run': str(billing.FIRSTPASS), 'no_probe': True, 'model_calls': 0})
        expected = {'schema': billing.SCHEMA, 'policy': deepcopy(billing.POLICY),
                    'failure_review': billing.pointer(failure_path),
                    'independent_review': billing.pointer(independent_path),
                    'resource_confirmation': billing.pointer(resource_path),
                    'firstpass_file_sha256': {'a': 'b'}, 'bindings': {},
                    'technical_code_sha256': {billing.TECHNICAL_CODE[0]: 'adapter-sha',
                                             billing.TECHNICAL_CODE[1]: 'wrapper-sha'}}
        review_path = self.root / 'ready.json'
        billing.save_new(review_path, {**expected, 'status': 'ready'})
        return review_path, resource_path, expected

    def test_unsealed_independent_evidence_is_accepted_by_exact_review_binding(self):
        path, resource, expected = self.fixture()
        with patch.object(billing, 'RESOURCE', resource), patch.object(billing, 'review_requirements', return_value=expected):
            guard = billing.BillingAdmission(path)
        self.assertEqual(guard.firstpass_inventory, {'a': 'b'})

    def test_independent_review_must_actually_pass_for_current_technical_code(self):
        for status, sha in [('pending', 'adapter-sha'), ('review_passed', 'wrong-code')]:
            with self.subTest(status=status, sha=sha), tempfile.TemporaryDirectory() as td:
                self.root = Path(td)
                path, resource, expected = self.fixture(independent_status=status, independent_sha=sha)
                with patch.object(billing, 'RESOURCE', resource), patch.object(billing, 'review_requirements', return_value=expected):
                    with self.assertRaisesRegex(ValueError, 'Independent technical review'):
                        billing.BillingAdmission(path)

    def test_ready_review_cannot_relax_policy_or_use_a_different_inventory(self):
        path, resource, expected = self.fixture()
        bad = deepcopy(expected); bad['firstpass_file_sha256'] = {'a': 'changed'}
        with patch.object(billing, 'RESOURCE', resource), patch.object(billing, 'review_requirements', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'Exact independent'):
                billing.BillingAdmission(path)

    def test_qa_resource_failures_block_but_known_transport_does_not_change_policy(self):
        ledger = self.root / 'conv_0/api/api_calls.jsonl'
        ledger.parent.mkdir(parents=True)
        for status in (401, 402, 403):
            ledger.write_text(json.dumps({'status': status, 'cache_hit': False}) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'resource/auth'):
                resume.qa_resource_gate(self.root)
        ledger.write_text(json.dumps({'status': 'transport_error', 'cache_hit': False}) + '\n', encoding='utf-8')
        self.assertEqual(resume.qa_resource_gate(self.root)['physical_attempts'], 1)

    def test_partial_qa_ledger_blocks(self):
        ledger = self.root / 'conv_0/api/api_calls.jsonl'
        ledger.parent.mkdir(parents=True); ledger.write_text('{"status":200}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Partial QA ledger'):
            resume.qa_resource_gate(self.root)

    def test_exclusive_runtime_namespace_rejects_already_started_recovery(self):
        with patch.object(billing, 'RUN', self.root / 'source'), patch.object(billing, 'QA', self.root / 'qa'), \
                patch.object(billing, 'LIVE', self.root / 'live'):
            resume.unused_namespaces()
            (billing.RUN / 'phases/r03/recovery_v1').mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, 'One reviewed continuation'):
                resume.unused_namespaces()

    def test_real_process_gate_distinguishes_venv_ancestor_from_competing_controller(self):
        own = {'pid': os.getpid(), 'parent_pid': 111, 'creation_utc': 'now',
               'command_line': 'python.exe tools/resume_gemini25_ablation_after_billing_v1.py run'}
        parent = {**own, 'pid': 111, 'parent_pid': 222}
        with patch.object(resume, 'controller_probe', return_value=[own, parent]):
            self.assertEqual(resume.assert_single_controller()['duplicate_controllers'], 0)
        other = {**own, 'pid': 333, 'parent_pid': 444,
                 'command_line': 'python.exe tools/run_gemini25_ablation_pipeline_v1.py run'}
        with patch.object(resume, 'controller_probe', return_value=[own, parent, other]):
            with self.assertRaisesRegex(ValueError, 'Another current'):
                resume.assert_single_controller()

    def test_process_probe_missing_command_or_current_identity_fails_closed(self):
        with patch.object(resume, 'controller_probe', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'identity unavailable'):
                resume.assert_single_controller()
        own = {'pid': os.getpid(), 'parent_pid': 1, 'creation_utc': 'now', 'command_line': 'python.exe test.py'}
        foreign = {'pid': 77, 'parent_pid': 1, 'creation_utc': 'now', 'command_line': None}
        with patch.object(resume, 'controller_probe', return_value=[own, foreign]):
            with self.assertRaisesRegex(ValueError, 'command line unavailable'):
                resume.assert_single_controller()


if __name__ == '__main__':
    unittest.main()
