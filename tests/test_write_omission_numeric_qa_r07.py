"""Numeric native compatibility must not alter frozen QA semantics or raw auditing."""
import copy
import importlib.util
from pathlib import Path
from types import FunctionType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from tools import run_write_omission_numeric_qa_r07 as runner
from tools import summarize_write_omission_numeric_r07 as summary
from tools import run_write_omission_qa_r07 as original_r07
from tools import run_evidence_exchange_qa_r01 as prior

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('numeric_original_r07_tests', ROOT/'tests/test_write_omission_qa_r07.py')
oldtests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oldtests)
base_scope = {**vars(oldtests.oldtests), 'runner': runner, 'summary': summary}
base_methods = {name: FunctionType(f.__code__, base_scope, name, f.__defaults__, f.__closure__)
    for name, f in vars(oldtests.oldtests.AdapterTests).items() if isinstance(f, FunctionType)
    and name != 'test_stop_rule_uses_fixed_1540_nulls_and_strict_boundary'}
InheritedStageTests = type('InheritedStageTests', (unittest.TestCase,), base_methods)
scope = {**vars(oldtests), 'runner': runner, 'summary': summary, 'R07InheritedBehaviorTests': InheritedStageTests}
methods = {name: FunctionType(f.__code__, scope, name, f.__defaults__, f.__closure__)
    for name, f in vars(oldtests.R07Contracts).items() if isinstance(f, FunctionType)
    and name not in ('test_source_validator_delegates_r07_schema_not_source_prefix_schema',
                     'test_summary_accepts_materializer_protocol_field_not_old_selector_field')}
NumericRawReplayTests = type('NumericRawReplayTests', (unittest.TestCase,), methods)

class NumericSourceSeamTests(unittest.TestCase):
    def source(self):
        return {'schema': runner.SOURCE_SCHEMA,
            'write_omission_protocol': {'path': 'science.json', 'sha256': 's'*64},
            'numeric_compatibility_protocol': {'path': 'tech.json', 'sha256': runner.POLICY['numeric_technical_protocol_sha256']},
            'effective_bank_gate': {'path': 'effective.json', 'sha256': 'e'*64},
            'selection_counts': {'jobs': 4620}}

    def test_only_new_numeric_gate_delegate_and_explicit_technical_bindings_accepted(self):
        source = self.source()
        calls = []
        fake = SimpleNamespace(validate_source=lambda p: calls.append(p) or source)
        with patch.dict(sys.modules, {'tools.materialize_write_omission_numeric_r07': fake}), \
             patch.object(runner, 'bound', return_value={}) as bound:
            self.assertIs(runner.source_validate('source.json'), source)
            self.assertEqual(calls, [Path('source.json')])
            self.assertEqual(bound.call_count, 2)
            for key in ('numeric_compatibility_protocol', 'effective_bank_gate'):
                bad = copy.deepcopy(source); del bad[key]
                fake.validate_source = lambda p, b=bad: b
                with self.assertRaisesRegex(ValueError, 'Explicit numeric source binding'):
                    runner.source_validate('old-strict-source.json')
            fake.validate_source = lambda p: {**source, 'numeric_compatibility_protocol': {'path':'wrong.json','sha256':'0'*64}}
            with self.assertRaisesRegex(ValueError, 'Numeric technical protocol differs'):
                runner.source_validate('other-tech-version.json')

    def test_summary_retains_original_failed_counts_and_new_gate_without_scoring_change(self):
        source = self.source()
        with patch.object(summary, '_summarize', return_value={'audit_status': 'no_invalid_artifacts'}), \
             patch.object(summary, 'read', return_value={'source_manifest': {'path':'source.json'}}), \
             patch.object(runner, 'source_validate', return_value=source), \
             patch.object(runner, 'proof_for', return_value={'reuse_revalidation': 'exact'}):
            result = summary.summarize('unused')
            self.assertEqual(result['original_native_firstpass'], {'complete':12, 'failed':4, 'not_run':4})
            self.assertEqual(result['numeric_compatibility_protocol'], source['numeric_compatibility_protocol'])
            self.assertEqual(result['effective_bank_gate'], source['effective_bank_gate'])
            self.assertEqual(result['write_omission_protocol'], source['write_omission_protocol'])
            self.assertEqual(result['selection_counts']['jobs'], 4620)
            source['selection_protocol'] = source.pop('write_omission_protocol')
            with self.assertRaisesRegex(ValueError, 'Explicit final write-omission'):
                summary.summarize('unused')

    def test_original_module_globals_and_scientific_serialization_unchanged(self):
        self.assertIsNot(runner._impl, original_r07._impl)
        self.assertEqual(original_r07._impl['SCHEMA'], original_r07.SCHEMA)
        self.assertEqual(original_r07._helpers['source_validate'], original_r07.source_validate)
        job = oldtests.oldtests.job()
        for stage, answer in (('reader','capture only'),('judge','No.')):
            self.assertEqual(runner.request_signature(job,stage,answer), original_r07.request_signature(job,stage,answer))
        self.assertIs(runner.evaluator, original_r07.evaluator)
        for field in ('conversation_workers','qa_threads_per_conversation'):
            self.assertEqual(runner.POLICY[field], original_r07.POLICY[field])

if __name__ == '__main__': unittest.main()
