"""Independent offline checks of Writer dispatch and admission boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from tools import run_gemini25_ablation_writer_v1 as writer
from tools import audit_gemini25_ablation_banks_v1 as audit
from jev_mem.local_bm25_context_v2 import forbid_network
from jev_mem import gemini25_ablation_provider_v1 as provider


class IndependentWriterReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gemini25_writer_review_')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        audit.idx.HASHES.clear()
        self.addCleanup(audit.idx.HASHES.clear)
        self.credentials = patch('jev_mem.providers.load_credentials',
                                 side_effect=AssertionError('Offline: no vault'))
        self.credentials.start()
        self.addCleanup(self.credentials.stop)

    def fixtures(self, conversations):
        (self.run / 'run_manifest.json').write_text('{}', encoding='utf-8')
        manifest = {'protocol_sha256': 'fixture'}
        rows, semantic_rows = [], []
        for ci in conversations:
            folder = self.run / f'conv_{ci}/session'
            journal_path = folder / 'journal.json'
            writer.grouped.save_new(journal_path, {'fixture': True})
            journal = audit.idx.pointer(journal_path)
            bank = {'complete': True, 'conversation': ci, 'arm': 'session',
                    'protocol_sha256': 'fixture', 'journal_files': [journal]}
            writer.grouped.save_new(folder / 'bank.json', bank)
            bank_ref = audit.idx.pointer(folder / 'bank.json')
            state = {'status': 'complete', 'protocol_sha256': 'fixture',
                     'conversation': ci, 'arm': 'session', 'bank': bank_ref}
            (folder / 'write_status.json').write_text(json.dumps(state), encoding='utf-8')
            writer.grouped.save_new(folder / 'write_outcome.json', {
                **state, 'write_status': audit.idx.pointer(folder / 'write_status.json')})
            row = {'conversation': ci, 'arm': 'session', 'bank': bank_ref,
                   'strict_passed': True, 'source_generation_persistence_passed': True,
                   'orphan_fact_ids': [], 'native_dir': str(folder / 'native'),
                   'native_inventory': {}, 'journals': [{'journal': journal}],
                   'semantic_review_targets': [{'journal': journal}]}
            rows.append(row)
            semantic_rows.append({'conversation': ci, 'arm': 'session', 'bank': bank_ref,
                                  'notes': ['Offline review fixture only.'],
                                  'blocking_implementation_error': False,
                                  'reviewed_journals': [journal]})
        structural = {'schema': audit.SCHEMA, 'strict_passed': True, 'exceptions': [],
                      'protocol_sha256': 'fixture', 'conversations': conversations,
                      'writer_manifest': audit.idx.pointer(self.run / 'run_manifest.json'),
                      'banks': rows, 'source_file_hashes': {}}
        structural_path = self.run / 'structural.json'
        writer.grouped.save_new(structural_path, structural)
        semantic = {'schema': audit.SEMANTIC_SCHEMA, 'status': 'reviewed_with_limitations',
                    'protocol_sha256': 'fixture', 'qa_gold_used': False,
                    'structural_audit': audit.idx.pointer(structural_path),
                    'banks': semantic_rows}
        semantic_path = self.run / 'semantic.json'
        writer.grouped.save_new(semantic_path, semantic)
        return manifest, structural_path, semantic_path

    def test_controller_dispatches_picklable_new_worker_for_two_session_banks(self):
        self.assertIs(pickle.loads(pickle.dumps(writer.worker)), writer.worker)
        self.assertIs(writer.run_write.__globals__['worker'], writer.worker)
        self.assertIs(writer.run_write.__globals__['manifest'], writer.manifest)
        self.assertIs(writer.run_write.__globals__['development_review'], writer.development_review)
        self.assertEqual(writer.run_write.__globals__['ARMS'], ('session',))
        calls = []

        class Future:
            def __init__(self, ci):
                self.ci = ci

            def result(self):
                return {'conversation': self.ci, 'arm': 'session', 'status': 'complete'}

        class Pool:
            def __init__(self, *, max_workers):
                calls.append(('workers', max_workers))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, run, ci, arm):
                calls.append((fn, ci, arm))
                return Future(ci)

        controller = writer.clone(writer.run_write,
            manifest=lambda run: {'protocol_sha256': 'fixture'},
            ProcessPoolExecutor=Pool, as_completed=lambda tasks: tasks)
        with forbid_network() as guards:
            result = controller(self.run, [0, 1], ['session'])
            self.assertEqual((result['jobs'], result['complete']), (2, 2))
            self.assertEqual(calls, [('workers', 2), (writer.worker, 0, 'session'),
                                     (writer.worker, 1, 'session')])
            self.assertFalse(any(guard.call_count for guard in guards))
        state = json.loads((self.run / 'controller_status.json').read_text(encoding='utf-8'))
        self.assertTrue(state['terminal'])
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['schema'], writer.SCHEMA)

    def test_failed_terminal_banks_block_audit_before_native_or_prompt_reads(self):
        manifest = {'protocol_sha256': 'fixture'}
        for ci in [0, 1]:
            folder = self.run / f'conv_{ci}/session'
            folder.mkdir(parents=True)
            state = {'status': 'failed', 'finished_utc': 'fixture'}
            (folder / 'write_status.json').write_text(json.dumps(state), encoding='utf-8')
            writer.grouped.save_new(folder / 'write_outcome.json', {
                'status': 'failed', 'protocol_sha256': 'fixture',
                'write_status': audit.idx.pointer(folder / 'write_status.json')})
        with patch.object(writer, 'manifest', return_value=manifest), \
             patch.object(audit.frozen.strict, 'prompt_environment', side_effect=AssertionError('No prompt reads')), \
             patch.object(audit.idx, 'load_extractor', side_effect=AssertionError('No NLP reads')), \
             forbid_network() as guards:
            result = audit.audit(self.run, [0, 1])
            self.assertEqual(result['status'], 'blocked_terminal_writer_failure')
            self.assertFalse(result['strict_passed'])
            self.assertEqual((result['banks'], result['model_calls']), ([], 0))
            self.assertEqual(len(result['failures']), 2)
            self.assertFalse(any(guard.call_count for guard in guards))

    def test_two_bank_publisher_and_admission_bind_new_schemas(self):
        manifest, structural, semantic = self.fixtures([0, 1])
        with patch.object(writer, 'manifest', return_value=manifest), forbid_network() as guards:
            result = audit.publish(self.run, structural, semantic)
            self.assertEqual(result['banks'], 2)
            value = audit.admit_review(self.run, conversations=[0, 1])
            self.assertEqual(value['review']['schema'], audit.DEVELOPMENT_SCHEMA)
            self.assertEqual(len(value['banks']), 2)
            self.assertEqual({r['arm'] for r in value['banks']}, {'session'})
            self.assertFalse(any(guard.call_count for guard in guards))

    def test_ten_bank_publisher_and_admission_bind_new_schemas(self):
        manifest, structural, semantic = self.fixtures(list(range(10)))
        with patch.object(writer, 'manifest', return_value=manifest), forbid_network() as guards:
            result = audit.publish(self.run, structural, semantic)
            self.assertEqual(result['banks'], 10)
            value = audit.admit_review(self.run)
            self.assertEqual(value['review']['schema'], audit.FULL_SCHEMA)
            self.assertEqual(len(value['banks']), 10)
            self.assertEqual({r['conversation'] for r in value['banks']}, set(range(10)))
            self.assertFalse(any(guard.call_count for guard in guards))

    def test_publisher_refuses_semantic_block_and_non_strict_audit(self):
        manifest, structural, semantic = self.fixtures([0, 1])
        original = audit.idx.verify(semantic)
        blocked = {k: deepcopy(v) for k, v in original.items() if k != 'content_sha256'}
        blocked['banks'][0]['blocking_implementation_error'] = True
        bad_semantic = self.run / 'blocked_semantic.json'
        writer.grouped.save_new(bad_semantic, blocked)
        with patch.object(writer, 'manifest', return_value=manifest), forbid_network():
            with self.assertRaisesRegex(ValueError, 'implementation error'):
                audit.publish(self.run, structural, bad_semantic)
            self.assertFalse((self.run / 'development_review.json').exists())
            structural_value = {k: v for k, v in audit.idx.verify(structural).items()
                                if k != 'content_sha256'}
            structural_value['strict_passed'] = False
            bad_structural = self.run / 'non_strict.json'
            writer.grouped.save_new(bad_structural, structural_value)
            with self.assertRaisesRegex(ValueError, 'Strict bound'):
                audit.publish(self.run, bad_structural, semantic)
            self.assertFalse((self.run / 'development_review.json').exists())

    def test_admission_rechecks_resealed_semantic_per_bank_obligations(self):
        root = self.run
        mutations = {
            'blocking_error': lambda row: row.update(blocking_implementation_error=True),
            'missing_notes': lambda row: row.update(notes=[]),
            'missing_required_journal': lambda row: row.update(reviewed_journals=[]),
            'wrong_bank': lambda row: row.update(bank={'path': 'outside', 'sha256': '0' * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                self.run = root / name
                self.run.mkdir()
                audit.idx.HASHES.clear()
                manifest, structural, semantic = self.fixtures([0, 1])
                with patch.object(writer, 'manifest', return_value=manifest), forbid_network():
                    audit.publish(self.run, structural, semantic)
                    gate_path = self.run / 'development_review.json'
                    gate = {k: v for k, v in audit.idx.verify(gate_path).items()
                            if k != 'content_sha256'}
                    value = {k: deepcopy(v) for k, v in audit.idx.verify(semantic).items()
                             if k != 'content_sha256'}
                    mutate(value['banks'][0])
                    bad = self.run / 'resealed_semantic.json'
                    writer.grouped.save_new(bad, value)
                    gate['semantic_review'] = audit.idx.pointer(bad)
                    gate_path.write_text(json.dumps({**gate, 'content_sha256': audit.idx.digest(gate)}),
                                         encoding='utf-8')
                    # Match a fresh admission process, not an in-process old-hash rejection.
                    audit.idx.HASHES.clear()
                    with self.assertRaises(ValueError):
                        audit.validate_review(self.run, manifest, development=True)
        self.run = root

    def test_raw_audit_rejects_budget_temperature_and_resolved_model_drift(self):
        import hashlib
        native = self.run / 'native'
        (native / 'api_cache').mkdir(parents=True)
        prompt, system = 'Unchanged session fixture', 'Unchanged system fixture'
        body = provider.generation_body(prompt, system=system, json_output=True,
                                         max_output=provider.WRITER_MAX_OUTPUT)
        journal = {'builder_call_count': 1, 'session_id': 's1', 'llm_calls': [{
            'kwargs': {'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': prompt}]},
            'response': '{"memory": []}', 'parsed_memory': []}]}
        for mutation in ('budget', 'thinking_level', 'temperature', 'resolved_model'):
            with self.subTest(mutation=mutation), forbid_network():
                audit.idx.HASHES.clear()
                changed = deepcopy(body)
                if mutation == 'budget':
                    changed['generationConfig']['thinkingConfig'] = {'thinkingBudget': 1024}
                elif mutation == 'thinking_level':
                    changed['generationConfig']['thinkingConfig'] = {'thinkingLevel': 'low'}
                elif mutation == 'temperature':
                    changed['generationConfig']['temperature'] = 0
                signature = {'provider': 'gemini', 'model': provider.MODEL,
                             'operation': 'generateContent', 'body': changed}
                key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                response = {'modelVersion': 'gemini-3.8-flash' if mutation == 'resolved_model' else provider.MODEL,
                            'candidates': [{'finishReason': 'STOP', 'content': {
                                'parts': [{'text': '{"memory": []}'}]}}]}
                (native / 'api_cache' / (key + '.json')).write_text(json.dumps({
                    'request': signature, 'response': response, 'stage': 'mem0_extract'}), encoding='utf-8')
                event = {'stage': 'mem0_extract', 'status': 200, 'provider': 'gemini',
                         'model': provider.MODEL, 'operation': 'generateContent', 'request_hash': key,
                         'context': {'session_id': 's1', 'conversation': 0, 'method': 'session'}}
                with self.assertRaises((ValueError, RuntimeError)):
                    audit.check_generation(journal, [event], native, prompt,
                        {'ADDITIVE_EXTRACTION_PROMPT': system}, 0, 'session')


if __name__ == '__main__':
    unittest.main()
