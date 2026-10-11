"""Offline lifecycle failure cases; all mutations/deletions confined to temp fixtures."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from tools import baseline_lifecycle_v2 as life
from tools import run_five_baselines_reviewed_chain_v2 as launch
from tools.baselines_common_v1 import save_new, sha
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='baseline_lifecycle_offline_')
        self.root = Path(self.tmp.name).resolve(); self.method = 'simplemem'
        self.stack = ExitStack(); self.addCleanup(self.stack.close); self.addCleanup(self.tmp.cleanup)
        self.actual = life.ROOT
        self.stack.enter_context(patch.object(life, 'ROOT', self.root))
        self.stack.enter_context(patch.object(life, 'RUN', self.root / 'runs/five_baselines_gemini25_v1_20261008'))
        self.stack.enter_context(patch.object(life, 'stopped'))
        self.stack.enter_context(patch('jev_mem.providers.load_credentials', side_effect=AssertionError('credentials forbidden')))
        self.stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch('socket.socket.connect', side_effect=AssertionError('network forbidden')))
        self.folder = life.folder(self.method)

    def fixture(self, *, archive_now=True):
        for rel in life.TOOL_FILES:
            p = self.root / rel; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((self.actual / rel).read_bytes())
        dataset = self.root / 'data/locomo10.json'; save_new(dataset, {'fixture': True})
        save_new(self.root / 'configs/sdk20_full1540_locked_v1.json', {'fixture': 'locked cohort'})
        review = self.root / 'scientific.json'; save_new(review, {'status': 'passed'})
        self.m = {'method': self.method, 'questions': [{}] * 1540, 'dataset_sha256': sha(dataset),
                  'code_sha256': {p: sha(self.root / p) for p in life.TOOL_FILES}, 'upstream_sha256': {},
                  'review': life.binding(review)}
        save_new(self.folder / 'run_manifest.json', self.m)
        summary = self.folder / 'results/summary.json'
        save_new(summary, {'schema': 'five-baseline-result-v1', 'method': self.method, 'audit_status': 'no_invalid_artifacts',
                           'effective': {'slots': 1540, 'reader_valid': 1540, 'judge_valid': 1540}})
        save_new(self.folder / 'status.json', {'stage': 'completed_awaiting_result_review', 'summary_sha256': sha(summary)})
        save_new(self.folder / 'results/raw_request_audit.json', {'status': 'passed', 'inventory': [], 'errors': []})
        self.raw = self.folder / 'work/api_method/api_cache/raw.json'
        save_new(self.raw, {'request': {'operation': 'generateContent'}, 'response': {'fixture': 'retained'}})
        self.vector = self.folder / 'work/api_method/api_cache/vector.json'
        save_new(self.vector, {'request': {'operation': 'embedContent'}, 'response': {'values': [1.0, 2.0]}})
        self.index = self.folder / 'work/native/c0/cache/index.bin'; self.index.parent.mkdir(parents=True); self.index.write_bytes(b'cache')
        evidence = self.root / 'evidence.json'; save_new(evidence, {'fixture': 'review evidence'})
        self.result_review = self.root / 'result_review.json'
        save_new(self.result_review, {'status': 'passed', 'method': self.method, 'summary_sha256': sha(summary),
                                     'evidence_sha256': {str(evidence): sha(evidence)},
                                     'reviewed_run_inventory': {p: rec['sha256'] for p, rec in life.inventory(self.folder).items()}})
        self.tool_review = self.root / 'tool_review.json'
        save_new(self.tool_review, {'status': 'passed', 'code_sha256': life.code_hashes()})
        self.stack.enter_context(patch.object(life.base, 'manifest', return_value=self.m))
        self.stack.enter_context(patch.object(life.base, 'audit_requests', return_value={'status': 'passed', 'inventory': [], 'errors': []}))
        self.stack.enter_context(patch.object(life, 'replay_all', return_value={'status': 'passed', 'fixture_only': True, 'model_calls': 0}))
        if not archive_now:
            return
        life.archive(self.method, self.result_review, self.tool_review)
        self.plan = life.read(life.plan_path(self.method))
        self.cleanup = self.root / 'cleanup_review.json'
        save_new(self.cleanup, {'status': 'passed', 'method': self.method,
                               'cleanup_plan_sha256': sha(life.plan_path(self.method)), 'archive_sha256': self.plan['archive']['sha256']})

    def remove_fixture(self):
        work = (self.folder / 'work').resolve()
        self.assertTrue(work.is_relative_to(self.root)); self.assertEqual(work, self.folder / 'work')
        shutil.rmtree(work)
        out = life.output_dir(self.method)
        base = {'method': self.method, 'cleanup_plan_sha256': sha(life.plan_path(self.method)), 'cleanup_review_sha256': sha(self.cleanup)}
        save_new(out / 'cleanup_execution_intent.json', {**base, 'schema': 'baseline-single-root-v2-cleanup-intent-v1', 'target': str(work)})
        save_new(out / 'removal_receipt.json', {**base, 'schema': 'baseline-single-root-v2-removal-receipt-v1', 'removed_path': str(work)})

    def test_real_archive_completion_vector_disposal_and_code_retention(self):
        self.fixture(); life.verify_archive(self.method)
        with life.Resolver(self.plan['archive']['path'], self.plan['archive_manifest']['path']) as r:
            self.assertEqual(r.bytes(self.raw), self.raw.read_bytes())
            for path in (self.vector, self.index):
                with self.assertRaisesRegex(ValueError, 'not retained'): r.bytes(path)
                self.assertIn(str(path), r.manifest['work_inventory'])
            for path in (self.root / life.TOOL_FILES[0], self.root / 'evidence.json', self.root / 'data/locomo10.json'):
                self.assertEqual(r.bytes(path), path.read_bytes())
        self.remove_fixture(); self.assertEqual(life.complete(self.method, self.cleanup)['status'], 'complete')
        self.assertEqual(life.verify_completion(self.method)['status'], 'passed')
        (self.folder / 'work').mkdir()
        with self.assertRaisesRegex(ValueError, 'reappeared'): life.verify_completion(self.method)

    def test_work_or_result_tamper_blocks_before_deletion(self):
        self.fixture(); self.vector.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Work changed'): life.verify_archive(self.method)
        self.assertTrue(self.raw.exists())
        summary = self.folder / 'results/summary.json'; summary.write_text('{}')
        with self.assertRaises(ValueError): life.verify_archive(self.method)

    def test_forged_receipt_and_missing_intent_cannot_complete(self):
        self.fixture()
        save_new(self.folder / 'results/cleanup_receipt.json', {'not': 'a receipt'})
        with self.assertRaises((KeyError, ValueError)): life.verify_completion(self.method)
        with self.assertRaisesRegex(ValueError, 'remains'): life.complete(self.method, self.cleanup)

    def test_result_review_requires_external_nonempty_canonical_full_inventory(self):
        self.fixture(); original = life.read(self.result_review)
        for bad in (None, {}, {'results/summary.json': '0' * 64}, {str(self.root / 'outside.json'): '0' * 64},
                    {str(self.folder / 'status.json'): 'not-a-sha'}):
            value = {**original, 'reviewed_run_inventory': bad}
            self.result_review.write_text(json.dumps(value), encoding='utf-8')
            with self.subTest(inventory=bad), self.assertRaisesRegex(ValueError, 'complete absolute run inventory'):
                life.verify_result(self.method, self.result_review, live_inventory=False)
        self.result_review.write_text(json.dumps(original), encoding='utf-8')
        inside = self.folder / 'result_review_inside.json'; save_new(inside, original)
        with self.assertRaisesRegex(ValueError, 'live outside'):
            life.verify_result(self.method, inside, live_inventory=False)

    def test_changed_live_raw_rejected_even_if_summary_still_matches(self):
        self.fixture(archive_now=False)
        self.assertEqual(life.verify_result(self.method, self.result_review), self.m)
        self.raw.write_text('{"request":{"operation":"generateContent"},"response":{"changed":true}}')
        with self.assertRaisesRegex(ValueError, 'Current run differs'):
            life.verify_result(self.method, self.result_review)

    def test_consistently_resealed_raw_cannot_change_independently_reviewed_snapshot(self):
        self.fixture(); plan_path = life.plan_path(self.method)
        mp = Path(self.plan['archive_manifest']['path']); zp = Path(self.plan['archive']['path'])
        manifest = life.read(mp); raw_path = str(self.raw)
        with zipfile.ZipFile(zp) as z:
            payloads = {name: z.read(name) for name in z.namelist() if name != 'archive_manifest.json'}
        old = manifest['members'][raw_path]['member']
        self.raw.write_text('{"request":{"operation":"generateContent"},"response":{"changed":true}}', encoding='utf-8')
        rec = {'sha256': sha(self.raw), 'size': self.raw.stat().st_size}
        manifest['work_inventory'][raw_path] = rec
        manifest['members'][raw_path] = {**rec, 'member': 'blobs/' + rec['sha256']}
        payloads.pop(old); payloads['blobs/' + rec['sha256']] = self.raw.read_bytes()
        mp.write_text(json.dumps(manifest), encoding='utf-8')
        with zipfile.ZipFile(zp, 'w') as z:
            z.writestr('archive_manifest.json', mp.read_bytes())
            for member, raw in payloads.items(): z.writestr(member, raw)
        # Update every local integrity seal; independent result review stays original.
        self.plan['archive'] = life.binding(zp); self.plan['archive_manifest'] = life.binding(mp)
        for p in (mp, zp): self.plan['keep_sha256'][str(p)] = sha(p)
        plan_path.write_text(json.dumps(self.plan), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Archived run differs from independently reviewed full inventory'):
            life.verify_archive(self.method)
        self.remove_fixture()
        with self.assertRaisesRegex(ValueError, 'Archived run differs from independently reviewed full inventory'):
            life.verify_archive(self.method, require_work=False)

    def test_scope_reparse_and_foreign_paths_rejected(self):
        with self.assertRaises(ValueError): life.folder('amem')
        with self.assertRaises(ValueError): life.safe_path(self.root.parent / 'outside')
        with self.assertRaises(ValueError): life.safe_path(self.root / '..' / self.root.name / 'x')
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Reparse'): life.safe_path(self.root / 'x')

    def test_crc_valid_altered_blob_and_added_members_rejected(self):
        self.fixture(); target = Path(self.plan['archive']['path']); mp = Path(self.plan['archive_manifest']['path'])
        with zipfile.ZipFile(target) as z: records = {p: z.read(p) for p in z.namelist()}
        blob = next(p for p in records if p.startswith('blobs/')); records[blob] = b'tampered but valid ZIP CRC'
        with zipfile.ZipFile(target, 'w') as z:
            for p, raw in records.items(): z.writestr(p, raw)
        with self.assertRaisesRegex(ValueError, 'SHA'): life.Resolver(target, mp)
        with zipfile.ZipFile(target, 'a') as z: z.writestr('unexpected', b'extra')
        with self.assertRaisesRegex(ValueError, 'ZIP entries'): life.Resolver(target, mp)

    def test_all_cli_and_module_controllers_detected_but_own_pid_exempt(self):
        names = ['run_five_baselines_gemini25_v1', 'run_five_baselines_format_v2',
                 'run_five_baselines_amem_native_v3', 'run_five_baselines_amem_native_v3_admission_v2',
                 'run_five_baselines_reviewed_chain_v2']
        for name in names:
            for command in (f'python.exe "D:\\a b\\{name}.py" run --method simplemem',
                            f'pythonw.exe -m tools.{name} run --method simplemem'):
                row = {'Name': 'pythonw.exe', 'ProcessId': 42, 'CommandLine': command}
                current={'Name':'python.exe','ProcessId':41,'CommandLine':'python -m unittest'}
                self.assertEqual(life.active_controllers([row,current], 41), [row])
                self.assertEqual(life.active_controllers([row], 42), [])

    def test_venv_launcher_exemption_is_exact_and_cannot_hide_other_controller(self):
        from tools import baseline_controller_guard_v2 as guard
        command=' "D:\\jev mem\\tools\\run_five_baselines_reviewed_chain_v2.py" run --method simplemem'
        parent={'Name':'python.exe','ProcessId':80,'CommandLine':'"venv.exe"'+command,
                'ExecutablePath':str(guard.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'),
                'CreatedUtc':'2026-10-08T20:00:00+00:00'}
        child={'Name':'python.exe','ProcessId':81,'ParentProcessId':80,'CommandLine':'"base.exe"'+command,
               'ExecutablePath':sys._base_executable,'CreatedUtc':'2026-10-08T20:00:01+00:00'}
        self.assertEqual(guard.active_controllers([parent,child],81),[])
        foreign={**parent,'ProcessId':82}
        self.assertEqual(guard.active_controllers([parent,child,foreign],81),[foreign])
        altered={**parent,'ExecutablePath':'C:/wrong/python.exe'}
        self.assertEqual(guard.active_controllers([altered,child],81),[altered])
        altered={**parent,'CommandLine':parent['CommandLine'].replace('simplemem','hypermem')}
        self.assertEqual(guard.active_controllers([altered,child],81),[altered])

    def test_external_evidence_closure_hashes_nested_test_files(self):
        p = self.root / 'fixture_test.py'; p.write_text('pass')
        review = self.root / 'review.json'; save_new(review, {'tests': {'files_sha256': {str(p): sha(p)}}})
        self.assertEqual(set(life.evidence_closure([(str(review), sha(review))])), {str(p), str(review)})
        p.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'): life.evidence_closure([(str(review), sha(review))])

    def test_original_raw_replay_all_1540_without_provider_or_original_raw_reads(self):
        method = self.method; root = self.folder; calls = []
        q = {'query_id': 'c0_q0', 'conversation': 0, 'question': 'Where did A walk?',
             'reference': 'park', 'category': 1, 'reference_date': '2023-05-08'}
        class Recorder:
            def __init__(self, run_dir, *, context): self.run_dir = Path(run_dir); self.context = context; self.events = []
            def _redact(self, value): return value
            def generate(self, prompt, *, model, system=None, json_output=False, max_output, stage):
                sig = {'provider': 'gemini', 'model': model, 'operation': 'generateContent',
                       'body': generation_body(prompt, system=system, json_output=json_output, max_output=max_output)}
                key = digest(sig); text = 'park' if stage == life.base.evaluator.READER_STAGE else '[[yes]]'
                p = self.run_dir / 'api_cache' / (key + '.json')
                save_new(p, {'request': sig, 'response': {'modelVersion': model, 'candidates': [
                    {'index': 0, 'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [{'text': text}]}}]}})
                self.events.append({'request_hash': key, 'cache_path': str(p), 'cache_sha256': sha(p),
                                    'stage': stage, 'context': self.context, 'error_type': None})
                calls.append(stage); return text
        metric = lambda answer, reference, category: float(answer == reference)
        with patch.object(life.base, 'BaselineAPI', Recorder), patch.object(life.base, 'verified_official_metric', return_value=(metric, {})):
            life.base.qa_one(root, method, q, {'context': 'A walked in the park.'})
        leaf = root / 'results/firstpass/c0_q0'
        reader = life.read(leaf / 'reader.json'); judge = life.read(leaf / 'judge.json')
        encode = lambda value: json.dumps(value).encode()
        docs = {}; questions = []; first = []; effective = []
        for p in (root / 'work').rglob('*.json'): docs[str(p)] = p.read_bytes()
        for ordinal in range(1540):
            current = {**q, 'query_id': 'c0_q' + str(ordinal)}; questions.append(current)
            current_leaf = root / 'results/firstpass' / current['query_id']
            for kind, stage in (('reader', reader), ('judge', judge)):
                value = deepcopy(stage); value['events'][0]['context']['query_id'] = current['query_id']
                docs[str(current_leaf / (kind + '.json'))] = encode(value)
            docs[str(root / 'results/c0/retrieval' / (current['query_id'] + '.json'))] = encode(
                {'status': 'complete', 'value': {'context': 'A walked in the park.'}})
            row = {**current, 'answer': 'park', 'official_f1': 1.0,
                   'custom_metrics': life.base.evaluator.lexical_metrics('park', 'park'),
                   'judge_correct': True, 'reader_status': 'complete', 'judge_status': 'complete'}
            first.append({**row, 'phase': 'firstpass'}); effective.append({**row, 'phase': 'effective'})
        docs[str(root / 'run_manifest.json')] = encode({'questions': questions})
        docs[str(root / 'results/firstpass_per_query.json')] = encode(first)
        docs[str(root / 'results/per_query.json')] = encode(effective)
        docs[str(root / 'results/summary.json')] = encode({'firstpass': life.aggregate(first), 'effective': life.aggregate(effective)})
        def pack(label):
            members = {}; work = {}; evidence = {}; blobs = {}
            for p, raw in docs.items():
                h = __import__('hashlib').sha256(raw).hexdigest()
                members[p] = {'sha256': h, 'size': len(raw), 'member': 'blobs/' + h}; blobs[h] = raw
                if Path(p).is_relative_to(root / 'work'): work[p] = {'sha256': h, 'size': len(raw)}
                else: evidence[p] = h
            mp = self.root / (label + '.json'); zp = self.root / (label + '.zip')
            save_new(mp, {'schema': life.SCHEMA, 'method': method, 'members': members,
                          'work_inventory': work, 'keep_sha256': {}, 'evidence_sha256': evidence})
            with zipfile.ZipFile(zp, 'x', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('archive_manifest.json', mp.read_bytes())
                for h, raw in blobs.items(): z.writestr('blobs/' + h, raw)
            return zp, mp
        # Delete only the two temporary raw fixtures; replay has no filesystem fallback.
        for p in (root / 'work').rglob('*.json'):
            self.assertTrue(p.resolve().is_relative_to(self.root)); p.unlink()
        with patch.object(life.base, 'questions', return_value=questions), \
             patch.object(life.base, 'verified_official_metric', return_value=(metric, {})), \
             patch.object(life.base, 'BaselineAPI', side_effect=AssertionError('provider forbidden')):
            with life.Resolver(*pack('full_cohort')) as resolver:
                self.assertEqual(life.replay_all(resolver)['firstpass_and_effective_leaves'], 3080)
            broken = deepcopy(effective); broken[3]['answer'] = 'tampered'
            docs[str(root / 'results/per_query.json')] = encode(broken)
            with life.Resolver(*pack('valid_crc_sha_but_false_row')) as resolver:
                with self.assertRaisesRegex(ValueError, 'full QA row'): life.replay_all(resolver)
        self.assertEqual(len(calls), 2)

    def test_actual_powershell_complete_flow_and_active_controller_refusal(self):
        self.fixture()
        # Run the actual PS body against a separately named temp workspace. Its
        # Python verifier uses the same narrow fixture mocks as fixture(); no
        # production path, process, provider or source file is modified.
        script = (self.actual / 'tools/cleanup_reviewed_chain_v2.ps1').read_text(encoding='utf-8')
        quote = lambda x: "'" + str(x).replace("'", "''") + "'"
        python = Path(sys.executable)
        stub = self.root / 'fixture_verify.py'
        stub.write_text("import sys\nfrom pathlib import Path\nfrom unittest.mock import patch\nsys.path.insert(0," + repr(str(self.actual)) + ")\n"
                        "from tools import baseline_lifecycle_v2 as x\nx.ROOT=Path(" + repr(str(self.root)) + ")\nx.RUN=x.ROOT/'runs/five_baselines_gemini25_v1_20261008'\n"
                        "x.stopped=lambda:None\nx.base.manifest=lambda method:x.read(x.folder(method)/'run_manifest.json')\n"
                        "x.replay_all=lambda r:{'status':'passed','fixture_only':True,'model_calls':0}\nx.main()\n", encoding='utf-8')
        script = script.replace("(Resolve-Path -LiteralPath 'D:\\jev mem').Path", "(Resolve-Path -LiteralPath " + quote(self.root) + ').Path')
        script = script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'", '$python = ' + quote(python))
        script = script.replace("$helper = Join-Path $workspace 'tools\\baseline_lifecycle_v2.py'", '$helper = ' + quote(stub))
        needle = "$ErrorActionPreference = 'Stop'"
        fixture_ps = self.root / 'OFFLINE_ONLY_cleanup.ps1'
        active = "function Get-CimInstance { [pscustomobject]@{Name='python.exe';ProcessId=42;CommandLine='python -m tools.run_five_baselines_reviewed_chain_v2 run --method simplemem'} }\n"
        fixture_ps.write_text(script.replace(needle, active + needle), encoding='utf-8')
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(fixture_ps),
                                 '-Method', self.method, '-CleanupReview', str(self.cleanup)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0); self.assertTrue(self.raw.exists())
        self.assertIn('controller is still active', result.stderr)
        fixture_ps.write_text(script.replace(needle, 'function Get-CimInstance { @() }\n' + needle), encoding='utf-8')
        # The only Remove-Item target is asserted to be inside this temp fixture.
        self.assertTrue((self.folder / 'work').resolve().is_relative_to(self.root))
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(fixture_ps),
                                 '-Method', self.method, '-CleanupReview', str(self.cleanup)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.folder / 'work').exists())
        self.assertEqual(life.verify_completion(self.method)['status'], 'passed')


class LaunchTests(unittest.TestCase):
    def test_private_dispatch_keeps_original_scientific_provider_and_routes_only_predecessor(self):
        from tools import run_five_baselines_amem_native_v3_admission_v2 as old
        from tools import baselines_common_v1 as common
        self.assertIsNot(launch.frozen,old)
        self.assertEqual(launch.frozen.main.__code__.co_code,old.main.__code__.co_code)
        self.assertEqual(launch.frozen.main.__code__.co_consts,old.main.__code__.co_consts)
        self.assertIs(launch.frozen.base.BaselineAPI,common.BaselineAPI)
        self.assertIs(launch.frozen.base.adapter,old._adapter)
        self.assertEqual(launch.frozen.NEW.name,'amem_native_v4')
        self.assertEqual(old.NEW.name,'amem_native_v3')

    def test_all_predecessors_strongly_verified_before_frozen_dispatch(self):
        from tools import archive_amem_continuation_v4 as amem
        for method in life.METHODS:
            events = []
            with patch.object(life, 'stopped'), patch.object(amem, 'verify_completion', side_effect=lambda p: events.append('amem') or {'status': 'passed'}), \
                 patch.object(life, 'verify_completion', side_effect=lambda m: events.append(m) or {'status': 'passed'}):
                launch.verify_predecessors(method)
            self.assertEqual(events, ['amem', *life.METHODS[:life.METHODS.index(method)]])

    def test_failed_strong_gate_prevents_frozen_main(self):
        frozen = launch.frozen
        with patch('sys.argv', ['entry', 'run', '--method', 'simplemem']), patch.object(frozen.base, 'manifest', return_value={'review': {'path': 'fake'}}), \
             patch.object(launch, 'verify_code_review'), patch.object(launch, 'verify_predecessors', side_effect=ValueError('bad predecessor')), \
             patch.object(frozen.base, 'paths'), patch.object(frozen.base, 'run') as dispatched:
            with self.assertRaisesRegex(ValueError, 'bad predecessor'): launch.main()
            dispatched.assert_not_called()

    def test_prepare_refuses_missing_new_helper_binding_before_frozen_main(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp) / 'review.json'; save_new(review, {'status': 'passed', 'code_sha256': {}})
            with patch('sys.argv', ['entry', 'prepare', '--method', 'simplemem', '--review', str(review)]), patch.object(launch.frozen, 'main') as dispatched:
                with self.assertRaisesRegex(ValueError, 'lifecycle helpers'): launch.main()
                dispatched.assert_not_called()


if __name__ == '__main__':
    import sys
    unittest.main()
