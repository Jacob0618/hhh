"""Independent bounded-compatibility and actual PowerShell launch-gate fixtures.

The launcher runs unchanged. Synthetic input uses a Join-Path temporary-root
seam and one fixed-parent hash stub; all other hashes and filesystem checks are
real. Start-Process is always intercepted; no production process is launched.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path('D:/jev mem')
LAUNCHER = ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/launch_hypermem_native_v2.ps1'
PARENT_SHA = '587e091852902ced9fab3081b9f44771e6993d736938f5f21eba0c13071e7b5d'
ENTRY = 'tools/run_five_baselines_hypermem_native_v2.py'
LAUNCHER_KEY = str(LAUNCHER.relative_to(ROOT)).replace('\\', '/')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return sha(p)


class Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = importlib.import_module('tools.baselines_hypermem_v1')
        cls.old_generation = cls.old._Generation
        cls.new = importlib.import_module('tools.baselines_hypermem_native_v2')

    def test_actual_all_ten_default_is_read_only_and_old_still_rejects(self):
        run = ROOT/'runs/five_baselines_gemini25_v1_20261008/hypermem'
        for ci in range(10):
            value = json.loads((run/f'work/native/c{ci}/calls/episode_000001_hypermem_boundary.json').read_text('utf-8'))['raw_response']
            before = copy.deepcopy(value)
            self.new._Generation._validate(value, 'hypermem_boundary', {})
            self.assertEqual(value, before)
            self.assertNotIn('should_wait', value)
            with self.assertRaises(self.old.HyperMemError):
                self.old._Generation._validate(value, 'hypermem_boundary', {})
        self.assertIs(self.old._Generation, self.old_generation)

    def test_null_wrong_types_and_missing_end_rejected(self):
        invalid = [{}, [], None, {'should_wait': True}, {'should_end': None},
                   {'should_end': 0}, {'should_end': 1}, {'should_end': 'false'},
                   *({'should_end': True, 'should_wait': x} for x in (None, 0, 1, 'true', [], {}))]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(self.new.HyperMemError):
                self.new._Generation._validate(value, 'hypermem_boundary', {})
        for end in (False, True):
            for wait in (False, True):
                value = {'should_end': end, 'should_wait': wait}
                self.new._Generation._validate(value, 'hypermem_boundary', {})

    def test_other_stages_and_scientific_functions_unchanged(self):
        for stage in ('hypermem_episode', 'hypermem_topic_match', 'hypermem_episode_roles', 'hypermem_fact_roles'):
            with self.subTest(stage=stage), self.assertRaises((self.new.HyperMemError, KeyError)):
                self.new._Generation._validate({'should_end': True}, stage, {})
        self.assertEqual(self.new.CONFIG, self.old.CONFIG)
        self.assertEqual(self.new._runtime._PACKAGE, '_jev_hypermem_native_v2')
        self.assertEqual(self.old._PACKAGE, '_jev_hypermem_pinned_v1')
        for name in ('build', 'retrieve', '_source_messages', '_Embedding'):
            if hasattr(self.old, name):
                original, current = getattr(self.old, name), getattr(self.new._runtime, name)
                if hasattr(original, '__code__'):
                    self.assertEqual(original.__code__.co_code, current.__code__.co_code)
                    self.assertEqual(original.__code__.co_consts, current.__code__.co_consts)
        self.assertEqual(sha(ROOT/'tools/baselines_hypermem_v1.py'), self.new.BASE_SHA256)


class Launcher(unittest.TestCase):
    def invoke(self, fault=None):
        with tempfile.TemporaryDirectory(prefix='hyper-launch-review-') as temporary:
            root = Path(temporary)
            old = root/'runs/five_baselines_gemini25_v1_20261008/hypermem'
            new = root/'runs/five_baselines_gemini25_v1_20261008/hypermem_native_v2'
            code = root/ENTRY; code.parent.mkdir(parents=True); code.write_text('# Never executed\n')
            metric = root/'research/metric.py'; metric.parent.mkdir(); metric.write_text('# fixture source\n')
            # Match the original launcher-bound relative path, with exact original bytes.
            launch_copy = root/LAUNCHER_KEY; launch_copy.parent.mkdir(parents=True, exist_ok=True); launch_copy.write_bytes(LAUNCHER.read_bytes())
            code_map = {ENTRY: sha(code), LAUNCHER_KEY: sha(LAUNCHER), 'research/metric.py': sha(metric)}
            original = {'method': 'hypermem', 'questions': list(range(1540)), 'generation_model': 'gemini-2.5-flash',
                        'thinkingBudget': 0, 'conversation_workers': 10, 'query_workers_per_conversation': 4,
                        'code_sha256': code_map, 'upstream_sha256': {'research/metric.py': sha(metric)}}
            if fault == 'science': original['thinkingBudget'] = 1
            save(old/'run_manifest.json', original)
            review = {'schema': 'hypermem-native-v2-independent-continuation-review-v1', 'status': 'passed', 'code_sha256': copy.deepcopy(code_map)}
            if fault == 'review_status': review['status'] = 'needs_review'
            if fault == 'review_code': review['code_sha256'][ENTRY] = '0'*64
            review_path = root/'review.json'; review_sha = save(review_path, review)
            proof = root/'proof.json'; proof_sha = save(proof, {'status': 'passed'})
            manifest = {'schema': 'hypermem-native-v2-continuation-manifest-v1', 'original_run': str(old), 'new_run': str(new),
                        'parent_manifest': {'path': str(old/'run_manifest.json'), 'sha256': PARENT_SHA},
                        'review': {'path': str(review_path), 'sha256': review_sha},
                        'terminal_proof': {'path': str(proof), 'sha256': proof_sha},
                        'code_sha256': copy.deepcopy(code_map), 'upstream_sha256': {'research/metric.py': sha(metric)}}
            if fault == 'wrong_namespace': manifest['new_run'] = str(old)
            if fault == 'launcher_binding': manifest['code_sha256'][LAUNCHER_KEY] = '0'*64
            manifest_sha = save(new/'run_manifest.json', manifest)
            if fault == 'existing_intent': save(new/'execution_intent.json', {})
            if fault == 'existing_work': (new/'work').mkdir()
            if fault == 'existing_recovery': save(new/'qa_recovery_plan.json', {})
            launch_root = root/'research/baseline_execution_2026-10-08/launch_hypermem_native_v2'
            if fault == 'existing_launch': launch_root.mkdir(parents=True)
            if fault == 'proof_hash': proof.write_text('{} changed')
            if fault == 'changed_source': metric.write_text('# changed')
            manifest_arg = '0'*64 if fault == 'manifest_hash' else manifest_sha
            entry_arg = '0'*64 if fault == 'entry_hash' else sha(code)
            active = "@([pscustomobject]@{Name='python3.11.exe';CommandLine='python3.11.exe -m tools.run_five_baselines_unknown run'})" if fault == 'active_controller' else '@()'
            driver = root/'driver.ps1'
            driver.write_text(r'''
$ErrorActionPreference='Stop'
$global:FixtureRoot='__ROOT__'
$global:Calls=0
function Join-Path {
 param([string]$Path,[string]$ChildPath)
 if ($Path -eq 'D:\jev mem') { $Path=$global:FixtureRoot }
 Microsoft.PowerShell.Management\Join-Path -Path $Path -ChildPath $ChildPath
}
function Get-FileHash {
 param([string]$LiteralPath,[string]$Algorithm)
 if ($LiteralPath -eq '__PARENT__') { return [pscustomobject]@{Hash='__PARENT_SHA__'} }
 $hasher=[Security.Cryptography.SHA256]::Create()
 try { [pscustomobject]@{Hash=([BitConverter]::ToString($hasher.ComputeHash([IO.File]::ReadAllBytes($LiteralPath)))).Replace('-','')} }
 finally { $hasher.Dispose() }
}
function Get-CimInstance { param($ClassName) __ACTIVE__ }
function Start-Process {
 param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,$RedirectStandardOutput,$RedirectStandardError,[switch]$PassThru)
 $global:Calls++
 if ($WorkingDirectory -ne 'D:\jev mem' -or $WindowStyle -ne 'Hidden' -or
     $FilePath -ne (Join-Path $global:FixtureRoot 'runtime\baseline_env_v1\Scripts\python.exe') -or
     $ArgumentList.Count -ne 2 -or $ArgumentList[0] -ne 'tools/run_five_baselines_hypermem_native_v2.py' -or $ArgumentList[1] -ne 'run') {throw 'Wrong actual launch parameters'}
 $i=Join-Path $global:FixtureRoot 'research\baseline_execution_2026-10-08\launch_hypermem_native_v2\launch_intent.json'
 if (!(Test-Path -LiteralPath $i)) {throw 'Start occurred before intent'}
 return [pscustomobject]@{Id=424242}
}
try {
 & '__LAUNCHER__' -ManifestSha256 '__MANIFEST__' -EntrySha256 '__ENTRY__' | Out-Null
 $result=@{passed=$true;calls=$global:Calls}
} catch { $result=@{passed=$false;calls=$global:Calls;error=$_.Exception.Message} }
$result | ConvertTo-Json -Compress
'''.replace('__ROOT__', str(root)).replace('__PARENT__', str(old/'run_manifest.json')).replace('__PARENT_SHA__', PARENT_SHA).replace('__ACTIVE__', active).replace('__LAUNCHER__', str(LAUNCHER)).replace('__MANIFEST__', manifest_arg).replace('__ENTRY__', entry_arg), encoding='utf-8')
            run = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(driver)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout.strip())
            if fault is None:
                self.assertEqual(result, {'passed': True, 'calls': 1})
                receipt = json.loads((launch_root/'receipt.json').read_text('utf-8-sig'))
                self.assertEqual(receipt['launcher_pid'], 424242)
                self.assertEqual(receipt['launcher_sha256'], sha(LAUNCHER))
            else:
                self.assertFalse(result['passed'], result)
                self.assertEqual(result['calls'], 0, result)
                self.assertNotIn('is not recognized', result.get('error', ''))
                self.assertFalse((launch_root/'launch_intent.json').exists())

    def test_real_powershell_positive_intercepted_start(self):
        self.invoke()

    def test_real_powershell_all_failure_gates_before_start(self):
        for fault in ('science', 'review_status', 'review_code', 'wrong_namespace', 'launcher_binding',
                      'existing_intent', 'existing_work', 'existing_recovery', 'existing_launch',
                      'proof_hash', 'changed_source', 'manifest_hash', 'entry_hash', 'active_controller'):
            with self.subTest(fault=fault): self.invoke(fault)


class Prefix(unittest.TestCase):
    def fixture(self, root):
        from tools import hypermem_prefix_api_v2 as prefix
        report = json.loads((ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/author_hypermem_boundary_terminal_v2.json').read_text('utf-8'))
        questions = json.loads((ROOT/'runs/five_baselines_gemini25_v1_20261008/hypermem/run_manifest.json').read_text('utf-8'))['questions']
        rows = report['rows']
        proof = {'rows': rows, 'registry': {r['request_hash']: r for r in rows}, 'questions': questions}
        return prefix.admitted_api(proof, root), rows

    def make(self, api_class, root, case, **kw):
        return api_class(root/'work/api_method', context=case['context'], keys={'GEMINI_API_KEY': 'synthetic-no-access'}, **kw)

    def call(self, api, case):
        s = case['signature']
        return api.request(s['provider'], s['model'], s['body'], operation=s['operation'], stage=case['stage'])

    def test_real_production_prefix_all_ten_exact_bytes_zero_transport(self):
        from jev_mem import providers
        import requests
        with tempfile.TemporaryDirectory(prefix='hyper-prefix-review-') as tmp, \
             patch.object(providers, 'load_credentials', side_effect=AssertionError('vault')), \
             patch.object(requests.Session, 'request', side_effect=AssertionError('transport')):
            root = Path(tmp); api_class, rows = self.fixture(root)
            for case in rows:
                api = self.make(api_class, root, case)
                original = Path(case['raw']['path']).read_bytes()
                result = self.call(api, case)
                self.assertEqual(result, json.loads(original)['response'])
                target = api.cache_dir/(case['request_hash']+'.json')
                self.assertEqual(target.read_bytes(), original)
                self.assertFalse((api.run_dir/'request_intents'/target.name).exists())
            ledger = [json.loads(x) for x in (root/'work/api_method/api_calls.jsonl').read_text('utf-8').splitlines()]
            events = [json.loads(x) for x in (root/'work/api_method/logical_events.jsonl').read_text('utf-8').splitlines()]
            self.assertEqual(len(ledger), 10)
            self.assertTrue(all(x['cache_hit'] is True for x in ledger))
            self.assertTrue(all(x['source'] == 'inherited_exact_request' for x in events))
            self.assertEqual(len(list((root/'work/prefix').glob('*.complete.json'))), 10)

    def test_mismatched_first_and_cross_context_cannot_dispatch(self):
        from tools.baselines_common_v1 import BaselineFatal
        from jev_mem import providers
        import requests
        with tempfile.TemporaryDirectory(prefix='hyper-prefix-review-') as tmp, \
             patch.object(providers, 'load_credentials', side_effect=AssertionError('vault')), \
             patch.object(requests.Session, 'request', side_effect=AssertionError('transport')):
            root=Path(tmp); api_class, rows=self.fixture(root); api=self.make(api_class, root, rows[0])
            wrong=copy.deepcopy(rows[0]); wrong['signature']['body']['generationConfig']['maxOutputTokens']=100
            with self.assertRaises(BaselineFatal): self.call(api, wrong)
            with self.assertRaises(BaselineFatal): self.call(api, rows[1])
            self.assertFalse((root/'work/prefix').exists())
            with self.assertRaises(BaselineFatal): self.make(api_class, root, rows[0], retries=2.0)

    def test_suffix_parallel_original_cache_and_unknown_never_redrawn(self):
        from tools.baselines_common_v1 import BaselineFatal, generation_body
        from jev_mem import providers
        import requests
        with tempfile.TemporaryDirectory(prefix='hyper-prefix-review-') as tmp, \
             patch.object(providers, 'load_credentials', side_effect=AssertionError('vault')):
            root=Path(tmp); api_class, rows=self.fixture(root); api=self.make(api_class, root, rows[0])
            with patch.object(requests.Session, 'request', side_effect=AssertionError('transport')):
                self.call(api, rows[0])
            barrier=threading.Barrier(2); calls=[]
            class Response:
                status_code=200;ok=True;headers={}
                def json(self): return {'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':'{}'}]}}], 'modelVersion':'gemini-2.5-flash'}
            def fake_post(session,url,**kw):
                calls.append(kw['json']); barrier.wait(timeout=3); return Response()
            def one(i):
                scoped=self.make(api_class, root, rows[0])
                return scoped.request('gemini','gemini-2.5-flash',generation_body('suffix-'+str(i),json_output=True,max_output=16384),stage='hypermem_boundary')
            with patch.object(requests.Session, 'post', fake_post), ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(one,range(2)))
            self.assertEqual(len(calls),2); self.assertEqual(len(results),2)
            body=generation_body('unknown-single-logical-request',json_output=True,max_output=16384)
            count=[]
            def unknown(*a,**kw):
                count.append(1); raise RuntimeError('synthetic unknown')
            with patch.object(requests.Session,'post',unknown):
                for _ in range(2):
                    current=self.make(api_class,root,rows[0])
                    with self.assertRaises(BaselineFatal):
                        current.request('gemini','gemini-2.5-flash',body,stage='hypermem_boundary')
            self.assertEqual(len(count),1)
            with patch.object(requests.Session,'post',side_effect=AssertionError('wrong-stage transport')):
                with self.assertRaises(BaselineFatal):
                    api.request('gemini','gemini-2.5-flash',generation_body('wrong stage',json_output=True,max_output=16384),stage='amem_analysis')


class Admission(unittest.TestCase):
    def test_independent_qualification_covers_whole_inventory_not_only_ten_rows(self):
        from tools import audit_hypermem_terminal_v2 as a
        report=json.loads((ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/author_hypermem_boundary_terminal_v2.json').read_text('utf-8'))
        changed=copy.deepcopy(report)
        session=next(p for p in changed['original_inventory'] if '/sessions/' in p.replace('\\','/'))
        changed['original_inventory'][session]='f'*64
        with patch.object(a,'assert_stopped'), patch.object(a,'_locked_terminal_audit',return_value=changed), \
             patch.object(a,'inventory',return_value=changed['original_inventory']):
            with self.assertRaises(ValueError): a.collect()

    def test_absent_or_null_token_telemetry_remains_unknown(self):
        from tools import audit_hypermem_terminal_v2 as a
        value=a.usage_presence([{'usage':{'promptTokenCount':3}}, {'usage':{'promptTokenCount':None}}, {'cache_hit':True,'usage':{'promptTokenCount':999}}])
        self.assertEqual(value['tokens']['promptTokenCount'],{'reported_calls':1,'unreported_calls':1,'reported_sum':3})
        self.assertEqual(value['tokens']['candidatesTokenCount'],{'reported_calls':0,'unreported_calls':2,'reported_sum':None})
        for invalid in (-1, True, 1.2):
            with self.subTest(value=invalid), self.assertRaises(ValueError): a.usage_presence([{'usage':{'promptTokenCount':invalid}}])


class Entry(unittest.TestCase):
    def setUp(self):
        from tests import test_hypermem_native_v2 as fixtures
        self.fixture_module=fixtures
        fixtures.Dispatch.setUp(self)

    def main(self,*args):
        return self.fixture_module.Dispatch.main(self,*args)

    def prepare(self):
        return self.fixture_module.Dispatch.prepare(self)

    def inventory(self):
        return {str(p):sha(p) for p in self.new.rglob('*') if p.is_file()}

    def test_guard_rejection_never_overwrites_an_existing_healthy_status(self):
        c=self.fixture_module.c
        self.prepare()
        save(self.new/'execution_intent.json',{'pid':987654,'fixture':'foreign active controller'})
        save(self.new/'status.json',{'stage':'firstpass','pid':987654,'fixture':True})
        before=self.inventory()
        c._OWNED_EXECUTION=True  # Must not inherit a previous invocation's ownership.
        with patch.object(c.admission,'assert_stopped',side_effect=ValueError('Foreign controller')):
            with self.assertRaises(ValueError): self.main('run')
        self.assertEqual(before,self.inventory())
        self.assertFalse(c._OWNED_EXECUTION)

    def test_manifest_and_prepare_reentry_are_read_only(self):
        c=self.fixture_module.c
        self.prepare(); before=self.inventory()
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(before,self.inventory())
        with patch.object(c.admission,'validate_proof',side_effect=ValueError('Changed parent inventory')):
            with self.assertRaises(ValueError): self.main('run')
        self.assertEqual(before,self.inventory())
        self.assertFalse((self.new/'status.json').exists())

    def test_actual_ten_threadpool_worker_boundary_no_provider(self):
        c=self.fixture_module.c
        self.prepare(); captured=[]; original=c.private_controller
        class WorkerBoundary(BaseException): pass
        def factory(m,proof,ticket):
            result=original(m,proof,ticket)
            result.load_data=lambda:[{'fixture':ci} for ci in range(10)]
            def worker(folder,method,ci,qs,item):
                self.assertTrue((self.new/'continuation_execution_intent.json').exists())
                self.assertTrue((self.new/'execution_intent.json').exists())
                self.assertFalse((self.new/'work').exists())
                self.assertTrue(issubclass(result.BaselineAPI,importlib.import_module('tools.hypermem_prefix_api_v2').PrefixAPI))
                captured.append((ci,len(qs),method,item))
                raise WorkerBoundary()
            result.conv_worker=worker
            return result
        from jev_mem import providers
        with patch.object(c,'private_controller',factory), patch.object(providers.ModelAPI,'__init__',side_effect=AssertionError('Provider before worker')), self.assertRaises(WorkerBoundary):
            self.main('run')
        self.assertEqual({x[0] for x in captured},set(range(10)))
        self.assertEqual(sum(x[1] for x in captured),1540)
        self.assertTrue(all(x[2]=='hypermem' for x in captured))
        self.assertEqual(self.gates,['hypermem','hypermem'])
        self.assertFalse((self.new/'work').exists())
        before=self.inventory()
        with self.assertRaises(ValueError): self.main('run')
        self.assertEqual(before,self.inventory())

    def test_failed_predecessor_has_no_execution_intent(self):
        c=self.fixture_module.c
        self.prepare(); before=self.inventory()
        with patch.object(c.chain.previous,'verify_predecessors',side_effect=ValueError('Actual prerequisite missing')):
            with self.assertRaises(ValueError): self.main('run')
        self.assertEqual(before,self.inventory())
        self.assertFalse((self.new/'continuation_execution_intent.json').exists())

    def test_supporting_evidence_changed_after_gate_blocks_ticket_and_main(self):
        c=self.fixture_module.c
        evidence=self.root/'independent_support.txt';evidence.write_text('fixed supporting evidence')
        review=json.loads(self.review.read_text('utf-8'))
        review['evidence_sha256'][str(evidence)]=sha(evidence);save(self.review,review)
        self.prepare();m,proof=c.manifest()
        ticket=c.GateTicket(m,proof,c.chain.strong_predecessors('hypermem'))
        evidence.write_text('changed after initial qualification')
        before=self.inventory()
        with self.assertRaises(ValueError): ticket.consume('hypermem')
        with self.assertRaises(ValueError): self.main('run')
        self.assertEqual(before,self.inventory())
        self.assertFalse((self.new/'continuation_execution_intent.json').exists())
        self.assertFalse((self.new/'status.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
