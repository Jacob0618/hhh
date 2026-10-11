"""Independent main-path, ticket and closure tests; temporary namespaces only."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

from tests.test_reviewed_chain_v6 import Dispatch as TemporaryNamespace
from tools import run_five_baselines_reviewed_chain_v6 as c
from tools.baselines_common_v1 import read,sha

class WorkerBoundary(BaseException):pass

class IndependentDispatch(unittest.TestCase):
    # Only the author's temporary filesystem/data builder is shared. Each test
    # below independently exercises the unchanged actual main and private run.
    setUp=TemporaryNamespace.setUp
    make_review=TemporaryNamespace.make_review
    main=TemporaryNamespace.main

    def test_all_three_actual_main_paths_reach_real_threadpool_worker_without_api(self):
        original_read=c.base.read
        for method in c.METHODS:
            review=self.review if method=='hypermem' else self.make_review(method)
            self.main('prepare','--method',method,'--review',str(review))
            self.main('check','--method',method)
            folder=self.runroot/method;seen=[]
            def reject_legacy(path):
                p=Path(path)
                if p.is_relative_to(self.runroot/'simplemem') or p.is_relative_to(self.runroot/'amem'):
                    raise AssertionError('Obsolete prior-run data read')
                return original_read(path)
            def stop_worker(actual_folder,actual_method,ci,questions,item):
                self.assertEqual((actual_folder,actual_method,ci),(folder,method,0))
                self.assertEqual(item,{'fixture_only':True})
                self.assertTrue(questions)
                original=read(folder/'execution_intent.json')
                technical=read(c.sidecar(method)/'execution_intent.json')
                self.assertLessEqual(technical['created_utc'],original['created_utc'])
                self.assertEqual(technical['ticket']['method'],method)
                self.assertEqual(original['manifest_sha256'],sha(folder/'run_manifest.json'))
                self.assertEqual(read(folder/'status.json')['stage'],'firstpass')
                seen.append(method)
                raise WorkerBoundary()
            with patch.object(c.base,'read',side_effect=reject_legacy), \
                 patch.object(c.base,'load_data',return_value=[{'fixture_only':True}]), \
                 patch.object(c.base,'conv_worker',side_effect=stop_worker), \
                 patch.object(c.BaselineAPI,'__init__',side_effect=AssertionError('Provider constructor forbidden')):
                with self.assertRaises(WorkerBoundary):self.main('run','--method',method)
            self.assertEqual(seen,[method])
            self.assertFalse((folder/'work').exists())
            with self.assertRaises(ValueError):self.main('run','--method',method)
        self.assertEqual(self.calls,['hypermem','hypermem','vmem','vmem','prism','prism'])

    def test_persistent_ticket_and_technical_manifest_have_real_strict_current_only_closure(self):
        self.main('prepare','--method','hypermem','--review',str(self.review))
        m,t=c.manifest('hypermem');proof=c.strong_predecessors('hypermem')
        ticket=c._GateTicket(c._ISSUER,'hypermem',m,t,proof);ticket.consume('hypermem')
        p=c.sidecar('hypermem')/'execution_intent.json';v=read(p)['ticket']
        attestation=v['predecessor_acceptance']
        self.assertEqual(attestation['schema'],'runtime-predecessor-acceptance-record-v1')
        self.assertIs(attestation['new_scientific_input'],False)
        self.assertIs(attestation['recursive_rearchive_claimed'],False)
        self.assertNotIn('strong_predecessors',v)
        self.assertEqual(attestation['verified_objects'],{name:{key+'_sha256':record['sha256'] for key,record in records.items()} for name,records in proof.items()})
        graph=c.previous.simple.evidence_closure([(str(p),sha(p)),(str(c.sidecar('hypermem')/'manifest.json'),sha(c.sidecar('hypermem')/'manifest.json'))])
        self.assertIn(str(self.hyper/'run_manifest.json'),graph)
        self.assertIn(str(self.review),graph)
        self.assertIn(str(c.ROOT/c.ENTRY),graph)
        self.assertFalse([p for p in graph if Path(p).is_relative_to(self.root/'predecessors')])
        self.assertFalse([p for p in graph if Path(p).is_relative_to(c.ROOT/'runs')])
        self.assertFalse([p for p in graph if 'simplemem_terminal_proof' in p])

    def test_changed_current_review_and_wrong_predecessor_set_cannot_consume(self):
        self.main('prepare','--method','hypermem','--review',str(self.review))
        m,t=c.manifest('hypermem');refs=c.strong_predecessors('hypermem')
        ticket=c._GateTicket(c._ISSUER,'hypermem',m,t,refs)
        text=self.review.read_bytes();self.review.write_bytes(text+b'\n')
        with self.assertRaisesRegex(ValueError,'review changed'):ticket.consume('hypermem')
        self.review.write_bytes(text)
        wrong={'amem':refs['amem'],'vmem':refs['simplemem']}
        ticket=c._GateTicket(c._ISSUER,'hypermem',m,t,wrong)
        with self.assertRaisesRegex(ValueError,'sequence differs'):ticket.consume('hypermem')
        with self.assertRaisesRegex(ValueError,'verified run entry'):c._GateTicket(object(),'hypermem',m,t,refs)
        self.assertFalse((c.sidecar('hypermem')/'execution_intent.json').exists())
        self.assertFalse((self.hyper/'execution_intent.json').exists())

    def test_full_real_scientific_specs_are_preserved_and_one_extra_field_change_rejected(self):
        for method in c.METHODS:
            p=self.review if method=='hypermem' else self.make_review(method)
            accepted=c.check_review(method,p)
            old=read(c.ROOT/c.PRIOR_REVIEW_ROOT/(method+'_preparation_review_v1.json'))
            self.assertEqual(accepted['specification'],old['specification'])
            self.assertGreater(len(accepted['specification']),len(c.base.adapter(method).describe()))
            changed=read(p);changed['specification']['generation']['thinkingBudget']=1
            p.write_text(json.dumps(changed),'utf8')
            with self.assertRaisesRegex(ValueError,'specification differs'):c.check_review(method,p)

    def test_global_scientific_api_or_adapter_replacement_fails_before_gate(self):
        for name,value in [('BaselineAPI',object()),('adapter',lambda method:None)]:
            with patch.object(c.base,name,value):
                with self.assertRaisesRegex(ValueError,'Fresh original'):self.main('prepare','--method','hypermem','--review',str(self.review))
        self.assertEqual(self.calls,[])
        self.assertFalse(c.sidecar('hypermem').exists())


class IndependentStaticAndGuard(unittest.TestCase):
    def test_root_launcher_parses_and_keeps_three_hashes_scope_and_hidden_gate(self):
        path=c.ROOT/c.LAUNCHER
        self.assertEqual(sha(path),'c9d6c73589aa59022f8bbf62adfba317c9cda4d3df852e104f6da27efd0a3a98')
        command="$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile('"+str(path)+"',[ref]$tokens,[ref]$errors) | Out-Null; if($errors.Count){$errors | Out-String; exit 1}; Write-Output 'PARSE_ONLY_OK'"
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        text=path.read_text('utf8');launch=text.index('$child=Start-Process')
        for required in ('$OriginalManifestSha256','$TechnicalManifestSha256','$EntrySha256',
                         "throw 'Fixed scientific scope differs'","throw 'Current technical preparation review has not passed'",
                         'technical_dispatch_v6\\execution_intent.json',"throw 'Active experiment controller prohibits dispatch'",
                         'New-Item -ItemType Directory -Path $launchRoot'):
            self.assertLess(text.index(required),launch)
        self.assertIn('-WorkingDirectory $taskRoot -WindowStyle Hidden',text)
        self.assertIn('technical_manifest_sha256=$TechnicalManifestSha256',text.split('$receipt=',1)[1])

    def test_private_run_body_matches_exact_original_except_one_prior_gate(self):
        text=(c.ROOT/'tools/run_five_baselines_gemini25_v1.py').read_text('utf8')
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        class Ticket:
            def consume(self,method):raise AssertionError('Not executed')
        actual=c.private_run(Ticket())
        self.assertIsNot(actual,c.base.run)
        desc=c.bridge_description();self.assertTrue(desc['scientific_ast_outside_gate_equal'])
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ThreadPoolExecutor' for n in ast.walk(node)),3)
        values=[kw.value.value for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ThreadPoolExecutor' for kw in n.keywords if kw.arg=='max_workers']
        self.assertEqual(values,[10,10,4])
        self.assertIs(actual.__globals__['qa_one'],c.base.qa_one)
        self.assertIs(actual.__globals__['conv_worker'],c.base.conv_worker)
        self.assertIs(actual.__globals__['BaselineAPI'],c.base.BaselineAPI)

    def test_synthetic_guard_rejects_foreign_wrong_parent_and_changed_tail(self):
        parent={'ProcessId':100,'ParentProcessId':99,'Name':'python.exe',
            'CommandLine':'"'+str(c.ROOT/'runtime/baseline_env_v1/Scripts/python.exe')+'" -B tools/run_five_baselines_reviewed_chain_v6.py run --method hypermem',
            'ExecutablePath':str(c.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'),'CreatedUtc':'2026-10-09T08:00:00+00:00'}
        child={**parent,'ProcessId':101,'ParentProcessId':100,'ExecutablePath':sys._base_executable,
            'CommandLine':'"'+sys._base_executable+'" -B tools/run_five_baselines_reviewed_chain_v6.py run --method hypermem','CreatedUtc':'2026-10-09T08:00:01+00:00'}
        self.assertEqual(c.active_controllers([parent,child],current_pid=101),[])
        for update in ({'ExecutablePath':'C:/foreign/python.exe'},{'CommandLine':parent['CommandLine']+' --other'}, {'CreatedUtc':'2026-10-09T08:01:00+00:00'}):
            self.assertEqual(len(c.active_controllers([{**parent,**update},child],current_pid=101)),1)
        foreign={**parent,'ProcessId':102,'ParentProcessId':1}
        self.assertEqual([p['ProcessId'] for p in c.active_controllers([parent,child,foreign],current_pid=101)],[102])

del TemporaryNamespace  # unittest must not rediscover the imported author class.
if __name__=='__main__':unittest.main()
