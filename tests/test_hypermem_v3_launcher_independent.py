"""Isolated PowerShell sentinels; the production launcher is never executed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/launch_hypermem_native_v3.ps1'
SHA='2b68be3f5db01eb202c58f618deaba194d60b13db50a8c319105955a894021d9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf8');return sha(p)


class Launcher(unittest.TestCase):
    def setUp(self):
        self.assertEqual(sha(LAUNCHER),SHA)

    def fixture(self,case):
        temp=tempfile.TemporaryDirectory(dir=ROOT/'research');self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        runs=root/'runs/five_baselines_gemini25_v1_20261008'
        old=runs/'hypermem';parent=runs/'hypermem_native_v2';new=runs/'hypermem_native_v3'
        entry=root/'tools/run_five_baselines_hypermem_native_v3.py';entry.parent.mkdir();entry.write_text('# fixture, never executed\n')
        ehash=sha(entry)
        leaf=root/'fixture_dependency.txt';leaf.write_text('bound')
        mapping={'fixture_dependency.txt':sha(leaf)}
        om=old/'run_manifest.json';oh=write(om,{'method':'hypermem','questions':[{}]*1540,'generation_model':'gemini-2.5-flash','thinkingBudget':0,'conversation_workers':10,'query_workers_per_conversation':4,'code_sha256':mapping,'upstream_sha256':mapping})
        pm=parent/'run_manifest.json';ph=write(pm,{'fixture_parent':True})
        script=root/'fixture_launcher.ps1'
        text=LAUNCHER.read_text(encoding='utf8').replace("$taskRoot = 'D:\\jev mem'",f"$taskRoot = '{root}'")
        text=text.replace('587e091852902ced9fab3081b9f44771e6993d736938f5f21eba0c13071e7b5d',oh).replace('474a62558f4f3905b3704e717cae3ac93de61aaa683baf41ffc9d24b5a53ffd0',ph)
        process="[pscustomobject]@{Name='python3.11.exe';CommandLine='python3.11.exe -m tools.run_five_baselines_hypermem_native_v3 run'}" if case=='active' else '@()'
        stubs="""
function Get-CimInstance { param($ClassName) PROCESS_RETURN }
function Start-Process {
    param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,$RedirectStandardOutput,$RedirectStandardError,[switch]$PassThru)
    $v=@{file=$FilePath;arguments=$ArgumentList;cwd=$WorkingDirectory;style=$WindowStyle;pass_thru=$PassThru.IsPresent;seed=$env:PYTHONHASHSEED}
    [IO.File]::WriteAllText((Join-Path $WorkingDirectory 'intercepted_start.json'),($v|ConvertTo-Json -Depth 5))
    throw 'INDEPENDENT_START_SENTINEL_NO_PROCESS_CREATED'
}
""".replace('PROCESS_RETURN',process)
        text=text.replace("$ErrorActionPreference = 'Stop'", "$ErrorActionPreference = 'Stop'\n"+stubs,1)
        script.write_text(text,encoding='utf8')
        launcherkey='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/launch_hypermem_native_v3.ps1'
        # The self-hash entry resolves to this instrumented copy inside its owned
        # fixture. Only this three-constant/stub environment changes the source.
        copy=root/launcherkey;copy.parent.mkdir(parents=True);copy.write_bytes(script.read_bytes())
        code={**mapping,'tools/run_five_baselines_hypermem_native_v3.py':ehash,launcherkey:sha(script)}
        proof=root/'proof.json';prh=write(proof,{'fixture':True})
        from tools import audit_hypermem_terminal_v3 as admission
        hash_environment=admission.hash_environment()
        review=root/'review.json';rh=write(review,{'schema':'hypermem-native-v3-independent-continuation-review-v1','status':'passed','code_sha256':code,'hash_environment':hash_environment,'proof_sha256':'0'*64 if case=='wrong_review_proof' else prh})
        manifest={'schema':'hypermem-native-v3-continuation-manifest-v1','original_run':str(old),'parent_run':str(parent),'new_run':str(new),
            'original_manifest':{'path':str(om),'sha256':oh},'parent_manifest':{'path':str(pm),'sha256':ph},
            'review':{'path':str(review),'sha256':rh},'terminal_proof':{'path':str(proof),'sha256':prh},'code_sha256':code,'upstream_sha256':mapping,'hash_environment':hash_environment}
        if case=='wrong_parent':manifest['parent_manifest']['sha256']='0'*64
        if case=='missing_entry':manifest['code_sha256']={k:v for k,v in code.items() if k!='tools/run_five_baselines_hypermem_native_v3.py'}
        if case=='wrong_launcher':manifest['code_sha256']={**code,launcherkey:'0'*64}
        if case=='wrong_seed':manifest['hash_environment']['seed']='779'
        if case=='wrong_qualification':manifest['hash_environment']['qualification']['sha256']='0'*64
        if case=='wrong_candidate':manifest['hash_environment']['candidate_report']['sha256']='0'*64
        mp=new/'run_manifest.json';mh=write(mp,manifest)
        if case=='work_exists':(new/'work').mkdir()
        if case=='changed_dependency':leaf.write_text('changed')
        shell=shutil.which('pwsh.exe')
        self.assertIsNotNone(shell,'Use the same bundled PowerShell 7 host as the root execution shell')
        result=subprocess.run([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),'-ManifestSha256',mh,'-EntrySha256',ehash],env={**os.environ,'PYTHONHASHSEED':'independent-prior-marker'},capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30)
        return root,new,result

    def test_valid_fixture_reaches_only_hidden_start_sentinel(self):
        root,new,r=self.fixture('valid')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('INDEPENDENT_START_SENTINEL_NO_PROCESS_CREATED',r.stderr)
        call=json.loads((root/'intercepted_start.json').read_text())
        self.assertEqual(call['arguments'],['tools/run_five_baselines_hypermem_native_v3.py','run'])
        self.assertEqual(call['cwd'],str(root));self.assertEqual(call['style'],'Hidden');self.assertTrue(call['pass_thru'])
        self.assertEqual(call['file'],str(root/'runtime/baseline_env_v1/Scripts/python.exe'))
        self.assertEqual(call['seed'],'780')
        self.assertEqual([x.name for x in new.iterdir()],['run_manifest.json'])
        self.assertFalse((root/'research/baseline_execution_2026-10-08/launch_hypermem_native_v3/receipt.json').exists())

    def test_wrong_bindings_pristine_or_active_controller_fail_before_start(self):
        for case in ('wrong_parent','missing_entry','wrong_launcher','work_exists','changed_dependency','active','wrong_review_proof','wrong_seed','wrong_qualification','wrong_candidate'):
            with self.subTest(case=case):
                root,new,r=self.fixture(case)
                self.assertNotEqual(r.returncode,0)
                self.assertFalse((root/'intercepted_start.json').exists())
                self.assertFalse((root/'research/baseline_execution_2026-10-08/launch_hypermem_native_v3').exists())


if __name__=='__main__':unittest.main()
