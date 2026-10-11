"""Independent isolated PowerShell launcher checks; zero real child processes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T1152/launch_hypermem_native_v4.ps1'
KEY='research/baseline_execution_2026-10-08/heartbeat_20261009T1152/launch_hypermem_native_v4.ps1'
ENTRY='tools/run_five_baselines_hypermem_native_v4.py'
BASE_SHA='77ce191aa5d30b17288a3d6adaf83813154998b8de82b82485d393d11ce47c58'
PROPOSAL_SHA='06540738a8cc0028743eb5f309756f1d95f06675c8ca2a05437c0a7723a8f6c8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value),encoding='utf-8')
    return sha(p)
def bound(p):return {'path':str(p),'sha256':sha(p)}


class Launcher(unittest.TestCase):
    def test_original_full_powershell_ast_parses_and_write_order(self):
        shell=shutil.which('pwsh.exe');self.assertIsNotNone(shell)
        code="$tokens=$null; $errors=$null; $ast=[System.Management.Automation.Language.Parser]::ParseFile('"+str(SOURCE)+"',[ref]$tokens,[ref]$errors); if($errors.Count){throw ($errors|Out-String)}; $c=@($ast.FindAll({param($a) $a -is [System.Management.Automation.Language.CommandAst]},$true)|ForEach-Object{[pscustomobject]@{name=$_.GetCommandName();offset=$_.Extent.StartOffset}}); [pscustomobject]@{parameters=@($ast.ParamBlock.Parameters.Name.VariablePath.UserPath);commands=$c}|ConvertTo-Json -Depth 8 -Compress"
        result=subprocess.run([shell,'-NoProfile','-Command',code],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);value=json.loads(result.stdout)
        self.assertEqual(value['parameters'],['ManifestSha256','EntrySha256','ReviewSha256','ProofSha256','HumanGrantSha256'])
        commands=value['commands']
        new=[c['offset'] for c in commands if c['name']=='New-Item']
        start=[c['offset'] for c in commands if c['name']=='Start-Process']
        cim=[c['offset'] for c in commands if c['name']=='Get-CimInstance']
        self.assertEqual(len(new),1);self.assertEqual(len(start),1);self.assertEqual(len(cim),1)
        self.assertLess(cim[0],new[0]);self.assertLess(new[0],start[0])

    def fixture(self,case):
        temp=tempfile.TemporaryDirectory(prefix='v4-launcher-independent-',dir=ROOT/'research')
        self.addCleanup(temp.cleanup);root=Path(temp.name)
        run=root/'runs/five_baselines_gemini25_v1_20261008/hypermem_native_v4'
        research=root/'research/baseline_execution_2026-10-08/hypermem_native_v4'
        launch=root/'research/baseline_execution_2026-10-08/launch_hypermem_native_v4'
        entry=root/ENTRY;entry.parent.mkdir();entry.write_text('# inert test source',encoding='utf-8')
        python=root/'runtime/baseline_env_v1/Scripts/python.exe';python.parent.mkdir(parents=True)
        python.write_bytes(b'inert fixture; Start-Process is intercepted')
        leaf=root/'dependency.txt';leaf.write_text('bound dependency',encoding='utf-8')
        qualification=root/'qualification.json';write(qualification,{'fixture':'qualification'})
        candidate=root/'candidate.json';write(candidate,{'fixture':'candidate'})
        hash_env={'seed':'780','base_interpreter':bound(python),'qualification':bound(qualification),'candidate_report':bound(candidate)}
        proposal=root/'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/new_suffix_diagnosis_v1/minimal_continuation_proposal_v1.json'
        targets=[]
        for ci in range(10):
            raw=root/f'original_raw_{ci}.json';write(raw,{'synthetic_raw':ci})
            targets.append({'conversation':ci,'request_hash':str(ci)*64,'stage':'hypermem_episode_roles',
                'signature':{'body':{'full':'synthetic '+str(ci)}},'context':{'method':'hypermem','conversation':ci,'phase':'build'},
                'original_raw':{'source_path':str(raw),'observed_sha256':sha(raw)}})
        write(proposal,{'targets':targets})
        source=SOURCE.read_text(encoding='utf-8')
        self.assertEqual(source.count("$taskRoot = 'D:\\jev mem'"),1)
        self.assertEqual(source.count(BASE_SHA),1)
        self.assertEqual(source.count(PROPOSAL_SHA),1)
        source=source.replace("$taskRoot = 'D:\\jev mem'",f"$taskRoot = '{root}'").replace(BASE_SHA,sha(python)).replace(PROPOSAL_SHA,sha(proposal))
        process="[pscustomobject]@{Name='python.exe';CommandLine='python tools/run_five_baselines_fixture run'}" if case=='active' else '@()'
        stubs=r'''
function Get-CimInstance { param($ClassName) PROCESS_RESULT }
function Start-Process {
 param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,$RedirectStandardOutput,$RedirectStandardError,[switch]$PassThru)
 $v=[ordered]@{file=$FilePath;args=$ArgumentList;cwd=$WorkingDirectory;style=$WindowStyle;seed=$env:PYTHONHASHSEED;intent=(Test-Path -LiteralPath (Join-Path $WorkingDirectory 'research/baseline_execution_2026-10-08/launch_hypermem_native_v4/launch_intent.json'))}
 [IO.File]::WriteAllText((Join-Path $WorkingDirectory 'intercepted.json'),($v|ConvertTo-Json -Depth 8))
 START_RESULT
}
'''.replace('PROCESS_RESULT',process).replace('START_RESULT',"throw 'INDEPENDENT_START_FAILURE'" if case=='start_failure' else '[pscustomobject]@{Id=12345}')
        source=source.replace("$ErrorActionPreference = 'Stop'","$ErrorActionPreference = 'Stop'\n"+stubs,1)
        script=root/KEY;script.parent.mkdir(parents=True);script.write_text(source,encoding='utf-8')
        codes={ENTRY:sha(entry),KEY:sha(script),'dependency.txt':sha(leaf)}
        upstream={'dependency.txt':sha(leaf)}
        sources={'proposal':bound(proposal)}
        policy={'six_exact_response_keys':sorted(r['request_hash'] for r in targets if r['conversation'] in (0,1,4,5,7,8)),
            'four_proposed_replacement_keys':sorted(r['request_hash'] for r in targets if r['conversation'] in (2,3,6,9))}
        proof=research/'terminal_candidate_v1.json';write(proof,{'fixture':True})
        review=research/'independent_candidate_review_v1.json'
        rv={'schema':'hypermem-native-v4-independent-candidate-review-v1','status':'passed_implementation_and_zero_api_preflight_only',
            'production_launch_authorized':False,'proof_sha256':sha(proof),'code_sha256':codes,'upstream_sha256':upstream,
            'source_bindings':sources,'continuation_policy':policy,'hash_environment':hash_env,'evidence_sha256':{'dependency.txt':sha(leaf)}}
        write(review,rv)
        grant=research/'human_authorization_v1.json'
        gv={'schema':'hypermem-native-v4-explicit-human-grant-v1','status':'authorized_by_explicit_user_instruction',
            'issued_by':'root_after_trusted_user_confirmation','human_confirmation':{'actor':'user','source':'direct_user_message',
                'message_id':'synthetic-not-human-approval','text':'fixture only','text_sha256':hashlib.sha256(b'fixture only').hexdigest()},
            'logical_limit_per_key':1,'old_consumed_key_permission':False,
            'old_consumed_key':'6758bd8d8dc914d7b627506008d436505d2cdcf597bf3ab15539eae1dd91fb6e',
            'implementation_review':bound(review),'terminal_proof':bound(proof),'source_bindings':sources,'code_sha256':codes,
            'accepted_compatibility_request_hashes':list(policy['six_exact_response_keys']),
            'replacements':[{k:v for k,v in t.items() if k!='original_raw'}|{'raw':{'path':t['original_raw']['source_path'],
                'sha256':t['original_raw']['observed_sha256']},'logical_limit':1} for t in targets if t['conversation'] in (2,3,6,9)]}
        if case=='wrong_text':gv['human_confirmation']['text']='changed'
        if case=='old_consumed_key':gv['old_consumed_key']='0'*64
        if case=='missing_compatibility':gv['accepted_compatibility_request_hashes'].pop()
        if case=='missing_replacement':gv['replacements'].pop()
        if case=='wrong_replacement_signature':gv['replacements'][0]['signature']={'body':{'changed':True}}
        if case=='wrong_replacement_raw':gv['replacements'][0]['raw']['sha256']='0'*64
        write(grant,gv)
        manifest=run/'run_manifest.json'
        mv={'schema':'hypermem-native-v4-continuation-candidate-manifest-v1','new_run':str(run),'review':bound(review),
            'terminal_proof':bound(proof),'human_grant':bound(grant),'code_sha256':codes,'upstream_sha256':upstream,
            'source_bindings':sources,'continuation_policy':policy,'hash_environment':hash_env}
        if case=='wrong_review':mv['review']['sha256']='0'*64
        if case=='wrong_proof':mv['terminal_proof']['sha256']='0'*64
        if case=='wrong_grant':mv['human_grant']['sha256']='0'*64
        if case=='wrong_seed':mv['hash_environment']={**hash_env,'seed':'779'}
        if case=='wrong_launcher':mv['code_sha256']={**codes,KEY:'0'*64}
        if case=='foreign_evidence':
            rv['evidence_sha256']={'D:/not_the_workspace/no_read.txt':'0'*64};write(review,rv)
            gv['implementation_review']=bound(review);write(grant,gv);mv['review']=bound(review);mv['human_grant']=bound(grant)
        write(manifest,mv)
        if case=='work_exists':(run/'work').mkdir()
        if case=='launch_exists':launch.mkdir()
        if case=='changed_evidence':leaf.write_text('changed',encoding='utf-8')
        params={'ManifestSha256':sha(manifest),'EntrySha256':sha(entry),'ReviewSha256':sha(review),'ProofSha256':sha(proof),'HumanGrantSha256':sha(grant)}
        if case=='no_grant':grant.unlink()
        wrapper=root/'invoke_isolated.ps1'
        argv=' '.join('-'+k+" '"+v+"'" for k,v in params.items())
        wrapper.write_text("$ErrorActionPreference='Stop'\n[Environment]::SetEnvironmentVariable('PYTHONHASHSEED','independent-prior-marker','Process')\n"
            +"try { & '"+str(script)+"' "+argv+"; $outcome='success' } catch { $outcome=$_.Exception.Message }\n"
            +"[pscustomobject]@{outcome=$outcome;seed=$env:PYTHONHASHSEED}|ConvertTo-Json -Compress\n",encoding='utf-8')
        shell=shutil.which('pwsh.exe');self.assertIsNotNone(shell)
        result=subprocess.run([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(wrapper)],
            cwd=root,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        outcome=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(outcome['seed'],'independent-prior-marker')
        return root,run,launch,outcome

    def test_valid_and_failed_start_restore_environment_and_never_spawn(self):
        for case in ('valid','start_failure'):
            with self.subTest(case=case):
                root,run,launch,outcome=self.fixture(case)
                self.assertEqual(outcome['outcome'],'success' if case=='valid' else 'INDEPENDENT_START_FAILURE')
                call=json.loads((root/'intercepted.json').read_text())
                self.assertEqual(call['args'],['-B',ENTRY,'run']);self.assertEqual(call['seed'],'780')
                self.assertEqual(call['style'],'Hidden');self.assertEqual(call['cwd'],str(root));self.assertTrue(call['intent'])
                self.assertEqual(sorted(p.name for p in run.iterdir()),['run_manifest.json'])
                self.assertEqual((launch/'receipt.json').exists(),case=='valid')

    def test_all_early_gates_reject_before_launch_directory_or_start(self):
        for case in ('no_grant','wrong_review','wrong_proof','wrong_grant','wrong_seed','wrong_launcher',
                'changed_evidence','foreign_evidence','work_exists','launch_exists','active','wrong_text','old_consumed_key',
                'missing_compatibility','missing_replacement','wrong_replacement_signature','wrong_replacement_raw'):
            with self.subTest(case=case):
                root,run,launch,outcome=self.fixture(case)
                self.assertNotEqual(outcome['outcome'],'success')
                self.assertFalse((root/'intercepted.json').exists())
                self.assertFalse((launch/'launch_intent.json').exists())
                self.assertFalse((launch/'receipt.json').exists())
                if case!='launch_exists':self.assertFalse(launch.exists())


if __name__=='__main__':unittest.main(verbosity=2)
