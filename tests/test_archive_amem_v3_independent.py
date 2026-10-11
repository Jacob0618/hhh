"""Independent archive-only evidence and PowerShell path safety regressions."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_amem_continuation_v3 as a


class IndependentArchiveTests(unittest.TestCase):
    def fixture(self, folder, *, corrupt=False, duplicate=False):
        root=Path(folder)
        paths=[root/name/'work/api_method/api_cache'/'same-request.json' for name in ('v1','v2','v3')]
        contents=[b'{"response":"original"}',b'{"response":"original"}',b'{"response":"replacement"}']
        records=[]
        blobs={}
        for path,raw in zip(paths,contents):
            digest=hashlib.sha256(raw).hexdigest()
            records.append({'path':str(path),'sha256':digest,'size':len(raw),'retained':True,'member':'blobs/'+digest})
            blobs['blobs/'+digest]=raw
        manifest={'schema':a.SCHEMA,'records':records}
        if duplicate:manifest['records'].append(records[0])
        mp=root/'manifest.json';mp.write_text(json.dumps(manifest),encoding='utf-8')
        target=root/'evidence.zip'
        with zipfile.ZipFile(target,'x') as archive:
            archive.writestr('archive_manifest.json',mp.read_bytes())
            for index,(member,raw) in enumerate(blobs.items()):
                archive.writestr(member,raw+b' ' if corrupt and index==0 else raw)
        return paths,contents,target,mp

    def test_same_request_key_across_roots_keeps_different_original_raws(self):
        with tempfile.TemporaryDirectory() as folder:
            paths,contents,target,mp=self.fixture(folder)
            self.assertTrue(all(not path.exists() for path in paths))
            with a.Resolver(target,mp) as resolver:
                self.assertEqual([resolver.bytes(path) for path in paths],contents)
                self.assertEqual(len(resolver.zip.namelist()),3)

    def test_crc_valid_but_wrong_sha_blob_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,target,mp=self.fixture(folder,corrupt=True)
            with self.assertRaisesRegex(ValueError,'blob SHA'):
                a.Resolver(target,mp)

    def test_duplicate_logical_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,target,mp=self.fixture(folder,duplicate=True)
            with self.assertRaisesRegex(ValueError,'Duplicate logical'):
                a.Resolver(target,mp)

    def test_review_evidence_sha_map_is_in_archive_closure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            evidence=root/'independent_native_semantics.json';evidence.write_text('{"status":"passed"}')
            report=root/'review.json';report.write_text(json.dumps({'evidence_sha256':{str(evidence):a.sha(evidence)}}))
            with patch.object(a,'ROOT',root):
                closure=a.closure([(str(report),a.sha(report))])
            self.assertEqual(closure,{str(report):a.sha(report),str(evidence):a.sha(evidence)})

    def test_real_powershell_path_helper_accepts_file_and_directory(self):
        source=Path(a.__file__).with_name('cleanup_amem_continuation_v3.ps1')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);child=root/'sub';child.mkdir();file=child/'proof.json';file.write_text('{}')
            command='''param([string]$Source,[string]$Root,[string]$File)
$ErrorActionPreference='Stop'
$workspace=$Root
$tokens=$null;$errors=$null
$tree=[System.Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw 'Script parse error' }
$fn=$tree.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'AssertUnderWorkspace'},$true)
if ($fn.Count -ne 1) { throw 'Path helper missing/ambiguous' }
Invoke-Expression $fn[0].Extent.Text
if ((AssertUnderWorkspace $File) -ne $File) { throw 'File helper differs' }
if ((AssertUnderWorkspace (Split-Path -LiteralPath $File)) -ne (Split-Path -LiteralPath $File)) { throw 'Directory helper differs' }
'''
            script=root/'probe.ps1';script.write_text(command,encoding='utf-8')
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),
                                   '-Source',str(source),'-Root',str(root),'-File',str(file)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
