"""Independent phase-2 checks; core is a reviewed dependency, not independently re-reviewed here."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_amem_retrieval_continuation_v1 as archive
from tools import run_five_baselines_reviewed_chain_v3 as dispatch
from tools.baselines_common_v1 import save_new,sha


class IndependentLifecycle(unittest.TestCase):
    def test_live_prearchive_path_passes_exact_result_review_to_identical_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            review=Path(tmp)/'review.json';save_new(review,{'status':'passed'})
            view=SimpleNamespace(manifest={'roots':{},'records':[]})
            def replay(value):
                self.assertIs(value,view)
                self.assertEqual(value.manifest['result_review'],{'path':str(review.resolve()),'sha256':sha(review)})
                return {'status':'passed'}
            with patch.object(archive.entry,'manifest',return_value=({},{})),patch.object(archive.engine,'LiveResolver',return_value=view),\
                 patch.object(archive,'read',return_value={}),patch.object(archive,'replay_all',side_effect=replay):
                self.assertEqual(archive.verify_native_seal(review),{'status':'passed'})

    def test_crc_valid_zip_cannot_discard_marker_or_original_text(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);roots={k:root/k for k in ('v1','v2','v3','v4')}
            stack.enter_context(patch.object(archive.engine,'ROOTS',roots))
            for index,p in enumerate((roots['v4']/'work/retrieval_continuation_v1/work/one_use_retrieval_replacement.json',
                                      roots['v3']/'work/native/c2/sessions/000453.json',
                                      roots['v4']/'work/retrieval_continuation_v1/native/c2/cache/audit.json')):
                mp=root/f'm{index}.json';zp=root/f'z{index}.zip'
                save_new(mp,{'schema':archive.SCHEMA,'records':[{'path':str(p),'sha256':'a'*64,'size':1,
                    'retained':False,'disposal_reason':'reconstructable numeric cache'}]})
                with zipfile.ZipFile(zp,'x') as z:z.writestr('archive_manifest.json',mp.read_bytes())
                with self.assertRaisesRegex(ValueError,'disposal'):archive.Resolver(zp,mp)

    def test_full1540_zip_replay_cannot_fallback_to_any_live_run_file(self):
        from tests.test_amem_retrieval_lifecycle_v1 import FullZipReplay
        original_open=Path.open;original_enter=archive.Resolver.__enter__;active=[False];blocked=[]
        def enter(resolver):
            value=original_enter(resolver);active[0]=True;return value
        def opened(path,*args,**kwargs):
            if active[0] and 'runs' in path.parts:
                blocked.append(str(path));raise AssertionError('A ZIP-only replay tried reading live run evidence: '+str(path))
            return original_open(path,*args,**kwargs)
        with patch.object(archive.Resolver,'__enter__',enter),patch.object(Path,'open',opened):
            FullZipReplay('test_actual_full1540_union_from_zip_with_no_original_work').test_actual_full1540_union_from_zip_with_no_original_work()
        self.assertTrue(active[0]);self.assertEqual(blocked,[])

    def test_failed_predecessor_stops_all_later_methods_before_dispatch(self):
        for method in dispatch.lifecycle.METHODS:
            with self.subTest(method=method),patch.object(dispatch.lifecycle,'stopped'),\
                 patch.object(dispatch.amem,'verify_completion',side_effect=ValueError('unaccepted union')),\
                 patch.object(dispatch.lifecycle,'verify_completion') as later:
                with self.assertRaisesRegex(ValueError,'unaccepted union'):dispatch.verify_predecessors(method)
                later.assert_not_called()

    def test_seed_certificate_counts_are_exact_integers_and_values_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs={};folder=Path(tmp)
            def put(path,value):
                raw=json.dumps(value,sort_keys=True).encode();docs[str(path)]=raw
                return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            view=SimpleNamespace(read=lambda p:json.loads(docs[str(p)]),sha=lambda p:hashlib.sha256(docs[str(p)]).hexdigest())
            state=put(folder/'state.json',{'memory_sha256':'original-memory'})
            vectors={'path':str(folder/'embeddings.npy'),'sha256':'a'*64}
            question={'question':'Which area flooded?'}
            proof={'bank':{'state':state,'embeddings':vectors},'missing_question':question}
            proof_ref=put(folder/'proof.json',proof)
            put(archive.FINAL/'run_manifest.json',{'proof':proof_ref})
            key=archive.entry.amem._v1._sha({'question':question['question'],'memory_sha256':'original-memory'})
            native={'seed_indices':list(range(10))}
            native_ref=put(archive.WORK_OWNER/'native/c2/retrieval'/(key+'.json'),native)
            capture_ref=put(archive.WORK_OWNER/'results/c2/retrieval/c2_q42.json',{'status':'complete','value':native})
            good={'schema':'amem-query-retrieval-seed-independent-review-v1','status':'passed','query_id':'c2_q42',
                  'model_calls':0,'credential_reads':0,'query_embedding_calls':1,'corpus_embedding_calls':0,
                  'query_embedding_recomputed':True,'source_inventory_before_after_equal':True,'full_result_equal':True,
                  'bank_state':state,'embeddings':vectors,'retrieval':capture_ref,'native_result':native_ref,'seed_indices':list(range(10))}
            seed_path=folder/'seed.json';review={'retrieval_seed_review':put(seed_path,good)}
            self.assertEqual(archive.verify_seed_review(review,view=view),review['retrieval_seed_review'])
            for field,value in [('model_calls',False),('credential_reads',False),('query_embedding_calls',True),
                                ('corpus_embedding_calls',False),('query_embedding_calls',2),('full_result_equal',1)]:
                with self.subTest(field=field,value=value):
                    altered={**good,field:value};review={'retrieval_seed_review':put(seed_path,altered)}
                    with self.assertRaises(ValueError):archive.verify_seed_review(review,view=view)

    def test_actual_single_root_powershell_v3_guard_delete_and_completion(self):
        from tests import test_baseline_lifecycle_v2 as factory
        from tools import baseline_lifecycle_v3 as version
        original=Path(__file__).resolve().parents[1]
        with patch.object(factory,'life',version.engine):
            fixture=factory.LifecycleTests();fixture.setUp()
            try:
                fixture.fixture();root=fixture.root
                self.assertTrue((fixture.folder/'work').resolve().is_relative_to(root.resolve()))
                stub=root/'independent_fixture_verify.py'
                stub.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(original))+")\n"
                    "from tools import baseline_lifecycle_v3 as module\nx=module.engine\nx.ROOT=Path("+repr(str(root))+")\n"
                    "x.RUN=x.ROOT/'runs/five_baselines_gemini25_v1_20261008'\nx.stopped=lambda:None\n"
                    "x.base.manifest=lambda m:x.read(x.folder(m)/'run_manifest.json')\n"
                    "x.replay_all=lambda r:{'status':'passed','fixture_only':True,'model_calls':0}\nx.main()\n",encoding='utf8')
                quote=lambda p:"'"+str(p).replace("'","''")+"'"
                script=(original/'tools/cleanup_reviewed_chain_v3.ps1').read_text(encoding='utf8')
                script=script.replace("(Resolve-Path -LiteralPath 'D:\\jev mem').Path","(Resolve-Path -LiteralPath "+quote(root)+").Path")
                script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'",'$python = '+quote(sys.executable))
                script=script.replace("$helper = Join-Path $workspace 'tools\\baseline_lifecycle_v3.py'",'$helper = '+quote(stub))
                path=root/'independent_cleanup_fixture.ps1';needle="$ErrorActionPreference = 'Stop'"
                active="function Get-CimInstance { [pscustomobject]@{Name='python3.11.exe';CommandLine='python -m tools.run_five_baselines_reviewed_chain_v3 run --method simplemem'} }\n"
                path.write_text(script.replace(needle,active+needle),encoding='utf8')
                cmd=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(path),'-Method','simplemem','-CleanupReview',str(fixture.cleanup)]
                refused=subprocess.run(cmd,capture_output=True,text=True)
                self.assertNotEqual(refused.returncode,0);self.assertIn('controller is still active',refused.stderr)
                self.assertTrue((fixture.folder/'work').exists())
                path.write_text(script.replace(needle,'function Get-CimInstance { @() }\n'+needle),encoding='utf8')
                removed=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(removed.returncode,0,removed.stdout+removed.stderr)
                self.assertFalse((fixture.folder/'work').exists())
                self.assertEqual(version.engine.verify_completion('simplemem')['status'],'passed')
                # A receipt is not permission to delete or continue a second time.
                repeated=subprocess.run(cmd,capture_output=True,text=True)
                self.assertNotEqual(repeated.returncode,0);self.assertIn('Cleanup already started',repeated.stderr)
            finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
